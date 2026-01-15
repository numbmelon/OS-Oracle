import os

# Prefer stable CUDA behavior
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True,max_split_size_mb:512"
)

import re
import json
import time
import hashlib
import argparse
from typing import Optional, Tuple, Dict, Any, List
from pathlib import Path

from tqdm import tqdm

from data_formatter import build_critic_messages  # NEW: import message builder


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Unified critic evaluation for desktop/mobile/web with sharding and TP auto-handling."
    )

    p.add_argument("--web_jsonl",    default=f"test_jsonl/web.jsonl")
    p.add_argument("--mobile_jsonl", default=f"test_jsonl/mobile.jsonl")
    p.add_argument("--desktop_jsonl", default=f"test_jsonl/desktop.jsonl")

    p.add_argument(
        "--critic_backend", choices=["oai", "qwen2_5-vl", "qwen3-vl"], default="oai",
        help="Critic backend: OpenAI GPT-4o (default) or local Qwen."
    )
    p.add_argument(
        "--critic_model_path",
        default="Qwen2.5-VL-7B-Instruct",
        help="Critic model path (not used  when --critic_backend=oai)."
    )
    p.add_argument(
        "--results_root",
        default=f"results",
        help="Root directory for outputs; final file is {results_root}/{checkpoint_name}/critic_merged.jsonl."
    )
    p.add_argument(
        "--checkpoint_name",
        default="",
        help="Directory name for results; if empty, use basename of critic_model_path."
    )

    p.add_argument("--rank", type=int, default=None, help="Global rank of this process (default: read from RANK).")
    p.add_argument("--world_size", type=int, default=None, help="Global world size (default: read from WORLD_SIZE).")
    p.add_argument("--wait_interval", type=int, default=10, help="Polling interval (seconds) when rank 0 waits flags.")
    p.add_argument("--wait_timeout", type=int, default=6000000000, help="Timeout (seconds) for rank 0 waiting flags.")

    p.add_argument(
        "--tp_size", type=int, default=0,
        help="Explicit tensor parallel size; 0 means auto (32B-like → 2, otherwise 1)."
    )

    return p


def _parse_visible_cuda_devices() -> List[int]:
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not cvd:
        return list(range(8))
    out: List[int] = []
    for tok in cvd.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            out.append(int(tok))
        except ValueError:
            pass
    return out if out else list(range(8))


def _is_big_model_32b_like(path: str) -> bool:
    s = (path or "").lower()
    return "32b" in s or "-32b" in s or "32b_" in s or "b32" in s


def _decide_tp_size(args, model_path: str) -> int:
    if args.tp_size and args.tp_size > 0:
        return args.tp_size
    return 2 if _is_big_model_32b_like(model_path) else 1


def _choose_two_gpus_for_process(visible: List[int], local_rank: int) -> List[int]:
    n = max(1, len(visible))
    a_idx = (2 * local_rank) % n
    b_idx = (a_idx + 1) % n
    return [visible[a_idx], visible[b_idx]]


def _prepare_device_env_for_backend(args) -> Dict[str, Any]:
    """
    Prepare CUDA-related environment variables for the critic backend.
    Returns a small dict with debug info about the chosen devices.
    """
    if os.environ.get("CRITIC_SILENT_RANK", "0") == "1":
        return {"mode": "silent"}

    info: Dict[str, Any] = {}
    critic_model_path = os.environ.get("CRITIC_MODEL_PATH", args.critic_model_path)
    tp_size = _decide_tp_size(args, critic_model_path)
    local_rank = int(
        os.environ.get(
            "LOCAL_RANK",
            os.environ.get("SLURM_LOCALID", os.environ.get("PMI_LOCAL_RANK", 0)),
        )
    )

    info.update(
        {
            "model_path": critic_model_path,
            "tp_size": tp_size,
            "local_rank": local_rank,
            "visible_before": _parse_visible_cuda_devices(),
        }
    )

    if tp_size <= 1:
        info["mode"] = "single"
        return info

    pair = _choose_two_gpus_for_process(info["visible_before"], local_rank)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, pair))
    os.environ["CRITIC_TP_SIZE"] = "2"
    os.environ["VLLM_TENSOR_PARALLEL_SIZE"] = "2"

    info["mode"] = "tp2"
    info["chosen_pair"] = pair
    info["visible_after"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    return info


def load_lines_from_jsonl(path: str, domain: str) -> List[Dict[str, Any]]:
    """
    Load a jsonl file and attach a 'domain' field to each example if missing.
    """
    if not path or not os.path.exists(path):
        return []
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for l in f:
            l = l.strip()
            if not l:
                continue
            obj = json.loads(l)
            if "domain" not in obj:
                obj["domain"] = domain
            out.append(obj)
    return out


def extract_action_from_prediction(prediction: str) -> Optional[str]:
    """
    Utility for extracting tool_call payload from a string.
    Kept for backward compatibility; you may ignore it if unused.
    """
    if not isinstance(prediction, str):
        return None
    m = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", prediction, flags=re.S | re.I)
    return m.group(1).strip() if m else prediction.strip()


def init_predictor(args):
    """
    Initialize critic backend (OpenAI or Qwen).
    """
    if os.environ.get("CRITIC_SILENT_RANK", "0") == "1":
        return None
    from inferencer import Qwen25VLBaseInferencer, OaiInferencer, Qwen3VLBaseInferencer

    if args.critic_backend in ["qwen2_5-vl", "qwen3-vl"]:
        critic_model_path = os.environ.get("CRITIC_MODEL_PATH", args.critic_model_path)
        os.environ.setdefault("CRITIC_MODEL_PATH", critic_model_path)
        os.environ.setdefault("CRITIC_BACKEND", args.critic_backend)
        if args.critic_backend == "qwen2_5-vl":
            return Qwen25VLBaseInferencer(critic_model_path)
        elif args.critic_backend == "qwen3-vl":
            return Qwen3VLBaseInferencer(critic_model_path)
    else:
        # You can change the default model name here if needed.
        return OaiInferencer(model_name="gemini-2.5-pro")


def _append_jsonl_lines(path: Path, rows: List[Dict[str, Any]]):
    """
    Write a list of dicts into a jsonl file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")



def main():
    args = build_argparser().parse_args()

    # Distributed / multi-node info
    rank_env = args.rank if args.rank is not None else int(os.environ.get("RANK", 0))
    world_size_env = (
        args.world_size if args.world_size is not None else int(os.environ.get("WORLD_SIZE", 1))
    )
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    node_rank = int(os.environ.get("NODE_RANK", 0))
    local_world_size = int(
        os.environ.get("LOCAL_WORLD_SIZE", os.environ.get("NPROC_PER_NODE", "8"))
    )

    def _is_32b_like(p: str) -> bool:
        s = (p or "").lower()
        return "32b" in s or "-32b" in s or "32b_" in s or "b32" in s or "72b" in s

    tp2 = _is_32b_like(os.environ.get("CRITIC_MODEL_PATH", args.critic_model_path)) or (
        getattr(args, "tp_size", 0) == 2
    )

    # For TP2 we may want only half of local ranks to be "active" critic ranks
    is_active = True
    if tp2 and local_world_size > 4:
        is_active = local_rank < (local_world_size // 2)

    effective_local_world = local_world_size if not tp2 else max(1, local_world_size // 2)
    est_nodes = max(1, world_size_env // max(1, local_world_size))
    effective_world = est_nodes * effective_local_world
    effective_rank = node_rank * effective_local_world + (
        local_rank if not tp2 else (local_rank % effective_local_world)
    )

    os.environ["CRITIC_SILENT_RANK"] = "0" if is_active else "1"

    device_info = _prepare_device_env_for_backend(args)
    print(
        "[Boot] PID=",
        os.getpid(),
        " RANK=",
        rank_env,
        " WORLD=",
        world_size_env,
        " CUDA_VISIBLE_DEVICES(before->after)=",
        device_info.get("visible_before"),
        "->",
        os.environ.get("CUDA_VISIBLE_DEVICES"),
        " LOCAL_RANK=",
        local_rank,
        " LOCAL_WORLD_SIZE=",
        local_world_size,
        " NODE_RANK=",
        node_rank,
        " MODE=",
        device_info.get("mode"),
        " TP_SIZE=",
        device_info.get("tp_size"),
        " CHOSEN_PAIR=",
        device_info.get("chosen_pair"),
        " MODEL_PATH=",
        device_info.get("model_path"),
        flush=True,
    )

    # Load raw eval data (already in unified "raw" format)
    web_lines = load_lines_from_jsonl(args.web_jsonl, "web")
    mobile_lines = load_lines_from_jsonl(args.mobile_jsonl, "mobile")
    desktop_lines = load_lines_from_jsonl(args.desktop_jsonl, "desktop")

    all_lines: List[Dict[str, Any]] = []
    all_lines.extend(web_lines)
    all_lines.extend(mobile_lines)
    all_lines.extend(desktop_lines)

    if not all_lines:
        raise FileNotFoundError("No input samples found from the three jsonl files.")

    ckpt_name = args.checkpoint_name.strip() or Path(args.critic_model_path).name
    result_dir = Path(args.results_root) / ckpt_name
    parts_dir = result_dir / "parts"
    flags_dir = result_dir / "flags"
    parts_dir.mkdir(parents=True, exist_ok=True)
    flags_dir.mkdir(parents=True, exist_ok=True)

    out_part_path = parts_dir / f"critic_rank{rank_env}.jsonl"
    flag_path = flags_dir / f"flag_rank{rank_env}.txt"
    merged_path = result_dir / "critic_merged.jsonl"

    if not is_active:
        # Inactive rank: produce empty part and mark flag as "silent"
        out_part_path.touch()
        flag_path.write_text("silent", encoding="utf-8")
        print(
            f"[Rank {rank_env}] Silent rank. Created empty part & flag and exit.",
            flush=True,
        )
        return

    model = init_predictor(args)

    # Simple sharding: strided slice over all_lines
    shard_world = effective_world
    shard_rank = effective_rank
    shard = all_lines[shard_rank::shard_world]

    results: List[Dict[str, Any]] = []
    for ex in tqdm(shard, desc=f"Rank {rank_env} unified critic"):
        ep_id = ex.get("episode_id")
        try:
            # Build critic_messages from the new raw-format fields (image, history, etc.)
            messages = build_critic_messages(ex, model_type=args.critic_backend)

            critic_result = model.predict(messages)

            orig_pred = ex.get("orig_prediction")
            out_obj = {
                "episode_id": ep_id,
                "domain": ex.get("domain"),
                "critic_messages": messages,
                "critic_output": critic_result,
                "orig_prediction": (
                    orig_pred if orig_pred is not None else ex.get("prediction")
                ),
                "answer": ex.get("answer"),
                "image_width": ex.get("image_width"),
                "image_height": ex.get("image_height"),
                "resized_image_width": ex.get("resized_image_width"),
                "resized_image_height": ex.get("resized_image_height"),
                "pred_label": ex.get("pred_label"),
            }
            results.append(out_obj)
        except Exception as e:
            print(f"[Rank {rank_env}] Error in {ep_id}: {e}", flush=True)

    _append_jsonl_lines(out_part_path, results)
    flag_path.write_text("done", encoding="utf-8")

    # Rank 0 merges all parts and performs a late-fix if some episodes are missing
    if rank_env == 0:
        print("[Rank 0] Waiting for all critic rank flag files...", flush=True)
        waited = 0
        while waited < args.wait_timeout:
            all_ready = True
            for i in range(world_size_env):
                fp = flags_dir / f"flag_rank{i}.txt"
                if not fp.exists():
                    all_ready = False
                    break
            if all_ready:
                break
            time.sleep(args.wait_interval)
            waited += args.wait_interval
        else:
            print("[Rank 0] Timeout waiting flags. Proceed to merge.", flush=True)

        merged_rows: List[Dict[str, Any]] = []
        for i in range(world_size_env):
            pp = parts_dir / f"critic_rank{i}.jsonl"
            if pp.exists():
                with pp.open("r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                merged_rows.append(json.loads(line))
                            except Exception:
                                pass
        _append_jsonl_lines(merged_path, merged_rows)
        print("[Rank 0] Merge complete:", merged_path, flush=True)

        # Late-fix: detect missing episode_ids and run critic on them on rank 0
        expected_examples = all_lines
        id2ex: Dict[str, Dict[str, Any]] = {}
        expected_ids: set[str] = set()
        for ex in expected_examples:
            eid = ex.get("episode_id")
            if not eid or eid in id2ex:
                continue
            id2ex[eid] = ex
            expected_ids.add(eid)

        actual_ids: set[str] = set()
        for row in merged_rows:
            eid = row.get("episode_id")
            if eid:
                actual_ids.add(eid)

        missing_ids = list(expected_ids - actual_ids)
        if missing_ids:
            print(
                f"[Rank 0] Missing {len(missing_ids)} episodes. Running late-fix...",
                flush=True,
            )
            missing_examples = [id2ex[eid] for eid in missing_ids if eid in id2ex]

            late_rows: List[Dict[str, Any]] = []
            for ex in tqdm(missing_examples, desc="Rank0 late-fix critic"):
                ep_id = ex.get("episode_id")
                try:
                    messages = build_critic_messages(ex, model_type=args.critic_backend)
                    critic_result = model.predict(messages)
                    orig_pred = ex.get("orig_prediction")
                    late_rows.append(
                        {
                            "episode_id": ep_id,
                            "domain": ex.get("domain"),
                            "critic_messages": messages,
                            "critic_output": critic_result,
                            "orig_prediction": (
                                orig_pred
                                if orig_pred is not None
                                else ex.get("prediction")
                            ),
                            "answer": ex.get("answer"),
                            "image_width": ex.get("image_width"),
                            "image_height": ex.get("image_height"),
                            "resized_image_width": ex.get("resized_image_width"),
                            "resized_image_height": ex.get("resized_image_height"),
                            "pred_label": ex.get("pred_label"),
                        }
                    )
                except Exception as e:
                    print(f"[Rank0 LateFix] Error in {ep_id}: {e}", flush=True)

            with merged_path.open("a", encoding="utf-8") as f:
                for r in late_rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print("[Rank 0] Late-fix appended. All done.", flush=True)
        else:
            print("[Rank 0] No missing episodes. All good.", flush=True)


if __name__ == "__main__":
    main()
