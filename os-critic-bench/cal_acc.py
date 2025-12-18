import argparse
import json
import os
import re
from collections import defaultdict, Counter
from typing import Dict, Any, Iterable, List
import glob

DEFAULT_DIR = "your_jsonl_path"


def _normalize_markup(text: str) -> str:
    return text or ""

_VERIFICATION_PATTERNS = [
    re.compile(
        r'Verification:\s*Does this action contribute to the completion of the task\?\s*\(Yes/No\)\s*'
        r'(?:\*\*|\*|__|_|["\'`])?\s*(?:\r?\n|\r)\s*(?:\*\*|\*|__|_|["\'`])?\s*(Yes|No)\s*(?:\*\*|\*|__|_|["\'`])?',
        re.IGNORECASE | re.DOTALL
    ),
    re.compile(
        r'(?:\*\*|\*|__|_|`|")?\s*Verification\s*:(?:\*\*|\*|__|_|`|")?\s*Does this action contribute to the completion of the task\?\s*\(Yes/No\)\s*'
        r'(?:\*\*|\*|__|_|["\'`])?\s*(?:\r?\n|\r)\s*(?:\*\*|\*|__|_|["\'`])?\s*(Yes|No)\s*(?:\*\*|\*|__|_|["\'`])?',
        re.IGNORECASE | re.DOTALL
    ),
    re.compile(
        r'Verification:\s*Does this action contribute to\s+(?:the\s+)?completion of the task\?\s*\(Yes/No\)\s*(?:\*\*)?\s*(Yes|No)\s*(?:\*\*)?',
        re.IGNORECASE | re.MULTILINE
    ),
    # re.compile(
    #     r'Verification:\s*Does this action contribute to completion of the task\?\s*\(Yes/No\)\s*(?:\*\*)?\s*(Yes|No)\s*(?:\*\*)?',
    #     re.IGNORECASE | re.MULTILINE
    # ),
    re.compile(
        r'(?:\*\*|\*|__|_|`|")?\s*Verification\s*:(?:\*\*|\*|__|_|`|")?\s*Does this action contribute to the completion of the task\?\s*\(Yes/No\)\s*(?:\*\*)?\s*(Yes|No)\s*(?:\*\*)?',
        re.IGNORECASE | re.MULTILINE
    ),
    re.compile(
        r'Verification:\s*Does this action contribute to the completion of the task\?\s*\(\s*(Yes|No)\s*\)',
        re.IGNORECASE | re.DOTALL
    ),
    re.compile(
        r'(?:\*\*|\*|__|_|`|")?\s*Verification\s*:(?:\*\*|\*|__|_|`|")?\s*\s*'
        r'Does this action contribute to the completion of the task\?\s*\(\s*(Yes|No)\s*\)',
        re.IGNORECASE | re.DOTALL
    ),
    re.compile(
        r'Verification:\s*Does this action contribute to the completion of the task\?\s*\((?:Yes/No|Yes|No)\)\s*(Yes|No)?',
        re.IGNORECASE | re.DOTALL
    ),
    re.compile(r'Verification:\s*(Yes|No)\b', re.IGNORECASE | re.DOTALL),
    re.compile(r'Grade:\s*(Yes|No)\b', re.IGNORECASE | re.DOTALL),
    re.compile(r'task\?\s*(Yes|No)\b', re.IGNORECASE | re.DOTALL),
    re.compile(r'\"Correctness\":\s*(True|False)\b', re.IGNORECASE | re.DOTALL),
    re.compile(r"<score>\s*(Correct|Incorrect)\s*</score>", re.IGNORECASE | re.DOTALL),
]

def extract_verification(critic_output: str) -> str | None:
    text = _normalize_markup(critic_output)
    last = None
    for pat in _VERIFICATION_PATTERNS:
        for m in pat.finditer(text):
            ans = (m.group(1) or "").strip().rstrip('.*!"\'`_)').lower()
            if ans in ("yes", "no"):
                last = ans
            elif ans in ("true", "yes", "y", "1", "correct"):
                last = "yes"
            elif ans in ("false", "no", "n", "0", "incorrect"):
                last = "no"
    return last

def iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            yield json.loads(s)

def safe_bool(v) -> bool | None:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true", "yes", "y", "1", "correct"):  return True
        if s in ("false", "no", "n", "0", "incorrect"):  return False
    return None

def update_confusion(c: Counter, gold: bool, pred: bool):
    if gold and pred:
        c["TP"] += 1
    elif (not gold) and (not pred):
        c["TN"] += 1
    elif gold and (not pred):
        c["FN"] += 1
    elif (not gold) and pred:
        c["FP"] += 1

def metrics_from_confusion(c: Counter) -> Dict[str, float]:
    TP, TN, FP, FN = c["TP"], c["TN"], c["FP"], c["FN"]
    valid = TP + TN + FP + FN
    def div(a, b): return a / b if b > 0 else 0.0
    acc  = div(TP + TN, valid)
    prec = div(TP, TP + FP)
    rec  = div(TP, TP + FN)
    f1   = div(2 * prec * rec, prec + rec)
    spec = div(TN, TN + FP)
    npv  = div(TN, TN + FN)
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "specificity": spec,
        "npv": npv,
    }

def pretty_pct(x: float) -> str:
    return f"{x*100:.2f}%"

def guess_domain_from_filename(path: str) -> str | None:
    name = os.path.basename(path).lower()
    if "desktop" in name: return "desktop"
    if "mobile" in name or "android" in name or "ac" in name or "odyssey" in name: return "mobile"
    if "web" in name or "scalecua" in name or "guiact" in name: return "web"
    return None

def collect_paths(inputs: List[str]) -> List[str]:
    paths: List[str] = []
    for p in inputs:
        if not p:
            continue
        matched = glob.glob(p)
        if matched:
            for mp in matched:
                if os.path.isdir(mp):
                    for root, _dirs, files in os.walk(mp):
                        for fn in files:
                            if fn.endswith(".jsonl"):
                                paths.append(os.path.join(root, fn))
                elif os.path.isfile(mp) and mp.endswith(".jsonl"):
                    paths.append(mp)
            continue
        if os.path.isdir(p):
            for root, _dirs, files in os.walk(p):
                for fn in files:
                    if fn.endswith(".jsonl"):
                        paths.append(os.path.join(root, fn))
        elif os.path.isfile(p) and p.endswith(".jsonl"):
            paths.append(p)

    seen = set(); uniq = []
    for x in paths:
        if x not in seen:
            uniq.append(x); seen.add(x)
    return uniq

def evaluate(paths: List[str], fallback_domain_mode: str | None):
    per_domain_cf: Dict[str, Counter] = defaultdict(Counter)       
    per_domain_total_with_gold: Counter = Counter()                
    per_domain_invalid_as_error: Counter = Counter()              
    per_domain_invalid_fpfn: Dict[str, Counter] = defaultdict(Counter)  

    def norm_domain(v: Any, src_path: str) -> str:
        d = (v or "").strip().lower()
        if d in ("desktop", "mobile", "web"):
            return d
        if fallback_domain_mode and fallback_domain_mode in ("auto", "filename"):
            g = guess_domain_from_filename(src_path)
            if g:
                return g
        if fallback_domain_mode in ("desktop", "mobile", "web"):
            return fallback_domain_mode
        return "unknown"

    total_files = 0

    for p in paths:
        total_files += 1
        try:
            for obj in iter_jsonl(p):
                domain = norm_domain(obj.get("domain"), p)
                gold = safe_bool(obj.get("pred_label"))
                if gold is None:
                    continue  
                per_domain_total_with_gold[domain] += 1

                v = extract_verification(obj.get("critic_output", "") or "")
                if v in ("yes", "no"):
                    pred = (v == "yes")
                    update_confusion(per_domain_cf[domain], gold, pred)
                else:
                    print(obj.get("critic_output", ""))
                    
                    per_domain_invalid_as_error[domain] += 1
                    if gold:
                        per_domain_cf[domain]["FN"] += 1
                        per_domain_invalid_fpfn[domain]["FN"] += 1
                    else:
                        per_domain_cf[domain]["FP"] += 1
                        per_domain_invalid_fpfn[domain]["FP"] += 1
        except FileNotFoundError:
            continue

    overall_cf = Counter()
    for d in per_domain_cf:
        overall_cf.update(per_domain_cf[d])

    overall_invalid_as_error = sum(per_domain_invalid_as_error.values())
    overall_invalid_fpfn = Counter()
    for d, cnt in per_domain_invalid_fpfn.items():
        overall_invalid_fpfn.update(cnt)

    ordered_domains = ["desktop", "mobile", "web", "unknown"]

    print("\n=== Critic Evaluation (by domain) ===")
    for d in ordered_domains:
        c = per_domain_cf[d]
        total_with_gold = per_domain_total_with_gold[d]
        if total_with_gold == 0:
            continue
        m = metrics_from_confusion(c)
        inv = per_domain_invalid_as_error[d]
        inv_fp = per_domain_invalid_fpfn[d]["FP"]
        inv_fn = per_domain_invalid_fpfn[d]["FN"]
        print(f"\n[Domain: {d}]")
        print(f"  Files Scanned: {total_files}  (global count)")
        print(f"  Total (with gold): {total_with_gold}")
        print(f"  Confusion: TP={c['TP']}  TN={c['TN']}  FP={c['FP']}  FN={c['FN']}")
        print(f"  Accuracy:    {pretty_pct(m['accuracy'])}")
        print(f"  Precision:   {pretty_pct(m['precision'])}")
        print(f"  Recall:      {pretty_pct(m['recall'])}")
        print(f"  F1:          {pretty_pct(m['f1'])}")
        print(f"  Specificity: {pretty_pct(m['specificity'])}")
        print(f"  NPV:         {pretty_pct(m['npv'])}")
      
        print(f"  Invalid-as-error: {inv}  ({pretty_pct(inv / total_with_gold)})  |  split FP={inv_fp}, FN={inv_fn}")

    print("\n=== Critic Evaluation (overall) ===")
    total_overall_with_gold = sum(per_domain_total_with_gold.values())
    valid_overall = overall_cf["TP"] + overall_cf["TN"] + overall_cf["FP"] + overall_cf["FN"]
    overall_m = metrics_from_confusion(overall_cf)
    print(f"  Files Scanned: {total_files}")
    print(f"  Total (with gold): {total_overall_with_gold} | Counted in confusion: {valid_overall}")
    print(f"  Confusion: TP={overall_cf['TP']}  TN={overall_cf['TN']}  FP={overall_cf['FP']}  FN={overall_cf['FN']}")
    print(f"  Accuracy:    {pretty_pct(overall_m['accuracy'])}")
    print(f"  Precision:   {pretty_pct(overall_m['precision'])}")
    print(f"  Recall:      {pretty_pct(overall_m['recall'])}")
    print(f"  F1:          {pretty_pct(overall_m['f1'])}")
    print(f"  Specificity: {pretty_pct(overall_m['specificity'])}")
    print(f"  NPV:         {pretty_pct(overall_m['npv'])}")
    print(f"  Invalid-as-error (overall): {overall_invalid_as_error}  "
          f"({pretty_pct(overall_invalid_as_error / total_overall_with_gold if total_overall_with_gold else 0.0)})  "
          f"| split FP={overall_invalid_fpfn['FP']}, FN={overall_invalid_fpfn['FN']}")

def main():
    ap = argparse.ArgumentParser(description="Evaluate critic outputs against pred_label=True/False, grouped by domain.")
    ap.add_argument("--jsonl", nargs="*", default=[DEFAULT_DIR],
                    help="Can be multiple .jsonl files, directories, or wildcards; defaults to the default directory.")
    ap.add_argument("--fallback-domain", default="auto",
                    choices=["auto", "filename", "desktop", "mobile", "web", "unknown"],
                    help="Fallback strategy when a sample is missing the domain field: "
                         "auto/filename = infer from filename; or force a specific value.")
    args = ap.parse_args()

    paths = collect_paths(args.jsonl)
    if not paths:
        print(f"[WARN] No .jsonl files found, using default directory: {DEFAULT_DIR}")
        paths = collect_paths([DEFAULT_DIR])
    if not paths:
        raise SystemExit(f"[ERROR] No .jsonl files found even in the default directory: {DEFAULT_DIR}")

    evaluate(paths, args.fallback_domain)

if __name__ == "__main__":
    main()
