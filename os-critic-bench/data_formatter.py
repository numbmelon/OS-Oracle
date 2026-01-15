#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Helpers for converting unified raw benchmark examples into 'critic_messages', which is the model input.

Each example has fields like:
- episode_id
- domain: "desktop" | "mobile" | "web"
- image: str (current screenshot, relative path under IMG_ROOT)
- history_images: List[str]
- instruction: str
- history_instructions: List[str]
- history_actions: List[str]
- current_action: str
- pred_label, answer, orig_prediction, ...

build_critic_messages(example) returns a list of messages, e.g.:

[
    {"role": "system", "content": [...]},
    {"role": "user", "content": [...]},
]
"""

import os
import math
from typing import Dict, Any, List, Tuple, Optional
import re
import ast
import json
import warnings

from PIL import Image 

# Root directory where converted images are stored
IMG_ROOT = "test_jsonl"

# Image resize configuration (aligned with your smart_resize logic)
IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 2560 * 28 * 28
MAX_RATIO = 200


# ===================== Resize helpers =====================

def round_by_factor(number: int, factor: int) -> int:
    """Return the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Return the smallest integer >= 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Return the largest integer <= 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


def smart_resize(
    height: int,
    width: int,
    factor: int = IMAGE_FACTOR,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
) -> Tuple[int, int]:
    """
    Rescale the image so that:

    1. Both dimensions (height and width) are divisible by 'factor'.
    2. Total pixels is within [min_pixels, max_pixels].
    3. Aspect ratio is preserved as much as possible.

    Returns:
        (h_bar, w_bar) in (height, width) order.
    """
    if height <= 0 or width <= 0:
        raise ValueError(f"Invalid image size: height={height}, width={width}.")

    ratio = max(height, width) / min(height, width)
    if ratio > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {ratio}"
        )

    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))

    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)

    return h_bar, w_bar


# ===================== Common utilities =====================

def _abs_image_path(rel_path: str) -> str:
    """
    Convert a relative image path (e.g., 'images/ac/ep_11840_02.png')
    into an absolute path under IMG_ROOT.
    """
    if not rel_path:
        return rel_path
    return os.path.join(IMG_ROOT, rel_path)


def _build_action_history_block(history_actions: List[str]) -> str:
    """
    Build the 'Action History' text block.

    Note: we only include the action JSON/dict strings, no natural language
    instruction, as requested. If history is empty, we return 'None'.
    """
    if not history_actions:
        return "None"
    lines: List[str] = []
    for idx, act in enumerate(history_actions, start=1):
        act_str = (act or "").strip().strip(';"')
        lines.append(f"Step {idx}: {act_str}")
    return "\n".join(lines)


def _get_resized_resolution_from_image(image_abs: str, fallback: Tuple[int, int]) -> Tuple[int, int]:
    """
    Compute resized (width, height) for the example using smart_resize
    on the actual image file.

    Args:
        image_abs: absolute path to the image file.
        fallback: (width, height) tuple used when the image cannot be read.

    Returns:
        (rw, rh) = (resized_width, resized_height).
    """
    if not image_abs or not os.path.exists(image_abs):
        return fallback

    try:
        with Image.open(image_abs) as im:
            width, height = im.size  # PIL gives (width, height)
        # smart_resize expects (height, width) and returns (h_bar, w_bar)
        h_bar, w_bar = smart_resize(height, width)
        return w_bar, h_bar
    except Exception:
        # If anything goes wrong, just fall back
        return fallback

def _parse_action_str(s: str) -> Dict[str, Any]:
    """
    Parse an action string into a dict.
    Supports:
      - JSON dict string
      - Python-literal dict string (single quotes)
      - Strings with a trailing quote (common in some logs)
    """
    s = s.strip().strip(';"')

    # e.g. "{'a': 1}'" or "{'a': 1}\""
    if (s.endswith('"') and (s.count("{") == s.count("}"))):
        # try to strip only the last quote if it looks dangling
        s2 = s[:-1].rstrip()
        # only keep if it ends with '}' (a dict)
        if s2.endswith("}"):
            s = s2

    # 1) Try JSON first
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # 2) Try python literal eval (for "{'name': 'mobile_use', ...}")
    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    raise ValueError(f"Cannot parse action string: {s[:120]}...")


def _scale_coords_in_action(action: Dict[str, Any], resized_width: int, resized_height: int) -> Dict[str, Any]:
    """
    Convert coordinate / coordinate2 from pixel [w, h] to [w/W*1000, h/H*1000].
    """
    if not isinstance(resized_width, int) or not isinstance(resized_height, int) or resized_width <= 0 or resized_height <= 0:
        raise ValueError("resized_width and resized_height must be positive integers.")

    args = action.get("arguments")
    if not isinstance(args, dict):
        return action

    def _convert(key: str) -> None:
        v = args.get(key)
        if (
            isinstance(v, (list, tuple))
            and len(v) == 2
            and isinstance(v[0], (int, float))
            and isinstance(v[1], (int, float))
        ):
            x, y = v
            nx = int(round(x / resized_width * 1000))
            ny = int(round(y / resized_height * 1000))
            args[key] = [nx, ny]

    _convert("coordinate")
    _convert("coordinate2")
    return action


def normalize_actions_qwen3vl(
    history_actions: List[str],
    current_action: str,
    resized_width: int,
    resized_height: int,
) -> Tuple[List[str], str]:
    """
    Parse history_actions/current_action, normalize coordinates, and return updated strings.
    - history_actions: returned as JSON strings (if parsed successfully)
                      otherwise keep original string and emit a warning.
    - current_action: returned as JSON string (if parsed successfully)
                      otherwise keep original string and emit a warning.
    """
    new_history: List[str] = []

    for i, s in enumerate(history_actions):
        try:
            d = _parse_action_str(s)
            d = _scale_coords_in_action(d, resized_width, resized_height)
            new_history.append(json.dumps(d, ensure_ascii=False))
        except Exception as e:
            warnings.warn(
                f"[normalize_actions_qwen3vl] Failed to parse/normalize history_actions[{i}]. "
                f"Keeping original. Error: {type(e).__name__}: {e}"
            )
            new_history.append(s)

    try:
        cur = _parse_action_str(current_action)
        cur = _scale_coords_in_action(cur, resized_width, resized_height)
        new_current = json.dumps(cur, ensure_ascii=False)
    except Exception as e:
        warnings.warn(
            f"[normalize_actions_qwen3vl] Failed to parse/normalize current_action. "
            f"Keeping original. Error: {type(e).__name__}: {e}"
        )
        new_current = current_action

    return new_history, new_current


# ===================== Mobile (mobile_use) =====================

def build_critic_messages_for_mobile(example: Dict[str, Any], model_type=None) -> List[Dict[str, Any]]:
    """
    Build critic_messages for mobile (mobile_use) examples,
    following the original template structure.
    """
    # Resolve image path
    image_rel = example.get("image", "")
    image_abs = _abs_image_path(image_rel)

    # Basic fields
    instruction = (example.get("instruction") or "").strip()
    history_actions: List[str] = example.get("history_actions") or []
    current_action = (example.get("current_action") or "").strip()

    # Resolution: compute from actual image via smart_resize
    # Fallback for mobile if the image cannot be opened
    rw, rh = _get_resized_resolution_from_image(image_abs, fallback=(924, 2100))
    input_w, input_h = rw, rh
    if model_type == "qwen3-vl":
        history_actions, current_action = normalize_actions_qwen3vl(history_actions, current_action, rw, rh)
        input_w, input_h = 1000, 1000

    # Common critic description header
    header = (
        "You are an expert GUI task evaluator. Your role is to analyze the provided action in context, "
        "generate an insightful textual critique, and grade the action's effectiveness.\n\n"
        "Your evaluation must be based on the current screen image, the overall task instruction, and "
        "the history of previous actions.\n"
        "At the end of your critique, you must provide a final grade in this exact format: "
        "\"Verification: Does this action contribute to the completion of the task? (Yes/No) X\", "
        "where X is either Yes or No.\n"
    )

    # Action space specification for mobile_use (mostly copied from original, with resolution injected)
    action_space = (
        "The following is the action space specification, which defines the model's output format to be evaluated:\n"
        "{\n"
        "  \"type\": \"function\",\n"
        "  \"function\": {\n"
        "    \"name_for_human\": \"mobile_use\",\n"
        "    \"name\": \"mobile_use\",\n"
        "    \"description\": \"Use a touchscreen to interact with a mobile device, and take screenshots.\\n"
        "* This is an interface to a mobile device with touchscreen. You can perform actions like clicking, typing, swiping, etc.\\n"
        "* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions.\\n"
        f"* The screen's resolution is {input_w}x{input_h}.\\n"
        "* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges.\", \n"
        "    \"parameters\": {\n"
        "      \"properties\": {\n"
        "        \"action\": {\n"
        "          \"description\": \"The action to perform. The available actions are:\\n"
        "* `key`: Perform a key event on the mobile device.\\n"
        "    - This supports adb's `keyevent` syntax.\\n"
        "    - Examples: \\\"volume_up\\\", \\\"volume_down\\\", \\\"power\\\", \\\"camera\\\", \\\"clear\\\".\\n"
        "* `click`: Click the point on the screen with coordinate (x, y).\\n"
        "* `long_press`: Press the point on the screen with coordinate (x, y) for specified seconds.\\n"
        "* `swipe`: Swipe from the starting point with coordinate (x, y) to the end point with coordinates2 (x2, y2).\\n"
        "* `type`: Input the specified text into the activated input box.\\n"
        "* `system_button`: Press the system button.\\n"
        "* `open`: Open an app on the device.\\n"
        "* `wait`: Wait specified seconds for the change to happen.\\n"
        "* `terminate`: Terminate the current task and report its completion status.\",\n"
        "          \"enum\": [\"key\", \"click\", \"long_press\", \"swipe\", \"type\", \"system_button\", \"open\", \"wait\", \"terminate\"],\n"
        "          \"type\": \"string\"\n"
        "        },\n"
        "        \"coordinate\": {\n"
        "          \"description\": \"(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. "
        "Required only by `action=click`, `action=long_press`, and `action=swipe`.\",\n"
        "          \"type\": \"array\"\n"
        "        },\n"
        "        \"coordinate2\": {\n"
        "          \"description\": \"(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. "
        "Required only by `action=swipe`.\",\n"
        "          \"type\": \"array\"\n"
        "        },\n"
        "        \"text\": {\n"
        "          \"description\": \"Required only by `action=key`, `action=type`, and `action=open`.\",\n"
        "          \"type\": \"string\"\n"
        "        },\n"
        "        \"time\": {\n"
        "          \"description\": \"The seconds to wait. Required only by `action=long_press` and `action=wait`.\",\n"
        "          \"type\": \"number\"\n"
        "        },\n"
        "        \"button\": {\n"
        "          \"description\": \"Back means returning to the previous interface, Home means returning to the desktop, "
        "Menu means opening the application background menu, and Enter means pressing the enter. "
        "Required only by `action=system_button`\",\n"
        "          \"enum\": [\"Back\", \"Home\", \"Menu\", \"Enter\"],\n"
        "          \"type\": \"string\"\n"
        "        },\n"
        "        \"status\": {\n"
        "          \"description\": \"The status of the task. Required only by `action=terminate`.\",\n"
        "          \"type\": \"string\",\n"
        "          \"enum\": [\"success\", \"failure\"]\n"
        "        }\n"
        "      },\n"
        "      \"required\": [\"action\"],\n"
        "      \"type\": \"object\"\n"
        "    },\n"
        "    \"args_format\": \"Format the arguments as a JSON object.\"\n"
        "  }\n"
        "}\n"
    )

    # Note about swipe direction (mobile-specific)
    swipe_note = (
        "Note:\n"
        "For the `swipe` action, pay attention to direction. Screen coordinates use a top-left origin, so the change in "
        "coordinates follows the finger's drag, while the page/content moves in the opposite direction. For reference:\n"
        " - If y1 > y2 → swipe up; the page moves to show content higher up (the viewport shifts upward).\n"
        " - If y1 < y2 → swipe down; the page moves to show content lower down (the viewport shifts downward).\n"
        " - If x1 > x2 → swipe left; the page moves to the left.\n"
        " - If x1 < x2 → swipe right; the page moves to the right.\n"
        "When both axes change, decide the primary direction by the larger |Δx| or |Δy|; otherwise treat it as a diagonal swipe.\n"
    )

    # Action history block: only actions, no natural language
    history_block = _build_action_history_block(history_actions)

    # Assemble the full user text
    text_parts = [
        header,
        action_space,
        swipe_note,
        "Task Instruction:\n" + instruction,
        "Action History:\n" + history_block,
        "Current Action:\n" + current_action,
        "Now please generate critiques and evaluate correctness.",
    ]
    user_text = "\n\n".join(text_parts)

    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "You are a helpful assistant."}
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_abs},
                {"type": "text", "text": user_text},
            ],
        },
    ]
    return messages


# ===================== Desktop / Web (computer_use) =====================

def build_critic_messages_for_desktop(example: Dict[str, Any], model_type=None) -> List[Dict[str, Any]]:
    """
    Build critic_messages for desktop (computer_use) examples,
    following the original template structure.
    """
    # Resolve image path
    image_rel = example.get("image", "")
    image_abs = _abs_image_path(image_rel)

    instruction = (example.get("instruction") or "").strip()
    history_actions: List[str] = example.get("history_actions") or []
    current_action = (example.get("current_action") or "").strip()

    # Resolution: compute from actual image via smart_resize
    # Fallback for desktop/web if the image cannot be opened
    rw, rh = _get_resized_resolution_from_image(image_abs, fallback=(1176, 784))
    input_w, input_h = rw, rh
    if model_type == "qwen3-vl":
        history_actions, current_action = normalize_actions_qwen3vl(history_actions, current_action, rw, rh)
        input_w, input_h = 1000, 1000

    # Common critic description header (same as mobile)
    header = (
        "You are an expert GUI task evaluator. Your role is to analyze the provided action in context, "
        "generate an insightful textual critique, and grade the action's effectiveness.\n\n"
        "Your evaluation must be based on the current screen image, the overall task instruction, and "
        "the history of previous actions.\n"
        "At the end of your critique, you must provide a final grade in this exact format: "
        "\"Verification: Does this action contribute to the completion of the task? (Yes/No) X\", "
        "where X is either Yes or No.\n"
    )

    # Action space specification for computer_use
    action_space = (
        "The following is the action space specification, which defines the model's output format to be evaluated:\n"
        "{\n"
        "  \"type\": \"function\",\n"
        "  \"function\": {\n"
        "    \"name_for_human\": \"computer_use\",\n"
        "    \"name\": \"computer_use\",\n"
        "    \"description\": \"Use a mouse and keyboard to interact with a computer, and take screenshots.\\n"
        "* This is an interface to a desktop GUI. You do not have access to a terminal or applications menu. "
        "You must click on desktop icons to start applications.\\n"
        "* Some applications may take time to start or process actions, so you may need to wait and take successive "
        "screenshots to see the results of your actions.\\n"
        f"* The screen's resolution is {input_w}x{input_h}.\\n"
        "* Whenever you intend to move the cursor to click on an element like an icon, you should consult a screenshot "
        "to determine the coordinates of the element before moving the cursor.\\n"
        "* If you tried clicking on a program or link but it failed to load, even after waiting, try adjusting your "
        "cursor position so that the tip of the cursor visually falls on the element that you want to click.\\n"
        "* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. "
        "Don't click boxes on their edges.\",\n"
        "    \"parameters\": {\n"
        "      \"properties\": {\n"
        "        \"action\": {\n"
        "          \"description\": \"The action to perform. The available actions are:\\n"
        "* `key`: Performs key down presses on the arguments passed in order, then performs key releases in reverse order.\\n"
        "* `type`: Type a string of text on the keyboard.\\n"
        "* `mouse_move`: Move the cursor to a specified (x, y) pixel coordinate on the screen.\\n"
        "* `left_click`: Click the left mouse button at a specified (x, y) pixel coordinate on the screen.\\n"
        "* `left_click_drag`: Click and drag the cursor to a specified (x, y) pixel coordinate on the screen.\\n"
        "* `right_click`: Click the right mouse button at a specified (x, y) pixel coordinate on the screen.\\n"
        "* `middle_click`: Click the middle mouse button at a specified (x, y) pixel coordinate on the screen.\\n"
        "* `double_click`: Double-click the left mouse button at a specified (x, y) pixel coordinate on the screen.\\n"
        "* `scroll`: Performs a scroll of the mouse scroll wheel. Positive values scroll up, negative values scroll down.\\n"
        "* `wait`: Wait specified seconds for the change to happen.\\n"
        "* `terminate`: Terminate the current task and report its completion status.\",\n"
        "          \"enum\": [\"key\", \"type\", \"mouse_move\", \"left_click\", \"left_click_drag\", "
        "\"right_click\", \"middle_click\", \"double_click\", \"scroll\", \"wait\", \"terminate\"],\n"
        "          \"type\": \"string\"\n"
        "        },\n"
        "        \"keys\": {\n"
        "          \"description\": \"Required only by `action=key`.\",\n"
        "          \"type\": \"array\"\n"
        "        },\n"
        "        \"text\": {\n"
        "          \"description\": \"Required only by `action=type`.\",\n"
        "          \"type\": \"string\"\n"
        "        },\n"
        "        \"coordinate\": {\n"
        "          \"description\": \"(x, y): The x (pixels from the left edge) and y (pixels from the top edge) "
        "coordinates to interact with. Required for pointer actions like `left_click`, `mouse_move`, `left_click_drag`, etc.\",\n"
        "          \"type\": \"array\"\n"
        "        },\n"
        "        \"pixels\": {\n"
        "          \"description\": \"The amount of scrolling to perform. Positive values scroll up, negative values scroll down. "
        "Required only by `action=scroll`.\",\n"
        "          \"type\": \"number\"\n"
        "        },\n"
        "        \"time\": {\n"
        "          \"description\": \"The seconds to wait. Required only by `action=wait`.\",\n"
        "          \"type\": \"number\"\n"
        "        },\n"
        "        \"status\": {\n"
        "          \"description\": \"The status of the task. Required only by `action=terminate`.\",\n"
        "          \"type\": \"string\",\n"
        "          \"enum\": [\"success\", \"failure\"]\n"
        "        }\n"
        "      },\n"
        "      \"required\": [\"action\"],\n"
        "      \"type\": \"object\"\n"
        "    },\n"
        "    \"args_format\": \"Format the arguments as a JSON object.\"\n"
        "  }\n"
        "}\n"
    )

    history_block = _build_action_history_block(history_actions)

    text_parts = [
        header,
        action_space,
        "Task Instruction:\n" + instruction,
        "Action History:\n" + history_block,
        "Current Action:\n" + current_action,
        "Now please generate critiques and evaluate correctness.",
    ]
    user_text = "\n\n".join(text_parts)

    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "You are a helpful assistant."}
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_abs},
                {"type": "text", "text": user_text},
            ],
        },
    ]
    return messages


def build_critic_messages_for_web(example: Dict[str, Any], model_type=None) -> List[Dict[str, Any]]:
    """
    Build critic_messages for web examples.

    Web uses the same action space (computer_use) and overall structure as desktop,
    so we simply delegate to build_critic_messages_for_desktop.
    """
    return build_critic_messages_for_desktop(example, model_type=None)




_GUI_CRITIC_R1_PREFIX = (
    "There is a multimodal agent that can perform a series of actions on a smart device (phone or PC) to automate the completion of user instructions. "
    "Possible actions include \"click\" / \"left_click\" at (x,y) position, \"long press\" at (x,y) position, \"swipe\" from (x1,y1) to (x2,y2), "
    "\"scroll\" down or up, \"drag\" from (x1,y1) to (x2,y2), \"type\" (text content), \"back\", \"home\", \"enter\" and so on. \n\n"
    "User instructions are usually complex and may include several detailed requirements. In some steps, the action decided by the mobile agent may be wrong.\n"
    "Now, you are a critic model used to evaluate the agent's decision. I will provide you with the following information:\n"
    "1. User instruction.\n"
    "2. History: The action history of the agent in the previous steps. \n"
    "3. Decision: The decision of the agent for this step.\n"
    "4. Image: The screenshot before executing this action. If the action contains positional parameters (such as click and swipe), the interaction area is marked with a translucent red circle or red arrow. \n\n"
    "Firstly, you need to understand the purpose of the decision. Pay attention to analyzing the interface elements in the screenshot (such as button position, text content, etc.). "
    "If there are red marks, focus on the action position. You can take appropriate account of the history information. \n"
    "Then, based on the given information, carefully analyze the decision given by the agent for the current step:\n"
    "1. Decision Analysis\n"
    "(1). Observation: Observe the screenshot and analyze the state without considering the user's instruction.\n"
    "- Focus on the operable or informative elements related to the operational decision. \n"
    "(2). Possible Result: Speculate the most possible result of executing this decision.\n"
    "- Predicts the screenshot change after the operation.\n"
    "- Whether to promote the progress of core tasks.\n"
    "(3). Critique: Determine whether the decision is correct and explain why.\n"
    "- Focus on historical operations. \n"
    "- Based on the previous analysis and the history, determine if this decision supports the completion of the instruction. \n"
    "- Only perform actions specified in the instructions. \n"
    "- Home button is the correct choice for switching apps. \n"
    "- Both clicking a suggestion and Enter are correct when searching. \n\n"
    "2. Based on the above analysis, determine whether this decision is \"Correct\" or \"Incorrect\".\n"
    "3. Reflection: If correct, retell the action; if incorrect, suggest a better action. Propose a one-step action for the current obversation, "
    "like click, swipe (with direction), type (with information), Home, Back, or Terminate (in 20 words).\n\n"
)

_GUI_CRITIC_R1_SUFFIX = (
    "\n\nAssess the current decision's correctness in the following format:\n"
    "<thinking>\n"
    "**Observation**: Describe the screenshot.\n"
    "**Possible Result**: Analysis from the possible result perspective.\n"
    "**Critique**: Criticize why the decision is correct or incorrect.\n"
    "</thinking>\n"
    "<score>\n"
    "Correct or Incorrect\n"
    "</score>\n"
    "<suggestion>\n"
    "If correct, provide a brief summary; if incorrect, suggest a better decision briefly.\n"
    "</suggestion>"
)


def _lower(x: Any) -> str:
    return str(x).strip().lower() if x is not None else ""


def _ensure_list(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return ["" if v is None else str(v) for v in x]
    if isinstance(x, str):
        return [x]
    return [str(x)]


def _strip_noise(s: str) -> str:
    s = (s or "").strip()
    s = s.strip(";")
    s = s.strip()
    if s.endswith('\\";') or s.endswith('";'):
        s = s[:-2].strip()
    if s.endswith('"') and s.count("{") >= 1:
        s = s[:-1].strip()
    return s


def _try_parse_json_obj(s: str) -> Optional[Dict[str, Any]]:
    if not isinstance(s, str) or not s.strip():
        return None
    t = s.strip()
    t = re.sub(r"^\s*<tool_call>\s*", "", t)
    t = re.sub(r"\s*</tool_call>\s*$", "", t)
    t = t.strip()

    try:
        obj = json.loads(t)
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None
    return None


def _extract_r1_decision(current_action: Any) -> Dict[str, Any]:
    """
    Returns a dict like {"action": "...", "coordinate": [...]} when possible.
    Falls back to {"raw": "..."} if parsing fails.
    """
    if current_action is None:
        return {}

    if isinstance(current_action, dict):
        obj = current_action
    elif isinstance(current_action, str):
        obj = _try_parse_json_obj(current_action)
        if obj is None:
            return {"raw": current_action.strip()}
    else:
        return {"raw": repr(current_action)}

    args = obj.get("arguments")
    if isinstance(args, dict) and isinstance(args.get("action"), str):
        out: Dict[str, Any] = {"action": args.get("action")}
        if "coordinate" in args:
            out["coordinate"] = args.get("coordinate")
        if "text" in args:
            out["text"] = args.get("text")
        if "pixels" in args:
            out["pixels"] = args.get("pixels")
        if "time" in args:
            out["time"] = args.get("time")
        if "status" in args:
            out["status"] = args.get("status")
        return out

    if isinstance(obj.get("action"), str):
        out2: Dict[str, Any] = {"action": obj.get("action")}
        if "coordinate" in obj:
            out2["coordinate"] = obj.get("coordinate")
        if "text" in obj:
            out2["text"] = obj.get("text")
        return out2

    return {"raw": obj}


def _build_r1_history_block(history_instructions: List[str], history_actions: List[str]) -> str:
    n = max(len(history_instructions), len(history_actions))
    if n == 0:
        return " \n"

    lines: List[str] = []
    for i in range(n):
        inst = history_instructions[i] if i < len(history_instructions) else ""
        act = history_actions[i] if i < len(history_actions) else ""
        inst = _strip_noise(inst)
        act = _strip_noise(act)

        step_text = inst if inst != "" else act
        step_text = step_text.strip()
        if step_text == "":
            continue

        prefix = " Step " if len(lines) == 0 else "; Step "
        lines.append(f"{prefix}{i+1}: {step_text}\n")

    if not lines:
        return " \n"

    return "".join(lines) + "; \n"


def build_critic_messages_for_gui_critic_r1(example: Dict[str, Any]) -> List[Dict[str, Any]]:
    user_instruction = (example.get("instruction") or example.get("user_instruction") or "").strip()

    history_instructions = _ensure_list(example.get("history_instructions"))
    history_actions = _ensure_list(example.get("history_actions"))

    history_block = _build_r1_history_block(history_instructions, history_actions)

    decision_obj = _extract_r1_decision(example.get("current_action"))
    decision_line = f"Action: {repr(decision_obj)}"

    prompt = (
        _GUI_CRITIC_R1_PREFIX
        + "Below is the information for the current step:\n"
        + "1. User instruction:\n"
        + f"{user_instruction}\n"
        + "2. History:\n"
        + history_block
        + "3. Decision:\n"
        + f"{decision_line}\n"
        + "4. Image is the screenshot of this step."
        + _GUI_CRITIC_R1_SUFFIX
    )

    image_rel = example.get("image", "")
    image_abs = _abs_image_path(image_rel)

    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant."}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_abs},
                {"type": "text", "text": prompt},
            ],
        },
    ]



# ===================== Unified entry =====================

def build_critic_messages(example: Dict[str, Any], model_type=None) -> List[Dict[str, Any]]:
    """
    Unified entry: choose the appropriate builder according to example['domain'].
    """
    domain = example.get("domain")
    # return build_critic_messages_for_gui_critic_r1(example)
    if domain == "desktop":
        return build_critic_messages_for_desktop(example, model_type)
    elif domain == "mobile":
        return build_critic_messages_for_mobile(example, model_type)
    elif domain == "web":
        return build_critic_messages_for_web(example, model_type)
    else:
        raise ValueError(
            f"Unknown domain for example: {domain}, episode_id={example.get('episode_id')}"
        )
