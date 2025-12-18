import os
import io
import json
import re
import base64
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Tuple, Union, Any, Dict, Optional
import random

import torch
from PIL import Image, ImageDraw
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoModelForImageTextToText

from qwen_vl_utils import process_vision_info
from data_formatter import smart_resize

from qwen_agent.tools.base import BaseTool, register_tool
import mimetypes
import time

import anthropic
# from __future__ import annotations


class QwenVLInferencerBase(ABC):
    """
    Abstract base class for Qwen2.5-VL-style inferencers.
    Handles model and processor initialization.
    Subclasses must implement the predict() method.
    """

    def __init__(self, model_path: str, min_pixels: int = 3136, max_pixels: int = 2007040):
        """
        Initialize the model and processor from the specified path.

        Args:
            model_path (str): Path to the pretrained Qwen2.5-VL model.
            min_pixels (int): Minimum image pixel count.
            max_pixels (int): Maximum image pixel count.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype="auto", device_map="auto"
        )
        # print(f"device_mamp: {self.model.hf_device_map}")
        if device == "cpu":
            print("[warning] Use cpu to load qwen model")
        self.processor = AutoProcessor.from_pretrained(
            model_path, min_pixels=min_pixels, max_pixels=max_pixels
        )

    @abstractmethod
    def predict(self, messages: List[dict]) -> str:
        """
        Abstract method to perform inference given multimodal input.

        Args:
            messages (List[dict]): List of message dicts with image and text.

        Returns:
            str: Model-generated response.
        """
        pass
    

class Qwen25VLBaseInferencer(QwenVLInferencerBase):
    """
    Concrete inferencer implementation for Qwen2.5-VL models.
    """

    def predict(self, messages: List[dict]) -> str:
        """
        Run inference on a given multimodal message.

        Args:
            messages (List[dict]): List of message dicts with image and text.

        Returns:
            str: Model-generated response.
        """
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # print(text)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        generated_ids = self.model.generate(**inputs, max_new_tokens=1024)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        print(output_text)
        return output_text

class Qwen3VLInferencerBase(ABC):
    """
    Abstract base class for Qwen3-VL-style inferencers.
    Handles model and processor initialization.
    Subclasses must implement the predict() method.
    """

    def __init__(
        self,
        model_path: str,
        *,
        dtype: str | torch.dtype = "auto",
        device_map: str | dict = "auto",
        use_flash_attn2: bool = False,
    ):
        """
        Initialize the model and processor from the specified path.

        Args:
            model_path: HF repo id or local path to the Qwen3-VL model.
            dtype: "auto" | torch.bfloat16 | torch.float16 | torch.float32
            device_map: e.g. "auto"
            use_flash_attn2: whether to enable flash_attention_2 for acceleration.
        """
        self.model_path = model_path

        model_kwargs: Dict[str, Any] = {
            "dtype": dtype,
            "device_map": device_map,
        }
        if use_flash_attn2:
            model_kwargs["attn_implementation"] = "flash_attention_2"

        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path, **model_kwargs
        )
        self.processor = AutoProcessor.from_pretrained(model_path)

        if not torch.cuda.is_available():
            print("[warning] Qwen3-VL is loaded on CPU; this will be slow.")

    @abstractmethod
    def predict(
        self,
        messages: List[Dict[str, Any]],
        *,
        max_new_tokens: int = 1024,
        **gen_kwargs: Any,
    ) -> str:
        """
        Abstract method to perform inference given multimodal input.

        Args:
            messages: Chat-style messages with images/videos and text.
            max_new_tokens: Generation length.
            gen_kwargs: Extra kwargs for `model.generate`.

        Returns:
            str: Model-generated response.
        """
        raise NotImplementedError


class Qwen3VLBaseInferencer(Qwen3VLInferencerBase):
    """
    Concrete inferencer implementation for Qwen3-VL models.
    """

    def predict(
        self,
        messages: List[Dict[str, Any]],
        *,
        max_new_tokens: int = 1024,
        **gen_kwargs: Any,
    ) -> str:
        """
        Run inference on a given multimodal message.

        Example messages format:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "/path/or/url/to/image.jpg"},
                    {"type": "text",  "text": "Describe this image."},
                ],
            }
        ]
        """
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            **gen_kwargs,
        )

        trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, outputs)]
        texts = self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        return texts[0] if texts else ""

class OaiInferencer:
    """
    OpenAI / Anthropic 
    """

    def __init__(
        self,
        model_name: str = "gpt-4o",
        openai_api_key: Optional[str] = None,
        openai_base_url: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        verbose: bool = False,
        smart_resize_images: bool = True,
        anthropic_img_max_bytes: int = 4_000_000,   
        anthropic_prefer_jpeg: bool = True,
    ):
        self.model = model_name
        self.verbose = verbose
        self.smart_resize_images = smart_resize_images
        self.anthropic_img_max_bytes = anthropic_img_max_bytes
        self.anthropic_prefer_jpeg = anthropic_prefer_jpeg

        self._b64 = base64
        self._mimetypes = mimetypes

        if "claude" in model_name.lower():
            import anthropic  # pip install anthropic
            self.provider = "anthropic"
            self.client = anthropic.Anthropic(
                api_key=(anthropic_api_key or os.getenv("ANTHROPIC_API_KEY"))
            )
        else:
            from openai import OpenAI  # pip install openai>=1.0
            self.provider = "openai"
            self.client = OpenAI(
                api_key=(openai_api_key or os.getenv("OPENAI_API_KEY")),
                base_url=(openai_base_url or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"),
            )


    @staticmethod
    def _read_bytes(path: str) -> Optional[bytes]:
        try:
            with open(path, "rb") as f:
                return f.read()
        except Exception:
            return None

    def _file_to_data_uri(self, path: str) -> Optional[str]:
        if path.startswith("file://"):
            path = path[len("file://"):]
        if not os.path.exists(path):
            print(f"no such image path: {path}")
            return None
        mime, _ = self._mimetypes.guess_type(path)
        mime = mime or "application/octet-stream"
        b = self._read_bytes(path)
        if not b:
            return None
        b64 = self._b64.b64encode(b).decode("ascii")
        return f"data:{mime};base64,{b64}"

    @staticmethod
    def _parse_data_uri(uri: str) -> Tuple[Optional[str], Optional[bytes]]:
        try:
            if not uri.startswith("data:"):
                return None, None
            head, b64 = uri.split(",", 1)
            media = "application/octet-stream"
            if ";" in head:
                media = head.split(":", 1)[1].split(";")[0] or media
            else:
                media = head.split(":", 1)[1] or media
            return media, base64.b64decode(b64)
        except Exception:
            return None, None

    @staticmethod
    def _pil_from_bytes(b: bytes):
        from PIL import Image
        return Image.open(io.BytesIO(b))

    def _encode_image_bytes(self, im, mime_hint: str, max_bytes: int) -> Tuple[str, bytes]:
        """将 PIL.Image 编码并尽量压到 max_bytes 内。"""
        from PIL import Image
        has_alpha = im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info)

        if not has_alpha and self.anthropic_prefer_jpeg:
            target = "JPEG"; mime = "image/jpeg"
        else:
            if mime_hint.endswith("png") or has_alpha:
                target = "PNG"; mime = "image/png"
            else:
                target = "JPEG"; mime = "image/jpeg"

        if target == "JPEG":
            if im.mode not in ("RGB", "L"):
                im = im.convert("RGB")
        else:
            if has_alpha and im.mode not in ("RGBA", "LA"):
                im = im.convert("RGBA")
            elif not has_alpha and im.mode not in ("RGB", "L"):
                im = im.convert("RGB")

        def _dump(quality: int = 95, optimize: bool = True) -> bytes:
            buf = io.BytesIO()
            if target == "JPEG":
                im.save(buf, format="JPEG", quality=quality, optimize=optimize, progressive=True)
            else:
                im.save(buf, format="PNG", optimize=True)
            return buf.getvalue()

        data = _dump()
        if len(data) <= max_bytes:
            return mime, data

        if target == "PNG" and (not has_alpha):
            im_j = im.convert("RGB")
            lo, hi, best = 30, 90, None
            while lo <= hi:
                q = (lo + hi) // 2
                buf = io.BytesIO()
                im_j.save(buf, format="JPEG", quality=q, optimize=True, progressive=True)
                cur = buf.getvalue()
                if len(cur) <= max_bytes:
                    best = cur; lo = q + 1
                else:
                    hi = q - 1
            if best:
                return "image/jpeg", best
            return mime, data  

        lo, hi, best = 30, 95, data
        while lo <= hi:
            q = (lo + hi) // 2
            cur = _dump(quality=q)
            if len(cur) <= max_bytes:
                best = cur; lo = q + 1
            else:
                hi = q - 1
        return mime, best


    def _convert_messages_openai(self, critic_messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out_msgs = []
        for msg in critic_messages:
            role = msg.get("role", "user")
            content_items = msg.get("content", [])
            new_content = []
            for it in content_items:
                if "text" in it and isinstance(it["text"], str):
                    new_content.append({"type": "text", "text": it["text"]})
                if "image" in it and isinstance(it["image"], str):
                    url = it["image"]
                    if url.startswith(("http://", "https://")):
                 
                        new_content.append({"type": "image_url", "image_url": {"url": url}})
                    elif url.startswith("data:"):
          
                        if self.smart_resize_images:
                            media, b = self._parse_data_uri(url)
                            if b:
                                from PIL import Image
                                im = self._pil_from_bytes(b)
                                rh, rw = smart_resize(im.height, im.width)
                                if (rw, rh) != (im.width, im.height):
                                    im = im.resize((rw, rh), Image.LANCZOS)
                                mime2, data = self._encode_image_bytes(im, media or "image/png", max_bytes=6_000_000)
                                data_uri = f"data:{mime2};base64,{base64.b64encode(data).decode('ascii')}"
                                new_content.append({"type": "image_url", "image_url": {"url": data_uri}})
                                continue
            
                        new_content.append({"type": "image_url", "image_url": {"url": url}})
                    else:
                        if url.startswith("file://"):
                            url = url[len("file://"):]
                        if os.path.exists(url):
                            b = self._read_bytes(url)
                            if b:
                                media = self._mimetypes.guess_type(url)[0] or "image/png"
                                if self.smart_resize_images:
                                    from PIL import Image
                                    im = self._pil_from_bytes(b)
                                    rh, rw = smart_resize(im.height, im.width)
                                    if (rw, rh) != (im.width, im.height):
                                        im = im.resize((rw, rh), Image.LANCZOS)
                                    mime2, data = self._encode_image_bytes(im, media, max_bytes=6_000_000)
                                    b64 = base64.b64encode(data).decode("ascii")
                                    new_content.append({"type": "image_url", "image_url": {"url": f"data:{mime2};base64,{b64}"}})
                                else:
                                    b64 = base64.b64encode(b).decode("ascii")
                                    new_content.append({"type": "image_url", "image_url": {"url": f"data:{media};base64,{b64}"}})
            if not new_content:
                new_content = [{"type": "text", "text": ""}]
            out_msgs.append({"role": role, "content": new_content})
        return out_msgs


    def _anthropic_prepare_image_source(self, img: str) -> Optional[Dict[str, Any]]:
        if img.startswith(("http://", "https://")):
            return {"type": "image", "source": {"type": "url", "url": img}}

        # data: URI
        if img.startswith("data:"):
            media, b = self._parse_data_uri(img)
            if not b:
                return None
            if self.smart_resize_images:
                from PIL import Image
                im = self._pil_from_bytes(b)
                rh, rw = smart_resize(im.height, im.width)
                if (rw, rh) != (im.width, im.height):
                    im = im.resize((rw, rh), Image.LANCZOS)
                mime2, data = self._encode_image_bytes(im, media or "image/png", self.anthropic_img_max_bytes)
                return {"type": "image", "source": {"type": "base64", "media_type": mime2, "data": base64.b64encode(data).decode("ascii")}}
   
            return {"type": "image", "source": {"type": "base64", "media_type": (media or "image/png"), "data": base64.b64encode(b).decode("ascii")}}

 
        path = img
        if path.startswith("file://"):
            path = path[len("file://"):]
        if not os.path.exists(path):
            return None
        b = self._read_bytes(path)
        if not b:
            return None
        media = self._mimetypes.guess_type(path)[0] or "image/png"
        if self.smart_resize_images:
            from PIL import Image
            im = self._pil_from_bytes(b)
            rh, rw = smart_resize(im.height, im.width)
            if (rw, rh) != (im.width, im.height):
                im = im.resize((rw, rh), Image.LANCZOS)
            mime2, data = self._encode_image_bytes(im, media, self.anthropic_img_max_bytes)
            return {"type": "image", "source": {"type": "base64", "media_type": mime2, "data": base64.b64encode(data).decode("ascii")}}

        return {"type": "image", "source": {"type": "base64", "media_type": media, "data": base64.b64encode(b).decode("ascii")}}

    def _convert_messages_anthropic(self, critic_messages: List[Dict[str, Any]]) -> Tuple[Optional[str], List[Dict[str, Any]]]:
        system_chunks: List[str] = []
        out_msgs = []
        for msg in critic_messages:
            role = msg.get("role", "user")
            content_items = msg.get("content", [])
            contents = []
            for it in content_items:
                if "text" in it and isinstance(it["text"], str):
                    if role == "system":
                        system_chunks.append(it["text"])
                    else:
                        contents.append({"type": "text", "text": it["text"]})
                if "image" in it and isinstance(it["image"], str):
                    src = self._anthropic_prepare_image_source(it["image"])
                    if src:
                        contents.append(src)
            if role != "system":
                if not contents:
                    contents = [{"type": "text", "text": ""}]
                role2 = "assistant" if role == "assistant" else "user"
                out_msgs.append({"role": role2, "content": contents})
        system_str = "\n\n".join([s for s in system_chunks if s])
        return (system_str or None), out_msgs


    def predict(self, critic_messages: List[Dict[str, Any]], max_tokens: int = 1024, temperature: float = 0.2) -> str:
        try:
            from openai import APIError, APITimeoutError, RateLimitError, APIConnectionError
        except Exception:
            APIError = APITimeoutError = RateLimitError = APIConnectionError = Exception

        max_tries, base_delay, max_delay = 5, 1.0, 30.0
        last_err = None

        for attempt in range(1, max_tries + 1):
            try:
                if self.provider == "anthropic":
                    system_str, ant_msgs = self._convert_messages_anthropic(critic_messages)
                    resp = self.client.messages.create(
                        model=self.model,
                        max_tokens=max_tokens,
                        system=system_str if system_str else None,
                        messages=ant_msgs,
                        temperature=temperature,
                    )
                    out_chunks = []
                    for blk in getattr(resp, "content", []) or []:
                        t = getattr(blk, "type", blk.get("type") if isinstance(blk, dict) else None)
                        txt = getattr(blk, "text", blk.get("text") if isinstance(blk, dict) else None)
                        if t == "text" and isinstance(txt, str):
                            out_chunks.append(txt)
                    return "\n".join(out_chunks).strip()

                # OpenAI
                openai_msgs = self._convert_messages_openai(critic_messages)
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=openai_msgs,
                    temperature=temperature,
                )
                return (resp.choices[0].message.content or "").strip()

            except (RateLimitError, APITimeoutError, APIConnectionError, APIError) as e:
                status = getattr(e, "status", None) or getattr(e, "http_status", None)
                if status in {400, 401, 403, 404}:
                    raise
                last_err = e
                if attempt < max_tries:
                    backoff = min(base_delay * (2 ** (attempt - 1)), max_delay) + random.uniform(0, 0.5)
                    if self.verbose:
                        print(f"[Infer] attempt {attempt}/{max_tries} failed ({type(e).__name__}); retry in {backoff:.2f}s...", flush=True)
                    time.sleep(backoff)
                else:
                    break
            except Exception:
                raise

        raise last_err if last_err is not None else RuntimeError("Request failed with unknown error")


@register_tool("mobile_use")
class MobileUse(BaseTool):
    @property
    def description(self):
        return f"""
Use a touchscreen to interact with a mobile device, and take screenshots.
* This is an interface to a mobile device with touchscreen. You can perform actions like clicking, typing, swiping, etc.
* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions.
* The screen's resolution is {self.display_width_px}x{self.display_height_px}.
* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.
""".strip()

    parameters = {
        "properties": {
            "action": {
                "description": """
The action to perform. The available actions are:
* `key`: Perform a key event on the mobile device.
    - This supports adb's `keyevent` syntax.
    - Examples: "volume_up", "volume_down", "power", "camera", "clear".
* `click`: Click the point on the screen with coordinate (x, y).
* `long_press`: Press the point on the screen with coordinate (x, y) for specified seconds.
* `swipe`: Swipe from the starting point with coordinate (x, y) to the end point with coordinates2 (x2, y2).
* `type`: Input the specified text into the activated input box.
* `system_button`: Press the system button.
* `open`: Open an app on the device.
* `wait`: Wait specified seconds for the change to happen.
* `terminate`: Terminate the current task and report its completion status.
""".strip(),
                "enum": [
                    "key",
                    "click",
                    "long_press",
                    "swipe",
                    "type",
                    "system_button",
                    "open",
                    "wait",
                    "terminate",
                ],
                "type": "string",
            },
            "coordinate": {
                "description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=click`, `action=long_press`, and `action=swipe`.",
                "type": "array",
            },
            "coordinate2": {
                "description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=swipe`.",
                "type": "array",
            },
            "text": {
                "description": "Required only by `action=key`, `action=type`, and `action=open`.",
                "type": "string",
            },
            "time": {
                "description": "The seconds to wait. Required only by `action=long_press` and `action=wait`.",
                "type": "number",
            },
            "button": {
                "description": "Back means returning to the previous interface, Home means returning to the desktop, Menu means opening the application background menu, and Enter means pressing the enter. Required only by `action=system_button`",
                "enum": [
                    "Back",
                    "Home",
                    "Menu",
                    "Enter",
                ],
                "type": "string",
            },
            "status": {
                "description": "The status of the task. Required only by `action=terminate`.",
                "type": "string",
                "enum": ["success", "failure"],
            },
        },
        "required": ["action"],
        "type": "object",
    }

    def __init__(self, cfg=None):
        self.display_width_px = cfg["display_width_px"]
        self.display_height_px = cfg["display_height_px"]
        super().__init__(cfg)

    def call(self, params: Union[str, dict], **kwargs):
        params = self._verify_json_format_args(params)
        action = params["action"]
        if action == "key":
            return self._key(params["text"])
        elif action == "click":
            return self._click(
                coordinate=params["coordinate"]
            )
        elif action == "long_press":
            return self._long_press(
                coordinate=params["coordinate"], time=params["time"]
            )
        elif action == "swipe":
            return self._swipe(
                coordinate=params["coordinate"], coordinate2=params["coordinate2"]
            )
        elif action == "type":
            return self._type(params["text"])
        elif action == "system_button":
            return self._system_button(params["button"])
        elif action == "open":
            return self._open(params["text"])
        elif action == "wait":
            return self._wait(params["time"])
        elif action == "terminate":
            return self._terminate(params["status"])
        else:
            raise ValueError(f"Unknown action: {action}")

    def _key(self, text: str):
        raise NotImplementedError()
        
    def _click(self, coordinate: Tuple[int, int]):
        raise NotImplementedError()

    def _long_press(self, coordinate: Tuple[int, int], time: int):
        raise NotImplementedError()

    def _swipe(self, coordinate: Tuple[int, int], coordinate2: Tuple[int, int]):
        raise NotImplementedError()

    def _type(self, text: str):
        raise NotImplementedError()

    def _system_button(self, button: str):
        raise NotImplementedError()

    def _open(self, text: str):
        raise NotImplementedError()

    def _wait(self, time: int):
        raise NotImplementedError()

    def _terminate(self, status: str):
        raise NotImplementedError()
    
@register_tool("computer_use")
class ComputerUse(BaseTool):
    @property
    def description(self):
        return f"""
Use a mouse and keyboard to interact with a computer, and take screenshots.
* This is an interface to a desktop GUI. You do not have access to a terminal or applications menu. You must click on desktop icons to start applications.
* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions. E.g. if you click on Firefox and a window doesn't open, try wait and taking another screenshot.
* The screen's resolution is {self.display_width_px}x{self.display_height_px}.
* Whenever you intend to move the cursor to click on an element like an icon, you should consult a screenshot to determine the coordinates of the element before moving the cursor.
* If you tried clicking on a program or link but it failed to load, even after waiting, try adjusting your cursor position so that the tip of the cursor visually falls on the element that you want to click.
* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.
""".strip()

    parameters = {
        "properties": {
            "action": {
                "description": """
The action to perform. The available actions are:
* `key`: Performs key down presses on the arguments passed in order, then performs key releases in reverse order.
* `type`: Type a string of text on the keyboard.
* `mouse_move`: Move the cursor to a specified (x, y) pixel coordinate on the screen.
* `left_click`: Click the left mouse button.
* `left_click_drag`: Click and drag the cursor to a specified (x, y) pixel coordinate on the screen.
* `right_click`: Click the right mouse button.
* `middle_click`: Click the middle mouse button.
* `double_click`: Double-click the left mouse button.
* `scroll`: Performs a scroll of the mouse scroll wheel.
* `wait`: Wait specified seconds for the change to happen.
* `terminate`: Terminate the current task and report its completion status.
""".strip(),
                "enum": [
                    "key",
                    "type",
                    "mouse_move",
                    "left_click",
                    "left_click_drag",
                    "right_click",
                    "middle_click",
                    "double_click",
                    "scroll",
                    "wait",
                    "terminate",
                ],
                "type": "string",
            },
            "keys": {
                "description": "Required only by `action=key`.",
                "type": "array",
            },
            "text": {
                "description": "Required only by `action=type`.",
                "type": "string",
            },
            "coordinate": {
                "description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=mouse_move` and `action=left_click_drag`.",
                "type": "array",
            },
            "pixels": {
                "description": "The amount of scrolling to perform. Positive values scroll up, negative values scroll down. Required only by `action=scroll`.",
                "type": "number",
            },
            "time": {
                "description": "The seconds to wait. Required only by `action=wait`.",
                "type": "number",
            },
            "status": {
                "description": "The status of the task. Required only by `action=terminate`.",
                "type": "string",
                "enum": ["success", "failure"],
            },
        },
        "required": ["action"],
        "type": "object",
    }

    def __init__(self, cfg=None):
        self.display_width_px = cfg["display_width_px"]
        self.display_height_px = cfg["display_height_px"]
        super().__init__(cfg)

    def call(self, params: Union[str, dict], **kwargs):
        params = self._verify_json_format_args(params)
        action = params["action"]
        if action in ["left_click", "right_click", "middle_click", "double_click"]:
            return self._mouse_click(action)
        elif action == "key":
            return self._key(params["keys"])
        elif action == "type":
            return self._type(params["text"])
        elif action == "mouse_move":
            return self._mouse_move(params["coordinate"])
        elif action == "left_click_drag":
            return self._left_click_drag(params["coordinate"])
        elif action == "scroll":
            return self._scroll(params["pixels"])
        elif action == "wait":
            return self._wait(params["time"])
        elif action == "terminate":
            return self._terminate(params["status"])
        else:
            raise ValueError(f"Invalid action: {action}")

    def _mouse_click(self, button: str):
        raise NotImplementedError()

    def _key(self, keys: List[str]):
        raise NotImplementedError()

    def _type(self, text: str):
        raise NotImplementedError()

    def _mouse_move(self, coordinate: Tuple[int, int]):
        raise NotImplementedError()

    def _left_click_drag(self, coordinate: Tuple[int, int]):
        raise NotImplementedError()

    def _scroll(self, pixels: int):
        raise NotImplementedError()

    def _wait(self, time: int):
        raise NotImplementedError()

    def _terminate(self, status: str):
        raise NotImplementedError()
    