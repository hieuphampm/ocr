"""Web-independent core of the API: bytes of one photo in -> result JSON out.

Kept apart from api.py so it can be tested (and reused) without FastAPI."""
from __future__ import annotations

import base64
import os
import threading
from types import SimpleNamespace
from typing import Any, Dict, Optional

# decompression-bomb guard for uploads: refuse to decode absurdly large pixel counts
os.environ.setdefault("OPENCV_IO_MAX_IMAGE_PIXELS", str(120_000_000))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import ocr  # noqa: E402


class BadImage(ValueError):
    """The upload is not a decodable image."""


class AiUnavailable(RuntimeError):
    """OCR of handwriting was requested but no model is configured."""


def _b64(ext: str, img: np.ndarray, params=()) -> str:
    ok, buf = cv2.imencode(ext, img, list(params))
    if not ok:
        raise RuntimeError(f"cannot encode {ext}")
    return base64.b64encode(buf.tobytes()).decode("ascii")


class Engine:
    def __init__(self, max_concurrency: int = 2):
        self.template: Optional[np.ndarray] = None
        self.client = None
        # alignment is CPU heavy (up to 4 orientations): cap how many photos are processed at once
        self._slots = threading.BoundedSemaphore(max(1, max_concurrency))

    def load(self) -> None:
        self.template = ocr.load_print_template(None)  # template_print.png next to ocr.py
        if self.template is None:
            print("[warn] Không có template_print.png: sẽ không dò được ghi chú cạnh ô.")
        if ocr.OPENAI_API_KEY and ocr.OPENAI_BASE_URL and ocr.MODEL_NAME:
            self.client = ocr.make_openai_client()
        else:
            print("[warn] Thiếu OPENAI_API_KEY / OPENAI_BASE_URL / MODEL_NAME: chỉ dùng được skip_ai=true.")

    @property
    def ai_configured(self) -> bool:
        return self.client is not None

    def info(self) -> Dict[str, Any]:
        return {"ai_configured": self.ai_configured, "model": ocr.MODEL_NAME if self.ai_configured else None,
                "notes_enabled": self.template is not None}

    def analyze(self, data: bytes, source: str, skip_ai: bool = False, verbose: bool = False,
                include_images: bool = False) -> Dict[str, Any]:
        if not skip_ai and self.client is None:
            raise AiUnavailable("Chưa cấu hình model OCR (OPENAI_API_KEY / OPENAI_BASE_URL / MODEL_NAME); dùng skip_ai=true để chỉ dò checkbox.")
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR) if data else None  # honours EXIF rotation
        if image is None:
            raise BadImage("không đọc được ảnh (file hỏng hoặc không phải ảnh)")
        args = SimpleNamespace(verbose=verbose, print_template=self.template)
        with self._slots:
            result, page, debug = ocr.analyze_image(image, source, None, None if skip_ai else self.client, args)
        if include_images:
            result["images"] = {"aligned_jpg_base64": _b64(".jpg", page, (cv2.IMWRITE_JPEG_QUALITY, 90)),
                                "debug_png_base64": _b64(".png", debug)}
        return result
