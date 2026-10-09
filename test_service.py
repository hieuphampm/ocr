"""Tests of the API core (service.Engine) - no FastAPI, no network, no real model needed."""
import base64
import json
import sys
import types

import cv2
import numpy as np

import ocr
from service import AiUnavailable, BadImage, Engine

TICKED = {"Nhăn mắt", "Rãnh mũi", "Ngạch cằm"}


def page(ticks=TICKED):
    img = np.full((ocr.H, ocr.W, 3), 245, np.uint8)
    for items in ocr.CHECKBOXES.values():
        for label, x, y in items:
            x, y = int(round(x)), int(round(y))
            cv2.rectangle(img, (x - 13, y - 13), (x + 13, y + 13), (70, 70, 70), 2)
            cv2.putText(img, label, (x + 28, y + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 60, 60), 1, cv2.LINE_AA)
            if label in ticks:
                cv2.polylines(img, [np.array([[x - 12, y], [x - 3, y + 11], [x + 24, y - 36]], np.int32)], False, (170, 40, 20), 3, cv2.LINE_AA)
    return img


def jpeg(img):
    return cv2.imencode(".jpg", img)[1].tobytes()


class Stub:  # stands in for the OpenAI client
    def __init__(self):
        self.chat = types.SimpleNamespace(completions=self)

    def create(self, **kw):
        msg = types.SimpleNamespace(message=types.SimpleNamespace(content='{"value":"abc","confidence":0.9}'))
        return types.SimpleNamespace(choices=[msg])


eng = Engine(max_concurrency=2)
eng.template = None  # notes off here; covered by test_synthetic.py
data = jpeg(cv2.resize(page(), (853, 1245)))
checks = {}

r = eng.analyze(data, "p.jpg", skip_ai=True)
got = {l for s in r["sections"].values() if isinstance(s, dict) for l in s.get("checked", [])}
checks["skip_ai: aligned, exactly the real ticks"] = r["alignment"]["ok"] and got == TICKED and r["source"] == "p.jpg"
checks["result is JSON serialisable"] = bool(json.dumps(r, ensure_ascii=False))
checks["no images unless asked"] = "images" not in r

r = eng.analyze(jpeg(cv2.rotate(cv2.resize(page(), (853, 1245)), cv2.ROTATE_90_CLOCKWISE)), "r.jpg", skip_ai=True)
checks["sideways photo is straightened"] = r["alignment"]["ok"] and r["alignment"]["rotation"] == 270

r = eng.analyze(data, "p.jpg", skip_ai=True, include_images=True)
a = cv2.imdecode(np.frombuffer(base64.b64decode(r["images"]["aligned_jpg_base64"]), np.uint8), 1)
d = cv2.imdecode(np.frombuffer(base64.b64decode(r["images"]["debug_png_base64"]), np.uint8), 1)
checks["include_images: aligned + debug decode to page size"] = a.shape[:2] == (ocr.H, ocr.W) and d.shape[:2] == (ocr.H, ocr.W)

r = eng.analyze(data, "p.jpg", skip_ai=True, verbose=True)
checks["verbose adds _debug"] = "_debug" in r


def raises(exc, **kw):
    try:
        eng.analyze(**kw)
    except exc:
        return True
    except Exception:
        return False
    return False


checks["garbage bytes -> BadImage"] = raises(BadImage, data=b"not an image", source="x", skip_ai=True)
checks["empty upload -> BadImage"] = raises(BadImage, data=b"", source="x", skip_ai=True)
checks["no model configured, AI wanted -> AiUnavailable"] = raises(AiUnavailable, data=data, source="x")

eng.client = Stub()
r = eng.analyze(data, "p.jpg")
checks["with model: header fields filled"] = r["header"]["ndh"]["value"] == "abc" and r["ocr"]["model"] == ocr.MODEL_NAME
checks["skip_ai still works when a model exists"] = eng.analyze(data, "p.jpg", skip_ai=True)["header"]["ndh"]["value"] == ""

blank = np.full((1245, 853, 3), 210, np.uint8)
cv2.randn(blank, 210, 12)
r = eng.analyze(jpeg(blank), "junk.jpg", skip_ai=True)
checks["junk photo: rejected, nothing ticked"] = (not r["alignment"]["ok"]) and not any(
    s.get("checked") for s in r["sections"].values() if isinstance(s, dict))

for name, ok in checks.items():
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
sys.exit(0 if all(checks.values()) else 1)
