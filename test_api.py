"""HTTP-level smoke test of api.py. Needs: pip install httpx.   Run: python test_api.py
Uses skip_ai=true, so no model is called."""
import sys

import cv2
import numpy as np
from fastapi.testclient import TestClient

import api
import ocr

img = np.full((ocr.H, ocr.W, 3), 245, np.uint8)
for items in ocr.CHECKBOXES.values():
    for label, x, y in items:
        x, y = int(round(x)), int(round(y))
        cv2.rectangle(img, (x - 13, y - 13), (x + 13, y + 13), (70, 70, 70), 2)
jpg = cv2.imencode(".jpg", cv2.resize(img, (853, 1245)))[1].tobytes()

checks = {}
with TestClient(api.app) as c:  # runs the lifespan (loads template, model client)
    checks["GET /health"] = c.get("/health").json()["status"] == "ok"
    r = c.post("/ocr?skip_ai=true", files={"file": ("a.jpg", jpg, "image/jpeg")})
    checks["POST /ocr -> 200, aligned"] = r.status_code == 200 and r.json()["alignment"]["ok"] and r.json()["source"] == "a.jpg"
    checks["bad image -> 400"] = c.post("/ocr?skip_ai=true", files={"file": ("x.jpg", b"nope", "image/jpeg")}).status_code == 400
    r = c.post("/ocr/batch?skip_ai=true", files=[("files", ("a.jpg", jpg, "image/jpeg")), ("files", ("x.jpg", b"nope", "image/jpeg"))])
    j = r.json()
    checks["batch: good one ok, bad one reported, request 200"] = r.status_code == 200 and j["count"] == 2 and "error" in j["results"][1] and "alignment" in j["results"][0]
    if not api.engine.ai_configured:
        checks["AI wanted but no model -> 503"] = c.post("/ocr", files={"file": ("a.jpg", jpg, "image/jpeg")}).status_code == 503
    api.API_TOKEN = "secret"
    checks["token set, no key -> 401"] = c.post("/ocr?skip_ai=true", files={"file": ("a.jpg", jpg, "image/jpeg")}).status_code == 401
    checks["token set, right key -> 200"] = c.post("/ocr?skip_ai=true", headers={"X-API-Key": "secret"}, files={"file": ("a.jpg", jpg, "image/jpeg")}).status_code == 200
    checks["/health stays open"] = c.get("/health").status_code == 200

for name, ok in checks.items():
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
sys.exit(0 if all(checks.values()) else 1)
