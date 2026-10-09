"""CHECKLIST HA - HTTP API (FastAPI).

    python api.py                     # http://0.0.0.0:33253   (docs: /docs)
    uvicorn api:app --host 0.0.0.0 --port 33253 --workers 2

Settings (environment / .env): PORT (33253), HOST (0.0.0.0), WORKERS (1), API_TOKEN (empty = no auth),
MAX_UPLOAD_MB (20), MAX_CONCURRENCY (2, per worker), MAX_BATCH_FILES (10), plus OPENAI_API_KEY /
OPENAI_BASE_URL / MODEL_NAME for the handwriting OCR."""
from __future__ import annotations

import hmac
import os
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import Depends, FastAPI, File, Header, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

import ocr  # noqa: F401  (also loads .env before the settings below are read)
from service import AiUnavailable, BadImage, Engine

PORT = int(os.environ.get("PORT", "33253"))
HOST = os.environ.get("HOST", "0.0.0.0")
API_TOKEN = os.environ.get("API_TOKEN", "")
MAX_UPLOAD = int(float(os.environ.get("MAX_UPLOAD_MB", "20")) * 1024 * 1024)
MAX_BATCH = int(os.environ.get("MAX_BATCH_FILES", "10"))

engine = Engine(max_concurrency=int(os.environ.get("MAX_CONCURRENCY", "2")))


@asynccontextmanager
async def lifespan(_: FastAPI):
    engine.load()  # once per worker process
    if not API_TOKEN:
        print("[warn] API_TOKEN trống: API mở cho mọi người truy cập được cổng này. Nên đặt API_TOKEN.")
    yield


app = FastAPI(title="CHECKLIST HA OCR", version="1.0", lifespan=lifespan)


def require_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    if API_TOKEN and not hmac.compare_digest((x_api_key or "").encode(), API_TOKEN.encode()):
        raise HTTPException(401, "Thiếu hoặc sai X-API-Key")


async def _read(file: UploadFile) -> bytes:
    data = await file.read(MAX_UPLOAD + 1)
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, f"Ảnh lớn hơn {MAX_UPLOAD // (1024 * 1024)} MB")
    return data


def _http(exc: Exception) -> HTTPException:
    if isinstance(exc, BadImage):
        return HTTPException(400, str(exc))
    if isinstance(exc, AiUnavailable):
        return HTTPException(503, str(exc))
    return HTTPException(500, f"{type(exc).__name__}: {exc}")


@app.get("/health")
def health():
    return {"status": "ok", **engine.info()}


@app.post("/ocr", dependencies=[Depends(require_key)])
async def ocr_one(
    file: UploadFile = File(..., description="Ảnh form (jpg/png/webp/bmp/tiff), có thể chụp nghiêng hoặc xoay ngang"),
    skip_ai: bool = Query(False, description="Chỉ dò checkbox + vùng ghi chú, không gọi model đọc chữ viết tay"),
    verbose: bool = Query(False, description="Kèm điểm từng checkbox (_debug)"),
    include_images: bool = Query(False, description="Kèm ảnh đã căn chỉnh và ảnh debug (base64)"),
):
    data = await _read(file)
    try:
        result = await run_in_threadpool(engine.analyze, data, file.filename or "upload", skip_ai, verbose, include_images)
    except Exception as exc:
        raise _http(exc)
    return JSONResponse(result)


@app.post("/ocr/batch", dependencies=[Depends(require_key)])
async def ocr_batch(
    files: List[UploadFile] = File(...),
    skip_ai: bool = Query(False),
    verbose: bool = Query(False),
    include_images: bool = Query(False),
):
    if len(files) > MAX_BATCH:
        raise HTTPException(413, f"Tối đa {MAX_BATCH} ảnh mỗi lần")
    results = []
    for f in files:  # one bad photo must not fail the others
        name = f.filename or "upload"
        try:
            data = await _read(f)
            results.append(await run_in_threadpool(engine.analyze, data, name, skip_ai, verbose, include_images))
        except HTTPException as exc:
            results.append({"source": name, "error": exc.detail})
        except AiUnavailable as exc:
            raise _http(exc)  # same for every file: fail the request
        except Exception as exc:
            results.append({"source": name, "error": f"{type(exc).__name__}: {exc}"})
    return JSONResponse({"count": len(results), "results": results})


if __name__ == "__main__":
    import uvicorn

    workers = int(os.environ.get("WORKERS", "1"))
    uvicorn.run("api:app" if workers > 1 else app, host=HOST, port=PORT, workers=workers)
