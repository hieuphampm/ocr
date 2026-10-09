#!/usr/bin/env bash
# Chạy: ./run.sh            (xử lý thư mục image/)   |   ./run.sh --skip-ai   (mọi tùy chọn của ocr.py)
cd "$(dirname "$0")"
exec .venv/bin/python ocr.py "$@"
