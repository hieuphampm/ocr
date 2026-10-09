#!/usr/bin/env bash
# Cài một lần: tạo môi trường ảo .venv và cài thư viện (Linux / macOS)
set -e
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
$PY -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
[ -f .env ] || { cp .env.example .env; echo ">> Đã tạo .env — hãy mở và điền OPENAI_API_KEY / OPENAI_BASE_URL / MODEL_NAME"; }
echo "Xong. Bỏ ảnh vào thư mục image/ rồi chạy: ./run.sh"
