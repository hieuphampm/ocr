#!/usr/bin/env bash
# Chạy API (cổng PORT trong .env, mặc định 33253)
cd "$(dirname "$0")"
exec .venv/bin/python api.py "$@"
