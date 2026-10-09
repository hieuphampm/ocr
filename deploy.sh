#!/usr/bin/env bash
# Triển khai trên server:  ./deploy.sh
# Lần đầu: tạo .env (kèm API_TOKEN ngẫu nhiên) rồi dừng để bạn điền thông tin model; chạy lại để build + khởi động.
set -e
cd "$(dirname "$0")"
if [ ! -f .env ]; then
  cp .env.example .env
  sed -i "s|^API_TOKEN=.*|API_TOKEN=$(openssl rand -hex 24)|" .env
  chmod 600 .env
  echo "Đã tạo .env. Hãy mở và điền OPENAI_API_KEY / OPENAI_BASE_URL / MODEL_NAME, rồi chạy lại ./deploy.sh"
  exit 0
fi
docker compose up -d --build
docker compose ps
echo "API: http://<server>:33253/docs   (health: /health)"
