FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 PORT=33253 HOST=0.0.0.0
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY ocr.py service.py api.py template_print.png ./
RUN useradd -r -u 10001 ocr && chown -R ocr /app
USER ocr
EXPOSE 33253
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s CMD python -c "import os,urllib.request as u; u.urlopen('http://127.0.0.1:%s/health' % os.environ.get('PORT','33253'), timeout=4)"
# mặc định chạy API (WORKERS tiến trình x MAX_CONCURRENCY ảnh song song mỗi tiến trình)
# chạy theo lô thư mục ảnh: docker run ... --entrypoint python checklist-ocr ocr.py
CMD ["python", "api.py"]
