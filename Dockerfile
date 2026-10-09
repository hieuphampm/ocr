FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY ocr.py service.py api.py template_print.png ./
EXPOSE 33253
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:33253/health', timeout=4)"
# mặc định chạy API; muốn chạy theo lô thư mục ảnh: docker run ... --entrypoint python checklist-ocr ocr.py
CMD ["python", "api.py"]
