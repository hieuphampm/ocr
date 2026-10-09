@echo off
rem Cai mot lan: tao moi truong ao .venv va cai thu vien (Windows)
cd /d "%~dp0"
py -3 -m venv .venv || python -m venv .venv
if errorlevel 1 (echo Khong tim thay Python 3.9+ - cai tu https://www.python.org/downloads/ ^(nho tick "Add python.exe to PATH"^) & pause & exit /b 1)
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
if not exist .env (copy .env.example .env >nul & echo Da tao .env - hay mo va dien OPENAI_API_KEY / OPENAI_BASE_URL / MODEL_NAME)
echo Xong. Bo anh vao thu muc image roi chay: run.bat
pause
