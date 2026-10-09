@echo off
rem Chay API (cong PORT trong .env, mac dinh 33253)
cd /d "%~dp0"
.venv\Scripts\python.exe api.py
pause
