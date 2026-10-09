@echo off
rem Chay: run.bat  (xu ly thu muc image)   |   run.bat --skip-ai  (moi tuy chon cua ocr.py)
cd /d "%~dp0"
.venv\Scripts\python.exe ocr.py %*
pause
