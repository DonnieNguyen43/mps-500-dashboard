@echo off
chcp 65001 >nul
title MPS-500 Multimodal Market Prediction System - Nhom 9
cd /d "%~dp0"

set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

echo ======================================================================
echo    MPS-500: Multimodal Market Prediction System (Nhom 9 - 50K33)
echo ======================================================================
echo  Dang khoi dong Quant Dashboard tai trinh duyet: http://localhost:8501
echo  Vui long giu cua so nay mo trong suot qua trinh su dung dashboard.
echo ======================================================================
echo.

:: 1. Chay truc tiep bang python -m streamlit
python -m streamlit run app.py
if %ERRORLEVEL% equ 0 goto end

:: 2. Neu that bai, thu qua WindowsApps Python
if exist "%LOCALAPPDATA%\Microsoft\WindowsApps\python.exe" (
    echo [*] Dang thu chay qua WindowsApps Python...
    "%LOCALAPPDATA%\Microsoft\WindowsApps\python.exe" -m streamlit run app.py
    if %ERRORLEVEL% equ 0 goto end
)

:: 3. Neu that bai, thu qua lenh streamlit truc tiep
where streamlit >nul 2>nul
if %ERRORLEVEL% equ 0 (
    echo [*] Dang thu chay qua lenh streamlit...
    streamlit run app.py
    if %ERRORLEVEL% equ 0 goto end
)

:: 4. Neu that bai, thu qua py launcher
where py >nul 2>nul
if %ERRORLEVEL% equ 0 (
    echo [*] Dang thu chay qua py -m streamlit...
    py -m streamlit run app.py
    if %ERRORLEVEL% equ 0 goto end
)

:: 5. Neu tat ca deu that bai, thu qua PowerShell script
echo [*] Dang chuyen tiep qua PowerShell run.ps1...
powershell -ExecutionPolicy Bypass -File .\run.ps1

:end
echo.
echo [!] Ung dung da dung lai. Nhan phim bat ky de dong cua so nay...
pause >nul
