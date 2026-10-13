@echo off
title MPS-500 Multimodal Market Prediction System - Nhom 9
cd /d "%~dp0"

echo [*] Dang khoi dong MPS-500 Multimodal Market Prediction System...

where streamlit >nul 2>nul
if %ERRORLEVEL% equ 0 (
    echo [+] Khoi chay streamlit truc tiep...
    streamlit run app.py
    goto end
)

set WIN_PYTHON="%LOCALAPPDATA%\Microsoft\WindowsApps\python.exe"
if exist %WIN_PYTHON% (
    echo [+] Khoi chay qua WindowsApps Python...
    %WIN_PYTHON% -m streamlit run app.py
    goto end
)

where python >nul 2>nul
if %ERRORLEVEL% equ 0 (
    echo [+] Khoi chay qua python -m streamlit...
    python -m streamlit run app.py
    goto end
)

set STORE_STREAMLIT="%LOCALAPPDATA%\Packages\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\LocalCache\local-packages\Python313\Scripts\streamlit.exe"
if exist %STORE_STREAMLIT% (
    echo [+] Khoi chay tu Windows Store Scripts...
    %STORE_STREAMLIT% run app.py
    goto end
)

powershell -ExecutionPolicy Bypass -File .\run.ps1

:end
pause
