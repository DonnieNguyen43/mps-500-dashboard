# ========================================================
# MPS-500: Multimodal Market Prediction System (Nhom 9)
# ========================================================

$ErrorActionPreference = "Continue"
$projectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectDir

[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "   MPS-500: Multimodal Market Prediction System (Nhom 9 - 50K33)      " -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "[*] Dang khoi dong MPS-500 tai trinh duyet: http://localhost:8501..." -ForegroundColor Yellow

# 1. Thu python -m streamlit truc tiep
if (Get-Command python -ErrorAction SilentlyContinue) {
    Write-Host "[+] Khoi chay qua: python -m streamlit run app.py" -ForegroundColor Green
    python -m streamlit run app.py
    exit
}

# 2. Thu qua WindowsApps python
$winAppPython = "$env:LOCALAPPDATA\Microsoft\WindowsApps\python.exe"
if (Test-Path $winAppPython) {
    Write-Host "[+] Khoi chay qua WindowsApps Python: $winAppPython" -ForegroundColor Green
    & $winAppPython -m streamlit run app.py
    exit
}

# 3. Thu streamlit truc tiep
if (Get-Command streamlit -ErrorAction SilentlyContinue) {
    Write-Host "[+] Tim thay streamlit trong PATH" -ForegroundColor Green
    streamlit run app.py
    exit
}

# 4. Thu py launcher
if (Get-Command py -ErrorAction SilentlyContinue) {
    Write-Host "[+] Khoi chay qua: py -m streamlit run app.py" -ForegroundColor Green
    py -m streamlit run app.py
    exit
}

Write-Host "[-] Khong tim thay python hoac streamlit executable." -ForegroundColor Red
Write-Host "[!] Vui long kiem tra lai moi truong Python." -ForegroundColor Yellow
pause
