# ========================================================
# MPS-500: Multimodal Market Prediction System (Nhom 9)
# ========================================================

$ErrorActionPreference = "Continue"
$projectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectDir

Write-Host "[*] Dang khoi dong MPS-500 Multimodal Market Prediction System..." -ForegroundColor Cyan

# 1. Thu qua WindowsApps python
$winAppPython = "$env:LOCALAPPDATA\Microsoft\WindowsApps\python.exe"
if (Test-Path $winAppPython) {
    Write-Host "[+] Khoi chay qua WindowsApps Python: $winAppPython" -ForegroundColor Green
    & $winAppPython -m streamlit run app.py
    exit
}

# 2. Thu streamlit trực tiếp
if (Get-Command streamlit -ErrorAction SilentlyContinue) {
    Write-Host "[+] Tim thay streamlit trong PATH" -ForegroundColor Green
    streamlit run app.py
    exit
}

# 3. Thu python -m streamlit
if (Get-Command python -ErrorAction SilentlyContinue) {
    Write-Host "[+] Khoi chay qua: python -m streamlit" -ForegroundColor Green
    python -m streamlit run app.py
    exit
}

# 4. Kiem tra thu muc Scripts cua Python 3.13 (Windows Store)
$storeScripts = "$env:LOCALAPPDATA\Packages\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\LocalCache\local-packages\Python313\Scripts"
if (Test-Path "$storeScripts\streamlit.exe") {
    Write-Host "[+] Tim thay streamlit tai Windows Store Python Scripts!" -ForegroundColor Green
    & "$storeScripts\streamlit.exe" run app.py
    exit
}

# 5. Tim kiem trong thu muc Packages neu ten thu muc co khac
$foundStreamlit = Get-ChildItem "$env:LOCALAPPDATA\Packages\PythonSoftwareFoundation*" -Recurse -Filter "streamlit.exe" -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty FullName
if ($foundStreamlit) {
    Write-Host "[+] Tim thay: $foundStreamlit" -ForegroundColor Green
    & $foundStreamlit run app.py
    exit
}

Write-Host "[-] Khong tim thay streamlit executable." -ForegroundColor Red
Write-Host "[!] Hay thu chay: python -m pip install streamlit" -ForegroundColor Yellow
pause
