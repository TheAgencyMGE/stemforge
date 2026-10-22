# Stemforge local setup for Windows (PowerShell). Run from the repository root:
#   powershell -ExecutionPolicy Bypass -File scripts\setup.ps1 [-Cpu]
param([switch]$Cpu)
$ErrorActionPreference = "Stop"

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
  Write-Host "Installing uv (Python manager) via winget..."
  winget install --id astral-sh.uv -e --accept-source-agreements --accept-package-agreements
  $env:Path = "$env:LOCALAPPDATA\Microsoft\WinGet\Links;$env:Path"
}

# Keep the virtualenv on a short path: PyTorch has files that exceed MAX_PATH inside deep folders.
$venv = "$env:LOCALAPPDATA\stemforge\venv"
uv python install 3.11
uv venv --python 3.11 $venv
$py = "$venv\Scripts\python.exe"
$index = if ($Cpu) { "https://download.pytorch.org/whl/cpu" } else { "https://download.pytorch.org/whl/cu128" }
uv pip install --python $py torch==2.11.0 torchaudio==2.11.0 --index-url $index
uv pip install --python $py --no-deps -r backend\requirements\lock.txt
uv pip install --python $py --no-deps -e backend

if (-not (Test-Path backend\.env)) {
  "STEMFORGE_DATA_DIR=$($env:LOCALAPPDATA -replace '\\','/')/stemforge/data" | Out-File -Encoding utf8 backend\.env
}
Push-Location backend
& "$venv\Scripts\stemforge-models.exe" download
Pop-Location

Push-Location frontend
npm ci
Pop-Location
Write-Host "`nDone. Start the app with: powershell -File scripts\dev.ps1"
