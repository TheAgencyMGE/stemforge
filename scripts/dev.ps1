# Starts API, worker and web UI in three windows. Stop each with Ctrl+C.
$ErrorActionPreference = "Stop"
$venv = "$env:LOCALAPPDATA\stemforge\venv\Scripts"
$root = Split-Path -Parent $PSScriptRoot
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$root\backend'; & '$venv\stemforge-api.exe' --port 8000"
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$root\backend'; & '$venv\stemforge-worker.exe'"
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$root\frontend'; npm run dev"
Write-Host "Open http://localhost:3000"
