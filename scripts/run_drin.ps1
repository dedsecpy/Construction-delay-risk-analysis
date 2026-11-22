# Launch DRIN: FastAPI (8000) + Vite UI (5173)
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$py = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }

Write-Host "Starting DRIN API on http://127.0.0.1:8000 ..." -ForegroundColor Cyan
Start-Process -FilePath $py -ArgumentList "-m","uvicorn","app.api:app","--host","127.0.0.1","--port","8000" -WorkingDirectory $Root

Start-Sleep -Seconds 2
Write-Host "Starting DRIN UI on http://127.0.0.1:5173 ..." -ForegroundColor Cyan
Set-Location (Join-Path $Root "web")
npm run dev
