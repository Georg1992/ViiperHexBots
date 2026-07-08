$ErrorActionPreference = "Stop"

$rootDir = $PSScriptRoot
$viiperOut = Join-Path $rootDir "VIIPER\dist\viiper.exe"

if (-not (Test-Path $viiperOut)) {
    Write-Host "Building viiper.exe..." -ForegroundColor Cyan
    Push-Location (Join-Path $rootDir "VIIPER")
    New-Item -ItemType Directory -Force "dist" | Out-Null
    $env:CGO_ENABLED = "0"
    go build -trimpath -o $viiperOut .\cmd\viiper
    if ($LASTEXITCODE -ne 0) { Pop-Location; exit $LASTEXITCODE }
    Pop-Location
}

Write-Host "Installing Python package (editable)..." -ForegroundColor Cyan
python -m pip install -e ".[dev]" -q
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ""
Write-Host "Done. Double-click run.pyw or run run.bat to start the Python bot." -ForegroundColor Green
Write-Host "  $viiperOut" -ForegroundColor Yellow
