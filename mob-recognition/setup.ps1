# Install mob recognition dependencies (run once)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
pip install -r (Join-Path $PSScriptRoot "requirements.txt")
Write-Host "MobRecognition Python deps installed." -ForegroundColor Green
Write-Host "Build descriptors separately, for example:" -ForegroundColor Yellow
Write-Host ".\scripts\build-mob-descriptor.ps1 -Mob horn -Force"
