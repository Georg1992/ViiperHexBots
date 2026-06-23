# Install mob recognition dependencies (run once)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
pip install -r (Join-Path $PSScriptRoot "requirements.txt")
Write-Host "MobRecognition Python deps installed." -ForegroundColor Green

$hornSpr = Join-Path $root "assets\mobs\horn\horn.spr"
$hornAct = Join-Path $root "assets\mobs\horn\horn.act"
if ((Test-Path $hornSpr) -and (Test-Path $hornAct)) {
    py -3 (Join-Path $PSScriptRoot "cli.py") build-simple-descriptor --mob horn
    Write-Host "Horn descriptor built in generated_descriptors\horn\simple\" -ForegroundColor Green
}
