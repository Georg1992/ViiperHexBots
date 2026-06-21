$ErrorActionPreference = "Stop"

$rootDir = $PSScriptRoot
$bridgeDir = Join-Path $rootDir "input-bridge"
$viiperOut = Join-Path $rootDir "VIIPER\dist\viiper.exe"
$embedDir = Join-Path $bridgeDir "embed"
$embedExe = Join-Path $embedDir "viiper.exe"
$bridgeOut = Join-Path $rootDir "viiper-input.exe"
$bridgeBuildOut = Join-Path $rootDir "viiper-input.exe.new"

New-Item -ItemType Directory -Force $embedDir | Out-Null

if (-not (Test-Path $viiperOut)) {
    Write-Host "Building viiper.exe..." -ForegroundColor Cyan
    Push-Location (Join-Path $rootDir "VIIPER")
    New-Item -ItemType Directory -Force "dist" | Out-Null
    $env:CGO_ENABLED = "0"
    go build -trimpath -o $viiperOut .\cmd\viiper
    if ($LASTEXITCODE -ne 0) { Pop-Location; exit $LASTEXITCODE }
    Pop-Location
}

Copy-Item $viiperOut $embedExe -Force

Push-Location $bridgeDir

Write-Host "Downloading Go modules..." -ForegroundColor Cyan
go mod download
if ($LASTEXITCODE -ne 0) { Pop-Location; exit $LASTEXITCODE }

Write-Host "Building viiper-input.exe..." -ForegroundColor Cyan
Remove-Item $bridgeBuildOut -Force -ErrorAction SilentlyContinue
go build -trimpath -o $bridgeBuildOut .
if ($LASTEXITCODE -ne 0) {
    Remove-Item $bridgeBuildOut -Force -ErrorAction SilentlyContinue
    Pop-Location
    exit $LASTEXITCODE
}

Remove-Item $bridgeOut -Force -ErrorAction SilentlyContinue
Move-Item $bridgeBuildOut $bridgeOut -Force

Pop-Location

Write-Host ""
Write-Host "Done. Run main.ahk with AutoHotkey after installing usbip-win2." -ForegroundColor Green
Write-Host "  $bridgeOut" -ForegroundColor Yellow
