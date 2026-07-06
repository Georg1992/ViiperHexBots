# Run mob-recognition tests only
$ErrorActionPreference = "Stop"
$ROOT = Split-Path -Parent $PSScriptRoot

Write-Host "=== Mob recognition tests ==="
Push-Location "$ROOT\mob-recognition"
try {
    python -m pytest tests/ -q --tb=short
    if ($LASTEXITCODE -ne 0) { throw "Recognition tests failed" }
} finally { Pop-Location }

Write-Host "`nAll recognition tests passed."
