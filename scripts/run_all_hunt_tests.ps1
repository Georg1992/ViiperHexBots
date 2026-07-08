# Run all Python unit tests (runtime hunt tests + mob-recognition tests)
$ErrorActionPreference = "Stop"
$ROOT = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = $ROOT

Write-Host "=== Python runtime hunt tests ==="
Push-Location $ROOT
try {
    python -m unittest discover -s pybot/runtime/tests -p "test_*.py" -v
    if ($LASTEXITCODE -ne 0) { throw "Runtime tests failed" }
} finally { Pop-Location }

Write-Host "`n=== Python app tests ==="
Push-Location $ROOT
try {
    python -m unittest discover -s pybot/app/tests -p "test_*.py" -v
    if ($LASTEXITCODE -ne 0) { throw "App tests failed" }
} finally { Pop-Location }

Write-Host "`n=== Mob recognition tests ==="
Push-Location "$ROOT\mob-recognition"
try {
    python -m pytest tests/ -q --tb=short
    if ($LASTEXITCODE -ne 0) { throw "Recognition tests failed" }
} finally { Pop-Location }

Write-Host "`nAll hunt tests passed."
