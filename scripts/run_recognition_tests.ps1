# Run recognition tests (tests/recognition)
$ErrorActionPreference = "Stop"
$ROOT = Split-Path -Parent $PSScriptRoot

Push-Location $ROOT
try {
    python -m pip install -e ".[dev]" -q
    if ($LASTEXITCODE -ne 0) { throw "pip install failed" }
    python -m pytest tests/recognition
    if ($LASTEXITCODE -ne 0) { throw "Recognition tests failed" }
} finally { Pop-Location }

Write-Host "`nAll recognition tests passed."
