# Run all Python tests via pytest (requires editable install)
$ErrorActionPreference = "Stop"
$ROOT = Split-Path -Parent $PSScriptRoot

Push-Location $ROOT
try {
    python -m pip install -e ".[dev]" -q
    if ($LASTEXITCODE -ne 0) { throw "pip install failed" }
    python -m pytest
    if ($LASTEXITCODE -ne 0) { throw "Tests failed" }
} finally { Pop-Location }

Write-Host "`nAll hunt tests passed."
