# Validate AutoHotkey v1 scripts (syntax / load errors only)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$ahkCandidates = @(
    "${env:ProgramFiles}\AutoHotkey\v1.1.37.01\AutoHotkeyU64.exe",
    "${env:ProgramFiles}\AutoHotkey\AutoHotkeyU64.exe",
    "${env:ProgramFiles}\AutoHotkey\v1.1.37.02\AutoHotkeyU64.exe",
    "${env:ProgramFiles}\AutoHotkey\AutoHotkey.exe"
)
$ahk = $ahkCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $ahk) {
    Write-Error "AutoHotkey v1 not found. Install from https://www.autohotkey.com/"
}

function Stop-VhbOrphanProcesses {
    Get-CimInstance Win32_Process | Where-Object {
        $cmd = $_.CommandLine
        if (-not $cmd) { return $false }
        ($cmd -match 'viiper-input\.exe') -or
        ($cmd -match 'viiper-hexbots.*\\viiper\.exe')
    } | ForEach-Object {
        Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
    }
    Start-Sleep -Milliseconds 300
}

$scripts = @(
    "main.ahk"
)

$failed = 0
foreach ($script in $scripts) {
    $path = Join-Path $root $script
    if (-not (Test-Path $path)) {
        Write-Host "SKIP $script (missing)"
        continue
    }

    $errFile = Join-Path $env:TEMP ("ahk_validate_" + [guid]::NewGuid().ToString() + ".txt")
    $scriptArgs = @("/ErrorStdOut", $script)
    if ($script -eq "main.ahk") {
        $scriptArgs += "--validate"
    }

    $proc = Start-Process -FilePath $ahk -ArgumentList $scriptArgs `
        -WorkingDirectory $root -PassThru -RedirectStandardError $errFile -WindowStyle Hidden
    $proc.WaitForExit(5000) | Out-Null
    if (-not $proc.HasExited) {
        Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
        Start-Sleep -Milliseconds 200
    }

    $stderr = ""
    if (Test-Path $errFile) {
        $raw = Get-Content $errFile -Raw -ErrorAction SilentlyContinue
        if ($raw) {
            $stderr = $raw.Trim()
        }
        Remove-Item $errFile -Force -ErrorAction SilentlyContinue
    }

    if ($stderr) {
        Write-Host "FAIL $script"
        Write-Host $stderr
        $failed++
    } else {
        Write-Host "OK   $script"
    }
}

Stop-VhbOrphanProcesses

if ($failed -gt 0) {
    exit 1
}

Write-Host "All AHK scripts validated."
