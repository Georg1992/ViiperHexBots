param(
    [Parameter(Mandatory = $true)]
    [string]$Mob,

    [switch]$Force
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$mobKey = $Mob.ToLowerInvariant()
$assetDir = Join-Path $root "assets\mobs\$mobKey"
$sprPath = Join-Path $assetDir "$mobKey.spr"
$actPath = Join-Path $assetDir "$mobKey.act"

if (-not (Test-Path $sprPath)) {
    throw "Missing SPR file: $sprPath"
}
if (-not (Test-Path $actPath)) {
    throw "Missing ACT file: $actPath"
}

$scriptArgs = @("-m", "pybot.recognition.cli", "build-descriptor", "--mob", $mobKey)
if ($Force) {
    $scriptArgs += "--force"
}

py -3 @scriptArgs
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host "Descriptor ready: assets\generated_descriptors\$mobKey\descriptor.json" -ForegroundColor Green
Write-Host "Restart the bot UI to load newly added descriptor mobs." -ForegroundColor Yellow
