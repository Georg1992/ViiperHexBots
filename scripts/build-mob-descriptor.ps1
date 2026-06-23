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
$cliPath = Join-Path $root "mob-recognition\cli.py"

if (-not (Test-Path $sprPath)) {
    throw "Missing SPR file: $sprPath"
}
if (-not (Test-Path $actPath)) {
    throw "Missing ACT file: $actPath"
}
if (-not (Test-Path $cliPath)) {
    throw "Missing mob recognition CLI: $cliPath"
}

$scriptArgs = @($cliPath, "build-simple-descriptor", "--mob", $mobKey)
if ($Force) {
    $scriptArgs += "--force"
}

py -3 @scriptArgs
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host "Descriptor ready: generated_descriptors\$mobKey\simple\descriptor.json" -ForegroundColor Green
Write-Host "Restart the bot UI to load newly added descriptor mobs." -ForegroundColor Yellow
