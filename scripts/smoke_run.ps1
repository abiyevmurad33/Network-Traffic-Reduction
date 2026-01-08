# Smoke run: run a short Tier A session and validate outputs.
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\smoke_run.ps1
# Optional:
#   powershell -ExecutionPolicy Bypass -File scripts\smoke_run.ps1 -Headless

param(
    [switch]$Headless
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Fail($msg) {
    Write-Error $msg
    exit 2
}

# Repo root = parent of scripts/
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $RepoRoot

# Deterministic session id for easy lookup (UTC timestamp)
$Ts = [DateTime]::UtcNow.ToString("yyyy-MM-ddTHH-mm-ssZ")
$SID = "session_smoke_$Ts"
$SessionDir = Join-Path $RepoRoot (Join-Path "data\raw_logs" $SID)

Write-Host "RepoRoot:     $RepoRoot"
Write-Host "SessionID:    $SID"
Write-Host "SessionDir:   $SessionDir"
Write-Host "Headless:     $Headless"
Write-Host ""

# Build logger command
$LoggerArgs = @(
    "-m", "src.logger.run_logger",
    "--duration-seconds", "10",
    "--protocol-tier", "A",
    "--seed", "12345",
    "--session-id", $SID
)

if ($Headless) {
    $LoggerArgs += "--headless"
}
else {
    $LoggerArgs += "--window-visible"
}

Write-Host "Running logger..."
python @LoggerArgs
if ($LASTEXITCODE -ne 0) { Fail "Logger failed with exit code $LASTEXITCODE" }

# Validate
Write-Host ""
Write-Host "Running validator..."
python -m scripts.validate_log $SessionDir
if ($LASTEXITCODE -ne 0) { Fail "Validator failed with exit code $LASTEXITCODE" }

# Print quick metadata preview
Write-Host ""
Write-Host "Smoke run PASS."
Write-Host "SessionDir: $SessionDir"
Write-Host ""
Write-Host "session.json (first 30 lines):"
Get-Content (Join-Path $SessionDir "session.json") -TotalCount 30
