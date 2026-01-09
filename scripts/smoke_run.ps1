# Smoke run: run short Tier A sessions and validate outputs (multiple scenarios).
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

Write-Host "RepoRoot:     $RepoRoot"
Write-Host "Headless:     $Headless"
Write-Host ""

# Deterministic session id base for easy lookup (UTC timestamp)
$Ts = [DateTime]::UtcNow.ToString("yyyy-MM-ddTHH-mm-ssZ")

function Run-One([string]$Suffix, [string]$ScenarioCfg, [string]$Profile) {
    $SID = "session_smoke_$Suffix" + "_" + $Ts
    $SessionDir = Join-Path $RepoRoot (Join-Path "data\raw_logs" $SID)

    Write-Host "------------------------------------------------------------"
    Write-Host "SessionID:    $SID"
    Write-Host "Scenario:     $ScenarioCfg"
    Write-Host "Profile:      $Profile"
    Write-Host "SessionDir:   $SessionDir"
    Write-Host ""

    $LoggerArgs = @(
        "-m", "src.logger.run_logger",
        "--duration-seconds", "10",
        "--protocol-tier", "A",
        "--seed", "12345",
        "--session-id", $SID,
        "--scenario-config", $ScenarioCfg,
        "--profile", $Profile
    )

    if ($Headless) {
        $LoggerArgs += "--headless"
    }
    else {
        $LoggerArgs += "--window-visible"
    }

    Write-Host "Running logger..."
    # IMPORTANT: Pipe external program output to Out-Host so it does NOT become function output.
    & python @LoggerArgs | Out-Host
    if ($LASTEXITCODE -ne 0) { Fail "Logger failed with exit code $LASTEXITCODE" }

    Write-Host ""
    Write-Host "Running validator..."
    & python -m scripts.validate_log $SessionDir | Out-Host
    if ($LASTEXITCODE -ne 0) { Fail "Validator failed with exit code $LASTEXITCODE" }

    Write-Host ""
    Write-Host "PASS for $SID"

    # Return ONLY the session dir (no other pipeline output)
    return $SessionDir
}

# Run both scenarios
$dir1 = Run-One "basic"  "scenarios\basic_movement.cfg" "basic_move"
$dir2 = Run-One "strafe" "scenarios\strafe_turn.cfg"    "strafe_turn"
$dir3 = Run-One "combat" "scenarios\combat_proxy.cfg" "combat_burst"

Write-Host ""
Write-Host "Smoke run PASS (all)."
Write-Host "Basic session:  $dir1"
Write-Host "Strafe session: $dir2"
Write-Host "Combat session: $dir3"

Write-Host ""
Write-Host "basic session.json (first 20 lines):"
Get-Content -LiteralPath (Join-Path $dir1 "session.json") -TotalCount 20

Write-Host ""
Write-Host "strafe session.json (first 20 lines):"
Get-Content -LiteralPath (Join-Path $dir2 "session.json") -TotalCount 20

Write-Host ""
Write-Host "combat session.json (first 20 lines):"
Get-Content -LiteralPath (Join-Path $dir3 "session.json") -TotalCount 20

