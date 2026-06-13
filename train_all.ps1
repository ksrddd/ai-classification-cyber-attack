# train_all.ps1 -- Train every model one at a time with below-normal OS priority.
#
# Below-normal priority lets Windows keep the machine responsive (mouse,
# browser, etc.) even when Python saturates the CPU. Each model still uses
# all available cores for its fits; the OS just services other tasks first.
#
# Usage:
#   .\train_all.ps1                  # full tuning (max accuracy)
#   .\train_all.ps1 -SkipTuning      # baseline only (faster, for smoke-test)

param(
    [switch]$SkipTuning
)

$ProjectRoot = $PSScriptRoot
$Python      = "$ProjectRoot\.venv\Scripts\python.exe"
$MainPy      = "$ProjectRoot\main.py"

# Order: cheapest first so you get early results fast.
# Stacking last (depends on same data, but its base models are self-contained).
$Models = @("lr", "rf", "cat", "xgb", "lgbm", "nn", "stack")

if (-not (Test-Path $Python)) {
    Write-Host "ERROR: venv not found at $Python" -ForegroundColor Red
    exit 1
}

$ExtraArgs = if ($SkipTuning) { "--skip-tuning" } else { "" }

$OverallStart = Get-Date
$Results      = @()

foreach ($model in $Models) {
    Write-Host ""
    Write-Host ("=" * 60) -ForegroundColor DarkGray
    Write-Host "  Training: $model  $(if ($SkipTuning) {'[skip-tuning]'} else {'[full tuning]'})" -ForegroundColor Cyan
    Write-Host ("=" * 60) -ForegroundColor DarkGray

    $ModelStart = Get-Date
    $ArgList    = "main.py --stage train --model $model"
    if ($SkipTuning) { $ArgList += " --skip-tuning" }

    $proc = Start-Process `
        -FilePath    $Python `
        -ArgumentList $ArgList `
        -PassThru    `
        -NoNewWindow `
        -WorkingDirectory $ProjectRoot

    # Drop to below-normal so the OS stays responsive.
    try {
        $proc.PriorityClass = [System.Diagnostics.ProcessPriorityClass]::BelowNormal
    } catch {
        Write-Host "  (Could not set priority -- continuing anyway)" -ForegroundColor Yellow
    }

    Wait-Process -Id $proc.Id
    $proc.Refresh()

    $elapsed = (Get-Date) - $ModelStart
    $status  = if ($proc.ExitCode -eq 0) { "OK" } else { "FAILED (exit $($proc.ExitCode))" }
    $color   = if ($proc.ExitCode -eq 0) { "Green" } else { "Red" }

    Write-Host "  $model : $status  [$($elapsed.ToString('mm\:ss'))]" -ForegroundColor $color
    $Results += [PSCustomObject]@{ Model = $model; Status = $status; Time = $elapsed.ToString('mm\:ss') }
}

$TotalElapsed = (Get-Date) - $OverallStart

Write-Host ""
Write-Host ("=" * 60) -ForegroundColor DarkGray
Write-Host "  Summary" -ForegroundColor White
Write-Host ("=" * 60) -ForegroundColor DarkGray
$Results | Format-Table -AutoSize
Write-Host "  Total time: $($TotalElapsed.ToString('hh\:mm\:ss'))" -ForegroundColor White
Write-Host ""
