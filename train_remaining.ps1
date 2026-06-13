# train_remaining.ps1 -- Train only the 3 models not yet saved to models/
# Runs each one at a time with BelowNormal OS priority so the machine stays usable.
#
# Usage:
#   .\train_remaining.ps1               # with hyperparameter tuning
#   .\train_remaining.ps1 -SkipTuning   # baseline only (faster smoke-test)

param(
    [switch]$SkipTuning
)

$ProjectRoot = $PSScriptRoot
$Python      = "$ProjectRoot\.venv\Scripts\python.exe"

$Models = @("lgbm", "nn", "stack")

if (-not (Test-Path $Python)) {
    Write-Host "ERROR: venv not found at $Python" -ForegroundColor Red
    exit 1
}

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
        -FilePath         $Python `
        -ArgumentList     $ArgList `
        -PassThru         `
        -NoNewWindow      `
        -WorkingDirectory $ProjectRoot

    # BelowNormal priority keeps mouse/browser responsive during training.
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

    if ($proc.ExitCode -ne 0) {
        Write-Host "  Stopping early -- fix the error above then re-run." -ForegroundColor Red
        break
    }
}

$TotalElapsed = (Get-Date) - $OverallStart

Write-Host ""
Write-Host ("=" * 60) -ForegroundColor DarkGray
Write-Host "  Summary" -ForegroundColor White
Write-Host ("=" * 60) -ForegroundColor DarkGray
$Results | Format-Table -AutoSize
Write-Host "  Total time: $($TotalElapsed.ToString('hh\:mm\:ss'))" -ForegroundColor White
Write-Host ""
