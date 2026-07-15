# Train all seven canonical models sequentially with one imbalance strategy.
# Completed model artifacts are reused when the same command is run again.
#
# Recommended overnight run (full tuning + trust checks):
#   .\train_all_imbalance.ps1
#
# Faster baseline run:
#   .\train_all_imbalance.ps1 -SkipTuning -SkipCV -SkipLabelShuffle

param(
    [ValidateSet("class_weight", "targeted", "random_over", "borderline_smote", "smoteenn")]
    [string]$Strategy = "targeted",

    [ValidateSet("8gb", "16gb", "32gb", "full")]
    [string]$Preset = "8gb",

    [ValidateRange(0.01, 1.0)]
    [double]$TargetRatio = 1.00,

    [ValidateRange(0.0, 1.0)]
    [double]$TargetMaxFpr = 0.02,

    [ValidateRange(0.01, 0.49)]
    [double]$ThresholdValidationSize = 0.20,

    [string]$TargetClass = "Infiltration",

    [string]$RunName = "all_models_$(Get-Date -Format 'yyyyMMdd')_fn_aware",

    [switch]$SkipTuning,
    [switch]$SkipCV,
    [switch]$SkipLabelShuffle,
    [switch]$Force,
    [switch]$StopOnError,
    [switch]$DryRun
)

$ProjectRoot = $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$MainPy = Join-Path $ProjectRoot "main.py"
$Models = @("lr", "rf", "cat", "xgb", "lgbm", "nn", "stack")
$CanonicalNames = @{
    lr = "logistic_regression"
    rf = "random_forest"
    cat = "catboost"
    xgb = "xgboost"
    lgbm = "lightgbm"
    nn = "mlp"
    stack = "stacking"
}
$ResultsDir = Join-Path $ProjectRoot "results\$RunName"

if (-not (Test-Path -LiteralPath $Python)) {
    Write-Host "ERROR: Python venv not found at $Python" -ForegroundColor Red
    exit 1
}
if (-not (Test-Path -LiteralPath $MainPy)) {
    Write-Host "ERROR: main.py not found at $MainPy" -ForegroundColor Red
    exit 1
}

New-Item -ItemType Directory -Path $ResultsDir -Force | Out-Null

Write-Host ""
Write-Host ("=" * 72) -ForegroundColor DarkGray
Write-Host "All-model imbalance training" -ForegroundColor Cyan
Write-Host "Run name       : $RunName"
Write-Host "Strategy       : $Strategy"
Write-Host "Target         : $TargetClass"
Write-Host "Target ratio   : $TargetRatio"
Write-Host "Target max FPR : $TargetMaxFpr"
Write-Host "Calibration    : $ThresholdValidationSize"
Write-Host "RAM preset     : $Preset"
Write-Host "Skip tuning    : $SkipTuning"
Write-Host "Skip CV        : $SkipCV"
Write-Host "Skip shuffle   : $SkipLabelShuffle"
Write-Host "Results        : $ResultsDir"
Write-Host ("=" * 72) -ForegroundColor DarkGray

$OverallStart = Get-Date
$Results = @()

foreach ($Model in $Models) {
    Write-Host ""
    Write-Host ("=" * 72) -ForegroundColor DarkGray
    Write-Host "Training model: $Model" -ForegroundColor Cyan
    Write-Host ("=" * 72) -ForegroundColor DarkGray

    $Arguments = @(
        "main.py",
        "--stage", "train",
        "--model", $Model,
        "--run-name", $RunName,
        "--preset", $Preset,
        "--imbalance-strategy", $Strategy,
        "--target-class", $TargetClass,
        "--target-ratio", $TargetRatio.ToString([Globalization.CultureInfo]::InvariantCulture),
        "--target-max-fpr", $TargetMaxFpr.ToString([Globalization.CultureInfo]::InvariantCulture),
        "--threshold-validation-size", $ThresholdValidationSize.ToString([Globalization.CultureInfo]::InvariantCulture),
        "--primary-metric", "target_f2"
    )
    if ($SkipTuning) { $Arguments += "--skip-tuning" }
    if ($SkipCV) { $Arguments += "--skip-cv" }
    if ($SkipLabelShuffle) { $Arguments += "--skip-label-shuffle" }
    if ($Force) { $Arguments += "--force" }

    if ($DryRun) {
        Write-Host "DRY RUN: $Python $($Arguments -join ' ')" -ForegroundColor Yellow
        $Results += [PSCustomObject]@{
            Model = $Model
            Strategy = $Strategy
            Status = "DRY RUN"
            StartedAt = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
            Duration = "00:00:00"
        }
        continue
    }

    $ModelStart = Get-Date
    $Process = Start-Process `
        -FilePath $Python `
        -ArgumentList $Arguments `
        -PassThru `
        -NoNewWindow `
        -WorkingDirectory $ProjectRoot

    try {
        $Process.PriorityClass = [System.Diagnostics.ProcessPriorityClass]::BelowNormal
    } catch {
        Write-Host "Could not set BelowNormal priority; continuing." -ForegroundColor Yellow
    }

    # WaitForExit() is required before reading ExitCode. Wait-Process alone can
    # leave ExitCode unset for the venv launcher on Windows, which previously
    # made successful runs appear as "FAILED (exit )".
    $Process.WaitForExit()
    $ExitCode = $Process.ExitCode

    $Elapsed = (Get-Date) - $ModelStart
    $CanonicalName = $CanonicalNames[$Model]
    $ExpectedArtifacts = @(
        (Join-Path $ResultsDir "$CanonicalName.joblib"),
        (Join-Path $ResultsDir "${CanonicalName}_metrics.json"),
        (Join-Path $ResultsDir "${CanonicalName}_per_class.csv"),
        (Join-Path $ResultsDir "${CanonicalName}_confusion_matrix.png")
    )
    $ArtifactsComplete = @(
        $ExpectedArtifacts | Where-Object { -not (Test-Path -LiteralPath $_) }
    ).Count -eq 0
    $Succeeded = ($ExitCode -eq 0) -and $ArtifactsComplete
    if ($Succeeded) {
        $Status = "OK"
    } elseif ($ExitCode -eq 0) {
        $Status = "FAILED (missing artifacts)"
    } else {
        $Status = "FAILED (exit $ExitCode)"
    }
    $Color = if ($Succeeded) { "Green" } else { "Red" }
    Write-Host "$Model : $Status [$($Elapsed.ToString('hh\:mm\:ss'))]" -ForegroundColor $Color

    $Results += [PSCustomObject]@{
        Model = $Model
        Strategy = $Strategy
        Status = $Status
        StartedAt = $ModelStart.ToString("yyyy-MM-dd HH:mm:ss")
        Duration = $Elapsed.ToString("hh\:mm\:ss")
    }

    $Results | Export-Csv `
        -LiteralPath (Join-Path $ResultsDir "training_session_summary.csv") `
        -NoTypeInformation `
        -Encoding UTF8

    if (-not $Succeeded -and $StopOnError) {
        Write-Host "Stopping because -StopOnError was requested." -ForegroundColor Red
        break
    }
}

$TotalElapsed = (Get-Date) - $OverallStart
$Results | Export-Csv `
    -LiteralPath (Join-Path $ResultsDir "training_session_summary.csv") `
    -NoTypeInformation `
    -Encoding UTF8

Write-Host ""
Write-Host ("=" * 72) -ForegroundColor DarkGray
Write-Host "Training summary" -ForegroundColor White
Write-Host ("=" * 72) -ForegroundColor DarkGray
$Results | Format-Table -AutoSize
Write-Host "Total time: $($TotalElapsed.ToString('hh\:mm\:ss'))"
Write-Host "Summary: $(Join-Path $ResultsDir 'training_session_summary.csv')"

if ($Results.Status -match "FAILED") {
    exit 1
}
exit 0
