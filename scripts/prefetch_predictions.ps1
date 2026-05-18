# DelphiAI — nightly prefetch of upcoming-event predictions.
#
# Wrapper around `python -m ml.prefetch_predictions`. Registered as a
# Windows Task Scheduler job (see WEEKLY_PIPELINE.md for the schtasks command).
#
# Run manually from the repo root:
#     powershell -ExecutionPolicy Bypass -File scripts\prefetch_predictions.ps1
#
# Logs go to scripts\logs\prefetch-<yyyy-MM-dd>.log (one file per day, appended).

param(
    [int]$Top = 4,
    [switch]$Force
)

$ErrorActionPreference = 'Continue'

# Force UTF-8 across the boundary — Python's stdout is UTF-8 (reconfigured in
# prefetch_predictions.py), and without this PowerShell 5.1 mis-decodes the
# byte stream as UTF-16, writing log files where every character has a NUL
# byte between it. Setting all three forces the right interpretation in both
# directions for native-exe pipelines.
[Console]::InputEncoding  = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding           = [System.Text.Encoding]::UTF8

# Resolve repo paths relative to THIS script, not the current working directory,
# so Task Scheduler can run us regardless of where it was registered from.
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot  = Split-Path -Parent $ScriptDir
$ModelsDir = Join-Path $RepoRoot 'DelphiAIApp\Models'
$LogsDir   = Join-Path $ScriptDir 'logs'
$LogFile   = Join-Path $LogsDir ("prefetch-{0:yyyy-MM-dd}.log" -f (Get-Date))

if (-not (Test-Path $LogsDir)) {
    New-Item -ItemType Directory -Path $LogsDir | Out-Null
}

# If a venv exists at repo-root .venv, activate it; otherwise rely on system Python.
$VenvActivate = Join-Path $RepoRoot '.venv\Scripts\Activate.ps1'
if (Test-Path $VenvActivate) {
    . $VenvActivate
}

Push-Location $ModelsDir
try {
    $argList = @('-m', 'ml.prefetch_predictions', '--top', $Top.ToString())
    if ($Force) { $argList += '--force' }

    Add-Content -Path $LogFile -Value "===== $(Get-Date -Format o) =====" -Encoding UTF8
    # Tee-Object in PS 5.1 hard-codes UTF-16 LE for -FilePath; Add-Content
    # honors -Encoding so the log is true UTF-8. Stream line-by-line so the
    # console mirrors what's being written.
    & python @argList | ForEach-Object {
        Write-Output $_
        Add-Content -Path $LogFile -Value $_ -Encoding UTF8
    }
    $ExitCode = $LASTEXITCODE
    Add-Content -Path $LogFile -Value "----- exit $ExitCode -----`n" -Encoding UTF8
    exit $ExitCode
} finally {
    Pop-Location
}
