param(
    [ValidateSet("firm", "soft", "glass", "unspecified")]
    [string]$MousepadCondition = "unspecified",
    [ValidateRange(1024, 65535)]
    [int]$Port = 8765
)

$ErrorActionPreference = "Stop"
$BundleRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$ExperimentPython = Join-Path $BundleRoot ".venv-experiment\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $ExperimentPython -PathType Leaf)) {
    $LocalPython = Get-Command python -ErrorAction SilentlyContinue
    if ($null -eq $LocalPython) {
        throw "Python is unavailable. Follow HUMAN_EXPERIMENT_GUIDE.md first."
    }
    $ExperimentPython = $LocalPython.Source
}

Set-Location -LiteralPath $BundleRoot
& $ExperimentPython -c "import aiohttp, numpy, pygame, yaml"
if ($LASTEXITCODE -ne 0) {
    throw "The local Python installation is missing an experiment dependency. Follow HUMAN_EXPERIMENT_GUIDE.md first."
}
& $ExperimentPython -m src.experiment.web_experiment_server `
    --input-backend windows_raw_input `
    --mousepad-condition $MousepadCondition `
    --port $Port
exit $LASTEXITCODE
