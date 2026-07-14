# Launch the human-detection sidecar on Windows. PowerShell parallel to
# start_sidecar.sh: bootstraps a venv + deps on first run, then launches.
#
#   .\start_sidecar.ps1
#   .\start_sidecar.ps1 --port 9000
#   .\start_sidecar.ps1 -Reinstall
#
# NOTE: validated logic mirrors the bash launcher, but run/confirm this on a
# real Windows pilot PC before relying on it in production.

param(
    [switch]$Reinstall,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$SidecarArgs
)

$ErrorActionPreference = "Stop"
$RepoDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $RepoDir

$PythonBin = if ($env:PYTHON_BIN) { $env:PYTHON_BIN } else { "python" }
$VenvDir = Join-Path $RepoDir ".venv"
$ReqFile = Join-Path $RepoDir "requirements.txt"
$StampFile = Join-Path $VenvDir ".requirements.sha256"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"

if (-not (Test-Path $VenvDir)) {
    Write-Host "[sidecar] creating virtualenv at .venv (first run)..."
    & $PythonBin -m venv $VenvDir
}

$currentHash = (Get-FileHash $ReqFile -Algorithm SHA256).Hash
$needInstall = $false
if ($Reinstall) { $needInstall = $true }
elseif (-not (Test-Path $StampFile)) { $needInstall = $true }
elseif ((Get-Content $StampFile -Raw).Trim() -ne $currentHash) { $needInstall = $true }

if ($needInstall) {
    Write-Host "[sidecar] installing dependencies (first run or requirements change)..."
    & $VenvPython -m pip install --upgrade pip | Out-Null
    & $VenvPython -m pip install -r $ReqFile
    Set-Content -Path $StampFile -Value $currentHash
}

& $VenvPython "scripts\run_sidecar.py" @SidecarArgs
