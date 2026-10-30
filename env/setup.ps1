# Build an environment on a Windows computer, from the repository folder:
#     powershell -ExecutionPolicy Bypass -File env\setup.ps1 -Environment main -Profiles "core,torch,unsupervised" -TorchIndex cpu
#     powershell -ExecutionPolicy Bypass -File env\setup.ps1 -Environment main -Profiles "core,torch" -TorchIndex cu130
# -Environment names the environment (.venv-<environment> in the repository folder, ignored by git); -Profiles lists
# the profiles installed into it, in order, separated by commas (one quoted list from Command Prompt, or a list typed
# in PowerShell itself), each from env\requirements-<profile>.txt with exact pins
# (docs\PROFILES.md says which a problem needs). -TorchIndex says where torch comes from: cpu (PyPI's Windows build
# has no CUDA), cu130 (an NVIDIA GPU, CUDA 13) or cu126 (a GPU older than Turing). Refuses if the environment exists.
# The exact versions installed go to artifacts\env\<environment>_<utc>.txt. Uses the py launcher's Python 3.12 (or
# 3.13 with -Python 3.13).
param(
    [Parameter(Mandatory = $true)][string]$Environment,
    [Parameter(Mandatory = $true)][string[]]$Profiles,
    [Parameter(Mandatory = $true)][ValidateSet("cpu", "cu130", "cu126")][string]$TorchIndex,
    [ValidateSet("3.12", "3.13")][string]$Python = "3.12"
)
$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
$Venv = Join-Path $Repo ".venv-$Environment"
$List = @($Profiles | ForEach-Object { $_ -split "," } | ForEach-Object { $_.Trim() } | Where-Object { $_ })
if ($List.Count -eq 0) { throw "-Profiles names no profile" }
foreach ($p in $List) {
    if (-not (Test-Path (Join-Path $Repo "env\requirements-$p.txt"))) { throw "no env\requirements-$p.txt" }
    if ($p -in @("rapids", "rapids-singlecell")) { throw "the $p profile needs Linux with an NVIDIA GPU (the node)" }
}
if (Test-Path $Venv) { throw "REFUSING: $Venv exists. Move it aside first if you mean to rebuild it." }
Write-Host "building $Venv with Python $Python from profiles: $($List -join ' ')"
py "-$Python" -m venv $Venv
if ($LASTEXITCODE -ne 0) { throw "could not make the environment with the py launcher (is Python $Python installed?)" }
$Py = Join-Path $Venv "Scripts\python.exe"
& $Py -m pip install --upgrade pip
# the environment's Python finds the package in src\ without PYTHONPATH (a .pth file naming this repository's src)
# (written by Python in the encoding it reads .pth files with; the code has no double quotes, which Windows
# PowerShell 5.1 would strip on the way to python.exe)
& $Py -c 'import sys, sysconfig, pathlib; pathlib.Path(sysconfig.get_paths()[''purelib''], ''sweepkit_src.pth'').write_text(sys.argv[1] + chr(10), encoding=''utf-8'' if sys.version_info >= (3, 13) else ''locale'')' (Join-Path $Repo "src")
if ($LASTEXITCODE -ne 0) { throw "could not write sweepkit_src.pth into the environment" }
foreach ($p in $List) {
    Write-Host "--- profile $p"
    if ($p -eq "torch") {
        $pin = (Select-String -Path (Join-Path $Repo "env\requirements-torch.txt") -Pattern '^torch==').Line
        & $Py -m pip install --index-url "https://download.pytorch.org/whl/$TorchIndex" $pin
    } elseif ($p -eq "pretrained") {
        $file = Join-Path $Repo "env\requirements-pretrained.txt"
        $vision = (Select-String -Path $file -Pattern '^torchvision==').Line
        & $Py -m pip install --index-url "https://download.pytorch.org/whl/$TorchIndex" $vision
        $rest = Join-Path $Venv "requirements-pretrained-rest.txt"
        Get-Content $file | Where-Object { $_ -notmatch '^torchvision==' } | Set-Content $rest
        & $Py -m pip install -r $rest
    } else {
        & $Py -m pip install -r (Join-Path $Repo "env\requirements-$p.txt")
    }
    if ($LASTEXITCODE -ne 0) { throw "pip failed for profile $p" }
}
$Stamp = (Get-Date).ToUniversalTime().ToString("yyyyMMdd'T'HHmmss'Z'")
$EnvDir = Join-Path $Repo "artifacts\env"
New-Item -ItemType Directory -Force -Path $EnvDir | Out-Null
$Freeze = Join-Path $EnvDir "${Environment}_$Stamp.txt"
("# profiles: $($List -join ' ')") | Set-Content $Freeze
& $Py -m pip freeze | Add-Content $Freeze
Write-Host "exact versions written to $Freeze"
Write-Host "next: $Py -m sweepkit.cli env-check --profiles $($List -join ' ')"
Write-Host "(or activate it: $Venv\Scripts\activate, then python -m sweepkit.cli ...)"
