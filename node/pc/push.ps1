# Send the code (and, with -Data, a folder of data/) to the node. Run on a Windows computer, from the repository
# folder:
#     powershell -ExecutionPolicy Bypass -File node\pc\push.ps1
#     powershell -ExecutionPolicy Bypass -File node\pc\push.ps1 -Data received_batch1
# Packs the code items (src, tests, sweeps, node, env, docs, templates, notebooks and the top-level files; the same
# list as sweepkit.hashing.CODE_ITEMS) into <project>_code_<utc>.tar.gz with its SHA-256, and copies it, with
# unpack.sh, to ~/<project>_incoming on the node in one scp (one password prompt). -Data <name> also sends data\<name>\
# as <project>_data_<name>_<utc>.tar. Then prints the line to run on the node and the code's tree digest. Nothing on
# the node changes until that line runs. The node's login (user@host) is read from node.local.txt beside README.md:
# one line, kept out of git, never sent anywhere.
param([string]$Data = "")
$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$NodeFile = Join-Path $Repo "node.local.txt"
if (-not (Test-Path $NodeFile)) { throw "node.local.txt is missing beside README.md: write the node's login (user@host) in it, one line" }
$Node = (Get-Content $NodeFile -TotalCount 1).Trim()
$Project = (Get-Content (Join-Path $Repo "project.json") -Raw | ConvertFrom-Json).project
if ($Project -cnotmatch '^[a-z][a-z0-9_]{0,23}$') { throw "project.json: the project name must be lowercase letters, digits and _" }
$Stamp = (Get-Date).ToUniversalTime().ToString("yyyyMMdd'T'HHmmss'Z'")
# Windows' own tar (bsdtar). A GNU tar from Git for Windows, if first on the PATH, would read "C:\..." as a remote
# host and fail, so the full path is used.
$Tar = Join-Path $env:SystemRoot "System32\tar.exe"
if (-not (Test-Path $Tar)) { throw "Windows tar not found at $Tar" }
$Stage = Join-Path $env:TEMP "${Project}_push_$Stamp"
$Inc = Join-Path $Stage "${Project}_incoming"
New-Item -ItemType Directory -Force -Path $Inc | Out-Null
# This computer's repository folder, so that node/pack.sh can print the whole line that pulls results back here
# (UTF-8 without a byte order mark, one line).
[System.IO.File]::WriteAllText((Join-Path $Inc "pc_folder.txt"), "windows $Repo`n")

function Write-Hash([string]$Path) {
    $name = Split-Path -Leaf $Path
    $hash = (Get-FileHash -Algorithm SHA256 $Path).Hash.ToLower()
    [System.IO.File]::WriteAllText("$Path.sha256", "$hash  $name`n")
    return $hash
}

$Code = "${Project}_code_$Stamp.tar.gz"
$Items = @("src", "tests", "sweeps", "node", "env", "docs", "templates", "notebooks", "README.md", "START_HERE.md",
           "CLAUDE.md", "TESTING.md", "LICENSE", "project.json", ".gitignore", ".gitattributes") |
         Where-Object { Test-Path (Join-Path $Repo $_) }
& $Tar -czf (Join-Path $Inc $Code) --exclude=__pycache__ --exclude=*.pyc -C $Repo @Items
if ($LASTEXITCODE -ne 0) { throw "tar failed for the code archive" }
$CodeHash = Write-Hash (Join-Path $Inc $Code)
Copy-Item (Join-Path $Repo "node\unpack.sh") $Inc
Write-Host "code archive $Code  sha256 $CodeHash"

$DataName = $null
if ($Data) {
    if ($Data -cnotmatch '^[A-Za-z0-9_.-]+$') { throw "-Data takes the name of a folder in data\ (letters, digits, _ . -)" }
    $Folder = Join-Path $Repo "data\$Data"
    if (-not (Test-Path $Folder)) { throw "no folder $Folder" }
    $DataName = "${Project}_data_${Data}_$Stamp.tar"
    & $Tar -cf (Join-Path $Inc $DataName) -C (Join-Path $Repo "data") $Data
    if ($LASTEXITCODE -ne 0) { throw "tar failed for the data archive" }
    $DataHash = Write-Hash (Join-Path $Inc $DataName)
    Write-Host "data archive $DataName  sha256 $DataHash"
}

scp -r $Inc "${Node}:"
if ($LASTEXITCODE -ne 0) { throw "scp failed; nothing was installed on the node" }
Write-Host ""
Write-Host "Copied to ~/${Project}_incoming on the node. There, run:"
Write-Host "    bash ~/${Project}_incoming/unpack.sh $Code"
if ($DataName) { Write-Host "    bash ~/${Project}_incoming/unpack.sh $DataName" }

# The tree digest of the code sent, so the node's install line can be checked against it (the same number
# node/tree_digest.sh prints there), from the project's environment when there is one, else from python on the PATH.
# It comes after the lines above, and anything that goes wrong here (no Python, or Windows' store alias for python,
# which only writes to stderr) leaves the digest out instead of stopping the script.
$Digest = $null
try {
    $ErrorActionPreference = "Continue"
    $Py = Join-Path $Repo ".venv-main\Scripts\python.exe"
    if (-not (Test-Path $Py)) { $Py = (Get-Command python -CommandType Application -ErrorAction SilentlyContinue |
                                       Select-Object -First 1).Source }
    if ($Py) {
        $env:PYTHONPATH = Join-Path $Repo "src"
        $Out = & $Py -m sweepkit.cli tree-digest --root $Repo 2>&1
        if ($LASTEXITCODE -eq 0) { $Digest = "$($Out | Select-Object -Last 1)".Trim() }
    }
} catch {
    $Digest = $null
}
if ($Digest -match '^[0-9a-f]{64}$') { Write-Host "Tree digest of the code sent: $Digest" }
else { Write-Host "(the tree digest was not computed here: no working Python; node/tree_digest.sh on the node prints it)" }
Write-Host "Local copy of what was sent: $Stage"
