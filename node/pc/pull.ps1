# Fetch results packed on the node by node/pack.sh, check them, and unpack them into artifacts\. Run on a Windows
# computer, from the repository folder, with the name pack.sh printed:
#     powershell -ExecutionPolicy Bypass -File node\pc\pull.ps1 <project>_results_<...>
# Checks the archive's SHA-256, refuses to overwrite anything already in artifacts\, unpacks, then checks every file
# against the list of hashes made on the node. The node's login is read from node.local.txt beside README.md.
param([Parameter(Mandatory = $true)][string]$Name)
$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$NodeFile = Join-Path $Repo "node.local.txt"
if (-not (Test-Path $NodeFile)) { throw "node.local.txt is missing beside README.md: write the node's login (user@host) in it, one line" }
$Node = (Get-Content $NodeFile -TotalCount 1).Trim()
$Project = (Get-Content (Join-Path $Repo "project.json") -Raw | ConvertFrom-Json).project
if ($Name -cnotmatch "^${Project}_results_[A-Za-z0-9_.-]+$") { throw "$Name is not an archive name of project $Project" }
$Artifacts = Join-Path $Repo "artifacts"
$Tar = Join-Path $env:SystemRoot "System32\tar.exe"
if (-not (Test-Path $Tar)) { throw "Windows tar not found at $Tar" }
$Incoming = Join-Path $Artifacts "_incoming"
New-Item -ItemType Directory -Force -Path $Incoming | Out-Null

# One remote pattern matches the archive and its .sha256, so scp makes one connection and asks for the password
# once. The destination is the folder itself, without a trailing backslash: Windows PowerShell passes a quoted path
# that ends in \" to scp as an escaped quote, which breaks the path.
scp "${Node}:${Project}_outgoing/$Name.tar.gz*" $Incoming
if ($LASTEXITCODE -ne 0) { throw "scp failed" }
$Archive = Join-Path $Incoming "$Name.tar.gz"
if (-not (Test-Path "$Archive.sha256")) { throw "the archive's .sha256 did not arrive; run the same command again" }
$Expected = ((Get-Content "$Archive.sha256" -Raw).Trim() -split '\s+')[0].ToLower()
$Actual = (Get-FileHash -Algorithm SHA256 $Archive).Hash.ToLower()
if ($Expected -ne $Actual) { throw "archive SHA-256 mismatch: expected $Expected, got $Actual" }

$Entries = & $Tar -tzf $Archive
if ($LASTEXITCODE -ne 0) { throw "cannot list $Archive" }
foreach ($e in $Entries) {
    if ($e.EndsWith("/")) { continue }
    if (Test-Path (Join-Path $Artifacts $e)) { throw "REFUSING: artifacts\$e already exists" }
}
& $Tar -xzf $Archive -C $Artifacts
if ($LASTEXITCODE -ne 0) { throw "tar failed" }

$List = Join-Path $Artifacts "_pulled\$Name.files.sha256"
$n = 0
foreach ($line in Get-Content $List) {
    if (-not $line.Trim()) { continue }
    $hash, $path = $line -split '\s+', 2
    $path = $path.TrimStart('*')
    $got = (Get-FileHash -Algorithm SHA256 (Join-Path $Artifacts $path)).Hash.ToLower()
    if ($got -ne $hash) { throw "hash mismatch after unpacking: $path" }
    $n++
}
Write-Host "pulled $Name : $n files unpacked into artifacts\ and checked against the node's hashes"
