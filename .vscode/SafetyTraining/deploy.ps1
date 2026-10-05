$ErrorActionPreference = 'Stop'
$safetyRoot = $PSScriptRoot
$bridgeRoot = (Resolve-Path (Join-Path $safetyRoot '../../Tools/YoloBridge')).Path
$report = Get-Content -LiteralPath (Join-Path $safetyRoot 'deployment_report.json') -Raw | ConvertFrom-Json
foreach ($kind in @('helmet','nohelmet','forklift')) {
    if (-not $report.evaluation.$kind.deployable) { throw "Validation failed: $kind" }
    $source = Join-Path $safetyRoot ($kind + '_warehouse_candidate.pt')
    $hash = (Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($hash -ne $report.sha256.$kind) { throw "Weights changed: $kind" }
}
$backup = Join-Path $bridgeRoot 'models.before_safety_training.json'
if (-not (Test-Path -LiteralPath $backup)) { Copy-Item -LiteralPath (Join-Path $bridgeRoot 'models.json') -Destination $backup }
foreach ($kind in @('helmet','nohelmet','forklift')) {
    Copy-Item -LiteralPath (Join-Path $safetyRoot ($kind + '_warehouse_candidate.pt')) -Destination (Join-Path $bridgeRoot ($kind + '_warehouse.pt'))
}
Copy-Item -LiteralPath (Join-Path $safetyRoot 'models.json') -Destination (Join-Path $bridgeRoot 'models.json')
$serverBackup = Join-Path $bridgeRoot 'server.before_safety_training.py'
if (-not (Test-Path -LiteralPath $serverBackup)) { Copy-Item -LiteralPath (Join-Path $bridgeRoot 'server.py') -Destination $serverBackup }
Copy-Item -LiteralPath (Join-Path $safetyRoot 'server.py') -Destination (Join-Path $bridgeRoot 'server.py')
$processes = Get-CimInstance Win32_Process -Filter "Name='python.exe'"
$byPid = @{}
foreach ($process in $processes) { $byPid[$process.ProcessId] = $process }
$venvPython = (Join-Path $bridgeRoot '.venv/Scripts/python.exe').Replace('/','\')
foreach ($process in $processes) {
    $parent = $byPid[$process.ParentProcessId]
    # Windows venv launchers spawn a base interpreter: verify that parent too.
    $isBridgePython = $process.ExecutablePath -eq $venvPython -or ($parent -and $parent.ExecutablePath -eq $venvPython)
    if ($isBridgePython -and $process.CommandLine -match 'SafetyTraining[/\\](train|pipeline)\.py') {
        Invoke-CimMethod -InputObject $process -MethodName Terminate -ErrorAction SilentlyContinue | Out-Null
        continue
    }
    if ($isBridgePython -and $process.CommandLine -match '(?:^|\s|["''])server\.py(?:\s|["'']|$)') {
        Invoke-CimMethod -InputObject $process -MethodName Terminate -ErrorAction SilentlyContinue | Out-Null
    }
}
foreach ($kind in @('helmet','nohelmet','forklift')) {
    $lock = Join-Path $safetyRoot ($kind + '.lock')
    if (Test-Path -LiteralPath $lock) { Remove-Item -LiteralPath $lock }
}
$server = Start-Process -FilePath (Join-Path $bridgeRoot '.venv/Scripts/python.exe') -ArgumentList @('-u','server.py','--device','cpu') -WorkingDirectory $bridgeRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $safetyRoot 'server.stdout.log') -RedirectStandardError (Join-Path $safetyRoot 'server.stderr.log') -PassThru
Write-Output "YOLO server restarted: PID $($server.Id)"
