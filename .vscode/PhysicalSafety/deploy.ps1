$ErrorActionPreference='Stop'
$bridgeRoot=(Resolve-Path (Join-Path $PSScriptRoot '../../Tools/YoloBridge')).Path
$report=Get-Content -LiteralPath (Join-Path $PSScriptRoot 'person_report.json') -Raw | ConvertFrom-Json
if($report.names.'0' -ne 'person' -or $report.task -ne 'detect' -or $report.precision -lt .6 -or $report.recall -lt .4){throw 'Person model validation failed'}
Copy-Item -LiteralPath (Join-Path $bridgeRoot 'models.json') -Destination (Join-Path $bridgeRoot 'models.before_person_training.json')
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'person_warehouse.pt') -Destination (Join-Path $bridgeRoot 'person_warehouse.pt')
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'models.json') -Destination (Join-Path $bridgeRoot 'models.json')
$processes=Get-CimInstance Win32_Process -Filter "Name='python.exe'"
$byPid=@{}
foreach($process in $processes){$byPid[$process.ProcessId]=$process}
$venvPython=(Join-Path $bridgeRoot '.venv/Scripts/python.exe').Replace('/','\')
foreach($process in $processes){
 $parent=$byPid[$process.ParentProcessId]
 if(($process.ExecutablePath -eq $venvPython -or ($parent -and $parent.ExecutablePath -eq $venvPython)) -and $process.CommandLine -match '(?:^|\s|["''])server\.py(?:\s|["'']|$)'){
  Invoke-CimMethod -InputObject $process -MethodName Terminate -ErrorAction SilentlyContinue | Out-Null
 }
}
$server=Start-Process -FilePath $venvPython -ArgumentList @('-u','server.py','--device','cpu') -WorkingDirectory $bridgeRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $PSScriptRoot 'server.stdout.log') -RedirectStandardError (Join-Path $PSScriptRoot 'server.stderr.log') -PassThru
Write-Output "Server restarted: $($server.Id)"
