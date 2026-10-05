$ErrorActionPreference='Stop'
$trainProcesses=Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match 'FireTraining[\\/](train_small|pipeline)\.py' }
foreach($trainProcess in $trainProcesses){ Invoke-CimMethod -InputObject $trainProcess -MethodName Terminate -ErrorAction SilentlyContinue | Out-Null }
$report=Get-Content 'FireTraining/deployment_report.json' -Raw | ConvertFrom-Json
$sha=(Get-FileHash 'FireTraining/fire_warehouse_candidate.pt' -Algorithm SHA256).Hash.ToLowerInvariant()
if($sha -ne $report.candidate_sha256){throw 'Validated candidate hash changed; refusing deployment'}
$bridgeRoot=(Resolve-Path '../Tools/YoloBridge').Path
if(-not (Test-Path (Join-Path $bridgeRoot 'models.before_unity_training.json'))){Copy-Item (Join-Path $bridgeRoot 'models.json') (Join-Path $bridgeRoot 'models.before_unity_training.json')}
Copy-Item -LiteralPath 'FireTraining/fire_warehouse_candidate.pt' -Destination (Join-Path $bridgeRoot 'fire_warehouse.pt')
Copy-Item -LiteralPath 'FireTraining/models.json' -Destination (Join-Path $bridgeRoot 'models.json')
Copy-Item -LiteralPath 'FireTraining/server.py' -Destination (Join-Path $bridgeRoot 'server.py')
$serverProcess=Get-CimInstance Win32_Process -Filter 'ProcessId=5856'
if($serverProcess){if($serverProcess.CommandLine -notmatch 'server.py'){throw 'Unexpected process at previous server PID'};Invoke-CimMethod -InputObject $serverProcess -MethodName Terminate -ErrorAction SilentlyContinue | Out-Null}
(Start-Process -FilePath (Join-Path $bridgeRoot '.venv/Scripts/python.exe') -ArgumentList @('-u','server.py','--device','cpu') -WorkingDirectory $bridgeRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path (Get-Location) 'FireTraining/server.stdout.log') -RedirectStandardError (Join-Path (Get-Location) 'FireTraining/server.stderr.log') -PassThru).Id

