param(
    [ValidateSet('cuda', 'cpu')][string]$Device = 'cuda',
    [int]$Port = 8765,
    [switch]$NoBrowser
)

$ErrorActionPreference = 'Stop'
$repoPath = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '../..')).Path
$serverPath = Join-Path $PSScriptRoot 'server.py'
$pythonPath = Join-Path $env:LOCALAPPDATA 'Programs/Python/Python312/python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    $pythonPath = (Get-Command python -ErrorAction Stop).Source
}
$demoUrl = "http://127.0.0.1:$Port"
$alreadyRunning = $false
try {
    $reply = Invoke-RestMethod -Uri "$demoUrl/api/status" -TimeoutSec 2
    $alreadyRunning = ($null -ne $reply.model -and $null -ne $reply.predictor)
} catch { }

if (-not $alreadyRunning) {
    $logPath = Join-Path $env:LOCALAPPDATA 'ResearchThesis/early_stopping_demo'
    New-Item -ItemType Directory -Path $logPath -Force | Out-Null
    $serverProcess = Start-Process -FilePath $pythonPath `
        -ArgumentList @('-B', "`"$serverPath`"", '--device', $Device, '--port', "$Port") `
        -WorkingDirectory $repoPath -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput (Join-Path $logPath 'server.stdout.log') `
        -RedirectStandardError (Join-Path $logPath 'server.stderr.log')
    $serverProcess.Id | Set-Content -LiteralPath (Join-Path $logPath 'server.pid')
    $deadline = [DateTime]::UtcNow.AddSeconds(20)
    while ([DateTime]::UtcNow -lt $deadline) {
        if ($serverProcess.HasExited) {
            throw "Demo server exited. See $(Join-Path $logPath 'server.stderr.log')"
        }
        try {
            $reply = Invoke-RestMethod -Uri "$demoUrl/api/status" -TimeoutSec 1
            $alreadyRunning = ($null -ne $reply.model)
            if ($alreadyRunning) { break }
        } catch { }
        Start-Sleep -Milliseconds 250
    }
    if (-not $alreadyRunning) { throw "Demo did not become ready. See $logPath" }
    Write-Host "Local demo server started (PID $($serverProcess.Id))."
}
Write-Host "Open $demoUrl"
if (-not $NoBrowser) {
    $chromePath = Join-Path $env:ProgramFiles 'Google/Chrome/Application/chrome.exe'
    if (Test-Path -LiteralPath $chromePath) {
        Start-Process -FilePath $chromePath -ArgumentList $demoUrl
    } else {
        Start-Process $demoUrl
    }
}
