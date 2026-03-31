param(
    [Parameter(Mandatory = $true)]
    [string]$ServerName
)

$ErrorActionPreference = 'Stop'

$workspaceRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$configCandidates = @()

if (-not [string]::IsNullOrWhiteSpace($env:COPILOT_MCP_CONFIG)) {
    $configCandidates += $env:COPILOT_MCP_CONFIG
}

$configCandidates += (Join-Path $workspaceRoot '.mcp_local.json')
$configCandidates += 'C:\Users\bhatt\.gemini\antigravity\mcp_config.json'

$configPath = $configCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1

if (-not $configPath) {
    throw "No MCP config found. Checked: $($configCandidates -join ', ')"
}

$config = Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json
$serverProperty = $config.mcpServers.PSObject.Properties | Where-Object { $_.Name -eq $ServerName } | Select-Object -First 1

if (-not $serverProperty) {
    throw "MCP server '$ServerName' was not found in $configPath"
}

$server = $serverProperty.Value
$command = [string]$server.command

if ([string]::IsNullOrWhiteSpace($command)) {
    throw "MCP server '$ServerName' does not define a command"
}

if ($server.env) {
    foreach ($envProperty in $server.env.PSObject.Properties) {
        $envValue = [string]$envProperty.Value

        if ($envValue -match '^\$\{(.+)\}$') {
            $envKey = $Matches[1]
            $resolvedValue = [Environment]::GetEnvironmentVariable($envKey, 'Process')

            if ([string]::IsNullOrWhiteSpace($resolvedValue)) {
                $resolvedValue = [Environment]::GetEnvironmentVariable($envKey, 'User')
            }

            if ([string]::IsNullOrWhiteSpace($resolvedValue)) {
                $resolvedValue = [Environment]::GetEnvironmentVariable($envKey, 'Machine')
            }

            if ([string]::IsNullOrWhiteSpace($resolvedValue)) {
                throw "Environment variable '$envKey' is required for MCP server '$ServerName'"
            }

            $envValue = $resolvedValue
        }

        [Environment]::SetEnvironmentVariable($envProperty.Name, $envValue, 'Process')
    }
}

$arguments = @()
if ($server.args) {
    foreach ($arg in $server.args) {
        $arguments += [string]$arg
    }
}

& $command @arguments

if ($null -ne $LASTEXITCODE) {
    exit $LASTEXITCODE
}