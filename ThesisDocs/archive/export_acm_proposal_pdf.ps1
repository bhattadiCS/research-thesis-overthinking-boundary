param(
    [string]$ChromePath = "C:\Program Files\Google\Chrome\Application\chrome.exe",
    [string]$InputHtml = "$PSScriptRoot\acm_thesis_proposal_email.html",
    [string]$OutputPdf = "$PSScriptRoot\acm_thesis_proposal_email.pdf"
)

if (-not (Test-Path $ChromePath)) {
    throw "Chrome was not found at $ChromePath"
}

if (-not (Test-Path $InputHtml)) {
    throw "Input HTML was not found at $InputHtml"
}

$resolvedHtml = (Resolve-Path $InputHtml).Path.Replace('\', '/')
$resolvedPdf = Join-Path (Resolve-Path $PSScriptRoot).Path ([System.IO.Path]::GetFileName($OutputPdf))
$fileUrl = "file:///$resolvedHtml"


& $ChromePath --headless --disable-gpu --no-pdf-header-footer --print-to-pdf="$resolvedPdf" "$fileUrl"

$created = $false
for ($i = 0; $i -lt 20; $i++) {
    if (Test-Path $resolvedPdf) {
        $created = $true
        break
    }
    Start-Sleep -Milliseconds 250
}

if (-not $created) {
    throw "PDF export did not produce $resolvedPdf"
}

Write-Output "Created PDF: $resolvedPdf"