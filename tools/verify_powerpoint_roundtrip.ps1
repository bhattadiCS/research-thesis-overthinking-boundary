param([Parameter(Mandatory=$true)][string]$Candidate)
$ErrorActionPreference = 'Stop'
$candidatePath = (Resolve-Path -LiteralPath $Candidate).Path
$taskBuild = Split-Path -Parent $candidatePath
$roundtripPath = Join-Path $taskBuild 'editable_roundtrip_check.pptx'
$application = New-Object -ComObject PowerPoint.Application
$initialCount = $application.Presentations.Count
$presentation = $null
try {
    # Reopen the saved candidate itself. The source candidate remains read-only;
    # all edits are applied to a disposable verification copy.
    $presentation = $application.Presentations.Open($candidatePath, -1, 0, -1)
    $presentation.Windows.Item(1).WindowState = 2
    if ($presentation.Slides.Count -ne 25) { throw 'Saved deck does not reopen with 25 slides' }
    $nativeTables = 0
    $nativeCharts = 0
    foreach ($slide in $presentation.Slides) {
        foreach ($shape in $slide.Shapes) {
            if ($shape.HasTable -eq -1) { $nativeTables++ }
            if ($shape.HasChart -eq -1) { $nativeCharts++ }
        }
    }
    if ($nativeTables -ne 13 -or $nativeCharts -ne 3) { throw 'Native object counts differ' }
    $chartShape = @($presentation.Slides.Item(15).Shapes | Where-Object { $_.HasChart -eq -1 })[0]
    $originalWidth = $chartShape.Width
    $chartShape.Width = $originalWidth + 12
    $chartShape.Chart.ChartData.Activate()
    $workbook = $chartShape.Chart.ChartData.Workbook
    $cell = $workbook.Worksheets.Item(1).Cells.Item(2, 2)
    $before = [double]$cell.Value2
    $cell.Value2 = $before + 0.001
    $workbook.Close($true)
    $tableShape = @($presentation.Slides.Item(3).Shapes | Where-Object { $_.HasTable -eq -1 })[0]
    $tableShape.Table.Cell(2, 1).Shape.TextFrame.TextRange.Text = 'Editable verification copy'
    $presentation.SaveAs($roundtripPath, 24, -1)
    $presentation.Close()
    $presentation = $application.Presentations.Open($roundtripPath, -1, 0, 0)
    $editedTable = @($presentation.Slides.Item(3).Shapes | Where-Object { $_.HasTable -eq -1 })[0]
    if ($editedTable.Table.Cell(2, 1).Shape.TextFrame.TextRange.Text -ne 'Editable verification copy') {
        throw 'Native table edit did not survive save/reopen'
    }
    $editedChart = @($presentation.Slides.Item(15).Shapes | Where-Object { $_.HasChart -eq -1 })[0]
    if ([Math]::Abs($editedChart.Width - ($originalWidth + 12)) -gt 0.01) { throw 'Chart resize did not persist' }
    $record = @{
        candidate_sha256 = (Get-FileHash -LiteralPath $candidatePath -Algorithm SHA256).Hash.ToLower()
        slide_count = 25; native_tables = $nativeTables; native_charts = $nativeCharts
        saved_candidate_reopened = $true; native_table_edit_persisted = $true
        chart_resize_persisted = $true; embedded_workbook_cell_edit_executed = $true
        workbook_cell_before = $before; workbook_cell_after = $before + 0.001
        verification_copy = $roundtripPath; source_candidate_modified = $false
    }
    $record | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $taskBuild 'roundtrip_audit.json') -Encoding UTF8
    $record | ConvertTo-Json -Depth 5
} finally {
    if ($null -ne $presentation) { $presentation.Close() }
    if ($initialCount -eq 0 -and $application.Presentations.Count -eq 0) { $application.Quit() }
    [System.Runtime.InteropServices.Marshal]::FinalReleaseComObject($application) | Out-Null
}
