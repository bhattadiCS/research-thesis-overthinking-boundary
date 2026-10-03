param(
    [string]$Source = 'ThesisDocs/defense/defense_slides.json',
    [string]$BuildRoot = 'tmp/presentations/defense'
)

# User-approved fallback: installed PowerPoint replaces the unavailable bundled
# renderer. This creates a new minimized presentation, never closes user files,
# and retains native editable text, tables, charts and embedded chart workbooks.
$ErrorActionPreference = 'Stop'
$taskRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$sourcePath = (Resolve-Path -LiteralPath (Join-Path $taskRoot $Source)).Path
$deck = Get-Content -LiteralPath $sourcePath -Raw -Encoding UTF8 | ConvertFrom-Json
if ($deck.slides.Count -ne 25) { throw 'Exactly 25 slides are required' }
if (($deck.slides | Measure-Object -Property seconds -Sum).Sum -ne 1800) { throw 'Speaker timings must total 1800 seconds' }
$buildId = Get-Date -Format 'yyyyMMdd_HHmmss'
$taskBuild = Join-Path (Join-Path $taskRoot $BuildRoot) $buildId
$renderPath = Join-Path $taskBuild 'rendered'
New-Item -ItemType Directory -Path $renderPath -Force | Out-Null
$candidate = Join-Path $taskBuild 'candidate.pptx'
$pdf = Join-Path $taskBuild 'defense_preview.pdf'
$script:layoutAudit = [System.Collections.Generic.List[object]]::new()
$script:navy = 0x493225
$script:ink = 0x3A332D
$script:blue = 0xB27200
$script:orange = 0x005ED5

function Add-Text($Slide, [string]$Text, [float]$X, [float]$Y, [float]$W, [float]$H, [float]$Size, [bool]$Bold = $false, [int]$Color = $script:ink) {
    $shape = $Slide.Shapes.AddTextbox(1, $X, $Y, $W, $H)
    $shape.TextFrame.MarginLeft = 0
    $shape.TextFrame.MarginRight = 0
    $shape.TextFrame.MarginTop = 0
    $shape.TextFrame.MarginBottom = 0
    $shape.TextFrame.WordWrap = -1
    $shape.TextFrame.AutoSize = 0
    $range = $shape.TextFrame.TextRange
    $range.Text = $Text
    $range.Font.Name = 'Arial'
    $range.Font.Size = $Size
    $range.Font.Bold = $(if ($Bold) { -1 } else { 0 })
    $range.Font.Color.RGB = $Color
    $range.ParagraphFormat.SpaceAfter = 10
    # PowerPoint's default textbox autofit can shrink the newly inserted empty
    # shape while margins are assigned. Restore the declared geometry after
    # setting all text properties and disable both legacy and modern autofit.
    $shape.TextFrame2.AutoSize = 0
    $shape.Left = $X
    $shape.Top = $Y
    $shape.Width = $W
    $shape.Height = $H
    return $shape
}

function Add-Bullets($Slide, $Lines, [float]$X, [float]$Y, [float]$W, [float]$H, [float]$Size = 24) {
    $shape = Add-Text $Slide ($Lines -join "`r") $X $Y $W $H $Size
    $range = $shape.TextFrame.TextRange
    $range.ParagraphFormat.Bullet.Visible = -1
    $range.ParagraphFormat.Bullet.Character = 8226
    $range.ParagraphFormat.Bullet.RelativeSize = 0.8
    $range.ParagraphFormat.SpaceAfter = 16
    $shape.TextFrame.Ruler.Levels.Item(1).FirstMargin = 0
    $shape.TextFrame.Ruler.Levels.Item(1).LeftMargin = 20
    return $shape
}

function Add-NativeTable($Slide, $Data, [float]$X = 40, [float]$Y = 140, [float]$W = 880, [float]$H = 300) {
    $rows = @($Data.rows)
    $columns = @($Data.headers)
    $shape = $Slide.Shapes.AddTable($rows.Count + 1, $columns.Count, $x, $y, $w, $h)
    $table = $shape.Table
    if ($Data.widths) {
        for ($column = 1; $column -le $columns.Count; $column++) {
            $table.Columns.Item($column).Width = [float]$Data.widths[$column - 1]
        }
    }
    for ($row = 1; $row -le $rows.Count + 1; $row++) {
        for ($column = 1; $column -le $columns.Count; $column++) {
            $cell = $table.Cell($row, $column).Shape
            $cell.TextFrame.MarginLeft = 8
            $cell.TextFrame.MarginRight = 8
            $cell.TextFrame.MarginTop = 6
            $cell.TextFrame.MarginBottom = 6
            $cell.TextFrame.VerticalAnchor = 3
            $range = $cell.TextFrame.TextRange
            $range.Text = $(if ($row -eq 1) { [string]$columns[$column - 1] } else { [string]$rows[$row - 2][$column - 1] })
            $range.Font.Name = 'Arial'
            $range.Font.Size = $(if ($Data.font_size) { [float]$Data.font_size } else { 18 })
            $range.Font.Bold = $(if ($row -eq 1) { -1 } else { 0 })
            $range.Font.Color.RGB = $(if ($row -eq 1) { 0xFFFFFF } else { $script:ink })
            $cell.Fill.ForeColor.RGB = $(if ($row -eq 1) { $script:navy } else { 0xFFFFFF })
        }
    }
    return $shape
}

function Add-NativeChart($Slide, $Data, [float]$Height = 338) {
    if ($Data.kind -notin @('bar', 'column', 'line')) { throw 'Supported editable chart kinds are bar, column and line' }
    $chartType = switch ($Data.kind) { 'bar' { 57 } 'column' { 51 } 'scatter' { -4169 } default { 65 } }
    $shape = $Slide.Shapes.AddChart($chartType, 40, 137, 650, $Height)
    $chart = $shape.Chart
    $chart.ChartArea.Font.Name = 'Arial'
    $chart.ChartArea.Font.Size = 15
    $chart.HasTitle = $false
    $chart.HasLegend = (@($Data.series).Count -gt 1)
    if ($chart.HasLegend) {
        $chart.Legend.Position = -4107
        $chart.Legend.Font.Name = 'Arial'
        $chart.Legend.Font.Size = 15
    }
    $chart.ChartArea.Format.Fill.ForeColor.RGB = 0xFFFFFF
    $chart.ChartArea.Format.Line.Visible = 0
    $chart.ChartData.Activate()
    $workbook = $chart.ChartData.Workbook
    $excelApplication = $workbook.Application
    $excelWasVisible = $excelApplication.Visible
    $hideOwnExcel = $excelApplication.Workbooks.Count -eq 1
    if ($hideOwnExcel) { $excelApplication.Visible = $false }
    try {
        $worksheet = $workbook.Worksheets.Item(1)
        $worksheet.Cells.Clear() | Out-Null
        while ($chart.SeriesCollection().Count -gt 0) { $chart.SeriesCollection().Item(1).Delete() }
        $seriesIndex = 0
        foreach ($series in @($Data.series)) {
            $seriesIndex++
            if ($Data.kind -eq 'scatter') {
                $xColumn = 2 * $seriesIndex - 1
                $yColumn = 2 * $seriesIndex
                $worksheet.Cells.Item(1, $xColumn).Value2 = [string]$Data.x_title
                $worksheet.Cells.Item(1, $yColumn).Value2 = [string]$series.name
                for ($i = 0; $i -lt @($series.values).Count; $i++) {
                    $worksheet.Cells.Item($i + 2, $xColumn).Value2 = [double]$series.x[$i]
                    $worksheet.Cells.Item($i + 2, $yColumn).Value2 = [double]$series.values[$i]
                }
            } else {
                $xColumn = 1
                $yColumn = $seriesIndex + 1
                $worksheet.Cells.Item(1, 1).Value2 = [string]$Data.x_title
                $worksheet.Cells.Item(1, $yColumn).Value2 = [string]$series.name
                for ($i = 0; $i -lt @($series.values).Count; $i++) {
                    $worksheet.Cells.Item($i + 2, 1).Value2 = [string]$Data.categories[$i]
                    $worksheet.Cells.Item($i + 2, $yColumn).Value2 = [double]$series.values[$i]
                }
            }
            $lastRow = @($series.values).Count + 1
        }
        $dataAddress = $worksheet.Range($worksheet.Cells.Item(1, 1), $worksheet.Cells.Item($lastRow, $seriesIndex + 1)).Address()
        $sheetName = $worksheet.Name
        $chart.SetSourceData("='$sheetName'!$dataAddress", 2)
        # SetSourceData restores Office's default chart styles and may add a
        # title. Apply the approved typography after binding the native data.
        $chart.ChartArea.Font.Name = 'Arial'
        $chart.ChartArea.Font.Size = 15
        $chart.HasTitle = $false
        $chart.HasLegend = (@($Data.series).Count -gt 1)
        if ($chart.SeriesCollection().Count -ne @($Data.series).Count) { throw 'Native chart series count differs from source' }
        for ($seriesIndex = 1; $seriesIndex -le @($Data.series).Count; $seriesIndex++) {
            $added = $chart.SeriesCollection().Item($seriesIndex)
            $color = $(if ($seriesIndex -eq 1) { $script:blue } elseif ($seriesIndex -eq 2) { $script:orange } else { $script:navy })
            $added.Format.Line.ForeColor.RGB = $color
            $added.Format.Line.Weight = 2
            $added.Format.Fill.ForeColor.RGB = $color
            if ($Data.kind -ne 'bar') { $added.MarkerSize = 8 }
        }
        # An embedded chart workbook has no ordinary Save path. Closing with
        # SaveChanges below commits it through its PowerPoint OLE container.
        foreach ($axisType in @(1, 2)) {
            $axis = $chart.Axes($axisType)
            $axis.HasTitle = $true
            $axis.AxisTitle.Text = $(if ($axisType -eq 1) { [string]$Data.x_title } elseif ($Data.y_title) { [string]$Data.y_title } else { [string]$Data.y_axis_title })
            $axis.AxisTitle.Font.Name = 'Arial'
            $axis.AxisTitle.Font.Size = 17
            $axis.TickLabels.Font.Name = 'Arial'
            $axis.TickLabels.Font.Size = 15
            if ($axisType -eq 1) { $axis.TickLabelPosition = -4134 }
        }
        if ($null -ne $Data.y_min) { $chart.Axes(2).MinimumScale = [double]$Data.y_min }
        if ($null -ne $Data.y_max) { $chart.Axes(2).MaximumScale = [double]$Data.y_max }
        if ($null -ne $Data.y_axis_min) { $chart.Axes(2).MinimumScale = [double]$Data.y_axis_min }
        if ($null -ne $Data.y_axis_max) { $chart.Axes(2).MaximumScale = [double]$Data.y_axis_max }
        if ($Data.y_format) { $chart.Axes(2).TickLabels.NumberFormat = [string]$Data.y_format }
        $chart.Refresh()
    } finally {
        $workbook.Close($true)
        if ($hideOwnExcel -and $excelWasVisible) { $excelApplication.Visible = $true }
    }
    return $shape
}

$powerpoint = New-Object -ComObject PowerPoint.Application
$initialPresentationCount = $powerpoint.Presentations.Count
$presentation = $null
try {
    $powerpoint.DisplayAlerts = 1
    # Office requires a document window to create native chart objects. Keep
    # this task's window minimized; never change another presentation's window.
    $presentation = $powerpoint.Presentations.Add(-1)
    $presentation.Windows.Item(1).WindowState = 2
    $presentation.PageSetup.SlideWidth = 960
    $presentation.PageSetup.SlideHeight = 540
    $index = 0
    foreach ($content in $deck.slides) {
        $index++
        $slide = $presentation.Slides.Add($index, 12)
        $slide.FollowMasterBackground = 0
        $slide.Background.Fill.ForeColor.RGB = 0xFFFFFF
        if ($index -eq 1) {
            $null = Add-Text $slide ([string]$content.title) 40 120 880 210 46 $true $script:navy
            $null = Add-Text $slide (@($content.bullets) -join "`r") 40 320 880 175 24
        } else {
            $null = Add-Text $slide ([string]$content.title) 40 32 880 92 32 $true $script:navy
            if ($content.table) {
                $null = Add-NativeTable $slide $content.table
                if ($content.table_caption) { $null = Add-Text $slide ([string]$content.table_caption) 40 452 880 48 17 }
            } elseif ($content.chart) {
                $chartHeight = $(if ($content.table_secondary) { 238 } else { 338 })
                $chartShape = Add-NativeChart $slide $content.chart $chartHeight
                if ($content.table_secondary) {
                    $chartShape.Chart.Axes(2).MajorUnit = 0.1
                    $null = Add-NativeTable $slide $content.table_secondary 40 395 880 108
                }
                $sideHeight = $(if ($content.table_secondary) { 240 } else { 305 })
                if ($content.bullets) { $null = Add-Text $slide (@($content.bullets) -join "`r`r") 725 150 195 $sideHeight 19 }
            } elseif ($content.figure) {
                $figurePath = (Resolve-Path -LiteralPath (Join-Path $taskRoot $content.figure)).Path
                $picture = $slide.Shapes.AddPicture($figurePath, 0, -1, 40, 138, -1, -1)
                $picture.LockAspectRatio = -1
                $scale = [Math]::Min(650 / $picture.Width, 335 / $picture.Height)
                $picture.Width *= $scale
                if ($content.bullets) { $null = Add-Text $slide (@($content.bullets) -join "`r`r") 725 150 195 305 19 }
            } else {
                $bodyHeight = $(if ($content.equation) { 215 } else { 300 })
                $null = Add-Bullets $slide @($content.bullets) 40 150 880 $bodyHeight 25
                if ($content.equation) { $null = Add-Text $slide ([string]$content.equation) 40 390 880 75 26 $false $script:blue }
            }
        }
        if ($content.footnote) { $null = Add-Text $slide ([string]$content.footnote) 40 485 850 28 12 }
        $null = Add-Text $slide ([string]$index) 900 508 20 18 11
        $notes = "Target duration: $($content.seconds) seconds.`r`r$($content.speaker_notes)"
        if ($content.sources) { $notes += "`r`rSources:`r" + (@($content.sources) -join "`r") }
        if ($content.references) {
            $referenceLines = foreach ($reference in @($content.references)) {
                if ($reference -is [string]) { $reference } else { "$($reference.title): $($reference.url)" }
            }
            $notes += "`r`rReferences:`r" + ($referenceLines -join "`r")
        }
        $slide.NotesPage.Shapes.Placeholders.Item(2).TextFrame.TextRange.Text = $notes
        $slide.SlideShowTransition.AdvanceOnTime = 0
        foreach ($shape in $slide.Shapes) {
            if ($shape.HasTextFrame -eq -1 -and $shape.TextFrame.HasText -eq -1) {
                $boundHeight = $shape.TextFrame.TextRange.BoundHeight
                if ($boundHeight -gt $shape.Height + 1) {
                    $script:layoutAudit.Add(@{slide = $index; shape = $shape.Name; issue = 'text exceeds shape height'; bound = $boundHeight; height = $shape.Height})
                }
            }
            if ($shape.Left -lt 0 -or $shape.Top -lt 0 -or ($shape.Left + $shape.Width) -gt 961 -or ($shape.Top + $shape.Height) -gt 541) {
                $script:layoutAudit.Add(@{slide = $index; shape = $shape.Name; issue = 'shape outside canvas'})
            }
        }
    }
    $presentation.SaveAs($candidate, 24, -1)
    $presentation.SaveAs($pdf, 32)
    for ($i = 1; $i -le $presentation.Slides.Count; $i++) {
        $presentation.Slides.Item($i).Export((Join-Path $renderPath ('slide_{0:00}.png' -f $i)), 'PNG', 1600, 900)
    }
    $record = @{
        source = $sourcePath; source_sha256 = (Get-FileHash -LiteralPath $sourcePath -Algorithm SHA256).Hash.ToLower()
        candidate = $candidate; candidate_sha256 = (Get-FileHash -LiteralPath $candidate -Algorithm SHA256).Hash.ToLower()
        preview_pdf = $pdf; rendered = $renderPath; slide_count = $presentation.Slides.Count
        timing_seconds = 1800; application = 'Installed Microsoft PowerPoint COM'; font_policy = @{basis = 'design'; families = @('Arial')}
        native_table_slides = @((1..25) | Where-Object { $deck.slides[$_ - 1].table })
        secondary_native_table_slides = @((1..25) | Where-Object { $deck.slides[$_ - 1].table_secondary })
        native_chart_slides = @((1..25) | Where-Object { $deck.slides[$_ - 1].chart })
        layout_findings = @($script:layoutAudit.ToArray()); visual_review = 'Required before final handoff'
    }
    $record | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath (Join-Path $taskBuild 'build_manifest.json') -Encoding UTF8
    $record | ConvertTo-Json -Depth 12
    if ($script:layoutAudit.Count -gt 0) { throw "PowerPoint layout audit found $($script:layoutAudit.Count) defects" }
} finally {
    if ($null -ne $presentation) { $presentation.Close() }
    if ($initialPresentationCount -eq 0 -and $powerpoint.Presentations.Count -eq 0) { $powerpoint.Quit() }
    [System.Runtime.InteropServices.Marshal]::FinalReleaseComObject($powerpoint) | Out-Null
}
