param([int]$StartTarget=0, [int]$EndTarget=49,
      [string]$Root=(Split-Path -Parent $PSScriptRoot))
$ErrorActionPreference = "Stop"

$root = $Root
$resultName = "comparison_bp3pool_seed2026_v2"

$experiments = @(
    @{Name="BP-3Layer-Pool"; Folder="Multi-scalePolytope"; Mode="pool-layerwise" }
)

$proxyNets = @(
    "DPN92", "SENet18", "ResNet50",
    "ResNeXt29_2x64d", "GoogLeNet", "MobileNetV2"
)

$checkpoints = @(
    "ckpt-%s-4800-dp0.200-droplayer0.000-seed1226.t7",
    "ckpt-%s-4800-dp0.250-droplayer0.000-seed1226.t7",
    "ckpt-%s-4800-dp0.300-droplayer0.000.t7"
)

if ($StartTarget -lt 0 -or $EndTarget -gt 49 -or $StartTarget -gt $EndTarget) {
    throw "Target range must be within 0..49"
}

$completed = 0
$skipped = 0

:targets for ($target = $StartTarget; $target -le $EndTarget; $target++) {

    foreach ($exp in $experiments) {

        $project = Join-Path $root (Split-Path -Leaf $PSScriptRoot)
        Set-Location $project

        if ($exp.Mode -eq "") {
            $logDir = Join-Path $project "$resultName\mean\800\$target"
        } else {
            $logDir = Join-Path $project "$resultName\$($exp.Mode)\mean\800\$target"
        }

        $logFile = Join-Path $logDir "log.txt"
        $poisonFile = Join-Path $logDir "poison_00799.pth"

        $alreadyDone = $false
        if ((Test-Path $poisonFile) -and (Test-Path $logFile)) {
            $text = Get-Content -LiteralPath $logFile -Raw
            $parts = $text -split "------SUMMARY------"
            $summary = $parts[-1]
            $alreadyDone = ($parts.Count -gt 1) -and
                ($summary -match "(?m)^\s*TARGET INDEX:\s*$target\s*$") -and
                ($summary -match "(?m)^\s*DenseNet121\s+[01]\s*$") -and
                ($summary -match "(?m)^\s*ResNet18\s+[01]\s*$")
        }

        if ($alreadyDone) {
            Write-Host "[SKIP] $($exp.Name) target=$target"
            $skipped++
            continue
        }

        Write-Host ""
        Write-Host "=============================================="
        Write-Host "[RUN] $($exp.Name) TARGET=$target"
        Write-Host "Time: $(Get-Date)"
        Write-Host "=============================================="

        $params = @(
            "craft_poisons_bp3pool.py",
            "--gpu", "0",
            "--device", "cuda",
            "--mode", "mean",
            "--seed", "2026",
            "--substitute-nets"
        )

        $params += $proxyNets
        $params += "--subs-chk-name"
        $params += $checkpoints
        $params += @(
            "--subs-dp", "0.2", "0.25", "0.3",
            "--target-net", "DenseNet121", "ResNet18",
            "--test-chk-name", "ckpt-%s-4800.t7",
            "--target-label", "6",
            "--target-index", "$target",
            "--poison-label", "8",
            "--poison-num", "5",
            "--poison-ites", "800",
            "--poison-epsilon", "0.1",
            "--retrain-epochs", "60",
            "--chk-path", $resultName
        )

        if ($exp.Mode -ne "") {
            $params += @("--feature-mode", $exp.Mode)
        }

        & python @params

        if ($LASTEXITCODE -ne 0) {
            throw "FAILED: $($exp.Name), target=$target, exit=$LASTEXITCODE"
        }

        if (!(Test-Path $poisonFile)) {
            throw "Final poison checkpoint missing: $poisonFile"
        }

        $completed++
        Write-Host "[DONE] $($exp.Name) TARGET=$target"
    }
}

Set-Location $root
Write-Host ""
Write-Host "Finished this session."
Write-Host "New completed runs: $completed"
Write-Host "Skipped completed runs: $skipped"
Write-Host "Current time: $(Get-Date)"
