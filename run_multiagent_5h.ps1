$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$resultName = "comparison_multiagent_seed2026_v2"
$deadline = (Get-Date).AddMinutes(275)

$experiments = @(
    @{Name="BP-Original"; Folder="Multi-scalePolytope"; Mode="original" },
    @{Name="MSP-Pool"; Folder="Multi-scalePolytope"; Mode="pool" },
    @{Name="MSP-Flat"; Folder="Multi-scalePolytope"; Mode="flat" }
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

$completed = 0
$skipped = 0

:targets for ($target = 0; $target -lt 50; $target++) {

    foreach ($exp in $experiments) {

        if ((Get-Date) -ge $deadline) {
            Write-Host "Time budget reached. Stopping new experiments."
            break targets
        }

        $project = $PSScriptRoot
        Set-Location $project

        if ($exp.Mode -eq "") {
            $logDir = Join-Path $project "$resultName\mean\800\$target"
        } else {
            $logDir = Join-Path $project "$resultName\$($exp.Mode)\mean\800\$target"
        }

        $logFile = Join-Path $logDir "log.txt"
        $poisonFile = Join-Path $logDir "poison_00799.pth"

        # Only skip when both the final poison and evaluation summary exist
        $alreadyDone = $false
        if ((Test-Path $poisonFile) -and (Test-Path $logFile)) {
            $alreadyDone = [bool](
                Select-String -Path $logFile `
                    -Pattern "SUMMARY" -Quiet
            )
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
            "craft_poisons_transfer.py",
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
