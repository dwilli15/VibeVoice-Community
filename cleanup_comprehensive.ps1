# Comprehensive D:\omen Cleanup Script with Dry-Run
# This script identifies and cleans up redundant Python installations, Docker images, and cache files

param(
    [switch]$DryRun = $true,
    [switch]$Force = $false
)

Write-Host "=== COMPREHENSIVE D:\OMEN CLEANUP ANALYSIS ===" -ForegroundColor Cyan
Write-Host "Dry Run Mode: $DryRun" -ForegroundColor Yellow
Write-Host ""

# Initialize cleanup summary
$totalSavedSpace = 0
$cleanupActions = @()

function Format-FileSize {
    param([long]$Size)
    if ($Size -gt 1GB) { return "{0:N2} GB" -f ($Size / 1GB) }
    elseif ($Size -gt 1MB) { return "{0:N2} MB" -f ($Size / 1MB) }
    elseif ($Size -gt 1KB) { return "{0:N2} KB" -f ($Size / 1KB) }
    else { return "$Size bytes" }
}

function Add-CleanupAction {
    param([string]$Action, [string]$Target, [long]$Size, [string]$Reason)
    $global:cleanupActions += [PSCustomObject]@{
        Action = $Action
        Target = $Target
        Size = $Size
        SizeFormatted = Format-FileSize $Size
        Reason = $Reason
    }
    $global:totalSavedSpace += $Size
}

# ============================
# 1. DOCKER IMAGE CLEANUP
# ============================
Write-Host "1. ANALYZING DOCKER IMAGES..." -ForegroundColor Green

try {
    $dockerImages = docker images --format "table {{.Repository}}\t{{.Tag}}\t{{.ID}}\t{{.Size}}" | ConvertFrom-Csv -Delimiter "`t" -Header Repository,Tag,ID,Size
    
    Write-Host "Found $($dockerImages.Count) Docker images" -ForegroundColor Yellow
    
    # Target redundant VibeVoice images (keep only streamlined)
    $vibeVoiceImages = $dockerImages | Where-Object { $_.Repository -like "*vibevoice*" }
    foreach ($image in $vibeVoiceImages) {
        if ($image.Tag -ne "streamlined" -and $image.Repository -eq "vibevoice-community") {
            $sizeGB = [regex]::Match($image.Size, '(\d+\.?\d*)GB').Groups[1].Value
            if ($sizeGB) {
                $sizeBytes = [long]([double]$sizeGB * 1GB)
                Add-CleanupAction "Docker Image Remove" "$($image.Repository):$($image.Tag)" $sizeBytes "Redundant VibeVoice version - keeping streamlined only"
            }
        }
    }
    
    # Target old PyTorch images
    $pytorchImages = $dockerImages | Where-Object { $_.Repository -like "*pytorch*" }
    foreach ($image in $pytorchImages) {
        $sizeGB = [regex]::Match($image.Size, '(\d+\.?\d*)GB').Groups[1].Value
        if ($sizeGB) {
            $sizeBytes = [long]([double]$sizeGB * 1GB)
            Add-CleanupAction "Docker Image Remove" "$($image.Repository):$($image.Tag)" $sizeBytes "Old PyTorch development image - replaced by VibeVoice streamlined"
        }
    }
    
    # Target dangling images
    $danglingImages = docker images -f "dangling=true" -q
    if ($danglingImages) {
        foreach ($imageId in $danglingImages) {
            Add-CleanupAction "Docker Image Remove" "dangling:$imageId" 0 "Dangling image without repository"
        }
    }
} catch {
    Write-Host "Docker not available or error accessing images: $($_.Exception.Message)" -ForegroundColor Red
}

# ============================
# 2. PYTHON CACHE CLEANUP
# ============================
Write-Host "2. ANALYZING PYTHON CACHE FILES..." -ForegroundColor Green

# Clean __pycache__ in virtual environments
$venvPaths = @(
    "D:\omen\VibeVoice-Community\.venv",
    "D:\omen\tools\audiblez-gui\venv"
)

foreach ($venvPath in $venvPaths) {
    if (Test-Path $venvPath) {
        Write-Host "Analyzing virtual environment: $venvPath" -ForegroundColor Yellow
        $pycacheSize = 0
        
        try {
            $pycacheDirs = Get-ChildItem -Path $venvPath -Name "__pycache__" -Directory -Recurse -ErrorAction SilentlyContinue
            foreach ($dir in $pycacheDirs) {
                $fullPath = Join-Path $venvPath $dir
                $dirSize = (Get-ChildItem $fullPath -Recurse -File -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum
                $pycacheSize += $dirSize
            }
            
            if ($pycacheSize -gt 0) {
                Add-CleanupAction "Python Cache Clear" $venvPath $pycacheSize "Python bytecode cache cleanup"
            }
        } catch {
            Write-Host "Error analyzing $venvPath : $($_.Exception.Message)" -ForegroundColor Red
        }
    }
}

# ============================
# 3. REDUNDANT PYTHON INSTALLATIONS
# ============================
Write-Host "3. ANALYZING PYTHON INSTALLATIONS..." -ForegroundColor Green

# System-wide Python installation analysis
$pythonInstalls = @(
    @{Path="C:\Users\armad\AppData\Local\Programs\Python\Python310"; Version="3.10"; Type="Standalone"},
    @{Path="C:\Users\armad\AppData\Local\Programs\Python\Python311"; Version="3.11"; Type="Standalone"},
    @{Path="C:\ProgramData\miniconda3"; Version="3.13"; Type="Miniconda"},
    @{Path="C:\Users\armad\AppData\Local\GaiaUi\runtime\python"; Version="Unknown"; Type="Application Bundle"},
    @{Path="C:\Users\armad\AppData\Local\GAIA\python"; Version="Unknown"; Type="Application Bundle"},
    @{Path="C:\Users\armad\AppData\Local\lemonade_server\python"; Version="Unknown"; Type="Application Bundle"}
)

Write-Host "Found multiple Python installations:" -ForegroundColor Yellow
foreach ($install in $pythonInstalls) {
    if (Test-Path $install.Path) {
        try {
            $size = (Get-ChildItem $install.Path -Recurse -File -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum
            Write-Host "  $($install.Type) $($install.Version): $($install.Path) - $(Format-FileSize $size)" -ForegroundColor Cyan
            
            # Mark standalone Python 3.10 for removal (keeping 3.11 and miniconda 3.13)
            if ($install.Version -eq "3.10" -and $install.Type -eq "Standalone") {
                Add-CleanupAction "Python Install Remove" $install.Path $size "Redundant Python 3.10 - keeping 3.11 and miniconda 3.13"
            }
        } catch {
            Write-Host "  Error measuring $($install.Path): $($_.Exception.Message)" -ForegroundColor Red
        }
    }
}

# ============================
# 4. DIRECTORY CLEANUP
# ============================
Write-Host "4. ANALYZING DIRECTORIES FOR CLEANUP..." -ForegroundColor Green

# Check temp and output directories
$tempDirs = @(
    "D:\omen\temp",
    "D:\omen\outputs",
    "D:\omen\logs"
)

foreach ($tempDir in $tempDirs) {
    if (Test-Path $tempDir) {
        try {
            $size = (Get-ChildItem $tempDir -Recurse -File -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum
            if ($size -gt 100MB) {
                Add-CleanupAction "Directory Clean" $tempDir $size "Temporary files and outputs cleanup"
            }
        } catch {
            Write-Host "Error analyzing $tempDir : $($_.Exception.Message)" -ForegroundColor Red
        }
    }
}

# Check for old model cache directories
$modelCacheDirs = Get-ChildItem -Path "D:\omen" -Name "*cache*", "*models*" -Directory -ErrorAction SilentlyContinue
foreach ($cacheDir in $modelCacheDirs) {
    $fullPath = "D:\omen\$cacheDir"
    try {
        $size = (Get-ChildItem $fullPath -Recurse -File -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum
        if ($size -gt 1GB) {
            Add-CleanupAction "Cache Directory Review" $fullPath $size "Large cache directory - manual review recommended"
        }
    } catch {
        Write-Host "Error analyzing cache directory $fullPath : $($_.Exception.Message)" -ForegroundColor Red
    }
}

# ============================
# 5. DOCKER CONTAINER CLEANUP
# ============================
Write-Host "5. ANALYZING DOCKER CONTAINERS..." -ForegroundColor Green

try {
    $exitedContainers = docker ps -a --filter "status=exited" --format "{{.Names}}"
    if ($exitedContainers) {
        foreach ($container in $exitedContainers) {
            Add-CleanupAction "Docker Container Remove" $container 0 "Exited container cleanup"
        }
    }
} catch {
    Write-Host "Docker container analysis failed: $($_.Exception.Message)" -ForegroundColor Red
}

# ============================
# SUMMARY AND EXECUTION
# ============================
Write-Host ""
Write-Host "=== CLEANUP SUMMARY ===" -ForegroundColor Cyan
Write-Host "Total actions identified: $($cleanupActions.Count)" -ForegroundColor Yellow
Write-Host "Estimated space savings: $(Format-FileSize $totalSavedSpace)" -ForegroundColor Green
Write-Host ""

# Group actions by type
$groupedActions = $cleanupActions | Group-Object Action
foreach ($group in $groupedActions) {
    Write-Host "$($group.Name) ($($group.Count) items):" -ForegroundColor Magenta
    $groupSize = ($group.Group | Measure-Object Size -Sum).Sum
    Write-Host "  Total size: $(Format-FileSize $groupSize)" -ForegroundColor Cyan
    
    foreach ($action in $group.Group) {
        Write-Host "    $($action.Target) - $($action.SizeFormatted) - $($action.Reason)" -ForegroundColor White
    }
    Write-Host ""
}

# ============================
# EXECUTION PHASE
# ============================
if (-not $DryRun) {
    Write-Host "=== EXECUTING CLEANUP ===" -ForegroundColor Red
    
    if (-not $Force) {
        $confirmation = Read-Host "This will permanently delete files and Docker resources. Continue? (y/N)"
        if ($confirmation -ne 'y' -and $confirmation -ne 'Y') {
            Write-Host "Cleanup cancelled by user" -ForegroundColor Yellow
            exit
        }
    }
    
    foreach ($action in $cleanupActions) {
        Write-Host "Executing: $($action.Action) - $($action.Target)" -ForegroundColor Yellow
        
        try {
            switch ($action.Action) {
                "Docker Image Remove" {
                    if ($action.Target -like "dangling:*") {
                        $imageId = $action.Target.Split(":")[1]
                        docker rmi $imageId
                    } else {
                        docker rmi $action.Target
                    }
                }
                "Docker Container Remove" {
                    docker rm $action.Target
                }
                "Python Cache Clear" {
                    Get-ChildItem -Path $action.Target -Name "__pycache__" -Directory -Recurse | 
                        ForEach-Object { Remove-Item (Join-Path $action.Target $_) -Recurse -Force }
                }
                "Python Install Remove" {
                    Remove-Item $action.Target -Recurse -Force
                }
                "Directory Clean" {
                    Get-ChildItem -Path $action.Target -File -Recurse | Remove-Item -Force
                }
                "Cache Directory Review" {
                    Write-Host "  Manual review required for: $($action.Target)" -ForegroundColor Cyan
                }
            }
            Write-Host "  Completed successfully" -ForegroundColor Green
        } catch {
            Write-Host "  Error: $($_.Exception.Message)" -ForegroundColor Red
        }
    }
    
    Write-Host ""
    Write-Host "=== CLEANUP COMPLETED ===" -ForegroundColor Green
    Write-Host "Estimated space freed: $(Format-FileSize $totalSavedSpace)" -ForegroundColor Cyan
} else {
    Write-Host "=== DRY RUN COMPLETE ===" -ForegroundColor Green
    Write-Host "To execute cleanup, run with -DryRun:`$false" -ForegroundColor Yellow
    Write-Host "To force execution without prompts, add -Force" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Cleanup script completed successfully!" -ForegroundColor Green
