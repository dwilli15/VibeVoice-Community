# D:\omen Drive Cleanup Script
# WARNING: This script will permanently delete files. Review carefully before execution.
# Run this script with administrative privileges for best results.

param(
    [switch]$WhatIf = $false,
    [switch]$Confirm = $true
)

Write-Host "=== D:\omen Drive Cleanup Script ===" -ForegroundColor Green
Write-Host "This script will clean up duplicates, caches, and regenerable files" -ForegroundColor Yellow
Write-Host ""

if ($WhatIf) {
    Write-Host "RUNNING IN PREVIEW MODE - No files will be deleted" -ForegroundColor Cyan
}

$totalSaved = 0

# Function to safely remove directory with size reporting
function Remove-DirectoryWithReport {
    param($Path, $Description)
    
    if (Test-Path $Path) {
        try {
            $size = (Get-ChildItem $Path -Recurse -File -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum
            $sizeGB = [math]::Round($size / 1GB, 3)
            
            if ($WhatIf) {
                Write-Host "WOULD DELETE: $Description - $Path ($sizeGB GB)" -ForegroundColor Yellow
            } else {
                if ($Confirm) {
                    $response = Read-Host "Delete $Description at $Path ($sizeGB GB)? (y/N)"
                    if ($response -ne 'y' -and $response -ne 'Y') {
                        Write-Host "Skipped: $Description" -ForegroundColor Gray
                        return 0
                    }
                }
                Remove-Item $Path -Recurse -Force -ErrorAction SilentlyContinue
                Write-Host "DELETED: $Description ($sizeGB GB)" -ForegroundColor Red
            }
            return $size
        } catch {
            Write-Host "ERROR: Could not process $Path - $($_.Exception.Message)" -ForegroundColor Red
            return 0
        }
    } else {
        Write-Host "NOT FOUND: $Path" -ForegroundColor Gray
        return 0
    }
}

# Function to remove Python cache files
function Remove-PythonCaches {
    Write-Host "`n--- Cleaning Python Cache Files ---" -ForegroundColor Cyan
    
    $cacheSize = 0
    $cacheCount = 0
    
    Get-ChildItem -Path "d:\omen" -Recurse -Directory -Name "__pycache__" -ErrorAction SilentlyContinue | ForEach-Object {
        $cachePath = "d:\omen\$_"
        $size = Remove-DirectoryWithReport $cachePath "Python cache directory"
        $cacheSize += $size
        $cacheCount++
    }
    
    # Also remove .pyc files scattered throughout
    Get-ChildItem -Path "d:\omen" -Recurse -File -Filter "*.pyc" -ErrorAction SilentlyContinue | ForEach-Object {
        $size = $_.Length
        if ($WhatIf) {
            Write-Host "WOULD DELETE: Python bytecode file - $($_.FullName)" -ForegroundColor Yellow
        } else {
            if (-not $Confirm -or (Read-Host "Delete $($_.FullName)? (y/N)") -eq 'y') {
                Remove-Item $_.FullName -Force -ErrorAction SilentlyContinue
                Write-Host "DELETED: $($_.FullName)" -ForegroundColor Red
                $cacheSize += $size
            }
        }
    }
    
    Write-Host "Python caches: $cacheCount directories, $([math]::Round($cacheSize / 1GB, 3)) GB total" -ForegroundColor Green
    return $cacheSize
}

# Function to remove virtual environments (with confirmation)
function Remove-VirtualEnvironments {
    Write-Host "`n--- Virtual Environment Cleanup ---" -ForegroundColor Cyan
    
    $venvSize = 0
    
    # .venv directories
    Get-ChildItem -Path "d:\omen" -Recurse -Directory -Name ".venv" -ErrorAction SilentlyContinue | ForEach-Object {
        $venvPath = "d:\omen\$_"
        Write-Host "Found virtual environment: $venvPath" -ForegroundColor Yellow
        Write-Host "WARNING: Virtual environments can be recreated but may require time to reinstall packages" -ForegroundColor Red
        $size = Remove-DirectoryWithReport $venvPath "Python virtual environment"
        $venvSize += $size
    }
    
    # venv directories  
    Get-ChildItem -Path "d:\omen" -Recurse -Directory -Name "venv" -ErrorAction SilentlyContinue | ForEach-Object {
        $venvPath = "d:\omen\$_"
        Write-Host "Found virtual environment: $venvPath" -ForegroundColor Yellow
        $size = Remove-DirectoryWithReport $venvPath "Python virtual environment"
        $venvSize += $size
    }
    
    Write-Host "Virtual environments total: $([math]::Round($venvSize / 1GB, 3)) GB" -ForegroundColor Green
    return $venvSize
}

# Function to clean Node.js modules
function Remove-NodeModules {
    Write-Host "`n--- Node.js Modules Cleanup ---" -ForegroundColor Cyan
    
    $nodeSize = 0
    
    Get-ChildItem -Path "d:\omen" -Recurse -Directory -Name "node_modules" -ErrorAction SilentlyContinue | ForEach-Object {
        $nodePath = "d:\omen\$_"
        Write-Host "Found Node modules: $nodePath" -ForegroundColor Yellow
        Write-Host "WARNING: Node modules can be recreated with 'npm install' but may take time" -ForegroundColor Red
        $size = Remove-DirectoryWithReport $nodePath "Node.js modules directory"
        $nodeSize += $size
    }
    
    Write-Host "Node modules total: $([math]::Round($nodeSize / 1GB, 3)) GB" -ForegroundColor Green
    return $nodeSize
}

# Function to clean build directories
function Remove-BuildDirectories {
    Write-Host "`n--- Build Directory Cleanup ---" -ForegroundColor Cyan
    
    $buildSize = 0
    
    @('build', 'dist', '.next', '.nuxt', 'out') | ForEach-Object {
        $buildType = $_
        Get-ChildItem -Path "d:\omen" -Recurse -Directory -Name $buildType -ErrorAction SilentlyContinue | ForEach-Object {
            $buildPath = "d:\omen\$_"
            $size = Remove-DirectoryWithReport $buildPath "Build directory ($buildType)"
            $buildSize += $size
        }
    }
    
    Write-Host "Build directories total: $([math]::Round($buildSize / 1GB, 3)) GB" -ForegroundColor Green
    return $buildSize
}

# Function to remove duplicate files (based on previous analysis)
function Remove-DuplicateFiles {
    Write-Host "`n--- Duplicate File Cleanup ---" -ForegroundColor Cyan
    
    $duplicateSize = 0
    
    # Known duplicates from analysis - remove carefully
    $knownDuplicates = @(
        @{
            Name = "blis.cp311-win_amd64.pyd"
            Paths = @(
                "D:\omen\tools\audiblez-gui\venv\Lib\site-packages\blis\blis.cp311-win_amd64.pyd",
                "D:\omen\VibeVoice-Community\.venv\Lib\site-packages\blis\blis.cp311-win_amd64.pyd"
            )
            KeepPath = "D:\omen\VibeVoice-Community\.venv\Lib\site-packages\blis\blis.cp311-win_amd64.pyd"
        },
        @{
            Name = "VibeVoice-1.5B model blob"
            Paths = @(
                "D:\omen\models\hf\models--VibeVoice-Audio--VibeVoice-1.5B\blobs\b6db833b5fb27bf9f86df56fcad13c0c55b8e81e5d1754cb48fbfc0b07f31e21",
                "D:\omen\VibeVoice-Community\.cache\huggingface\hub\models--VibeVoice-Audio--VibeVoice-1.5B\blobs\b6db833b5fb27bf9f86df56fcad13c0c55b8e81e5d1754cb48fbfc0b07f31e21"
            )
            KeepPath = "D:\omen\VibeVoice-Community\.cache\huggingface\hub\models--VibeVoice-Audio--VibeVoice-1.5B\blobs\b6db833b5fb27bf9f86df56fcad13c0c55b8e81e5d1754cb48fbfc0b07f31e21"
        }
    )
    
    foreach ($duplicate in $knownDuplicates) {
        Write-Host "Processing duplicates: $($duplicate.Name)" -ForegroundColor Yellow
        
        foreach ($path in $duplicate.Paths) {
            if ($path -ne $duplicate.KeepPath -and (Test-Path $path)) {
                $size = (Get-Item $path).Length
                if ($WhatIf) {
                    Write-Host "WOULD DELETE: Duplicate $($duplicate.Name) - $path ($([math]::Round($size / 1MB, 1)) MB)" -ForegroundColor Yellow
                } else {
                    if ($Confirm) {
                        $response = Read-Host "Delete duplicate $($duplicate.Name) at $path? Keeping copy at $($duplicate.KeepPath) (y/N)"
                        if ($response -ne 'y' -and $response -ne 'Y') {
                            continue
                        }
                    }
                    Remove-Item $path -Force -ErrorAction SilentlyContinue
                    Write-Host "DELETED: Duplicate $($duplicate.Name) ($([math]::Round($size / 1MB, 1)) MB)" -ForegroundColor Red
                    $duplicateSize += $size
                }
            }
        }
    }
    
    Write-Host "Duplicate files total: $([math]::Round($duplicateSize / 1GB, 3)) GB" -ForegroundColor Green
    return $duplicateSize
}

# Function to clean temporary files
function Remove-TempFiles {
    Write-Host "`n--- Temporary Files Cleanup ---" -ForegroundColor Cyan
    
    $tempSize = 0
    
    # Common temp file patterns
    $tempPatterns = @('*.tmp', '*.temp', '*.log', '*.cache', '.DS_Store', 'Thumbs.db')
    
    foreach ($pattern in $tempPatterns) {
        Get-ChildItem -Path "d:\omen" -Recurse -File -Filter $pattern -ErrorAction SilentlyContinue | ForEach-Object {
            $size = $_.Length
            if ($WhatIf) {
                Write-Host "WOULD DELETE: Temp file - $($_.FullName)" -ForegroundColor Yellow
            } else {
                if (-not $Confirm -or (Read-Host "Delete temp file $($_.FullName)? (y/N)") -eq 'y') {
                    Remove-Item $_.FullName -Force -ErrorAction SilentlyContinue
                    Write-Host "DELETED: $($_.FullName)" -ForegroundColor Red
                    $tempSize += $size
                }
            }
        }
    }
    
    Write-Host "Temporary files total: $([math]::Round($tempSize / 1GB, 3)) GB" -ForegroundColor Green
    return $tempSize
}

# Main execution
Write-Host "Starting cleanup process..." -ForegroundColor Green
Write-Host "Use -WhatIf switch to preview changes without deleting" -ForegroundColor Cyan
Write-Host ""

# Execute cleanup functions
$totalSaved += Remove-PythonCaches
$totalSaved += Remove-BuildDirectories  
$totalSaved += Remove-DuplicateFiles
$totalSaved += Remove-TempFiles

# Virtual environments and Node modules with extra warnings
Write-Host "`n=== DESTRUCTIVE OPERATIONS (Regenerable but time-consuming) ===" -ForegroundColor Red
$response = Read-Host "Do you want to remove virtual environments and Node modules? This will require reinstallation. (y/N)"
if ($response -eq 'y' -or $response -eq 'Y') {
    $totalSaved += Remove-VirtualEnvironments
    $totalSaved += Remove-NodeModules
}

# Summary
Write-Host "`n=== CLEANUP SUMMARY ===" -ForegroundColor Green
Write-Host "Total space that would be/was freed: $([math]::Round($totalSaved / 1GB, 2)) GB" -ForegroundColor Green

if ($WhatIf) {
    Write-Host "`nThis was a preview run. Use the script without -WhatIf to actually delete files." -ForegroundColor Cyan
} else {
    Write-Host "`nCleanup completed. Remember to:" -ForegroundColor Yellow
    Write-Host "1. Recreate virtual environments: python -m venv .venv" -ForegroundColor Yellow  
    Write-Host "2. Reinstall Node dependencies: npm install" -ForegroundColor Yellow
    Write-Host "3. Rebuild any projects as needed" -ForegroundColor Yellow
}

Write-Host "`nScript completed at $(Get-Date)" -ForegroundColor Green
