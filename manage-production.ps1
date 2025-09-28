#!/usr/bin/env pwsh
# VibeVoice Production Container Management Script

param(
    [Parameter(Position=0)]
    [ValidateSet("start", "stop", "restart", "build", "logs", "shell", "clean", "status")]
    [string]$Action = "start",
    
    [Parameter()]
    [int]$Port = 7862,
    
    [Parameter()]
    [switch]$Force,
    
    [Parameter()]
    [switch]$NoBrowser
)

$ContainerName = "vibevoice-production"
$ImageName = "vibevoice-community:production"
$ComposeFile = "docker-compose.production.yml"

function Write-Status {
    param([string]$Message, [string]$Type = "INFO")
    $timestamp = Get-Date -Format "HH:mm:ss"
    switch ($Type) {
        "INFO" { Write-Host "[$timestamp] [INFO] $Message" -ForegroundColor Green }
        "WARN" { Write-Host "[$timestamp] [WARN] $Message" -ForegroundColor Yellow }
        "ERROR" { Write-Host "[$timestamp] [ERROR] $Message" -ForegroundColor Red }
        "DEBUG" { Write-Host "[$timestamp] [DEBUG] $Message" -ForegroundColor Cyan }
    }
}

function Test-ContainerRunning {
    $running = docker ps --filter "name=$ContainerName" --format "{{.Names}}" 2>$null
    return $running -eq $ContainerName
}

function Test-ImageExists {
    $exists = docker images --filter "reference=$ImageName" --format "{{.Repository}}:{{.Tag}}" 2>$null
    return $exists -eq $ImageName
}

switch ($Action) {
    "build" {
        Write-Status "Building production Docker image..."
        if ($Force) {
            Write-Status "Force rebuild requested"
            docker build --no-cache -f Dockerfile.production -t $ImageName .
        } else {
            docker build -f Dockerfile.production -t $ImageName .
        }
        
        if ($LASTEXITCODE -eq 0) {
            Write-Status "Build completed successfully"
        } else {
            Write-Status "Build failed" "ERROR"
            exit 1
        }
    }
    
    "start" {
        if (Test-ContainerRunning) {
            Write-Status "Container already running"
            if (-not $NoBrowser) {
                Write-Status "Opening browser..."
                Start-Process "http://localhost:$Port"
            }
            exit 0
        }
        
        if (-not (Test-ImageExists)) {
            Write-Status "Image not found, building..."
            & $PSCommandPath build
            if ($LASTEXITCODE -ne 0) { exit 1 }
        }
        
        Write-Status "Starting production container..."
        docker-compose -f $ComposeFile up -d
        
        if ($LASTEXITCODE -eq 0) {
            Write-Status "Container started successfully"
            Write-Status "GUI available at: http://localhost:$Port"
            
            if (-not $NoBrowser) {
                Start-Sleep 3
                Write-Status "Opening browser..."
                Start-Process "http://localhost:$Port"
            }
        } else {
            Write-Status "Failed to start container" "ERROR"
        }
    }
    
    "stop" {
        Write-Status "Stopping production container..."
        docker-compose -f $ComposeFile down
        Write-Status "Container stopped"
    }
    
    "restart" {
        Write-Status "Restarting production container..."
        & $PSCommandPath stop
        & $PSCommandPath start -Port $Port -NoBrowser:$NoBrowser
    }
    
    "logs" {
        Write-Status "Showing container logs (Ctrl+C to exit)..."
        docker-compose -f $ComposeFile logs -f
    }
    
    "shell" {
        if (-not (Test-ContainerRunning)) {
            Write-Status "Container not running, starting first..." "WARN"
            & $PSCommandPath start -NoBrowser
            Start-Sleep 5
        }
        
        Write-Status "Opening shell in container..."
        docker exec -it $ContainerName /bin/bash
    }
    
    "clean" {
        Write-Status "Cleaning up old containers and images..."
        
        # Stop current container
        docker-compose -f $ComposeFile down 2>$null
        
        # Remove old images (keep only production)
        $oldImages = @(
            "vibevoice-community:latest",
            "vibevoice-community-vibe:latest", 
            "local/vibevoice-community:gpu",
            "vibevoice-gui:latest"
        )
        
        foreach ($img in $oldImages) {
            $exists = docker images --filter "reference=$img" --format "{{.Repository}}:{{.Tag}}" 2>$null
            if ($exists) {
                Write-Status "Removing old image: $img"
                docker rmi $img 2>$null
            }
        }
        
        # Clean up dangling images
        Write-Status "Cleaning up dangling images..."
        docker image prune -f
        
        Write-Status "Cleanup completed"
    }
    
    "status" {
        Write-Status "=== VIBEVOICE PRODUCTION STATUS ==="
        
        # Container status
        if (Test-ContainerRunning) {
            Write-Status "Container: RUNNING" "INFO"
            $containerInfo = docker ps --filter "name=$ContainerName" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
            Write-Host $containerInfo
        } else {
            Write-Status "Container: STOPPED" "WARN"
        }
        
        # Image status
        if (Test-ImageExists) {
            Write-Status "Image: EXISTS" "INFO"
            $imageInfo = docker images --filter "reference=$ImageName" --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"
            Write-Host $imageInfo
        } else {
            Write-Status "Image: NOT FOUND" "ERROR"
        }
        
        # Port status
        try {
            $response = Invoke-WebRequest -Uri "http://localhost:$Port" -TimeoutSec 5 -UseBasicParsing 2>$null
            Write-Status "GUI: ACCESSIBLE at http://localhost:$Port" "INFO"
        } catch {
            Write-Status "GUI: NOT ACCESSIBLE" "WARN"
        }
    }
    
    default {
        Write-Host @"
VibeVoice Production Container Manager

USAGE:
    .\manage-production.ps1 <action> [options]

ACTIONS:
    start     Start the production container (default)
    stop      Stop the production container  
    restart   Restart the production container
    build     Build the production Docker image
    logs      Show container logs
    shell     Open shell in running container
    clean     Remove old containers and images
    status    Show current status

OPTIONS:
    -Port <port>    Port to use (default: 7862)
    -Force          Force rebuild (with build action)
    -NoBrowser      Don't open browser automatically

EXAMPLES:
    .\manage-production.ps1 start
    .\manage-production.ps1 build -Force
    .\manage-production.ps1 logs
    .\manage-production.ps1 status
"@
    }
}
