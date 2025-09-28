# VibeVoice Container Management Script for Windows
# This script properly handles GPU support and port cleanup

param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("build", "start", "stop", "restart", "logs", "status", "compose", "multimodel", "desktop", "ebook", "ebook-py311")]
    [string]$Action
)

$CONTAINER_NAME = "vibe"
$MULTIMODEL_CONTAINER = "vibe-multimodel"
$DESKTOP_CONTAINER = "vibe-desktop"
$IMAGE_NAME = "vibevoice-community"
$PORT = 7860
$MULTIMODEL_PORT = 7861
$DESKTOP_PORT = 6080

# Function to check prerequisites
function Test-Prerequisites {
    Write-Host "Checking prerequisites..." -ForegroundColor Cyan
    
    # Check if Docker is installed and running
    try {
        $dockerVersion = docker version --format "{{.Server.Version}}" 2>$null
        if ($dockerVersion) {
            Write-Host "Docker is running (version: $dockerVersion)" -ForegroundColor Green
        } else {
            Write-Host "Docker is not running or not installed" -ForegroundColor Red
            return $false
        }
    } catch {
        Write-Host "Docker is not available" -ForegroundColor Red
        return $false
    }
    
    # Check if Docker Compose is available
    try {
        $composeVersion = docker-compose version --short 2>$null
        if ($composeVersion) {
            Write-Host "Docker Compose is available (version: $composeVersion)" -ForegroundColor Green
        } else {
            Write-Host "Docker Compose not found - some features may not work" -ForegroundColor Yellow
        }
    } catch {
        Write-Host "Docker Compose not available - some features may not work" -ForegroundColor Yellow
    }
    
    # Check if required files exist
    $requiredFiles = @("Dockerfile", "docker-compose.yml")
    foreach ($file in $requiredFiles) {
        if (Test-Path $file) {
            Write-Host "Found $file" -ForegroundColor Green
        } else {
            Write-Host "Missing $file" -ForegroundColor Red
            return $false
        }
    }
    
    return $true
}

# Function to stop and remove existing container
function Remove-VibeContainer {
    param(
        [string]$ContainerName = $CONTAINER_NAME,
        [int]$PortToClean = $PORT
    )
    
    Write-Host "Cleaning up existing container: $ContainerName..." -ForegroundColor Yellow
    
    $containerExists = docker ps -a --format "table {{.Names}}" | Select-String "^$ContainerName$"
    if ($containerExists) {
        Write-Host "Stopping container: $ContainerName" -ForegroundColor Yellow
        docker stop $ContainerName 2>$null
        Write-Host "Removing container: $ContainerName" -ForegroundColor Yellow
        docker rm $ContainerName 2>$null
    }
    
    # Kill any processes using the port
    Write-Host "Cleaning up port $PortToClean..." -ForegroundColor Yellow
    try {
        $processes = Get-NetTCPConnection -LocalPort $PortToClean -ErrorAction SilentlyContinue
        foreach ($process in $processes) {
            Stop-Process -Id $process.OwningProcess -Force -ErrorAction SilentlyContinue
        }
    } catch {
        # Port not in use or error getting processes
    }
}

# Function to build the image
function Invoke-ImageBuild {
    Write-Host "Building VibeVoice image..." -ForegroundColor Green
    docker build -t $IMAGE_NAME .
}

# Function to run the container with proper GPU support
function Start-VibeContainer {
    Write-Host "Starting VibeVoice container with GPU support..." -ForegroundColor Green
    
    # Create directories if they don't exist
    New-Item -ItemType Directory -Force -Path ".\outputs" | Out-Null
    New-Item -ItemType Directory -Force -Path ".\models" | Out-Null
    
    $currentPath = (Get-Location).Path
    
    docker run -d `
        --name $CONTAINER_NAME `
        --gpus all `
        --ipc=host `
        --ulimit memlock=-1 `
        --ulimit stack=67108864 `
        -p "${PORT}:${PORT}" `
        -v "${currentPath}\outputs:/workspace/outputs" `
        -v "${currentPath}\demo\voices:/workspace/demo/voices" `
        -v "${currentPath}\models:/models" `
        -e NVIDIA_VISIBLE_DEVICES=all `
        -e NVIDIA_DRIVER_CAPABILITIES=compute,utility `
        -e CUDA_VISIBLE_DEVICES=0 `
        $IMAGE_NAME
    
    Write-Host "Container started! Checking status..." -ForegroundColor Green
    docker ps | Select-String $CONTAINER_NAME
    
    Write-Host "Logs (first 20 lines):" -ForegroundColor Cyan
    Start-Sleep -Seconds 5
    docker logs $CONTAINER_NAME | Select-Object -First 20
    
    Write-Host ""
    Write-Host "Access the demo at: http://localhost:$PORT" -ForegroundColor Green
    Write-Host "View logs: docker logs -f $CONTAINER_NAME" -ForegroundColor Yellow
    Write-Host "Stop container: docker stop $CONTAINER_NAME" -ForegroundColor Red
}

# Function to show logs
function Get-VibeLogs {
    Write-Host "Showing logs for $CONTAINER_NAME..." -ForegroundColor Cyan
    docker logs -f $CONTAINER_NAME
}

# Function to show container status
function Get-VibeStatus {
    param(
        [string]$ContainerName = $CONTAINER_NAME,
        [int]$PortToCheck = $PORT
    )
    
    Write-Host "Container status for ${ContainerName}:" -ForegroundColor Cyan
    $containerStatus = docker ps -a | Select-String $ContainerName
    if ($containerStatus) {
        $containerStatus
    } else {
        Write-Host "Container '${ContainerName}' not found" -ForegroundColor Red
    }
    
    Write-Host ""
    Write-Host "Port ${PortToCheck} status:" -ForegroundColor Cyan
    try {
        $portStatus = Get-NetTCPConnection -LocalPort $PortToCheck -ErrorAction SilentlyContinue
        if ($portStatus) {
            $portStatus | Format-Table -AutoSize
        } else {
            Write-Host "Port ${PortToCheck} is free" -ForegroundColor Green
        }
    } catch {
        Write-Host "Port ${PortToCheck} is free" -ForegroundColor Green
    }
    
    Write-Host ""
    Write-Host "GPU status:" -ForegroundColor Cyan
    try {
        if (docker ps | Select-String $ContainerName) {
            docker exec $ContainerName nvidia-smi 2>$null
        } else {
            Write-Host "Container '${ContainerName}' is not running - cannot check GPU" -ForegroundColor Yellow
        }
    } catch {
        Write-Host "Cannot access GPU or container not running" -ForegroundColor Red
    }
}

# Main script logic
if (-not (Test-Prerequisites)) {
    Write-Host "Prerequisites check failed. Please ensure Docker is installed and running." -ForegroundColor Red
    exit 1
}

switch ($Action) {
    "build" {
        Invoke-ImageBuild
    }
    "start" {
        Remove-VibeContainer
        Invoke-ImageBuild
        Start-VibeContainer
    }
    "stop" {
        Remove-VibeContainer
    }
    "restart" {
        Remove-VibeContainer
        Invoke-ImageBuild
        Start-VibeContainer
    }
    "logs" {
        Get-VibeLogs
    }
    "status" {
        Get-VibeStatus
    }
    "compose" {
        Write-Host "Using Docker Compose for standard container..." -ForegroundColor Blue
        docker-compose down
        docker-compose up --build -d $CONTAINER_NAME
        docker-compose logs -f $CONTAINER_NAME
    }
    "multimodel" {
        Write-Host "Starting Multi-Model TTS container..." -ForegroundColor Magenta
        docker-compose down
        docker-compose up --build -d $MULTIMODEL_CONTAINER
        Write-Host "Multi-Model TTS available at: http://localhost:$MULTIMODEL_PORT" -ForegroundColor Green
        docker-compose logs -f $MULTIMODEL_CONTAINER
    }
    "desktop" {
        Write-Host "Starting Desktop GUI container..." -ForegroundColor Cyan
        docker-compose down
        docker-compose up --build -d $DESKTOP_CONTAINER
        Write-Host "Desktop GUI (VNC) available at: http://localhost:$DESKTOP_PORT" -ForegroundColor Green
        Write-Host "VNC Password: vibevoice" -ForegroundColor Yellow
        docker-compose logs -f $DESKTOP_CONTAINER
    }
    "ebook" {
        Write-Host "Starting Ebook to Audiobook Converter..." -ForegroundColor Blue
        docker-compose down
        docker-compose up --build -d vibe-ebook
        Write-Host "Ebook Converter available at: http://localhost:7862" -ForegroundColor Green
        Write-Host "Supports: PDF, TXT, DOCX, EPUB -> Audiobook" -ForegroundColor Yellow
        Write-Host "Engine: VibeVoice (Python 3.13)" -ForegroundColor Cyan
        docker-compose logs -f vibe-ebook
    }
    "ebook-py311" {
        Write-Host "Starting Ebook Converter with Coqui AI Support..." -ForegroundColor Magenta
        docker-compose down
        docker-compose up --build -d vibe-ebook-py311
        Write-Host "Ebook Converter (Full) available at: http://localhost:7863" -ForegroundColor Green
        Write-Host "Supports: PDF, TXT, DOCX, EPUB -> Audiobook" -ForegroundColor Yellow
        Write-Host "Engines: VibeVoice + Coqui AI (Python 3.11)" -ForegroundColor Cyan
        Write-Host "Formats: WAV, MP3, M4B with chapter markers" -ForegroundColor White
        docker-compose logs -f vibe-ebook-py311
    }
}
