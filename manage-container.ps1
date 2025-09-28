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

# Centralized status writer
function Write-Status {
    param(
        [Parameter(Mandatory=$true)][string]$Message,
        [ValidateSet('Black','DarkBlue','DarkGreen','DarkCyan','DarkRed','DarkMagenta','DarkYellow','Gray','DarkGray','Blue','Green','Cyan','Red','Magenta','Yellow','White')]
        [string]$Color = 'Gray'
    )
    Write-Host $Message -ForegroundColor $Color
}

# Function to check prerequisites
function Test-Prerequisite {
    Write-Status "Checking prerequisites..." Cyan
    
    # Check if Docker is installed and running
    try {
        $dockerVersion = docker version --format "{{.Server.Version}}" 2>$null
        if ($dockerVersion) {
            Write-Status "Docker is running (version: $dockerVersion)" Green
        } else {
            Write-Status "Docker is not running or not installed" Red
            return $false
        }
    } catch {
        Write-Status "Docker is not available" Red
        return $false
    }
    
    # Check if Docker Compose is available
    try {
        $composeVersion = docker-compose version --short 2>$null
        if ($composeVersion) {
            Write-Status "Docker Compose is available (version: $composeVersion)" Green
        } else {
            Write-Status "Docker Compose not found - some features may not work" Yellow
        }
    } catch {
        Write-Status "Docker Compose not available - some features may not work" Yellow
    }
    
    # Check if required files exist
    $requiredFiles = @("Dockerfile", "docker-compose.yml")
    foreach ($file in $requiredFiles) {
        if (Test-Path $file) {
            Write-Status "Found $file" Green
        } else {
            Write-Status "Missing $file" Red
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
    
    Write-Status "Cleaning up existing container: $ContainerName..." Yellow
    
    $containerExists = docker ps -a --format "table {{.Names}}" | Select-String "^$ContainerName$"
    if ($containerExists) {
        Write-Status "Stopping container: $ContainerName" Yellow
        docker stop $ContainerName 2>$null
        Write-Status "Removing container: $ContainerName" Yellow
        docker rm $ContainerName 2>$null
    }
    
    # Kill any processes using the port
    Write-Status "Cleaning up port $PortToClean..." Yellow
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
function Invoke-VibeImageBuild {
    Write-Status "Building VibeVoice image..." Green
    docker build -t $IMAGE_NAME .
}

# Function to run the container with proper GPU support
function Start-VibeContainer {
    Write-Status "Starting VibeVoice container with GPU support..." Green
    
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
    
    Write-Status "Container started! Checking status..." Green
    docker ps | Select-String $CONTAINER_NAME
    
    Write-Status "Logs (first 20 lines):" Cyan
    Start-Sleep -Seconds 5
    docker logs $CONTAINER_NAME | Select-Object -First 20
    
    Write-Output ""
    Write-Status "Access the demo at: http://localhost:$PORT" Green
    Write-Status "View logs: docker logs -f $CONTAINER_NAME" Yellow
    Write-Status "Stop container: docker stop $CONTAINER_NAME" Red
}

# Function to show logs
function Get-VibeLogs {
    Write-Status "Showing logs for $CONTAINER_NAME..." Cyan
    docker logs -f $CONTAINER_NAME
}

# Function to show container status
function Get-VibeStatus {
    param(
        [string]$ContainerName = $CONTAINER_NAME,
        [int]$PortToCheck = $PORT
    )
    
    Write-Status "Container status for ${ContainerName}:" Cyan
    $containerStatus = docker ps -a | Select-String $ContainerName
    if ($containerStatus) {
        $containerStatus
    } else {
        Write-Status "Container '${ContainerName}' not found" Red
    }
    
    Write-Output ""
    Write-Status "Port ${PortToCheck} status:" Cyan
    try {
        $portStatus = Get-NetTCPConnection -LocalPort $PortToCheck -ErrorAction SilentlyContinue
        if ($portStatus) {
            $portStatus | Format-Table -AutoSize
        } else {
            Write-Status "Port ${PortToCheck} is free" Green
        }
    } catch {
        Write-Status "Port ${PortToCheck} is free" Green
    }
    
    Write-Output ""
    Write-Status "GPU status:" Cyan
    try {
        if (docker ps | Select-String $ContainerName) {
            docker exec $ContainerName nvidia-smi 2>$null
        } else {
            Write-Status "Container '${ContainerName}' is not running - cannot check GPU" Yellow
        }
    } catch {
        Write-Status "Cannot access GPU or container not running" Red
    }
}

# Main script logic
if (-not (Test-Prerequisite)) {
    Write-Status "Prerequisites check failed. Please ensure Docker is installed and running." Red
    throw "Prerequisites check failed."
}

switch ($Action) {
    "build" {
        Invoke-VibeImageBuild
    }
    "start" {
        Remove-VibeContainer
        Invoke-VibeImageBuild
        Start-VibeContainer
    }
    "stop" {
        Remove-VibeContainer
    }
    "restart" {
        Remove-VibeContainer
        Invoke-VibeImageBuild
        Start-VibeContainer
    }
    "logs" {
        Get-VibeLogs
    }
    "status" {
        Get-VibeStatus
    }
    "compose" {
        Write-Status "Using Docker Compose for standard container..." Blue
        docker-compose down
        docker-compose up --build -d $CONTAINER_NAME
        docker-compose logs -f $CONTAINER_NAME
    }
    "multimodel" {
        Write-Status "Starting Multi-Model TTS container..." Magenta
        docker-compose down
        docker-compose up --build -d $MULTIMODEL_CONTAINER
        Write-Status "Multi-Model TTS available at: http://localhost:$MULTIMODEL_PORT" Green
        docker-compose logs -f $MULTIMODEL_CONTAINER
    }
    "desktop" {
        Write-Status "Starting Desktop GUI container..." Cyan
        docker-compose down
        docker-compose up --build -d $DESKTOP_CONTAINER
        Write-Status "Desktop GUI (VNC) available at: http://localhost:$DESKTOP_PORT" Green
        Write-Status "VNC Password: vibevoice" Yellow
        docker-compose logs -f $DESKTOP_CONTAINER
    }
    "ebook" {
        Write-Status "Starting Ebook to Audiobook Converter..." Blue
        docker-compose down
        $ebookSvc = 'vibe-ebook'
        $ebookPort = 7862
        docker-compose up --build -d $ebookSvc
        Write-Status "Ebook Converter available at: http://localhost:$ebookPort" Green
        Write-Status "Supports: PDF, TXT, DOCX, EPUB -> Audiobook" Yellow
        Write-Status "Engine: VibeVoice (Python 3.13)" Cyan
        docker-compose logs -f $ebookSvc
    }
    "ebook-py311" {
        Write-Status "Starting Ebook Converter with Coqui AI Support..." Magenta
        docker-compose down
        $ebookSvc = 'vibe-ebook-py311'
        $ebookPort = 7863
        docker-compose up --build -d $ebookSvc
        Write-Status "Ebook Converter (Full) available at: http://localhost:$ebookPort" Green
        Write-Status "Supports: PDF, TXT, DOCX, EPUB -> Audiobook" Yellow
        Write-Status "Engines: VibeVoice + Coqui AI (Python 3.11)" Cyan
        Write-Status "Formats: WAV, MP3, M4B with chapter markers" White
        docker-compose logs -f $ebookSvc
    }
}
