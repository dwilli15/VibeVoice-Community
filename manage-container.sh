#!/bin/bash

# VibeVoice Container Management Script
# This script properly handles GPU support and port cleanup

CONTAINER_NAME="vibe"
IMAGE_NAME="vibevoice-community"
PORT=7860

# Function to stop and remove existing container
cleanup_container() {
    echo "🧹 Cleaning up existing container..."
    if docker ps -a --format "table {{.Names}}" | grep -q "^${CONTAINER_NAME}$"; then
        echo "Stopping container: $CONTAINER_NAME"
        docker stop $CONTAINER_NAME 2>/dev/null
        echo "Removing container: $CONTAINER_NAME"
        docker rm $CONTAINER_NAME 2>/dev/null
    fi
    
    # Kill any processes using the port
    echo "🧹 Cleaning up port $PORT..."
    lsof -ti:$PORT | xargs -r kill -9 2>/dev/null || true
}

# Function to build the image
build_image() {
    echo "🔨 Building VibeVoice image..."
    docker build -t $IMAGE_NAME .
}

# Function to run the container with proper GPU support
run_container() {
    echo "🚀 Starting VibeVoice container with GPU support..."
    
    # Create directories if they don't exist
    mkdir -p ./outputs
    mkdir -p ./models
    
    docker run -d \
        --name $CONTAINER_NAME \
        --gpus all \
        --ipc=host \
        --ulimit memlock=-1 \
        --ulimit stack=67108864 \
        -p $PORT:$PORT \
        -v "$(pwd)/outputs:/workspace/outputs" \
        -v "$(pwd)/demo/voices:/workspace/demo/voices" \
        -v "$(pwd)/models:/models" \
        -e NVIDIA_VISIBLE_DEVICES=all \
        -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
        -e CUDA_VISIBLE_DEVICES=0 \
        $IMAGE_NAME
    
    echo "✅ Container started! Checking status..."
    docker ps | grep $CONTAINER_NAME
    
    echo "📝 Logs (first 20 lines):"
    sleep 5
    docker logs $CONTAINER_NAME | head -20
    
    echo ""
    echo "🌐 Access the demo at: http://localhost:$PORT"
    echo "📊 View logs: docker logs -f $CONTAINER_NAME"
    echo "🛑 Stop container: docker stop $CONTAINER_NAME"
}

# Function to show logs
show_logs() {
    echo "📊 Showing logs for $CONTAINER_NAME..."
    docker logs -f $CONTAINER_NAME
}

# Function to show container status
show_status() {
    echo "📈 Container status:"
    docker ps -a | grep $CONTAINER_NAME || echo "Container not found"
    
    echo ""
    echo "🌐 Port status:"
    lsof -i:$PORT || echo "Port $PORT is free"
    
    echo ""
    echo "🖥️ GPU status:"
    docker exec $CONTAINER_NAME nvidia-smi 2>/dev/null || echo "Cannot access GPU or container not running"
}

# Main script logic
case "$1" in
    "build")
        build_image
        ;;
    "start")
        cleanup_container
        build_image
        run_container
        ;;
    "stop")
        cleanup_container
        ;;
    "restart")
        cleanup_container
        build_image
        run_container
        ;;
    "logs")
        show_logs
        ;;
    "status")
        show_status
        ;;
    "compose")
        echo "🐳 Using Docker Compose..."
        docker-compose down
        docker-compose up --build -d
        docker-compose logs -f
        ;;
    *)
        echo "🎙️ VibeVoice Container Management"
        echo ""
        echo "Usage: $0 {build|start|stop|restart|logs|status|compose}"
        echo ""
        echo "Commands:"
        echo "  build   - Build the Docker image"
        echo "  start   - Clean up and start the container"
        echo "  stop    - Stop and remove the container"
        echo "  restart - Stop, rebuild, and start the container"
        echo "  logs    - Show container logs (follow mode)"
        echo "  status  - Show container and port status"
        echo "  compose - Use docker-compose for management"
        echo ""
        echo "GPU Requirements:"
        echo "  - NVIDIA GPU with Docker support"
        echo "  - nvidia-docker2 installed"
        echo "  - Sufficient VRAM (recommended: 8GB+)"
        ;;
esac
