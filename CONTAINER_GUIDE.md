# VibeVoice Container Management

This guide helps you properly run VibeVoice with GPU support and handle common issues.

## 🚨 GPU Compatibility Issues

Your **RTX 5070 Laptop GPU** is newer than the PyTorch container supports. Here are solutions:

### Option 1: Use CPU Mode (Recommended for testing)
```bash
# Set environment variable to disable GPU
$env:CUDA_VISIBLE_DEVICES=""
docker run -it --rm -p 7860:7860 vibevoice-community
```

### Option 2: Update to Newer PyTorch Container
Edit `Dockerfile` and change the base image to:
```dockerfile
FROM nvcr.io/nvidia/pytorch:24.12-py3  # or latest available
```

### Option 3: Use Local PyTorch Installation
Build with a custom PyTorch version that supports RTX 5070:
```dockerfile
FROM nvidia/cuda:12.4-devel-ubuntu22.04
# Then install PyTorch 2.5+ manually
```

## 🛠️ Container Management

### Quick Start (Windows PowerShell)
```powershell
# Clean start with proper GPU flags
.\manage-container.ps1 start

# Check status
.\manage-container.ps1 status

# View logs
.\manage-container.ps1 logs

# Stop cleanly
.\manage-container.ps1 stop
```

### Using Docker Compose (Recommended)
```bash
# Start with docker-compose (handles GPU automatically)
docker-compose up --build -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

### Manual Docker Commands
```bash
# Build with GPU support
docker build -t vibevoice-community .

# Run with proper GPU flags
docker run -d \
  --name vibe \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -p 7860:7860 \
  -v "$(pwd)/outputs:/workspace/outputs" \
  -e NVIDIA_VISIBLE_DEVICES=all \
  vibevoice-community
```

## 🔧 Troubleshooting

### Port Already in Use
The scripts automatically handle port cleanup, but you can manually clear:
```powershell
# Windows
Get-Process -Id (Get-NetTCPConnection -LocalPort 7860).OwningProcess | Stop-Process -Force

# Linux/Mac
lsof -ti:7860 | xargs kill -9
```

### GPU Not Detected
1. **Check NVIDIA Docker support:**
   ```bash
   docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu22.04 nvidia-smi
   ```

2. **Install nvidia-docker2** if missing:
   ```bash
   # Ubuntu/Debian
   sudo apt-get install nvidia-docker2
   sudo systemctl restart docker
   ```

3. **Windows with WSL2:** Ensure you have:
   - Windows 11 or Windows 10 version 21H2+
   - NVIDIA driver 470.76+ 
   - Docker Desktop with WSL2 backend
   - WSL2 with Ubuntu

### Container Won't Start
Check the logs:
```bash
docker logs vibe
```

Common issues:
- **CUDA version mismatch:** Use CPU mode or update base image
- **Insufficient memory:** Add `--shm-size=2g` flag
- **Port conflicts:** Use the management scripts to auto-clean

## 🌐 Gradio Configuration

The container now includes:
- **Port cleanup** on startup
- **Share mode enabled** by default (set `--share` flag)
- **Proper server binding** to 0.0.0.0 for container access

To disable public sharing, modify the Dockerfile:
```dockerfile
CMD ["bash","-lc","lsof -ti:7860 | xargs -r kill -9; python demo/gradio_demo.py --model_path microsoft/VibeVoice-1.5B"]
```

## 📁 Directory Structure
```
outputs/          # Generated audio files (mounted volume)
models/           # Model cache (mounted volume)  
demo/voices/      # Voice samples (mounted volume)
```

## ⚡ Performance Tips

1. **Use SSD storage** for model cache
2. **Increase shared memory:** `--shm-size=4g`
3. **Monitor GPU memory:** `nvidia-smi -l 1`
4. **CPU fallback** for compatibility: `CUDA_VISIBLE_DEVICES=""`

## 🔗 URLs
- **Local Demo:** http://localhost:7860
- **Public Share:** Shown in container logs when `--share` is enabled
- **API Docs:** http://localhost:7860/docs (if enabled)
