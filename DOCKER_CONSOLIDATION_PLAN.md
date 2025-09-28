# Docker Consolidation Plan - VibeVoice Community

## Current Situation
- **Native Environment**: Most developed with latest fixes
- **Docker Containers**: Outdated, missing critical improvements
- **Goal**: Consolidate everything into one GPU-enabled Docker container

## Development State
- ✅ Enhanced GUI with port management (82KB ebook_gui.py)
- ✅ Fixed TTS backend with GPU support (34KB tts_backend.py) 
- ✅ VibeVoice TTS working with external tokenizer
- ✅ Unicode encoding fixes for Windows
- ✅ PyTorch 2.5.1+cu121 with RTX 5070 support
- ✅ 408 packages in virtual environment

## Consolidation Steps

### 1. Container Cleanup
```bash
# Stop and remove outdated containers
docker stop vibe vibevoice-community
docker rm vibe vibevoice-community

# Remove outdated images (keep only the latest)
docker rmi local/vibevoice-community:gpu
docker rmi vibevoice-community-vibe
docker rmi vibevoice-gui
```

### 2. Create Production Dockerfile
- Base: `nvcr.io/nvidia/pytorch:24.08-py3` (CUDA 12.4)
- Include all native environment fixes
- GPU-optimized PyTorch
- Port management system
- Enhanced GUI with all features

### 3. Requirements Capture
```bash
# Export current working environment
pip freeze > requirements-production.txt
```

### 4. Docker Build Strategy
- Single production image: `vibevoice-community:production`
- GPU-enabled with RTX 5070 support
- All current fixes included
- Port 7862 with port management

### 5. Container Management
- Single container: `vibevoice-production`
- Volume mounts for persistence
- GPU pass-through configured
- Health checks included

## Target Architecture
```
vibevoice-community:production
├── GPU PyTorch 2.5.1+cu121
├── Enhanced GUI (port management)
├── Fixed TTS backend
├── VibeVoice with external tokenizer
├── All voice libraries
└── Production-ready startup scripts
```

## Success Criteria
- [ ] Single Docker container running
- [ ] GPU acceleration working
- [ ] All recent fixes included
- [ ] Port management operational
- [ ] VibeVoice TTS functional
- [ ] Old containers removed
