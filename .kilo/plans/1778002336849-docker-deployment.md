# Docker Deployment Plan for LuxTTS

## Overview

This plan provides a comprehensive Docker deployment strategy for LuxTTS (OpenAI-compatible TTS API server) following best practices and common patterns. The setup works with the existing git repository and does not modify any git configuration.

## Configuration Decisions (User Preferences)

| Decision | Choice | Reason |
|----------|--------|--------|
| CUDA Version | **12.1** | Current stable, best PyTorch compatibility, matches RTX 3050 Ti Ampere architecture |
| Image Type | **Runtime** | ~4GB compressed, inference only, no build tools needed |
| Voice Storage | **Bind Mount** | Easy to add/modify voices without rebuilding image |

---

## Files to Create

### 1. `Dockerfile`

**Purpose**: Build a production-ready LuxTTS container image based on PyTorch CUDA 12.1 runtime.

**Key decisions**:
- Base image: `pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime` (official PyTorch, ~4GB)
- Python 3.10 required (llvmlite dependency constraint)
- Install zipvoice from GitHub (pip package is empty placeholder)
- Copy only necessary files (api_server.py, static/, requirements.txt)
- Expose port 8000 (default)
- Health check endpoint: `/health`

**Best practices applied**:
- Multi-stage build consideration: Not needed (runtime image already minimal)
- Layer ordering: Dependencies first, then code (better caching)
- `.dockerignore` to exclude `.venv`, `.git`, `__pycache__`, logs
- Non-root user: Optional but recommended for production
- Environment variables via `.env` file (not hardcoded)

---

### 2. `docker-compose.yml`

**Purpose**: Orchestrate LuxTTS container with GPU support, bind mounts, and health checks.

**Key configuration**:

```yaml
services:
  luxtts:
    build: .
    ports:
      - "8000:8000"
    volumes:
      # Bind mounts for user data
      - ./voice_samples:/app/voice_samples
      - ./.env:/app/.env:ro
      # HuggingFace cache (prevents re-downloading model on restart)
      - hf_cache:/root/.cache/huggingface
    environment:
      - MODEL_PATH=YatharthS/LuxTTS
      - DEVICE=cuda
      - HOST=0.0.0.0
      - PORT=8000
      - ENABLE_TF32=true
      - ENABLE_FP16=false
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 120s  # Model download takes time
    restart: unless-stopped

volumes:
  hf_cache:
    driver: local
```

**Best practices applied**:
- GPU reservation via `deploy.resources.reservations.devices` (Docker Compose spec)
- Named volume for HuggingFace cache (persistent across container rebuilds)
- Bind mount `.env:ro` (read-only for security)
- Health check with long `start_period` (model download ~1-2GB)
- Restart policy: `unless-stopped` (survives host reboot)

---

### 3. `.dockerignore`

**Purpose**: Exclude unnecessary files from Docker build context (reduces build time and image size).

**Contents**:
```
# Virtual environment (rebuilt in container)
.venv/
venv/
env/

# Git (not needed in container)
.git/
.gitignore

# Python cache
__pycache__/
*.py[cod]
*$py.class
*.pyo

# IDE/Editor
.idea/
.vscode/
*.swp
*.swo

# Kilo config (not needed in container)
.kilo/

# Logs
*.log
server_stdout.log
server_stderr.log

# OS files
.DS_Store
Thumbs.db

# Build artifacts
dist/
build/
*.egg-info/

# User data (mounted at runtime, not copied)
# voice_samples/ - mounted via bind mount

# Test/temp files
temp/
tmp/
*.tmp
output_speech.wav
test.wav
```

---

### 4. `.env.docker.example`

**Purpose**: Template environment file for Docker deployment with sensible defaults.

**Contents**:
```env
# LuxTTS Docker Configuration
# Copy this file to .env before running docker-compose up

# Model Settings
MODEL_PATH=YatharthS/LuxTTS
DEVICE=cuda

# Server Settings
HOST=0.0.0.0
PORT=8000

# GPU Optimizations (RTX 3050 4GB)
ENABLE_TF32=true
ENABLE_FP16=false

# Default TTS Parameters
DEFAULT_RMS=0.001
DEFAULT_T_SHIFT=0.5
DEFAULT_NUM_STEPS=4
DEFAULT_SPEED=1.0
DEFAULT_RETURN_SMOOTH=false
DEFAULT_REF_DURATION=3

# Optional: HuggingFace token for private models
# HF_TOKEN=your_token_here
```

---

## Bind Mounts Details

| Mount | Source | Target | Mode | Purpose |
|-------|--------|--------|------|---------|
| voice_samples | `./voice_samples` | `/app/voice_samples` | rw | User voice samples (add/modify without rebuild) |
| .env | `./.env` | `/app/.env` | ro | Configuration (read-only for security) |
| hf_cache | Named volume | `/root/.cache/huggingface` | rw | Model cache (prevents re-download) |

**Why bind mounts instead of copying into image**:
1. **Voice samples**: Users frequently add new voices; bind mount avoids rebuilds
2. **Configuration**: `.env` contains secrets/settings; mount allows easy changes
3. **HuggingFace cache**: Named volume persists across container recreates (~1-2GB model)

---

## GPU Support Requirements

### Host System Requirements

1. **NVIDIA Driver**: Version >= 525.60.13 (for CUDA 12.1)
   - Check: `nvidia-smi` (should show driver version)

2. **NVIDIA Container Toolkit**: Required for Docker GPU access
   - Install on Linux:
     ```bash
     # Ubuntu/Debian
     curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
     curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
       sed 's#libnvidia-container https://#libnvidia-container [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
       sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
     sudo apt-get update
     sudo apt-get install -y nvidia-container-toolkit
     sudo nvidia-ctk runtime configure --runtime=docker
     sudo systemctl restart docker
     ```
   - Windows: Docker Desktop includes NVIDIA support automatically

3. **Verify GPU access**:
   ```bash
   docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
   ```

---

## Deployment Steps

### Step 1: Prerequisites

1. Ensure NVIDIA driver installed (`nvidia-smi` works)
2. Install NVIDIA Container Toolkit (Linux) or use Docker Desktop (Windows)
3. Verify GPU access in Docker (see command above)

### Step 2: Create Files

1. Copy `.env.docker.example` to `.env` and customize
2. Add voice samples to `voice_samples/default.wav` (or other .wav/.mp3 files)
3. Dockerfile, docker-compose.yml, .dockerignore will be created

### Step 3: Build and Run

```bash
# Build image (first time takes ~5-10 minutes for dependencies)
docker compose build

# Start container (model downloads on first run)
docker compose up -d

# Watch logs (model loading takes 1-2 minutes)
docker compose logs -f

# Check health
curl http://localhost:8000/health
```

### Step 4: Test

```bash
# Test API
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello from Docker!", "voice": "default", "response_format": "wav"}' \
  --output test_docker.wav

# Or use the Web UI
open http://localhost:8000
```

---

## Common Operations

### Stop container
```bash
docker compose down
```

### Restart container
```bash
docker compose restart
```

### Update code (rebuild)
```bash
docker compose build --no-cache
docker compose up -d
```

### Add new voice
```bash
# Just copy file to voice_samples/ (no rebuild needed)
cp new_voice.wav voice_samples/new_voice.wav
# API automatically detects it
```

### View logs
```bash
docker compose logs -f luxtts
```

### GPU memory check inside container
```bash
docker compose exec luxtts python -c "import torch; print(torch.cuda.memory_allocated()/1e9, 'GB used')"
```

---

## Troubleshooting

### Issue: "Could not find GPU"
**Solution**: Verify NVIDIA Container Toolkit installed and Docker configured
```bash
# Check Docker daemon config
cat /etc/docker/daemon.json  # Should have nvidia runtime
# Restart Docker daemon
sudo systemctl restart docker
```

### Issue: "CUDA out of memory"
**Solution**: Reduce `max_chunk_chars` in API request, or enable FP16
```env
ENABLE_FP16=true  # Halves VRAM usage
```

### Issue: "Model download takes forever"
**Solution**: Check network, or use pre-cached model volume
```bash
# First download is ~1-2GB, subsequent runs use cache
docker volume ls  # Check hf_cache volume exists
```

### Issue: "Voice not found"
**Solution**: Ensure voice file exists in mounted directory
```bash
ls voice_samples/  # Should show .wav or .mp3 files
```

### Issue: "Container exits immediately"
**Solution**: Check logs for startup errors
```bash
docker compose logs luxtts
```

---

## Production Recommendations

1. **Use `.env` for secrets**: Never hardcode API keys in Dockerfile
2. **Read-only mounts where possible**: `.env` mounted `:ro` prevents accidental modification
3. **Health checks**: Already configured in docker-compose.yml
4. **Named volumes for cache**: HuggingFace cache persists across rebuilds
5. **Restart policy**: `unless-stopped` ensures availability
6. **Port binding**: Default 8000, change in `.env` if needed

---

## File Summary

| File | Purpose | Git Status |
|------|---------|------------|
| `Dockerfile` | Build container image | New file (to be created) |
| `docker-compose.yml` | Container orchestration | New file (to be created) |
| `.dockerignore` | Exclude files from build context | New file (to be created) |
| `.env.docker.example` | Environment template | New file (to be created) |
| `.env` | User configuration (mounted) | Existing, not modified |
| `voice_samples/` | Voice samples (mounted) | Existing, not modified |

---

## Next Steps

After plan approval, I will create:
1. `Dockerfile` - Full build instructions
2. `docker-compose.yml` - GPU-enabled orchestration
3. `.dockerignore` - Build exclusions
4. `.env.docker.example` - Configuration template

All new files will be added to git (no changes to existing files except potential `.gitignore` update to track `.env.docker.example`).