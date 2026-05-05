# LuxTTS Dockerfile - OpenAI-Compatible TTS API Server
# Base: PyTorch CUDA 12.1 Runtime (~4GB)
# Purpose: Production-ready container for LuxTTS inference

# =============================================================================
# WHAT: LuxTTS API container build
# WHAT IT DOES: Creates a minimal production image with PyTorch CUDA support
# WHY: Enables GPU-accelerated TTS inference in isolated Docker environment
# =============================================================================

# Base image: Official PyTorch with CUDA 12.1 and cuDNN 9
# Runtime image (~4GB) - no build tools, only inference needed
# Python 3.10 included (required for llvmlite dependency)
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

# Metadata labels for container management
LABEL maintainer="LuxTTS"
LABEL description="OpenAI-Compatible TTS API Server"
LABEL version="1.1.0"

# Set working directory
WORKDIR /app

# Environment variables for build
# PYTHONUNBUFFERED: Ensures logs flush immediately (important for Docker logs)
# PYTHONDONTWRITEBYTECODE: Prevents .pyc files (cleaner image)
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# =============================================================================
# STAGE 1: Install Python dependencies
# WHY FIRST: Better layer caching - dependencies change less than code
# =============================================================================

# Copy requirements file first (for layer caching)
COPY requirements.txt .

# Install dependencies with pip
# --find-links: Required for piper_phonemize (special wheel location)
# --no-cache-dir: Reduces image size by not caching pip downloads
# zipvoice: Must install from GitHub (pip package is empty placeholder)
RUN pip install --no-cache-dir --find-links https://k2-fsa.github.io/icefall/piper_phonemize.html \
    -r requirements.txt && \
    pip install --no-cache-dir "zipvoice @ git+https://github.com/ysharma3501/LuxTTS.git"

# =============================================================================
# STAGE 2: Copy application code
# WHY SECOND: Code changes more frequently, separate layer for faster rebuilds
# =============================================================================

# Copy only necessary application files
# api_server.py: Main FastAPI server
# static/: Web UI files (index.html)
# voice_samples/: Not copied - mounted at runtime via bind mount
COPY api_server.py .
COPY static/ ./static/

# Create voice_samples directory placeholder
# Actual voice files mounted at runtime via docker-compose bind mount
RUN mkdir -p voice_samples

# =============================================================================
# STAGE 3: Configure container
# =============================================================================

# Expose default port (can be overridden via PORT env var)
EXPOSE 8000

# Health check - verifies model loaded and server responding
# interval: Check every 30s after startup
# timeout: 10s to respond
# retries: 3 failures before marking unhealthy
# start_period: 120s grace period for model download (~1-2GB)
HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=120s \
    CMD curl -f http://localhost:8000/health || exit 1

# =============================================================================
# STAGE 4: Runtime configuration
# =============================================================================

# Default environment variables (can be overridden via .env or docker-compose)
# These are defaults; actual values come from mounted .env file
ENV MODEL_PATH=YatharthS/LuxTTS
ENV DEVICE=cuda
ENV HOST=0.0.0.0
ENV PORT=8000
ENV ENABLE_TF32=true
ENV ENABLE_FP16=false

# Run the API server
# uvicorn handles graceful shutdown and signal handling
# Uses HOST and PORT env vars for binding
CMD ["python", "api_server.py"]