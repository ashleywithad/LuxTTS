# LuxTTS Setup Guide

## Recommended: Using uv (Fast Package Manager)

uv is recommended for faster installation. The project requires Python 3.10 due to dependency constraints.

```bash
# Install uv
pip install uv

# Create virtual environment with Python 3.10
uv venv --python 3.10

# Install dependencies (automatically uses the venv)
uv pip install -r requirements.txt
```

## Alternative: Using pip

```bash
# Create virtual environment with Python 3.10
python3.10 -m venv .venv
.venv/Scripts/activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

**Note**: Python 3.10 is required due to `llvmlite` dependency constraints.

## Windows Setup (If using pip with system Python)

The main dependency that causes issues on Windows is `piper_phonemize`. Follow these steps to fix it:

### Step 1: Install Visual C++ Redistributable

Download and install the latest Microsoft Visual C++ Redistributable:
- Download from: https://aka.ms/vs/17/release/vc_redist.x64.exe
- Run the installer and complete the installation

### Step 2: Reinstall piper_phonemize

```bash
pip uninstall piper_phonemize -y
pip install piper_phonemize --find-links https://k2-fsa.github.io/icefall/piper_phonemize.html
```

### Step 3: Verify Installation

```bash
python -c "from zipvoice.luxvoice import LuxTTS; print('Success!')"
```

## Quick Start (After Fixing Dependencies)

### 1. Add a Voice Sample

Place a voice reference audio file (at least 3 seconds) in `voice_samples/`:

```bash
# Add your voice sample
cp your_audio_file.wav voice_samples/default.wav
```

### 2. Start the API Server

```bash
# Default (CUDA, port 8000)
python api_server.py

# CPU mode
set DEVICE=cpu && python api_server.py

# Custom port
set PORT=8080 && python api_server.py
```

### 3. Test the API

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello, world!", "voice": "default", "response_format": "wav"}' \
  --output test.wav
```

Or use the Python client:

```bash
python client_example.py
```

## Troubleshooting

### "DLL load failed" Error

This means the Visual C++ Redistributable is missing. Install it from:
https://aka.ms/vs/17/release/vc_redist.x64.exe

### "Voice not found" Error

Make sure you have placed an audio file in `voice_samples/` named `default.wav` or `default.mp3`.

### CUDA Not Available

The model will automatically fall back to CPU if CUDA is not available. You can also explicitly set:
```bash
set DEVICE=cpu
python api_server.py
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `YatharthS/LuxTTS` | HuggingFace model identifier |
| `DEVICE` | `cuda` | Device: cuda, cpu, or mps |
| `HOST` | `0.0.0.0` | Server host |
| `PORT` | `8000` | Server port |
