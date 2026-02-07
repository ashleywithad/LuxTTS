# LuxTTS - OpenAI-Compatible API Server

LuxTTS is a lightweight text-to-speech model designed for high quality voice cloning and realistic generation at speeds exceeding 150x realtime. This repository includes an **OpenAI-compatible API server** for easy integration.

## Features

- **Voice cloning**: SOTA voice cloning on par with models 10x larger
- **Clarity**: Clear 48kHz speech generation unlike most TTS models (24kHz)
- **Speed**: Reaches speeds of 150x realtime on GPU and faster than realtime on CPU
- **Efficiency**: Fits within 1GB VRAM
- **OpenAI-Compatible API**: Use standard OpenAI TTS API format

## Quick Start

### 1. Installation

**Recommended: Using uv** (fast package manager):

```bash
# Clone the repository
git clone https://github.com/ysharma3501/LuxTTS.git
cd LuxTTS

# Install uv if needed
pip install uv

# Create venv with Python 3.10 and install dependencies
uv venv --python 3.10
uv pip install -r requirements.txt
```

**Alternative: Using pip**:

```bash
python3.10 -m venv .venv
.venv/Scripts/activate  # Windows
pip install -r requirements.txt
```

**Note**: Python 3.10 is required due to dependency constraints. See [SETUP.md](SETUP.md) for detailed instructions.

### 2. Add a Voice Sample

Place a voice reference audio file (at least 3 seconds) in the `voice_samples` directory:

```bash
# Add your voice sample as default.wav
cp your_audio_file.wav voice_samples/default.wav
```

### 3. Start the API Server

```bash
# Start with default settings (CUDA, float32, port 8000)
python api_server.py

# Or use the batch script (Windows)
start_server.bat

# For faster inference with float16 (~2x speed)
start_server_fp16.bat
# Or manually: set DTYPE=float16 && python api_server.py

# Or with custom settings
MODEL_PATH=YatharthS/LuxTTS DEVICE=cpu PORT=8080 DTYPE=float32 python api_server.py
```

The API will be available at `http://localhost:8000`

**Web UI & Documentation:**
- **Web UI**: `http://localhost:8000` - Beautiful web interface for generating speech
- **API Docs**: `http://localhost:8000/docs` - Interactive Swagger API documentation
- **Health Check**: `http://localhost:8000/health`

### Web UI Features

The web interface provides:

- **Text Input** - Enter any text you want to convert to speech
- **Voice Selection** - Choose from available voice presets or enter custom voice name
- **Real-time Controls**:
  - Speed slider (0.25x - 4x)
  - Volume (RMS) control
  - Sampling steps (quality vs speed)
  - Temperature (creativity vs accuracy)
  - Reference duration
  - Smooth audio toggle
- **Audio Player** - Listen to generated speech directly in the browser
- **Download** - Save generated audio as WAV, MP3, or PCM
- **Server Status** - Real-time health check indicator

**Float16 vs Float32**:
- **Float32** (default): Best quality, standard precision
- **Float16** (~2x faster): Slightly less precision but usually imperceptible difference. GPU only.

## API Usage (OpenAI-Compatible)

### Generate Speech

**Endpoint**: `POST /v1/audio/speech`

**Request Body**:
```json
{
  "model": "luxtts",
  "input": "Hello, this is a test of the LuxTTS API!",
  "voice": "default",
  "response_format": "wav",
  "speed": 1.0
}
```

**Example with curl**:
```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "luxtts",
    "input": "Hello, this is a test!",
    "voice": "default",
    "response_format": "wav"
  }' \
  --output output.wav
```

**Example with Python**:
```python
import requests

response = requests.post(
    "http://localhost:8000/v1/audio/speech",
    json={
        "model": "luxtts",
        "input": "Hello, world!",
        "voice": "default",
        "response_format": "wav"
    }
)

with open("output.wav", "wb") as f:
    f.write(response.content)
```

**Streaming Support**:
Set `stream: true` in your request for streaming audio (compatible with Open WebUI):
```python
response = requests.post(
    "http://localhost:8000/v1/audio/speech",
    json={
        "model": "luxtts",
        "input": "Hello, world!",
        "voice": "default",
        "response_format": "wav",
        "stream": true
    },
    stream=True
)

with open("output.wav", "wb") as f:
    for chunk in response.iter_content(chunk_size=8192):
        f.write(chunk)
```

response = requests.post(
    "http://localhost:8000/v1/audio/speech",
    json={
        "model": "luxtts",
        "input": "Hello, world!",
        "voice": "default",
        "response_format": "wav"
    }
)

with open("output.wav", "wb") as f:
    f.write(response.content)
```

### List Models

**Endpoint**: `GET /v1/models`

```bash
curl http://localhost:8000/v1/models
```

## API Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | string | "luxtts" | Model identifier |
| `input` | string | (required) | Text to generate speech from |
| `voice` | string | "default" | Voice preset to use (filename without extension from `voice_samples/`) |
| `response_format` | string | "mp3" | Audio format: "mp3", "wav", or "pcm" |
| `speed` | float | 1.0 | Speed of audio (0.25 to 4.0) |
| `stream` | bool | false | Enable streaming response (for Open WebUI) |
| `rms` | float | 0.01 | Audio volume/RMS (0.0 to 1.0) |
| `t_shift` | float | 0.9 | Sampling temperature (0.0 to 1.0) |
| `num_steps` | int | 4 | Number of sampling steps (1 to 10) |
| `return_smooth` | bool | false | Enable smoother audio |
| `ref_duration` | int | 5 | Reference duration for voice encoding |

## Voice Management

### Adding New Voices

1. Place your audio file in `voice_samples/` directory
2. Name it `{voice_name}.wav` or `{voice_name}.mp3`
3. Use it in API requests with `"voice": "{voice_name}"`

Example:
```bash
# Add a new voice
cp my_voice.mp3 voice_samples/sarah.mp3

# Use it in API
curl -X POST http://localhost:8000/v1/audio/speech \
  -d '{"input": "Hello!", "voice": "sarah", "response_format": "wav"}' \
  --output output.wav
```

## Python Library Usage

You can also use LuxTTS directly in Python:

```python
from zipvoice.luxvoice import LuxTTS
import soundfile as sf

# Load model (float32 default)
lux_tts = LuxTTS('YatharthS/LuxTTS', device='cuda')

# Or load with float16 for faster inference (~2x speed)
# lux_tts = LuxTTS('YatharthS/LuxTTS', device='cuda', dtype='float16')

# Encode reference audio
encoded_prompt = lux_tts.encode_prompt('audio_file.wav', rms=0.01)

# Generate speech
final_wav = lux_tts.generate_speech(
    "Hello, world!",
    encoded_prompt,
    num_steps=4
)

# Save audio
sf.write('output.wav', final_wav.numpy().squeeze(), 48000)
```

## Configuration

You can configure the server using a `.env` file in the project root:

```env
# Model Settings
MODEL_PATH=YatharthS/LuxTTS
DEVICE=cuda
DTYPE=float32

# Server Settings
HOST=0.0.0.0
PORT=8000
```

Or use environment variables:
- `MODEL_PATH`: Model identifier or path (default: "YatharthS/LuxTTS")
- `DEVICE`: Device to use - "cuda", "cpu", or "mps" (default: "cuda")
- `DTYPE`: Data type - "float32" (default) or "float16" (~2x faster, GPU only)
- `HOST`: Server host (default: "0.0.0.0")
- `PORT`: Server port (default: "8000")

## Tips

- Use at minimum a 3 second audio file for voice cloning
- Use `return_smooth = True` if you hear metallic sounds
- Lower `t_shift` for fewer pronunciation errors but possibly worse quality
- The model automatically detects available hardware (CUDA/MPS/CPU)

## License

Apache-2.0 license. See LICENSE for details.

## Acknowledgments

- [ZipVoice](https://github.com/ysharma3501/LinaCodec) for their excellent code and model
- [Vocos](https://github.com/gemelo-ai/vocos) for their great vocoder
