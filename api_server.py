"""
OpenAI-Compatible API Server for LuxTTS

This server provides an OpenAI-compatible TTS API using LuxTTS.
It follows the OpenAI API specification for text-to-speech.

API Endpoints:
- POST /v1/audio/speech - Generate speech from text (OpenAI compatible, supports stream parameter)
- POST /v1/audio/speech/stream - Explicit streaming endpoint
- GET /v1/models - List available models (OpenAI compatible)
- GET /v1/voices - List available voice presets
- GET /health - Health check endpoint
- GET / - Web UI

Streaming Support:
- Set stream=true in the request body for streaming response
- Compatible with Open WebUI and other OpenAI-compatible clients
"""

import os
import io
import gc
import tempfile
import logging
import warnings
from typing import Optional, Literal
from pathlib import Path
from contextlib import asynccontextmanager
from dotenv import load_dotenv

os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "max_split_size_mb:128,garbage_collection_threshold:0.6,expandable_segments:True",
)

warnings.filterwarnings("ignore", message="Failed import k2")
warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`")
logging.getLogger("root").setLevel(logging.ERROR)

load_dotenv()

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, JSONResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import torch
import soundfile as sf
import numpy as np

from zipvoice.luxvoice import LuxTTS

# Configuration
MODEL_PATH = os.getenv("MODEL_PATH", "YatharthS/LuxTTS")
DEVICE = os.getenv("DEVICE", "cuda")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
ENABLE_TF32 = os.getenv("ENABLE_TF32", "true").lower() in ("true", "1", "yes")
ENABLE_FP16 = os.getenv("ENABLE_FP16", "false").lower() in ("true", "1", "yes")

# Apply Ampere GPU optimizations (RTX 3050, etc.)
if torch.cuda.is_available():
    if ENABLE_TF32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.cufft_plan_cache.max_size = 64
    try:
        torch.cuda.set_per_process_memory_fraction(0.85, device=0)
    except Exception:
        pass

# Audio samples directory for voice presets
VOICE_SAMPLES_DIR = Path(__file__).parent / "voice_samples"
VOICE_SAMPLES_DIR.mkdir(exist_ok=True)

# Global model instance
lux_tts: Optional[LuxTTS] = None


def _apply_half_precision(lux_tts_instance: LuxTTS) -> None:
    """Apply half-precision (float16) to model components for reduced VRAM.

    Note: This may be slower than float32 on consumer Ampere GPUs (RTX 3050/3060)
    due to complex FFT operations falling back to slower emulated execution.
    Use only if VRAM is constrained.
    """
    if lux_tts_instance.device == "cpu":
        return
    lux_tts_instance.model = lux_tts_instance.model.half()
    lux_tts_instance.vocos = lux_tts_instance.vocos.half()


def _cleanup_gpu_memory() -> None:
    """Aggressively free GPU memory between requests."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown events"""
    global lux_tts
    print(f"Loading LuxTTS model from {MODEL_PATH} on {DEVICE}...")
    print(f"  TF32: {'enabled' if ENABLE_TF32 else 'disabled'}")
    print(f"  FP16: {'enabled' if ENABLE_FP16 else 'disabled'}")

    try:
        lux_tts = LuxTTS(MODEL_PATH, device=DEVICE)

        if ENABLE_FP16 and DEVICE != "cpu":
            print("Applying half-precision (FP16)...")
            _apply_half_precision(lux_tts)

        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024 ** 3)
            reserved = torch.cuda.memory_reserved() / (1024 ** 3)
            print(f"GPU memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")

        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

    yield

    print("Shutting down API server...")
    lux_tts = None
    _cleanup_gpu_memory()


app = FastAPI(
    title="LuxTTS OpenAI-Compatible API",
    description="OpenAI-compatible TTS API powered by LuxTTS",
    version="1.0.0",
    lifespan=lifespan,
)

static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


class TTSRequest(BaseModel):
    """OpenAI-compatible TTS request model"""
    model: str = Field(default="luxtts", description="Model identifier")
    input: str = Field(..., min_length=1, description="Text to generate speech from")
    voice: str = Field(default="default", description="Voice preset to use")
    response_format: Literal["mp3", "wav", "pcm"] = Field(default="wav", description="Audio format")
    speed: float = Field(default=1.0, ge=0.25, le=4.0, description="Speed of audio")
    stream: bool = Field(default=False, description="Enable streaming response (for compatibility)")

    rms: float = Field(default=0.001, ge=0.0, le=1.0, description="Audio volume (RMS)")
    t_shift: float = Field(default=0.5, ge=0.0, le=1.0, description="Sampling temperature")
    num_steps: int = Field(default=4, ge=1, le=10, description="Number of sampling steps")
    return_smooth: bool = Field(default=False, description="Enable smoother audio")
    ref_duration: int = Field(default=3, ge=1, le=1000, description="Reference duration in seconds")
    guidance_scale: float = Field(default=3.0, ge=0.1, le=10.0, description="Guidance scale")


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str


class ModelsList(BaseModel):
    object: str = "list"
    data: list[ModelInfo]


@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": lux_tts is not None}


@app.get("/")
async def root():
    return FileResponse(str(static_dir / "index.html"))


@app.get("/v1/models", response_model=ModelsList)
async def list_models():
    return ModelsList(
        data=[
            ModelInfo(
                id="luxtts",
                created=1704067200,
                owned_by="LuxTTS",
            )
        ]
    )


@app.get("/v1/voices")
async def list_voices():
    voices = []
    for ext in ['*.wav', '*.mp3']:
        voices.extend([f.stem for f in VOICE_SAMPLES_DIR.glob(ext)])
    return {"voices": sorted(set(voices))}


def _find_voice_file(voice: str) -> Path:
    for ext in ('.wav', '.mp3'):
        voice_file = VOICE_SAMPLES_DIR / f"{voice}{ext}"
        if voice_file.exists():
            return voice_file
    raise HTTPException(
        status_code=400,
        detail=f"Voice '{voice}' not found. Place your audio file in voice_samples/ as {voice}.wav or {voice}.mp3",
    )


def _encode_prompt(voice_file: Path, rms: float, ref_duration: int):
    print(f"[TTS] Encoding prompt audio: {voice_file} (rms={rms}, duration={ref_duration}s)")
    return lux_tts.encode_prompt(
        str(voice_file),
        rms=rms,
        duration=ref_duration,
    )


def _generate_audio(request: TTSRequest, voice_file: Path):
    encoded_prompt = _encode_prompt(voice_file, request.rms, request.ref_duration)

    dtype = torch.float16 if ENABLE_FP16 and DEVICE != "cpu" else None

    with torch.inference_mode():
        audio = lux_tts.generate_speech(
            text=request.input,
            encode_dict=encoded_prompt,
            num_steps=request.num_steps,
            guidance_scale=request.guidance_scale,
            t_shift=request.t_shift,
            speed=request.speed,
            return_smooth=request.return_smooth,
        )

    audio_numpy = audio.numpy().squeeze()
    print(f"[TTS] Generated audio shape: {audio_numpy.shape}, duration: {len(audio_numpy)/48000:.2f}s")

    _cleanup_gpu_memory()
    return audio_numpy


def _encode_audio(audio_numpy: np.ndarray, fmt: str) -> tuple[bytes, str]:
    if fmt == "wav":
        with io.BytesIO() as buf:
            sf.write(buf, audio_numpy, 48000, format="WAV")
            return buf.getvalue(), "audio/wav"

    if fmt == "pcm":
        with io.BytesIO() as buf:
            sf.write(buf, audio_numpy, 48000, format="RAW", subtype="PCM_16")
            return buf.getvalue(), "audio/pcm"

    # fmt == "mp3" — try soundfile native MP3 first (requires libsndfile >= 1.1)
    try:
        with io.BytesIO() as buf:
            sf.write(buf, audio_numpy, 48000, format="MP3")
            return buf.getvalue(), "audio/mpeg"
    except Exception:
        pass

    # Fallback: pydub/ffmpeg (slower but always available)
    try:
        from pydub import AudioSegment
        with io.BytesIO() as wav_buf:
            sf.write(wav_buf, audio_numpy, 48000, format="WAV")
            wav_data = wav_buf.getvalue()
        audio_segment = AudioSegment.from_wav(io.BytesIO(wav_data))
        with io.BytesIO() as mp3_buf:
            audio_segment.export(mp3_buf, format="mp3")
            return mp3_buf.getvalue(), "audio/mpeg"
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"MP3 conversion failed: {e}")


@app.post("/v1/audio/speech")
async def create_speech(request: TTSRequest):
    if lux_tts is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    print(f"[TTS Request] voice={request.voice}, format={request.response_format}, "
          f"stream={request.stream}, text_len={len(request.input)}")

    voice_file = _find_voice_file(request.voice)

    if request.stream:
        return await create_speech_stream(request)

    try:
        audio_numpy = _generate_audio(request, voice_file)
        content, media_type = _encode_audio(audio_numpy, request.response_format)
        print(f"[TTS] Returning {request.response_format}: {len(content)} bytes")
        return Response(content=content, media_type=media_type)
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"[TTS Error] {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error generating speech: {e}")


@app.post("/audio/speech")
async def create_speech_alias(request: TTSRequest):
    return await create_speech(request)


@app.post("/v1/audio/speech/stream")
async def create_speech_stream(request: TTSRequest):
    if lux_tts is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    voice_file = _find_voice_file(request.voice)

    async def audio_generator():
        try:
            audio_numpy = _generate_audio(request, voice_file)
            audio_data, media_type = _encode_audio(audio_numpy, request.response_format)

            chunk_size = 8192
            for i in range(0, len(audio_data), chunk_size):
                yield audio_data[i:i + chunk_size]
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error generating speech: {e}")

    fmt = request.response_format
    content_type = "audio/wav" if fmt == "wav" else "audio/mpeg" if fmt == "mp3" else "audio/pcm"
    return StreamingResponse(audio_generator(), media_type=content_type)


@app.post("/audio/speech/stream")
async def create_speech_stream_alias(request: TTSRequest):
    return await create_speech_stream(request)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"error": {"message": str(exc), "type": "internal_error"}},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)