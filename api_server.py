"""
OpenAI-Compatible API Server for LuxTTS

This server provides an OpenAI-compatible TTS API using LuxTTS.
It follows the OpenAI API specification for text-to-speech.

API Endpoints:
- POST /v1/audio/speech - Generate speech from text (OpenAI compatible)
- GET /v1/models - List available models (OpenAI compatible)
- GET /health - Health check endpoint
"""

import os
import io
import tempfile
from typing import Optional, Literal
from pathlib import Path
from contextlib import asynccontextmanager
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import torch
import soundfile as sf

from zipvoice.luxvoice import LuxTTS

# Configuration
MODEL_PATH = os.getenv("MODEL_PATH", "YatharthS/LuxTTS")
DEVICE = os.getenv("DEVICE", "cuda")  # cuda, cpu, or mps
DTYPE = os.getenv("DTYPE", "float32")  # float32 or float16 (GPU only)
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

# Audio samples directory for voice presets
VOICE_SAMPLES_DIR = Path(__file__).parent / "voice_samples"
VOICE_SAMPLES_DIR.mkdir(exist_ok=True)

# Global model instance
lux_tts: Optional[LuxTTS] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown events"""
    global lux_tts
    # Startup
    try:
        print(f"Loading LuxTTS model from {MODEL_PATH} on {DEVICE} with dtype={DTYPE}...")
        lux_tts = LuxTTS(MODEL_PATH, device=DEVICE, dtype=DTYPE)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

    yield

    # Shutdown
    print("Shutting down API server...")


app = FastAPI(
    title="LuxTTS OpenAI-Compatible API",
    description="OpenAI-compatible TTS API powered by LuxTTS",
    version="1.0.0",
    lifespan=lifespan
)

# Mount static files for the web UI
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


class TTSRequest(BaseModel):
    """OpenAI-compatible TTS request model"""
    model: str = Field(default="luxtts", description="Model identifier")
    input: str = Field(..., min_length=1, description="Text to generate speech from")
    voice: str = Field(default="default", description="Voice preset to use")
    response_format: Literal["mp3", "wav", "pcm"] = Field(default="mp3", description="Audio format")
    speed: float = Field(default=1.0, ge=0.25, le=4.0, description="Speed of audio")

    # LuxTTS specific parameters (optional)
    rms: float = Field(default=0.01, ge=0.0, le=1.0, description="Audio volume (RMS)")
    t_shift: float = Field(default=0.9, ge=0.0, le=1.0, description="Sampling temperature")
    num_steps: int = Field(default=4, ge=1, le=10, description="Number of sampling steps")
    return_smooth: bool = Field(default=False, description="Enable smoother audio")
    ref_duration: int = Field(default=5, ge=1, le=1000, description="Reference duration")


class ModelInfo(BaseModel):
    """Model information response"""
    id: str
    object: str = "model"
    created: int
    owned_by: str


class ModelsList(BaseModel):
    """List of available models"""
    object: str = "list"
    data: list[ModelInfo]


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": lux_tts is not None}


@app.get("/")
async def root():
    """Serve the web UI"""
    return FileResponse(str(static_dir / "index.html"))


@app.get("/v1/models", response_model=ModelsList)
async def list_models():
    """List available models (OpenAI compatible)"""
    return ModelsList(
        data=[
            ModelInfo(
                id="luxtts",
                created=1704067200,
                owned_by="LuxTTS"
            )
        ]
    )


@app.get("/v1/voices")
async def list_voices():
    """List available voice presets"""
    voices = []
    for ext in ['*.wav', '*.mp3']:
        voices.extend([f.stem for f in VOICE_SAMPLES_DIR.glob(ext)])
    return {"voices": voices}


@app.post("/v1/audio/speech")
async def create_speech(request: TTSRequest):
    """
    Generate speech from text (OpenAI compatible endpoint)

    This endpoint follows the OpenAI TTS API specification:
    https://platform.openai.com/docs/api-reference/audio/createSpeech
    """
    if lux_tts is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Get voice sample path
    voice_file = VOICE_SAMPLES_DIR / f"{request.voice}.wav"
    if not voice_file.exists():
        # Try mp3
        voice_file = VOICE_SAMPLES_DIR / f"{request.voice}.mp3"
        if not voice_file.exists():
            raise HTTPException(
                status_code=400,
                detail=f"Voice '{request.voice}' not found. Place your audio file in voice_samples/ as {request.voice}.wav or {request.voice}.mp3"
            )

    try:
        # Encode the prompt audio
        encoded_prompt = lux_tts.encode_prompt(
            str(voice_file),
            rms=request.rms,
            duration=request.ref_duration
        )

        # Generate speech
        audio = lux_tts.generate_speech(
            text=request.input,
            encode_dict=encoded_prompt,
            num_steps=request.num_steps,
            t_shift=request.t_shift,
            speed=request.speed,
            return_smooth=request.return_smooth
        )

        # Convert to numpy array
        audio_numpy = audio.numpy().squeeze()

        # Return based on format
        if request.response_format == "wav":
            with io.BytesIO() as buffer:
                sf.write(buffer, audio_numpy, 48000, format="WAV")
                return Response(content=buffer.getvalue(), media_type="audio/wav")
        elif request.response_format == "mp3":
            # For mp3, we'll use pydub to convert
            from pydub import AudioSegment
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
                sf.write(temp_wav.name, audio_numpy, 48000, format="WAV")
                audio_segment = AudioSegment.from_wav(temp_wav.name)
                with io.BytesIO() as buffer:
                    audio_segment.export(buffer, format="mp3")
                    mp3_data = buffer.getvalue()
                os.unlink(temp_wav.name)
                return Response(content=mp3_data, media_type="audio/mpeg")
        else:  # pcm
            with io.BytesIO() as buffer:
                sf.write(buffer, audio_numpy, 48000, format="RAW", subtype="PCM_16")
                return Response(content=buffer.getvalue(), media_type="audio/pcm")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating speech: {str(e)}")


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    return JSONResponse(
        status_code=500,
        content={"error": {"message": str(exc), "type": "internal_error"}}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)
