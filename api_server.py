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
import tempfile
import logging
import warnings
from typing import Optional, Literal
from pathlib import Path
from contextlib import asynccontextmanager
from dotenv import load_dotenv

# Suppress warnings
warnings.filterwarnings("ignore", message="Failed import k2")
warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`")
logging.getLogger("root").setLevel(logging.ERROR)

# Load environment variables from .env file
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
    stream: bool = Field(default=False, description="Enable streaming response (for compatibility)")

    # LuxTTS specific parameters (optional)
    rms: float = Field(default=0.01, ge=0.0, le=1.0, description="Audio volume (RMS)")
    t_shift: float = Field(default=0.9, ge=0.0, le=1.0, description="Sampling temperature")
    num_steps: int = Field(default=4, ge=1, le=10, description="Number of sampling steps")
    return_smooth: bool = Field(default=False, description="Enable smoother audio")
    ref_duration: int = Field(default=5, ge=1, le=1000, description="Reference duration")
    transcription: Optional[str] = Field(default=None, description="Custom transcription text for voice reference (bypasses Whisper, allows longer audio files)")


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

    Set stream=true for streaming response (compatible with Open WebUI).
    """
    if lux_tts is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    print(f"[TTS Request] voice={request.voice}, format={request.response_format}, stream={request.stream}, text_length={len(request.input)}")

    # Get voice sample path
    voice_file = VOICE_SAMPLES_DIR / f"{request.voice}.wav"
    if not voice_file.exists():
        # Try mp3
        voice_file = VOICE_SAMPLES_DIR / f"{request.voice}.mp3"
        if not voice_file.exists():
            print(f"[TTS Error] Voice file not found: {voice_file}")
            raise HTTPException(
                status_code=400,
                detail=f"Voice '{request.voice}' not found. Place your audio file in voice_samples/ as {request.voice}.wav or {request.voice}.mp3"
            )

    print(f"[TTS] Using voice file: {voice_file}")

    # Determine transcription text (in order: request parameter > .txt file > Whisper)
    transcription_text = request.transcription
    if transcription_text is None:
        # Check for .txt file alongside voice sample
        txt_file = VOICE_SAMPLES_DIR / f"{request.voice}.txt"
        if txt_file.exists():
            with open(txt_file, 'r', encoding='utf-8') as f:
                transcription_text = f.read().strip()
            print(f"[TTS] Loaded transcription from {txt_file}")

    # Cap ref_duration at 30 seconds only if using Whisper (no custom transcription)
    if transcription_text is None:
        actual_duration = min(request.ref_duration, 30)
        if request.ref_duration > 30:
            print(f"[TTS] Warning: ref_duration capped at 30 seconds (use transcription parameter or .txt file to bypass)")
    else:
        actual_duration = request.ref_duration  # Use full requested duration with custom transcription
        print(f"[TTS] Using custom transcription (allows longer audio files)")

    # If streaming is requested, use streaming response
    if request.stream:
        return await create_speech_stream(request)

    try:
        # Encode the prompt audio
        print(f"[TTS] Encoding prompt audio...")
        encoded_prompt = lux_tts.encode_prompt(
            str(voice_file),
            rms=request.rms,
            duration=actual_duration,
            transcription_text=transcription_text
        )

        # Generate speech
        print(f"[TTS] Generating speech...")
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
        print(f"[TTS] Generated audio shape: {audio_numpy.shape}")

        # Return based on format
        if request.response_format == "wav":
            with io.BytesIO() as buffer:
                sf.write(buffer, audio_numpy, 48000, format="WAV")
                print(f"[TTS] Returning WAV audio: {len(buffer.getvalue())} bytes")
                return Response(content=buffer.getvalue(), media_type="audio/wav")
        elif request.response_format == "mp3":
            # For mp3, we'll use pydub to convert (in-memory to avoid file locking issues)
            from pydub import AudioSegment
            try:
                # Write WAV to memory first
                with io.BytesIO() as wav_buffer:
                    sf.write(wav_buffer, audio_numpy, 48000, format="WAV")
                    wav_data = wav_buffer.getvalue()

                # Create AudioSegment from WAV data in memory
                audio_segment = AudioSegment.from_wav(io.BytesIO(wav_data))

                # Export to MP3 in memory
                with io.BytesIO() as mp3_buffer:
                    audio_segment.export(mp3_buffer, format="mp3")
                    mp3_data = mp3_buffer.getvalue()

                print(f"[TTS] Returning MP3 audio: {len(mp3_data)} bytes")
                return Response(content=mp3_data, media_type="audio/mpeg")
            except Exception as e:
                print(f"[TTS Error] MP3 conversion failed: {e}")
                raise HTTPException(status_code=500, detail=f"MP3 conversion failed: {str(e)}")
        else:  # pcm
            with io.BytesIO() as buffer:
                sf.write(buffer, audio_numpy, 48000, format="RAW", subtype="PCM_16")
                print(f"[TTS] Returning PCM audio: {len(buffer.getvalue())} bytes")
                return Response(content=buffer.getvalue(), media_type="audio/pcm")

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"[TTS Error] {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error generating speech: {str(e)}")


# Alias endpoint for Open WebUI compatibility (some clients expect /audio/speech)
@app.post("/audio/speech")
async def create_speech_alias(request: TTSRequest):
    """Alias endpoint for /v1/audio/speech - redirects to main endpoint"""
    return await create_speech(request)


@app.post("/v1/audio/speech/stream")
async def create_speech_stream(request: TTSRequest):
    """
    Generate speech with streaming response (explicit streaming endpoint)

    This endpoint streams audio in chunks, allowing clients to start playback
    before the entire audio is generated. Note: LuxTTS generates the full audio
    internally, then chunks it for streaming.

    You can also use the main /v1/audio/speech endpoint with stream=true parameter.
    """
    if lux_tts is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Get voice sample path
    voice_file = VOICE_SAMPLES_DIR / f"{request.voice}.wav"
    if not voice_file.exists():
        voice_file = VOICE_SAMPLES_DIR / f"{request.voice}.mp3"
        if not voice_file.exists():
            raise HTTPException(
                status_code=400,
                detail=f"Voice '{request.voice}' not found. Place your audio file in voice_samples/ as {request.voice}.wav or {request.voice}.mp3"
            )

    # Determine transcription text (in order: request parameter > .txt file > Whisper)
    transcription_text = request.transcription
    if transcription_text is None:
        txt_file = VOICE_SAMPLES_DIR / f"{request.voice}.txt"
        if txt_file.exists():
            with open(txt_file, 'r', encoding='utf-8') as f:
                transcription_text = f.read().strip()
            print(f"[TTS] Loaded transcription from {txt_file}")

    async def audio_generator():
        """Generate audio and yield it in chunks"""
        try:
            # Cap ref_duration at 30 seconds only if using Whisper (no custom transcription)
            if transcription_text is None:
                actual_duration = min(request.ref_duration, 30)
                if request.ref_duration > 30:
                    print(f"[TTS] Warning: ref_duration capped at 30 seconds (use transcription parameter or .txt file to bypass)")
            else:
                actual_duration = request.ref_duration
                print(f"[TTS] Using custom transcription (allows longer audio files)")

            # Encode the prompt audio
            encoded_prompt = lux_tts.encode_prompt(
                str(voice_file),
                rms=request.rms,
                duration=actual_duration,
                transcription_text=transcription_text
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

            # Determine media type and format
            if request.response_format == "wav":
                media_type = "audio/wav"
                with io.BytesIO() as buffer:
                    sf.write(buffer, audio_numpy, 48000, format="WAV")
                    audio_data = buffer.getvalue()
            elif request.response_format == "mp3":
                from pydub import AudioSegment
                # Write WAV to memory first
                with io.BytesIO() as wav_buffer:
                    sf.write(wav_buffer, audio_numpy, 48000, format="WAV")
                    wav_data = wav_buffer.getvalue()
                # Create AudioSegment from WAV data in memory
                audio_segment = AudioSegment.from_wav(io.BytesIO(wav_data))
                # Export to MP3 in memory
                with io.BytesIO() as buffer:
                    audio_segment.export(buffer, format="mp3")
                    audio_data = buffer.getvalue()
                media_type = "audio/mpeg"
            else:  # pcm
                media_type = "audio/pcm"
                with io.BytesIO() as buffer:
                    sf.write(buffer, audio_numpy, 48000, format="RAW", subtype="PCM_16")
                    audio_data = buffer.getvalue()

            # Stream in chunks (8KB chunks for smooth playback)
            chunk_size = 8192
            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i:i + chunk_size]
                yield chunk

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error generating speech: {str(e)}")

    return StreamingResponse(
        audio_generator(),
        media_type="audio/wav" if request.response_format == "wav" else "audio/mpeg" if request.response_format == "mp3" else "audio/pcm"
    )


# Alias streaming endpoint for Open WebUI compatibility
@app.post("/audio/speech/stream")
async def create_speech_stream_alias(request: TTSRequest):
    """Alias endpoint for /v1/audio/speech/stream - redirects to main streaming endpoint"""
    return await create_speech_stream(request)


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
