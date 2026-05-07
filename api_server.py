"""
OpenAI-Compatible API Server for LuxTTS

This server provides an OpenAI-compatible TTS API using LuxTTS.
It follows the OpenAI API specification for text-to-speech.

API Endpoints:
- POST /v1/audio/speech - Generate speech from text (OpenAI compatible, supports stream parameter)
- POST /v1/audio/speech/stream - Explicit streaming endpoint (raw PCM progressive)
- GET /v1/models - List available models (OpenAI compatible)
- GET /v1/voices - List available voice presets
- GET /health - Health check endpoint
- GET / - Web UI

Streaming Support:
- Set stream=true for progressive PCM streaming (raw int16 LE, 48kHz, mono)
- Audio is sent chunk-by-chunk as each text segment finishes generating
- No cross-fade in streaming (lowest latency); non-streaming retains cross-fade
- Compatible with Open WebUI and other OpenAI-compatible clients

Chunking Support:
- Long text is automatically split into chunks to prevent CUDA OOM on low-VRAM GPUs
- Chunks are generated separately and cross-fade concatenated for seamless output
- Use max_chunk_chars=0 to disable chunking (single-pass generation)
"""

import os
import sys
import io
import re
import gc
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
from zipvoice.utils.infer import cross_fade_concat

# Configuration
MODEL_PATH = os.getenv("MODEL_PATH", "YatharthS/LuxTTS")
DEVICE = os.getenv("DEVICE", "cuda")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
ENABLE_TF32 = os.getenv("ENABLE_TF32", "true").lower() in ("true", "1", "yes")
ENABLE_FP16 = os.getenv("ENABLE_FP16", "false").lower() in ("true", "1", "yes")
# WHAT: Auto-restart when CUDA allocator corruption detected
# WHAT IT DOES: Forces process exit on corruption, Docker restart policy brings it back
# WHY: Prevents manual intervention when allocator is broken from previous OOM
AUTO_RESTART_ON_CORRUPTION = os.getenv("AUTO_RESTART_ON_CORRUPTION", "true").lower() in ("true", "1", "yes")

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

# Chunking constants
SAMPLE_RATE = 48000
CROSSFADE_DURATION = 0.05  # 50ms cross-fade between chunks
# WHAT: Default chunk size for text splitting
# WHAT IT DOES: Limits text chunk size to prevent CUDA OOM on 4GB VRAM GPUs
# WHY: Reduced to 150 chars for Docker memory overhead (host can use larger)
DEFAULT_MAX_CHUNK_CHARS = 150  # ~60 tokens, conservative for Docker + 4GB VRAM

# Global model instance
lux_tts: Optional[LuxTTS] = None


def _apply_half_precision(lux_tts_instance: LuxTTS) -> None:
    if lux_tts_instance.device == "cpu":
        return
    lux_tts_instance.model = lux_tts_instance.model.half()
    lux_tts_instance.vocos = lux_tts_instance.vocos.half()


def _cleanup_gpu_memory() -> None:
    """WHAT: Force GPU memory cleanup
    WHAT IT DOES: Runs Python garbage collector and PyTorch CUDA cache clear
    WHY: Prevents memory accumulation between chunk generations on low-VRAM GPUs"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _is_cuda_error(error: Exception) -> bool:
    """WHAT: Detect if an exception is CUDA-related
    WHAT IT DOES: Checks error message for CUDA/OOM keywords
    WHY: RuntimeError with 'CUDA out of memory' is thrown instead of OutOfMemoryError"""
    error_str = str(error).lower()
    return (
        "cuda" in error_str
        or "out of memory" in error_str
        or "!handles_.at(i)" in str(error)  # Allocator corruption signature
        or "INTERNAL ASSERT FAILED" in str(error)
    )


def _reset_cuda_allocator() -> bool:
    """WHAT: Attempt to recover from CUDA allocator corruption
    WHAT IT DOES: Synchronize, clear cache, reset peak stats
    WHY: After allocator corruption (!handles_.at(i)), attempts graceful recovery
    Returns: True if recovery attempted (not guaranteed success)"""
    if not torch.cuda.is_available():
        return False
    try:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        gc.collect()
        return True
    except Exception:
        return False


def _check_cuda_allocator_health() -> bool:
    """WHAT: Test if CUDA allocator is functional before generation
    WHAT IT DOES: Attempts small memory allocation, detects corruption signature
    WHY: Once corrupted (!handles_.at(i)), all subsequent requests fail until restart
         Pre-flight check prevents failing mid-generation, gives immediate feedback"""
    if not torch.cuda.is_available():
        return True  # CPU mode, no CUDA allocator
    try:
        # WHAT: Small test allocation to verify allocator is functional
        # WHAT IT DOES: Allocates 1024 floats, immediately frees them
        # WHY: If allocator corrupted, even small allocation triggers RuntimeError
        test_tensor = torch.empty(1024, device='cuda')
        del test_tensor
        torch.cuda.empty_cache()
        return True
    except RuntimeError as e:
        # WHAT: Check for allocator corruption signature
        # WHAT IT DOES: Detect !handles_.at(i) or INTERNAL ASSERT FAILED
        # WHY: These signatures indicate allocator state is permanently broken
        error_str = str(e)
        if "!handles_.at(i)" in error_str or "INTERNAL ASSERT FAILED" in error_str:
            print(f"[CUDA] Allocator corruption detected in pre-flight check: {error_str}")
            return False
        # WHAT: Other RuntimeErrors might be transient
        # WHAT IT DOES: Return True to allow generation attempt
        # WHY: Not all CUDA errors indicate permanent corruption
        return True


def _handle_allocator_corruption(context: str = "generation") -> None:
    """WHAT: Handle detected CUDA allocator corruption
    WHAT IT DOES: Either auto-restarts process or raises HTTPException 503
    WHY: Once corrupted, allocator cannot recover - restart required
    Args: context - Description of where corruption was detected (for logging)"""
    print(f"[CUDA] Allocator corrupted during {context}. Auto-restart: {AUTO_RESTART_ON_CORRUPTION}")

    if AUTO_RESTART_ON_CORRUPTION:
        print("[CUDA] Triggering auto-restart via os._exit(1)...")
        # WHAT: Force immediate process termination
        # WHAT IT DOES: Bypasses Python cleanup, Docker sees non-zero exit
        # WHY: Faster than graceful shutdown, Docker restart policy brings it back
        sys.stdout.flush()
        os._exit(1)

    # WHAT: If auto-restart disabled, raise HTTPException for client feedback
    raise HTTPException(
        status_code=503,
        detail=f"CUDA allocator corrupted during {context}. "
               "Container restart required. Set AUTO_RESTART_ON_CORRUPTION=true for automatic recovery."
    )


def _chunk_text(text: str, max_chars: int = DEFAULT_MAX_CHUNK_CHARS) -> list[str]:
    """Split text into chunks at sentence boundaries, each <= max_chars.

    Splits on sentence-ending punctuation (. ! ? ; : , \n) and merges
    short segments together up to max_chars. Keeps punctuation attached
    to the preceding sentence.
    """
    if max_chars <= 0 or len(text) <= max_chars:
        return [text]

    # Split on sentence boundaries, keeping the delimiter with the preceding text
    parts = re.split(r'([.!?,;:\n]+)', text)
    segments: list[str] = []
    current = ""
    i = 0
    while i < len(parts):
        part = parts[i]
        i += 1
        # If the next part is a delimiter, merge it with the current segment
        if i < len(parts) and re.match(r'[.!?,;:\n]+', parts[i]):
            part += parts[i]
            i += 1
        candidate = current + part if not current else current + " " + part
        candidate = candidate.strip()
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                segments.append(current.strip())
            # If a single segment exceeds max_chars, force-split by words
            if len(part.strip()) > max_chars:
                words = part.strip().split()
                forced = ""
                for word in words:
                    if forced and len(forced) + 1 + len(word) > max_chars:
                        segments.append(forced.strip())
                        forced = word
                    else:
                        forced = (forced + " " + word).strip() if forced else word
                current = forced
            else:
                current = part.strip()

    if current.strip():
        segments.append(current.strip())

    # Merge tiny trailing segments into the previous one if possible
    merged: list[str] = []
    for seg in segments:
        if merged and len(seg) < 20 and len(merged[-1]) + 1 + len(seg) <= max_chars:
            merged[-1] = (merged[-1] + " " + seg).strip()
        else:
            merged.append(seg)

    return [s for s in merged if s]


def _crossfade_numpy(chunks: list[np.ndarray], fade_samples: int) -> np.ndarray:
    """Cross-fade concatenate numpy audio chunks.

    Args:
        chunks: List of 1D numpy audio arrays at SAMPLE_RATE.
        fade_samples: Number of samples for cross-fade overlap.

    Returns:
        Single concatenated 1D numpy array.
    """
    if len(chunks) == 0:
        return np.array([], dtype=np.float32)
    if len(chunks) == 1:
        return chunks[0]
    if fade_samples <= 0:
        return np.concatenate(chunks)

    result = chunks[0]
    for chunk in chunks[1:]:
        k = min(fade_samples, len(result), len(chunk))
        if k <= 0:
            result = np.concatenate([result, chunk])
            continue
        fade_out = np.linspace(1.0, 0.0, k, dtype=np.float32)
        fade_in = np.linspace(0.0, 1.0, k, dtype=np.float32)
        overlapped = result[-k:] * fade_out + chunk[:k] * fade_in
        result = np.concatenate([result[:-k], overlapped, chunk[k:]])
    return result


@asynccontextmanager
async def lifespan(app: FastAPI):
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
    version="1.1.0",
    lifespan=lifespan,
)

static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


class TTSRequest(BaseModel):
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
    max_chunk_chars: int = Field(
        default=DEFAULT_MAX_CHUNK_CHARS, ge=0, le=5000,
        description="Max characters per chunk (0=disable chunking, prevents OOM on long text)"
    )


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


def _generate_single(text: str, encoded_prompt: dict, num_steps: int,
                      guidance_scale: float, t_shift: float, speed: float,
                      return_smooth: bool) -> np.ndarray:
    """WHAT: Generate audio for a single text chunk
    WHAT IT DOES: Runs LuxTTS generation inside inference_mode, returns numpy array
    WHY: inference_mode reduces memory overhead; explicit tensor deletion prevents accumulation"""
    with torch.inference_mode():
        audio = lux_tts.generate_speech(
            text=text,
            encode_dict=encoded_prompt,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            t_shift=t_shift,
            speed=speed,
            return_smooth=return_smooth,
        )
        # WHAT: Convert to numpy and explicitly delete CUDA tensor
        # WHAT IT DOES: Breaks reference chain before cleanup
        # WHY: Allows PyTorch to free GPU memory immediately
        audio_numpy = audio.numpy().squeeze()
        del audio
    _cleanup_gpu_memory()
    return audio_numpy


def _generate_audio(request: TTSRequest, voice_file: Path) -> np.ndarray:
    # WHAT: Pre-flight check for CUDA allocator corruption
    # WHAT IT DOES: Verify allocator is functional before attempting generation
    # WHY: If corrupted from previous OOM, fail immediately instead of mid-generation
    if not _check_cuda_allocator_health():
        _handle_allocator_corruption("pre-flight check (non-streaming)")

    encoded_prompt = _encode_prompt(voice_file, request.rms, request.ref_duration)

    # Determine if chunking is needed
    chunks = _chunk_text(request.input, request.max_chunk_chars)

    if len(chunks) <= 1:
        # Single-pass generation (short text)
        audio_numpy = _generate_single(
            request.input, encoded_prompt,
            request.num_steps, request.guidance_scale,
            request.t_shift, request.speed, request.return_smooth,
        )
        print(f"[TTS] Generated audio: shape={audio_numpy.shape}, "
              f"duration={len(audio_numpy)/SAMPLE_RATE:.2f}s")
        return audio_numpy

    # Chunked generation (long text — prevents OOM on low-VRAM GPUs)
    # WHAT: Split text into chunks, generate each separately, cross-fade concatenate
    # WHAT IT DOES: Prevents CUDA OOM by limiting per-chunk memory usage
    # WHY: 4GB VRAM can't hold full text generation state; chunks allow progressive generation
    print(f"[TTS] Chunking text into {len(chunks)} segments "
          f"({', '.join(str(len(c)) for c in chunks)} chars)")
    fade_samples = int(CROSSFADE_DURATION * SAMPLE_RATE)
    audio_chunks: list[np.ndarray] = []

    for i, chunk_text in enumerate(chunks):
        # WHAT: Aggressive cleanup before each chunk (except first)
        # WHAT IT DOES: Synchronize + empty_cache to release previous chunk memory
        # WHY: Prevents memory accumulation across chunks in Docker environment
        if i > 0 and torch.cuda.is_available():
            torch.cuda.synchronize()
            _cleanup_gpu_memory()

        print(f"[TTS] Generating chunk {i+1}/{len(chunks)}: "
              f"{len(chunk_text)} chars, '{chunk_text[:50]}...'")
        try:
            chunk_audio = _generate_single(
                chunk_text, encoded_prompt,
                request.num_steps, request.guidance_scale,
                request.t_shift, request.speed, request.return_smooth,
            )
            # Ensure 1D
            if chunk_audio.ndim > 1:
                chunk_audio = chunk_audio.squeeze()
            audio_chunks.append(chunk_audio)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            # WHAT: Catch both OutOfMemoryError and CUDA RuntimeError
            # WHAT IT DOES: Prevents allocator corruption cascade
            # WHY: RuntimeError 'CUDA out of memory' is thrown, not OutOfMemoryError
            _cleanup_gpu_memory()

            # WHAT: Check for allocator corruption signature
            # WHAT IT DOES: Trigger auto-restart or raise 503
            # WHY: !handles_.at(i) means allocator state is permanently broken
            if "!handles_.at(i)" in str(e) or "INTERNAL ASSERT FAILED" in str(e):
                _handle_allocator_corruption(f"chunk {i+1}/{len(chunks)} generation")

            # WHAT: Normal CUDA OOM (not corruption)
            # WHAT IT DOES: Return 507 with actionable advice
            # WHY: Client can retry with smaller chunks without allocator damage
            if _is_cuda_error(e):
                raise HTTPException(
                    status_code=507,
                    detail=f"CUDA OOM on chunk {i+1}/{len(chunks)} "
                           f"({len(chunk_text)} chars). Try reducing max_chunk_chars "
                           f"(current: {request.max_chunk_chars}) or shorter text.",
                )

            # WHAT: Non-CUDA RuntimeError — re-raise
            # WHAT IT DOES: Let FastAPI handle as 500 error
            # WHY: Should not corrupt allocator, allow debugging
            raise

    result = _crossfade_numpy(audio_chunks, fade_samples)
    total_duration = len(result) / SAMPLE_RATE
    print(f"[TTS] Chunked generation complete: {len(chunks)} chunks, "
          f"total duration={total_duration:.2f}s")
    return result


def _generate_audio_chunks(request: TTSRequest, voice_file: Path):
    """WHAT: Yield audio chunks progressively for streaming endpoint
    WHAT IT DOES: Generates each chunk separately, sends PCM bytes immediately
    WHY: Progressive streaming for real-time playback; prevents OOM with chunking"""

    # WHAT: Pre-flight check for CUDA allocator corruption
    # WHAT IT DOES: Verify allocator is functional before attempting generation
    # WHY: If corrupted from previous OOM, fail immediately instead of mid-stream
    if not _check_cuda_allocator_health():
        _handle_allocator_corruption("pre-flight check (streaming)")

    encoded_prompt = _encode_prompt(voice_file, request.rms, request.ref_duration)
    chunks = _chunk_text(request.input, request.max_chunk_chars)

    for i, chunk_text in enumerate(chunks):
        # WHAT: Aggressive cleanup before each chunk (except first)
        # WHAT IT DOES: Synchronize + empty_cache to release previous chunk memory
        # WHY: Prevents memory accumulation in streaming mode
        if i > 0 and torch.cuda.is_available():
            torch.cuda.synchronize()
            _cleanup_gpu_memory()

        print(f"[TTS Stream] Generating chunk {i+1}/{len(chunks)}: {len(chunk_text)} chars")
        try:
            chunk_audio = _generate_single(
                chunk_text, encoded_prompt,
                request.num_steps, request.guidance_scale,
                request.t_shift, request.speed, request.return_smooth,
            )
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            # WHAT: Catch both OutOfMemoryError and CUDA RuntimeError
            # WHAT IT DOES: Prevents allocator corruption cascade in streaming
            # WHY: Streaming generator errors handled differently than batch
            _cleanup_gpu_memory()

            # WHAT: Check for allocator corruption signature
            # WHAT IT DOES: Trigger auto-restart or raise 503
            # WHY: !handles_.at(i) means allocator state is permanently broken
            if "!handles_.at(i)" in str(e) or "INTERNAL ASSERT FAILED" in str(e):
                _handle_allocator_corruption(f"chunk {i+1}/{len(chunks)} streaming")

            # WHAT: Normal CUDA OOM
            if _is_cuda_error(e):
                raise HTTPException(
                    status_code=507,
                    detail=f"CUDA OOM on chunk {i+1}/{len(chunks)} "
                           f"({len(chunk_text)} chars). Try reducing max_chunk_chars "
                           f"(current: {request.max_chunk_chars}) or shorter text.",
                )

            raise

        if chunk_audio.ndim > 1:
            chunk_audio = chunk_audio.squeeze()
        yield chunk_audio, i, len(chunks)


def _encode_audio(audio_numpy: np.ndarray, fmt: str) -> tuple[bytes, str]:
    if fmt == "wav":
        with io.BytesIO() as buf:
            sf.write(buf, audio_numpy, SAMPLE_RATE, format="WAV")
            return buf.getvalue(), "audio/wav"

    if fmt == "pcm":
        with io.BytesIO() as buf:
            sf.write(buf, audio_numpy, SAMPLE_RATE, format="RAW", subtype="PCM_16")
            return buf.getvalue(), "audio/pcm"

    # fmt == "mp3" — try soundfile native MP3 first (requires libsndfile >= 1.1)
    try:
        with io.BytesIO() as buf:
            sf.write(buf, audio_numpy, SAMPLE_RATE, format="MP3")
            return buf.getvalue(), "audio/mpeg"
    except Exception:
        pass

    # Fallback: pydub/ffmpeg (slower but always available)
    try:
        from pydub import AudioSegment
        with io.BytesIO() as wav_buf:
            sf.write(wav_buf, audio_numpy, SAMPLE_RATE, format="WAV")
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
          f"stream={request.stream}, text_len={len(request.input)}, "
          f"max_chunk_chars={request.max_chunk_chars}")

    voice_file = _find_voice_file(request.voice)

    if request.stream:
        async def audio_generator():
            for chunk_audio, idx, total in _generate_audio_chunks(request, voice_file):
                pcm_bytes = (chunk_audio * 32767).clip(-32768, 32767).astype(np.int16).tobytes()
                yield pcm_bytes

        return StreamingResponse(
            audio_generator(),
            media_type="audio/pcm",
            headers={
                "X-Sample-Rate": str(SAMPLE_RATE),
                "X-Format": "s16le",
                "X-Channels": "1",
            },
        )

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
        for chunk_audio, idx, total in _generate_audio_chunks(request, voice_file):
            pcm_bytes = (chunk_audio * 32767).clip(-32768, 32767).astype(np.int16).tobytes()
            print(f"[TTS Stream] Sending chunk {idx+1}/{total}: "
                  f"{len(pcm_bytes)} bytes PCM, "
                  f"duration={len(chunk_audio)/SAMPLE_RATE:.2f}s")
            yield pcm_bytes

    return StreamingResponse(
        audio_generator(),
        media_type="audio/pcm",
        headers={
            "X-Sample-Rate": str(SAMPLE_RATE),
            "X-Format": "s16le",
            "X-Channels": "1",
        },
    )


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