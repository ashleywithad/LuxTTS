# LuxTTS - Project Status & Agent Notes

## Project Overview
OpenAI-compatible TTS API server built on LuxTTS (ZipVoice-based voice cloning).
Running on **NVIDIA RTX 3050 Ti Laptop 4GB VRAM**.
Fork: `https://github.com/ashleywithad/LuxTTS`

## Key Architecture Facts
- `LuxTTS.__init__` only accepts `(model_path, device, threads)` — **no dtype parameter**
- `encode_prompt` accepts `(prompt_audio, duration=5, rms=0.001)` — **no transcription_text**
- `generate_speech` default `t_shift=0.5` (not 0.9), `rms` default is 0.001 (not 0.01)
- Model uses `torch.inference_mode()` internally in `process_audio` and `generate`
- Whisper-base loaded on GPU uses ~140MB VRAM
- Model at float32 uses ~1GB VRAM
- CUDA context overhead ~300-500MB
- Net free for inference: ~2.2GB on 4GB card
- `zipvoice` pip package is empty/placeholder — install from `git+https://github.com/ysharma3501/LuxTTS.git`
- Soundfile libsndfile 1.2.2 — supports native MP3 write

## Completed Work

### 1. Fixed Broken Virtual Environment
- Deleted corrupted `.venv` (missing Scripts/ directory)
- Recreated with `uv venv --python 3.10`
- Installed deps with `uv pip install -r requirements.txt`
- Installed zipvoice from LuxTTS GitHub repo

### 2. RTX 3050 4GB Optimizations
- `PYTORCH_CUDA_ALLOC_CONF` env var for memory fragmentation prevention
- TF32 enabled for ~7x matmul speedup on Ampere (`ENABLE_TF32=true`)
- cuDNN benchmark enabled, deterministic disabled
- cuFFT plan cache: 4096 → 64
- `torch.cuda.set_per_process_memory_fraction(0.85)` for headroom
- `torch.inference_mode()` wrapping generation
- GPU memory cleanup (`gc.collect()` + `torch.cuda.empty_cache()`) between requests
- Manual FP16 support via `_apply_half_precision()` (`.half()` on model/vocos)
  - Controlled by `ENABLE_FP16` env var (default: false)
  - Note: FP16 may be **slower** on consumer Ampere GPUs due to complex FFT fallback
- Removed broken `DTYPE` env var (was silently ignored by LuxTTS library)
- Default `response_format`: `mp3` → `wav` (eliminates ~50-300ms encode overhead)
- Default `ref_duration`: 5 → 3 (faster prompt encoding)
- Default `rms`: 0.01 → 0.001, `t_shift`: 0.9 → 0.5 (match library defaults)
- MP3 encoding: soundfile native first (fast), pydub fallback (slow)

### 3. Bug Fixes
- Removed `transcription_text` param from `encode_prompt` calls (not supported)
- Removed `DTYPE` logic (LuxTTS library has no dtype param)
- Added `guidance_scale` to API (was in library but missing from API)
- Renamed `voice_samples/zipvoice.wav` → `default.wav` (API default voice works OOTB)
- Restored `static/index.html` from git (was deleted, caused 500 on `/`)
- Updated `start_server_fp16.bat` to use `ENABLE_FP16` instead of broken `DTYPE`

### 4. Text Chunking for OOM Prevention
- `_chunk_text()`: regex-based sentence splitter, max chars per chunk (default 200 ≈ 80 tokens)
- `_crossfade_numpy()`: cross-fade concatenation with 50ms overlap at 48kHz
- `_generate_single()`: generates audio for a single text chunk
- Chunked generation: encode prompt once, generate each chunk, cross-fade results
- GPU memory cleanup between chunks
- `max_chunk_chars` API parameter (0=disable, default=200)
- Progressive streaming: chunks generated incrementally
- CUDA OOM error returns 507 with actionable message
- Max Chunk slider added to Web UI advanced settings

## Environment Setup
```bash
uv venv --python 3.10
uv pip install -r requirements.txt
uv pip install "zipvoice @ git+https://github.com/ysharma3501/LuxTTS.git"
```

## Configuration (.env)
```
MODEL_PATH=YatharthS/LuxTTS
DEVICE=cuda
ENABLE_TF32=true
ENABLE_FP16=false
HOST=0.0.0.0
PORT=8020
DEFAULT_RMS=0.001
DEFAULT_T_SHIFT=0.5
DEFAULT_NUM_STEPS=4
DEFAULT_SPEED=1.0
DEFAULT_RETURN_SMOOTH=false
DEFAULT_REF_DURATION=3
```

## Tested Configurations
**Working Docker settings** (with `shm_size: '2gb'`):
```json
{
  "speed": 0.5,
  "num_steps": 10,
  "ref_duration": 20,
  "max_chunk_chars": 500
}
```
- Same settings that worked on host now work in Docker after shared memory fix
- Docker overhead requires proper `/dev/shm` allocation (2GB minimum)
- Without `shm_size: '2gb'`, even conservative settings fail with allocator corruption

### 5. Progressive Streaming (Live Playback)
- `/v1/audio/speech/stream` now sends raw PCM (int16 LE, 48kHz, mono) per text chunk as it generates
- `stream=true` on `/v1/audio/speech` also uses progressive PCM streaming
- Server yields each chunk's PCM bytes immediately — no waiting for all chunks
- Custom headers: `X-Sample-Rate`, `X-Format`, `X-Channels` for client auto-detection
- No cross-fade in streaming mode (lowest latency); non-streaming retains cross-fade for quality
- OOM errors in streaming handled per-chunk (HTTPException raised inside generator)
- Web UI: "Live Streaming" toggle (default ON) using Web Audio API + ReadableStream
  - `pcmToAudioBuffer()`: int16 → float32 conversion for `AudioContext.createBuffer()`
  - `generateSpeechStream()`: fetch with `response.body.getReader()`, schedule `AudioBufferSourceNode` per chunk
  - Progressive playback: audio starts playing first chunk while remaining chunks generate
  - Progress bar + chunk count indicator during streaming
  - Stop button to abort streaming mid-generation
  - Download button: uses non-streaming endpoint (cross-faded WAV/MP3) for quality
- Non-streaming mode (toggle OFF): original blob-based playback with `<audio>` element, cross-fade quality

### 6. Docker Deployment
- **Dockerfile**: PyTorch CUDA 12.1 runtime base (~4GB), install git/curl for pip deps and healthcheck
- **docker-compose.yml**: GPU support via `deploy.resources.reservations.devices`, bind mounts, health check
- **CRITICAL: Shared Memory Configuration**
  - Added `shm_size: '2gb'` to docker-compose.yml
  - WHY: PyTorch/CUDA allocator requires >64MB shared memory for GPU inference
  - Docker default `/dev/shm` = 64MB → causes allocator corruption (`!handles_.at(i)` internal assert)
  - Host OS `/dev/shm` = half of system RAM (works fine on host, fails in Docker)
  - Symptom: OOM → allocator corruption → all subsequent requests fail with "CUDA error: unknown error"
  - Fix: Increase shared memory to 2GB (same settings that work on host now work in Docker)
- **CRITICAL: PyTorch Allocator Configuration**
  - Added `PYTORCH_CUDA_ALLOC_CONF` to docker-compose.yml environment variables
  - Config: `max_split_size_mb:128,garbage_collection_threshold:0.6,expandable_segments:True`
  - WHY: Prevents memory fragmentation during diffusion inference on low-VRAM GPUs
  - `expandable_segments:True` — Supported in Linux container (NOT supported on Windows host)
  - Must be set via environment BEFORE Python process starts (not via `os.environ.setdefault()` in code)
  - Note: `expandable_segments:True` listed as "not supported on Windows" in Known Issues but IS supported in Linux container
- **Bind mounts**: 
  - `voice_samples/` → `/app/voice_samples` (rw) — add voices without rebuild
  - `.env` → `/app/.env` (ro) — config without rebuild
  - Named volume `hf_cache` → HuggingFace cache (prevents model re-download ~1-2GB)
- **Port**: 8020 (8000 is reserved for Budgie Dialer)
- **Health check**: 120s start_period for model download, 30s interval after startup
- **Restart policy**: `unless-stopped` (survives host reboot)
- `.dockerignore`: Excludes `.venv`, `.git`, `__pycache__`, logs, voice files (mounted at runtime)
- `.env.docker.example`: Template with Docker-specific defaults
- `.gitignore`: Updated with `!.env.docker.example` to track example in git
- **Running**: `docker compose up -d` — model loads in ~40s, GPU memory ~0.79GB
- **Access**: Web UI at http://localhost:8020, API docs at http://localhost:8020/docs

### 7. CUDA Memory Management Improvements (2026-05-06)
- **Problem**: OOM errors caused CUDA allocator corruption, cascading 500 errors for all subsequent requests
- **Root cause analysis**:
  - RuntimeError "CUDA out of memory" thrown (NOT `torch.cuda.OutOfMemoryError`)
  - Uncatched RuntimeError → allocator internal state corrupted (`!handles_.at(i)` assertion failure)
  - Once corrupted, only container restart could fix — no programmatic recovery possible
- **Error handling fixes**:
  - Catch both `torch.cuda.OutOfMemoryError` AND `RuntimeError` in chunk generation
  - Detect allocator corruption signature: `!handles_.at(i)` or `INTERNAL ASSERT FAILED`
  - Return HTTP 503 (Service Unavailable) with "Container restart recommended" message
  - Return HTTP 507 (Insufficient Storage) for normal CUDA OOM with actionable advice
- **Memory cleanup improvements**:
  - Added `torch.cuda.synchronize()` BEFORE each chunk (releases previous chunk memory)
  - Explicit `del audio` after `.numpy()` conversion (breaks reference chain)
  - Pre-chunk cleanup prevents memory accumulation across chunks
- **Default chunk size reduction**: 200 → 150 chars (extra safety for Docker overhead)
- **Added helper functions**:
  - `_is_cuda_error()` — Detects CUDA-related RuntimeErrors
  - `_reset_cuda_allocator()` — Attempts recovery (limited success, mainly for error detection)
- **Result**: Proper HTTP status codes, allocator remains functional after errors, actionable user feedback

### 8. CUDA Allocator Pre-Flight Check + Auto-Restart (2026-05-07)
- **Problem**: Once allocator corrupted from OOM, ALL subsequent requests failed with 503 until manual restart
- **Root cause**: Previous fixes caught corruption in-flight, but allocator was already dead before request started
- **New fixes**:
  - `_check_cuda_allocator_health()`: Pre-flight check that attempts small allocation before generation
    - Tests allocator with 1024-float tensor allocation
    - Returns False if corruption signature detected (`!handles_.at(i)` or `INTERNAL ASSERT FAILED`)
    - Called at start of `_generate_audio()` and `_generate_audio_chunks()`
  - `_handle_allocator_corruption()`: Unified handler for corruption detection
    - If `AUTO_RESTART_ON_CORRUPTION=true`: calls `os._exit(1)` for immediate process termination
    - Docker's `restart: unless-stopped` policy brings container back automatically
    - If disabled: raises HTTP 503 with actionable message
  - `AUTO_RESTART_ON_CORRUPTION` env var (default: true)
    - Set in docker-compose.yml
    - Documented in .env.docker.example
- **Result**: 
  - Pre-flight check fails fast with clear message before generation starts
  - Auto-restart eliminates need for manual intervention
  - Service self-heals after OOM corruption within ~40 seconds (model reload time)

## Known Issues / Future Work
- `expandable_segments:True` in PYTORCH_CUDA_ALLOC_CONF not supported on Windows host, but IS supported in Docker Linux container
- Streaming has no cross-fade between chunks (minor quality tradeoff for lowest latency)
  - Sentence boundary pauses in TTS output make the stitch mostly seamless
  - Could add client-side cross-fade using Web Audio API gain nodes (fade-out/fade-in)
- The `zipvoice/` local directory files show as deleted in git (replaced by pip package)
- No `default.wav` in git (ignored by .gitignore) — users need to add their own voice sample
- Docker container shows ffmpeg warning (expected) — MP3 uses soundfile native (fast) or pydub fallback
- Port 8000 reserved for Budgie Dialer — LuxTTS uses 8020 by default
- `_reset_cuda_allocator()` has limited success — once corrupted, container restart is required
- Very long texts (>1500 chars) may still OOM even with chunking on 4GB VRAM — reduce `max_chunk_chars` or enable FP16