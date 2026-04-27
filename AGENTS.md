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
PORT=8000
DEFAULT_RMS=0.001
DEFAULT_T_SHIFT=0.5
DEFAULT_NUM_STEPS=4
DEFAULT_SPEED=1.0
DEFAULT_RETURN_SMOOTH=false
DEFAULT_REF_DURATION=3
```

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

## Known Issues / Future Work
- `expandable_segments:True` in PYTORCH_CUDA_ALLOC_CONF not supported on Windows
- Streaming has no cross-fade between chunks (minor quality tradeoff for lowest latency)
  - Sentence boundary pauses in TTS output make the stitch mostly seamless
  - Could add client-side cross-fade using Web Audio API gain nodes (fade-out/fade-in)
- The `zipvoice/` local directory files show as deleted in git (replaced by pip package)
- No `default.wav` in git (ignored by .gitignore) — users need to add their own voice sample