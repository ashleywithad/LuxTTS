# Fix Plan: CUDA Allocator Pre-Check + Auto-Restart + RMS Slider Fix

**Date:** 2026-05-07
**Issue:** 503 errors on every request after initial OOM corrupts allocator
**Status:** 🔧 IMPLEMENTING

## Problem Analysis

From the log:
```
10:19:53 - Request: yenna2, 1354 chars, max_chunk_chars=250, rms=0.1
10:19:54 - Chunking into 6 segments
10:19:54 - Generating chunk 1/6...
10:19:54 - 503 Service Unavailable (0.7s after chunk start)
```

**Key findings:**
1. **503 on chunk 1** — allocator was already corrupted from a previous request's OOM
2. **0.7s failure** — too fast for OOM on 238-char chunk, confirms pre-existing corruption
3. **`rms=0.1`** — client sending 100x louder than default (0.001)
4. **`max_chunk_chars=250`** — client overriding safe default (150)

## Three Fixes

### Fix 1: Pre-Flight Corruption Check

**Goal:** Detect corrupted allocator BEFORE generation attempt, return 503 immediately

**Implementation:**
```python
def _check_cuda_allocator_health() -> bool:
    """WHAT: Test if CUDA allocator is functional
    WHAT IT DOES: Attempts small memory allocation, returns False if corrupted
    WHY: Once corrupted (!handles_.at(i)), all requests fail - detect early"""
    if not torch.cuda.is_available():
        return True
    try:
        # Small test allocation to verify allocator is functional
        test_tensor = torch.empty(1024, device='cuda')
        del test_tensor
        torch.cuda.empty_cache()
        return True
    except RuntimeError as e:
        if "!handles_.at(i)" in str(e) or "INTERNAL ASSERT FAILED" in str(e):
            return False
        # Other errors might be transient
        return True
```

**Call locations:**
- Start of `_generate_audio()` (non-streaming)
- Start of `_generate_audio_chunks()` (streaming)

**Result:** Instead of failing during generation, we fail immediately with clear message: "CUDA allocator corrupted from previous OOM. Container restart required."

---

### Fix 2: Auto-Restart Mechanism

**Goal:** Automatically restart server when corruption detected (no manual intervention)

**Implementation:**
1. Add env var `AUTO_RESTART_ON_CORRUPTION=true` (default: true)
2. On corruption detection, call `os._exit(1)` to force immediate process termination
3. Docker's `restart: unless-stopped` policy will restart the container

**Why `os._exit(1)`:**
- Immediate termination, bypasses Python cleanup
- Docker sees non-zero exit, triggers restart
- Faster than graceful shutdown (model already broken)

```python
AUTO_RESTART = os.getenv("AUTO_RESTART_ON_CORRUPTION", "true").lower() in ("true", "1", "yes")

# In corruption detection handler:
if not _check_cuda_allocator_health():
    if AUTO_RESTART:
        print("[CUDA] Allocator corrupted - triggering auto-restart...")
        os._exit(1)  # Force immediate termination, Docker will restart
    raise HTTPException(status_code=503, detail="CUDA allocator corrupted...")
```

---

### Fix 3: Fix RMS Slider Max

**Goal:** Prevent client sending `rms=0.1` (100x louder than intended)

**Current state:**
- Web UI slider: `min=0, max=0.1, value=0.001`
- API default: `0.001`
- Problem: Slider max is 100x the default, user accidentally sends 0.1

**Fix:** Change slider max to 0.01 (10x default, reasonable range)

**File:** `static/index.html` line 540
```html
<!-- OLD -->
<input type="range" id="rmsSlider" min="0" max="0.1" step="0.001" value="0.001" ...>

<!-- NEW -->
<input type="range" id="rmsSlider" min="0" max="0.01" step="0.001" value="0.001" ...>
```

---

## Implementation Order

1. Add `_check_cuda_allocator_health()` function
2. Add `AUTO_RESTART_ON_CORRUPTION` env var
3. Call pre-flight check in `_generate_audio()` and `_generate_audio_chunks()`
4. Update RMS slider max in Web UI
5. Update docker-compose.yml with new env var
6. Update .env.docker.example with documentation

## Files to Modify

| File | Changes |
|------|---------|
| `api_server.py` | Add `_check_cuda_allocator_health()`, add AUTO_RESTART env var, pre-flight checks |
| `static/index.html` | RMS slider max: 0.1 → 0.01 |
| `docker-compose.yml` | Add AUTO_RESTART_ON_CORRUPTION env var |
| `.env.docker.example` | Document AUTO_RESTART_ON_CORRUPTION |

## Testing Plan

1. Restart container
2. Send request that causes OOM (large text, high max_chunk_chars)
3. Send second request immediately — should either:
   - Return 503 immediately with clear message (pre-flight detected)
   - Trigger auto-restart (if AUTO_RESTART=true)
4. After restart, subsequent requests work again

## Summary

The core issue is **corrupted allocator from previous OOM persists until manual restart**.

Fixes:
1. **Pre-flight check** — fail fast with clear message instead of failing mid-generation
2. **Auto-restart** — no manual intervention needed, container restarts automatically
3. **RMS slider fix** — prevent accidentally sending 100x louder audio