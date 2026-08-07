"""Speech-to-text using faster-whisper (CTranslate2 backend).

Uses the 'base' model with int8 quantization (~500-800MB VRAM) — light enough
to stay resident alongside Z-Image-Turbo without competing for VRAM.

faster-whisper requires FFmpeg installed on the system to decode non-WAV
audio formats (e.g. WebM/Opus from browser MediaRecorder).
"""

import time
from pathlib import Path
import psutil
import os

from faster_whisper import WhisperModel

_model: WhisperModel | None = None


def load_whisper_model(
    model_size: str = "base",
    device: str = "cpu",
    compute_type: str = "int8",
) -> WhisperModel:
    """Load the Whisper model into memory.

    Called once at server startup (in the FastAPI lifespan).  The model
    stays resident — at ~1GB (int8) it's small enough to coexist with
    Z-Image-Turbo's ~17GB footprint on a 24GB GPU.

    Parameters
    ----------
    model_size : str
        Whisper model variant.  'base' is the default — good accuracy
        for short spoken prompts without heavy RAM cost.
    device : str
        Hardcoded to 'cpu' as requested to save GPU VRAM.
    compute_type : str
        'int8' for minimal RAM (~1-1.5GB), 'float16' for slightly better
        accuracy.
    """
    global _model
    print(f"Loading Whisper '{model_size}' model (device={device}, compute_type={compute_type})...")
    _model = WhisperModel(model_size, device=device, compute_type=compute_type)
    
    process = psutil.Process(os.getpid())
    ram_mb = process.memory_info().rss / (1024 * 1024)
    print(f"✅ Whisper model loaded. Current process RAM usage: {ram_mb:.1f} MB")
    
    return _model


def transcribe_audio(audio_path: Path) -> str:
    """Transcribe an audio file to text.

    Parameters
    ----------
    audio_path : Path
        Path to the audio file.  Any format FFmpeg can decode is
        accepted (WAV, MP3, WebM, OGG, etc.).

    Returns
    -------
    str
        The transcribed text, with segments joined by spaces.

    Raises
    ------
    RuntimeError
        If the Whisper model hasn't been loaded yet.
    """
    if _model is None:
        raise RuntimeError("Whisper model not loaded. Call load_whisper_model() first.")

    start_time = time.time()
    segments, info = _model.transcribe(str(audio_path), beam_size=1, language="en", vad_filter=True, vad_parameters=dict(min_silence_duration_ms=500))
    transcript = " ".join(seg.text.strip() for seg in segments).strip()
    end_time = time.time()
    
    processing_time = end_time - start_time
    
    process = psutil.Process(os.getpid())
    ram_mb = process.memory_info().rss / (1024 * 1024)

    print(f"[STT] Transcribed ({info.language}, audio length: {info.duration:.1f}s): {transcript!r}")
    print(f"[STT] Processing Time: {processing_time:.2f} seconds")
    print(f"[STT] Current Process RAM: {ram_mb:.1f} MB")
    
    return transcript
