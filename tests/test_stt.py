"""
GhostFrame STT Test Script
===========================
Tests the speech-to-text pipeline in isolation.

Usage:
    python tests/test_stt.py              → Records from your mic for 5 seconds, transcribes it
    python tests/test_stt.py myfile.wav   → Transcribes an existing audio file

Prerequisites:
    pip install faster-whisper sounddevice soundfile
    FFmpeg must be installed (winget install ffmpeg)
"""

import sys
import os
import tempfile
from pathlib import Path

# ── Step 0: Check if faster-whisper is installed ──────────────────────
try:
    from faster_whisper import WhisperModel
    print("✅ faster-whisper is installed")
except ImportError:
    print("❌ faster-whisper is NOT installed.")
    print("   Run: pip install faster-whisper")
    sys.exit(1)

# ── Step 1: Get audio (record or use provided file) ──────────────────
audio_path = None

if len(sys.argv) > 1:
    # User provided an audio file
    audio_path = Path(sys.argv[1])
    if not audio_path.exists():
        print(f"❌ File not found: {audio_path}")
        sys.exit(1)
    print(f"✅ Using provided audio file: {audio_path}")

else:
    # Record from microphone
    try:
        import sounddevice as sd
        import soundfile as sf
        print("✅ sounddevice + soundfile are installed")
    except ImportError:
        print("❌ sounddevice or soundfile is NOT installed.")
        print("   Run: pip install sounddevice soundfile")
        print("   Or provide an audio file: python tests/test_stt.py myfile.wav")
        sys.exit(1)

    DURATION = 5  # seconds
    SAMPLE_RATE = 16000

    print(f"\n🎤 Recording for {DURATION} seconds... Speak your prompt NOW!")
    print("=" * 50)

    recording = sd.rec(
        int(DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
    )
    sd.wait()  # Block until recording is done

    print("=" * 50)
    print("✅ Recording complete!")

    # Save to a temp WAV file
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp.close()  # Close the file handle on Windows so soundfile can use it
    sf.write(tmp.name, recording, SAMPLE_RATE)
    audio_path = Path(tmp.name)
    print(f"   Saved to: {audio_path}")

# ── Step 2: Load Whisper model ────────────────────────────────────────
print(f"\n⏳ Loading Whisper 'base' model on CPU (first time downloads ~500MB)...")
import psutil
import time

try:
    model = WhisperModel("base", device="cpu", compute_type="int8")
    print("✅ Whisper model loaded on CPU (int8)")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    sys.exit(1)

# ── Step 3: Transcribe ────────────────────────────────────────────────
print(f"\n⏳ Transcribing audio...")

start_time = time.time()
segments, info = model.transcribe(str(audio_path), beam_size=1, language="en", vad_filter=True, vad_parameters=dict(min_silence_duration_ms=500))
transcript = " ".join(seg.text.strip() for seg in segments).strip()
end_time = time.time()

processing_time = end_time - start_time
process = psutil.Process(os.getpid())
ram_mb = process.memory_info().rss / (1024 * 1024)

print(f"\n{'=' * 50}")
print(f"📝 TRANSCRIPT: {transcript}")
print(f"{'=' * 50}")
print(f"   Language: {info.language} (confidence: {info.language_probability:.1%})")
print(f"   Audio Duration: {info.duration:.1f}s")
print(f"   ⏱️ Processing Time: {processing_time:.2f} seconds")
print(f"   💻 RAM Usage: {ram_mb:.1f} MB (Expected for 'base' int8: ~500-800 MB)")

if transcript:
    print(f"\n✅ STT is working! Your prompt was: \"{transcript}\"")
else:
    print(f"\n⚠️  No speech detected. Try speaking louder or check your microphone.")

# Cleanup temp file if we recorded
if len(sys.argv) <= 1 and audio_path:
    audio_path.unlink(missing_ok=True)
