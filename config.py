"""Ghost Frame — Unified Configuration

Single source of truth for all ports, URLs, model paths, and runtime settings
across the entire Ghost Frame project (cloud backend, local engine, frontend).
"""
import os
from pathlib import Path

# ── Project Paths ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
BACKEND_DIR = PROJECT_ROOT / "backend"
LOCAL_ENGINE_DIR = PROJECT_ROOT / "local_engine"
FRONTEND_DIR = PROJECT_ROOT / "frontend"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Cloud Backend (Port 8000) ──────────────────────────────────────────────
CLOUD_HOST = os.environ.get("GHOST_CLOUD_HOST", "0.0.0.0")
CLOUD_PORT = int(os.environ.get("GHOST_CLOUD_PORT", "8000"))
CLOUD_API_URL = os.environ.get("GHOST_CLOUD_URL", f"http://localhost:{CLOUD_PORT}")

# ── Local Engine (Port 8001) ───────────────────────────────────────────────
LOCAL_HOST = os.environ.get("GHOST_LOCAL_HOST", "0.0.0.0")
LOCAL_PORT = int(os.environ.get("GHOST_LOCAL_PORT", "8001"))
LOCAL_API_URL = os.environ.get("GHOST_LOCAL_URL", f"http://localhost:{LOCAL_PORT}")
LOCAL_WS_URL = os.environ.get("GHOST_LOCAL_WS", f"ws://localhost:{LOCAL_PORT}")

# ── Webcam Settings ────────────────────────────────────────────────────────
CAMERA_INDEX = int(os.environ.get("GHOST_CAMERA", "0"))
CAMERA_WIDTH = int(os.environ.get("GHOST_CAM_W", "640"))
CAMERA_HEIGHT = int(os.environ.get("GHOST_CAM_H", "360"))
MIRROR_FEED = os.environ.get("GHOST_MIRROR", "true").lower() == "true"

# ── Model Paths ────────────────────────────────────────────────────────────
# Local Engine models
RVM_MODEL_PATH = os.environ.get(
    "GHOST_RVM_PATH",
    str(LOCAL_ENGINE_DIR / "models" / "rvm_resnet50.pth")
)
HAND_LANDMARKER_PATH = os.environ.get(
    "GHOST_HAND_MODEL",
    str(LOCAL_ENGINE_DIR / "models" / "hand_landmarker.task")
)
POSE_LANDMARKER_PATH = os.environ.get(
    "GHOST_POSE_MODEL",
    str(LOCAL_ENGINE_DIR / "models" / "pose_landmarker_full.task")
)

# Cloud Backend models
GGUF_TRANSFORMER_PATH = os.environ.get(
    "ZIMAGE_GGUF_PATH",
    str(BACKEND_DIR / "models" / "gguf" / "z_image_turbo-Q4_K_M.gguf")
)
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:4b")

# Whisper STT
WHISPER_MODEL_SIZE = os.environ.get("GHOST_WHISPER_MODEL", "base")
WHISPER_DEVICE = os.environ.get("GHOST_WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE_TYPE = os.environ.get("GHOST_WHISPER_COMPUTE", "int8")

# ── Heartbeat Settings ─────────────────────────────────────────────────────
HEARTBEAT_INTERVAL_SECONDS = 5
