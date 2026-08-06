# Ghost Stream 🎮👻

**Real-time AI-powered prop and background generation for live streamers, with body-tracking and OBS integration.**

Ghost Stream lets streamers voice or type a prompt, generates an AI image in seconds, and instantly equips it as a tracked body prop (sword, shield, hat) or background in their OBS scene — all in real-time.

---

## Architecture

Ghost Stream runs as a **two-machine system**:

```
┌─────────────────────────────────────────────────┐
│               Streamer's Local PC               │
│                                                 │
│  ┌──────────────┐    ┌────────────────────────┐ │
│  │  Ghost Engine │───▶│ OBS Studio             │ │
│  │  (Port 8001)  │    │  ├─ Window Capture     │ │
│  │  ├─ Webcam    │    │  └─ Browser Source     │ │
│  │  ├─ MediaPipe │    │     (overlay.html)     │ │
│  │  ├─ RVM       │    └────────────────────────┘ │
│  │  └─ Props     │                               │
│  └──────────────┘                               │
└─────────────────────────────────────────────────┘
         ▲                    │
         │ Event-driven       │ Voice/Text
         │ hot-swap           │ prompts
         ▼                    ▼
┌─────────────────────────────────────────────────┐
│             Cloud GPU Server                     │
│                                                 │
│  ┌──────────────────────────────────────────┐   │
│  │  FastAPI Backend (Port 8000)              │   │
│  │  ├─ Whisper STT (Speech-to-Text)          │   │
│  │  ├─ Ollama Agent (Prompt Rewriting)       │   │
│  │  ├─ Z-Image-Turbo (Image Generation)      │   │
│  │  └─ rembg (Prop Background Removal)       │   │
│  └──────────────────────────────────────────┘   │
│                                                 │
│  ┌──────────────────────────────────────────┐   │
│  │  Web UI (Browser Dashboard)               │   │
│  │  ├─ Voice prompt (push-to-talk)           │   │
│  │  ├─ Text prompt (textarea + generate)     │   │
│  │  ├─ Custom prop upload                    │   │
│  │  └─ Local Engine health indicator         │   │
│  └──────────────────────────────────────────┘   │
└─────────────────────────────────────────────────┘
```

---

## Project Structure

```
ghost-stream/
├── config.py                  # Unified configuration (ports, URLs, models)
├── backend/                   # Cloud GPU Server (Port 8000)
│   ├── main.py                # FastAPI: voice/text generation, uploads, WebSocket
│   ├── engine/                # Z-Image-Turbo image generation pipeline
│   ├── agent/                 # Ollama agent (prompt rewriting + safety)
│   ├── stt/                   # Whisper speech-to-text
│   ├── requirements.txt
│   └── setup.sh
├── local_engine/              # Local PC Vision Engine (Port 8001)
│   ├── ghost_engine.py        # Camera loop + Local API + WebSocket tracking
│   ├── tracker.py             # MediaPipe pose/hand tracking
│   ├── prop_config.py         # Prop attachment taxonomy
│   ├── prop_manager.py        # Prop image preprocessing
│   ├── live_equip.py          # Real-time prop rendering
│   ├── background_remover.py  # RVM video matting
│   ├── download_models.py     # Model weight downloader
│   └── requirements.txt
├── frontend/                  # Web UI + OBS Overlay
│   ├── index.html             # Dashboard (voice + text + props)
│   ├── script.js              # Dashboard logic + health monitoring
│   ├── style.css              # Dark theme styling
│   ├── overlay.html           # OBS Browser Source (transparent)
│   └── overlay.js             # Real-time prop tracking overlay
└── outputs/                   # Generated images
```

---

## Quick Start

### 1. Cloud Backend (GPU Server)

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Run full setup (installs deps + downloads models)
bash setup.sh

# Start the server
python main.py
```

The backend will be available at `http://localhost:8000`.
Open `http://localhost:8000/app/` for the Web UI.

### 2. Local Vision Engine (Your PC)

```bash
cd local_engine

# Create virtual environment  
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Download model weights (~125 MB total)
python download_models.py

# Start the engine
python ghost_engine.py
```

The local engine starts on `http://localhost:8001` with WebSocket tracking at `ws://localhost:8001/ws/anchor`.

### 3. OBS Studio Setup

1. **Video Feed**: Add a **Window Capture** source → select the "Ghost Stream" OpenCV window.
2. **Prop Overlay**: Add a **Browser Source** → URL: `http://localhost:8000/app/overlay.html` → Set width/height to match your canvas → Check "Shutdown source when not visible".

---

## Configuration

All settings are in `config.py` and can be overridden via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `GHOST_CLOUD_PORT` | `8000` | Cloud backend port |
| `GHOST_LOCAL_PORT` | `8001` | Local engine port |
| `GHOST_CAMERA` | `0` | Webcam device index |
| `GHOST_CAM_W` | `1280` | Camera width |
| `GHOST_CAM_H` | `720` | Camera height |
| `GHOST_MIRROR` | `true` | Selfie mirror mode |
| `OLLAMA_MODEL` | `qwen3:4b` | Ollama agent model |
| `GHOST_WHISPER_MODEL` | `base` | Whisper STT model size |

---

## Requirements

### Cloud Server
- **GPU**: NVIDIA GPU with ≥16GB VRAM (24GB recommended)
- **CUDA**: 12.1+
- **Python**: 3.10+
- **Ollama**: For the agent model

### Local PC
- **Webcam**: Any USB/built-in camera
- **Python**: 3.10+
- **OBS Studio**: 28+ (for Window Capture + Browser Source)

---

## License

This project is proprietary. All rights reserved.
