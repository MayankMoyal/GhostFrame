# Ghost Frame — Project Context (for new chat reference)

## What Ghost Frame Is
A real-time AI prop and background system for livestreamers. The streamer speaks or types a prompt → AI generates an image → it's instantly classified as a prop or background → tracked and composited onto the streamer's body in OBS.

## Architecture (Two-Machine Split)

### Cloud GPU (Lightning AI T4, 16GB VRAM)
- **Backend**: `Ghost Frame/backend/main.py` (FastAPI, port 8000)
- **Image Gen**: Z-Image-Turbo GGUF Q5_K_M (~4s generation)
- **Agent**: Command-R 7B via Ollama (classifies prompts, rewrites for image gen)
- **STT**: faster-whisper (voice → text)
- **Lightning URL**: Dynamically assigned (e.g., `https://8000-xxxxx.lightning.ai`)

### Local Laptop (RTX 5050, 8GB VRAM)
- **Local Engine**: `Ghost Frame/local_engine/ghost_engine.py` (FastAPI, port 8001)
- **RVM**: Real-time background removal at 30 FPS (640x360, 16:9)
- **MediaPipe**: Pose + hand tracking
- **Props**: Depth-aware compositing (swords in front, capes behind via z_index)
- **OBS**: Window Capture of "Ghost Frame" OpenCV window

### Standalone Vision (for testing/demo)
- **Location**: `C:\Users\Mayank Moyal\Desktop\vision\ghost_engine.py`
- **Run**: `python ghost_engine.py --bg dungeon_bg.jpg --props sword.png`
- **No server, no API** — pure CLI for testing the vision pipeline

## Communication Flow
```
Browser Dashboard → Lightning AI Backend (generates image)
                  → Browser Relay (pushToLocalEngine in script.js)
                  → Local Engine localhost:8001 (composites in OBS)
```

## Key Files

### Backend (Cloud)
| File | Purpose |
|---|---|
| `backend/main.py` | FastAPI server, /generate, /generate-voice endpoints |
| `backend/agent/client.py` | **Active: Command-R 7B** agent (classifies + rewrites) |
| `backend/agent/client_qwen_3.py` | Archived: Qwen 3 4B agent (rollback) |
| `backend/agent/router.py` | Normalizes agent output, validates anchor_type |
| `backend/agent/safety.py` | Safety gate (is_safe check) |
| `backend/engine/zimage_turbo.py` | Z-Image-Turbo GGUF pipeline |
| `backend/stt/transcriber.py` | faster-whisper STT |
| `backend/setup.sh` | **Complete one-stop setup** (installs everything) |

### Local Engine
| File | Purpose |
|---|---|
| `local_engine/ghost_engine.py` | Camera loop + FastAPI server + WebSocket |
| `local_engine/background_remover.py` | RVM ResNet50 with edge denoising |
| `local_engine/tracker.py` | MediaPipe pose + hand with One Euro Filters |
| `local_engine/live_equip.py` | Prop rendering with temporal persistence |
| `local_engine/prop_config.py` | Prop categories + z_index depth ordering |
| `local_engine/prop_manager.py` | Image processing + category detection |

### Frontend
| File | Purpose |
|---|---|
| `frontend/index.html` | Dashboard UI |
| `frontend/script.js` | Dashboard controller + browser relay to local engine |
| `frontend/overlay.js` | OBS overlay WebSocket tracking |

### Config
| File | Purpose |
|---|---|
| `config.py` | All ports, URLs, camera settings (640x360) |

## Agent Output Schema (Command-R 7B)
```json
{
  "rewritten_prompt": "Enhanced prompt for image generation",
  "is_safe": true,
  "safety_reason": "",
  "style": "fantasy",
  "type": "prop",
  "anchor_type": "hand_held",
  "prop_category": "hand_held"
}
```

### Valid anchor_type / prop_category values:
- `background` — full scene replacement
- `hand_held` — sword, wand, torch, axe, staff
- `shield` — shield, buckler
- `head_wear` — crown, helmet, hat, tiara
- `neck_wear` — necklace, pendant, scarf
- `wrist_wear` — bracelet, gauntlet
- `ear_wear` — earring, ear cuff
- `face_wear` — mask, glasses, goggles
- `body_wear` — cape, cloak, armor, wings (z_index = -1, renders BEHIND user)

## Setup Commands
### Lightning AI (Cloud Backend):
```bash
cd backend && bash setup.sh   # installs everything
python main.py                # starts the server
```

### Laptop (Local Engine):
```powershell
cd "C:\Users\Mayank Moyal\Desktop\Ghost Frame\local_engine"
python ghost_engine.py
```

### Laptop (Vision standalone test):
```powershell
cd "C:\Users\Mayank Moyal\Desktop\vision"
python ghost_engine.py --bg dungeon_bg.jpg --props sword.png
```
