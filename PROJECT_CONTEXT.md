# Ghost Stream — Project Context (for new chat reference)

## Project Location
- **Merged project**: `c:\Users\Mayank Moyal\Desktop\ghost-stream\`
- **Teammate's cloud GPU**: `/home/gpu26/GhostFrame/`
- **Teammate's local PC**: `D:\AIML\ML Projects\GhostFrame\`

## What Ghost Stream Is
A real-time AI-powered streaming tool that:
1. **Generates images** from voice/text prompts (Z-Image-Turbo on cloud GPU)
2. **Tracks the streamer's body** with MediaPipe (local PC)
3. **Attaches generated props** (swords, hats, shields) to tracked body parts
4. **Composites everything** into a live OBS stream with depth-aware rendering

## Architecture (Two-Machine Split)
- **Cloud GPU Server (Port 8000)**: FastAPI backend with Z-Image-Turbo image gen, Whisper STT, Ollama agent, rembg
- **Local PC (Port 8001)**: Ghost Engine with webcam, MediaPipe tracking, RVM video matting, prop rendering, OBS Window Capture

## Key Architectural Decisions
| Decision | Choice |
|----------|--------|
| WebSocket Tracking | Local Engine (Port 8001) hosts WS directly for zero latency |
| Prop Hot-Swap | Event-driven: Cloud pushes directly to Local Engine |
| Background Removal | RVM locally (video matting) + rembg on cloud (prop images) |
| OBS Feed | Window Capture (cv2.imshow) for lowest latency |
| Input Methods | Both voice AND text prompts |
| Config | Unified config.py with env var overrides |

## File Structure
```
ghost-stream/
├── config.py                  # Unified config (ports, URLs, models)
├── backend/                   # Cloud GPU Server (Port 8000)
│   ├── main.py                # FastAPI: /generate, /generate-voice, /upload-prop, /clear-props, /ws/anchor
│   ├── engine/zimage_turbo.py # Z-Image-Turbo pipeline (GGUF Q5_K_M)
│   ├── agent/                 # Ollama agent (prompt rewriting + safety)
│   └── stt/transcriber.py     # Whisper STT
├── local_engine/              # Local PC (Port 8001)
│   ├── ghost_engine.py        # Camera loop + FastAPI + WebSocket (threaded)
│   ├── tracker.py             # MediaPipe pose/hand tracking
│   ├── prop_config.py         # Prop attachment taxonomy
│   ├── prop_manager.py        # Prop preprocessing (dynamic pivots)
│   ├── live_equip.py          # Depth-aware prop rendering
│   ├── background_remover.py  # RVM video matting
│   └── download_models.py     # Model weight downloader
├── frontend/
│   ├── index.html             # Dashboard (voice + text + health indicator)
│   ├── script.js              # Dashboard logic
│   ├── style.css              # Dark theme
│   ├── overlay.html           # OBS Browser Source
│   └── overlay.js             # Dual WebSocket overlay (local tracking + cloud events)
└── outputs/                   # Generated images
```

## Source Folders (Original, Pre-Merge)
- vision/ — Local vision engine (tracker, props, RVM)
- newver/ — Frontend + STT
- q8/ — Cloud backend (Z-Image-Turbo + Ollama agent)

## Fixes Already Applied
1. prop_manager.py: Made rembg import optional (try/except) — cloud handles bg removal
2. zimage_turbo.py: Fixed relative path to absolute path for GGUF loading. Changed Q8_0 to Q5_K_M
3. ghost_engine.py: Fixed _build_tracking_payload() — hand.palm is a tuple (x,y) not palm_x/palm_y, hand.handedness not hand.side, pose.head_angle not head_tilt_deg

## Key Data Classes (from tracker.py)
```python
@dataclass
class HandResult:
    palm: Tuple[float, float]          # (x, y) normalized 0-1
    index_mcp: Tuple[float, float]
    pinky_mcp: Tuple[float, float]
    wrist: Tuple[float, float]
    middle_mcp: Tuple[float, float]
    hand_angle: float                  # smoothed radians
    handedness: str                    # "Left" or "Right"
    confidence: float

@dataclass
class PoseResult:
    nose, left_eye, right_eye, left_ear, right_ear: Optional[Tuple]
    left_shoulder, right_shoulder: Optional[Tuple]
    left_elbow, right_elbow: Optional[Tuple]
    left_wrist_pose, right_wrist_pose: Optional[Tuple]
    head_center: Optional[Tuple[float, float]]
    head_angle: Optional[float]        # degrees
    neck_point: Optional[Tuple[float, float]]
    shoulder_width: Optional[float]    # normalised
    shoulder_angle: Optional[float]    # degrees

@dataclass
class TrackResult:
    hands: List[HandResult]
    pose: Optional[PoseResult]
```

## How to Run
- Cloud: cd backend && python main.py (port 8000)
- Local: cd local_engine && python ghost_engine.py (port 8001)
- Web UI: http://localhost:8000/app/
- OBS: Window Capture + Browser Source overlay.html

## Conversation ID (for @mention)
ac4cb6f5-c632-406e-91b1-33e6d3bb4320
