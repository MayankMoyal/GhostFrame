<div align="center">

# Ghost Frame 👻🎮

**Speak it. See it. Stream it.**

*Real-time AI-powered prop and background generation for livestreamers — driven by voice, tracked to your body, rendered on your stream in seconds.*

</div>

---

Ghost Frame is a real-time AI-powered prop and background generation system for livestreamers. The streamer speaks or types a natural language prompt (e.g. *"give me a flaming sword"*), and within seconds, an AI-generated image of that prop appears on their live stream — tracked to their body in real-time using computer vision. Backgrounds can also be swapped instantly by voice command.

No pre-made assets. No graphic design skills. Just your imagination, your voice, and your stream.

---

## 🔴 The Problem

Livestreaming visuals are stuck in the past:

- **Static overlays, static world** — Current tools like OBS and Streamlabs only offer pre-made, manually-placed overlays. There's no way to conjure something new on the fly.
- **No dynamic generation** — Streamers have zero ability to create visual content in real-time during a live broadcast. Every asset must be prepared before going live.
- **AI exists, but it's disconnected** — Tools like Midjourney and DALL·E can generate incredible images, but they're too slow and completely disconnected from streaming software.
- **Audiences crave magic** — There's a growing demand for interactive, audience-engaging content that feels spontaneous and alive — not canned and rehearsed.

---

## 💡 Our Solution

Ghost Frame bridges the gap between generative AI and live content creation with a seamless, end-to-end pipeline:

- **Voice-to-Visual Pipeline** — Speak a prompt → AI generates an image → it appears on your stream. The entire loop completes in seconds.
- **Real-Time Body Tracking** — MediaPipe Pose + Hand tracking ensures every generated prop follows the streamer's movements naturally at 30 FPS.
- **Intelligent AI Agent** — Command-R 7B understands context: say *"give me a crown"* and it auto-detects it's headwear, placing it on your head — not your hand.
- **Instant Background Swap** — Backgrounds skip the heavy background-removal step entirely, enabling zero-delay scene changes.
- **Native OBS Integration** — Works inside OBS Studio, the industry-standard streaming software used by millions of creators worldwide.

---

## ✨ Key Features

| | Feature | Description |
|---|---|---|
| 🎤 | **Voice-Controlled Generation** | Push-to-talk microphone triggers AI image generation via Whisper STT |
| ⌨️ | **Text Prompt Support** | Type detailed prompts with style selection (Cinematic, Fantasy, Cyberpunk, etc.) |
| 🦾 | **Real-Time Body Tracking** | MediaPipe Pose + Hand tracking at 30 FPS with 9 anchor zones (hand, head, shoulders, wrists, etc.) |
| 🎨 | **Z-Image-Turbo** | Quantized GGUF diffusion model generates 1024×576 images in ~4–5 seconds on a T4 GPU |
| 🤖 | **AI Agent Router** | Command-R 7B classifies prompts, rewrites them for optimal image quality, and routes to the correct body anchor |
| 🖼️ | **Automatic Background Removal** | rembg strips backgrounds from props asynchronously so it doesn't block rendering |
| 📺 | **OBS Integration** | Transparent browser source overlay + dedicated streamer control panel |
| 🔄 | **Event-Driven Architecture** | WebSocket-based real-time communication between all components |
| 🛡️ | **Content Safety** | Built-in safety filter blocks inappropriate prompt generation |

---

## 🏗️ Architecture Overview

Ghost Frame operates on a **two-machine architecture** that separates compute-heavy AI workloads from lightweight real-time rendering:

```
┌─────────────────────────────────┐         SSH Tunnel         ┌─────────────────────────────────┐
│      ☁️  CLOUD GPU SERVER       │◄──────────────────────────►│      🖥️  STREAMER'S LOCAL PC    │
│          (Port 8000)            │    WebSocket + REST API     │          (Port 8001)            │
│                                 │                             │                                 │
│  ┌───────────────────────────┐  │                             │  ┌───────────────────────────┐  │
│  │  Z-Image-Turbo            │  │                             │  │  Webcam Capture            │  │
│  │  (Image Generation)       │  │         Generated           │  │  (Video Input)             │  │
│  ├───────────────────────────┤  │         Images              │  ├───────────────────────────┤  │
│  │  Whisper STT              │  │ ──────────────────────────► │  │  MediaPipe Tracking        │  │
│  │  (Speech-to-Text)         │  │                             │  │  (Body + Hand Landmarks)   │  │
│  ├───────────────────────────┤  │                             │  ├───────────────────────────┤  │
│  │  Command-R 7B             │  │                             │  │  Real-Time Compositing     │  │
│  │  (AI Agent / Router)      │  │                             │  │  (Prop Rendering)          │  │
│  ├───────────────────────────┤  │                             │  ├───────────────────────────┤  │
│  │  rembg (U²-Net)           │  │                             │  │  OBS Output Window         │  │
│  │  (Background Removal)     │  │                             │  │  (Browser Source)          │  │
│  └───────────────────────────┘  │                             │  └───────────────────────────┘  │
└─────────────────────────────────┘                             └─────────────────────────────────┘
```

- **Cloud GPU Server** (Port 8000) — Runs the heavy AI models: Z-Image-Turbo for image generation, Whisper for speech-to-text, Command-R 7B for intelligent routing, and rembg for background removal.
- **Streamer's Local PC** (Port 8001) — Runs the lightweight vision engine: webcam capture, MediaPipe body tracking, real-time prop compositing, and the OBS-visible output window.
- **Connected via SSH tunnel** — A browser-based relay forwards generated images from the cloud server to the local engine in real time.

---

## 🛠️ Tech Stack

| Component | Technology |
|:---|:---|
| **Image Generation** | Z-Image-Turbo (GGUF Q5_K_M quantized) |
| **Speech-to-Text** | faster-whisper (CTranslate2) |
| **AI Agent** | Command-R 7B via Ollama |
| **Body Tracking** | MediaPipe Pose + Hand Landmarks |
| **Background Removal** | rembg (U²-Net) |
| **Video Matting** | RVM (Robust Video Matting) |
| **Backend Framework** | FastAPI + Uvicorn |
| **Frontend** | Vanilla HTML / CSS / JS |
| **Streaming Software** | OBS Studio (Browser Source) |
| **Communication** | WebSocket (real-time) + REST API |
| **GPU** | NVIDIA T4 (16 GB VRAM) |
| **Quantization** | GGUF Q5_K_M + BitsAndBytes NF4 |

---

## 📁 Project Structure

```
ghost-frame/
├── config.py                  # Unified configuration
├── backend/                   # Cloud GPU Server
│   ├── main.py                # FastAPI server (REST + WebSocket)
│   ├── engine/                # Z-Image-Turbo pipeline
│   ├── agent/                 # AI routing agent (Command-R 7B)
│   ├── stt/                   # Whisper speech-to-text
│   └── setup.sh               # One-command GPU setup
├── local_engine/              # Streamer's Local PC
│   ├── background_remover.py  # Background removal module
│   ├── download_models.py     # Model weight downloader
│   ├── ghost_engine.py        # Vision engine (camera + tracking + compositing)
│   ├── tracker.py             # MediaPipe tracking
│   ├── prop_config.py         # Prop attachment taxonomy
│   ├── prop_manager.py        # Prop state and attachment management
│   ├── live_equip.py          # Real-time prop rendering
│   └── requirements.txt       # Local dependencies
├── frontend/                  # Web UI + OBS Overlay
│   ├── index.html             # Analytics dashboard
│   ├── panel.html             # Streamer control panel
│   ├── overlay.html           # OBS transparent overlay
│   ├── overlay.js             # Real-time prop tracking
│   ├── script.js              # Frontend logic
│   └── style.css              # Styling
└── outputs/                   # Generated images
```

---

## 📋 System Requirements

### ☁️ Cloud Server
| Requirement | Minimum |
|:---|:---|
| GPU | NVIDIA with ≥ 16 GB VRAM (T4 / A100 recommended) |
| CUDA | 12.1+ |
| Python | 3.10+ |
| Ollama | Latest (for AI agent model) |

### 🖥️ Streamer's PC
| Requirement | Minimum |
|:---|:---|
| Webcam | Any USB or built-in camera |
| Python | 3.10+ |
| OBS Studio | 28+ |

---

## 🚀 Why Ghost Frame?

- **First of its kind** — The first system to combine real-time AI image generation with body-tracked prop rendering for livestreaming.
- **Democratizes content creation** — Streamers don't need graphic design skills to create stunning, on-the-fly visuals.
- **Unlocks audience interaction** — Opens new possibilities where viewers can suggest props via chat, creating truly interactive streams.
- **Scalable architecture** — Cleanly separates compute-heavy AI from lightweight local rendering, enabling flexible deployment.
- **Future-proof & modular** — Swap in better AI models as they emerge without redesigning the system.

---

## 👥 Team

Built by **Team Ghost Frame**

---

## 📄 License

This project is proprietary. All rights reserved.

---

<div align="center">

📖 **Getting Started** → See [START_HERE.md](START_HERE.md) for complete setup and launch instructions.

</div>
