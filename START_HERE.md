# Ghost Frame — START HERE

> **Ghost Frame** is a real-time AI prop & background system for livestreamers.
> Speak or type a prompt → AI generates an image → it appears on your stream, tracked to your body.

---

## Architecture

```
┌──────────────────────────────┐    SSH Tunnel :8000    ┌──────────────────────────────┐
│       YOUR LAPTOP            │◄══════════════════════►│    CLUB GPU (gpu26)          │
│                              │                        │                              │
│  Local Engine (:8001)        │                        │  Backend (:8000)             │
│  ├─ Webcam + MediaPipe       │                        │  ├─ Z-Image-Turbo (GGUF)     │
│  ├─ RVM background removal   │                        │  ├─ Command-R 7B agent       │
│  ├─ Prop compositing         │                        │  ├─ Whisper STT              │
│  └─ OBS Window Capture       │                        │  └─ rembg                    │
│                              │                        │                              │
│  Panel (browser)             │                        │  Ollama (:11434)             │
│  http://localhost:8000/app/  │                        │                              │
│        panel.html            │                        │                              │
└──────────────────────────────┘                        └──────────────────────────────┘
```

---

## Step 1: Setup the Cloud GPU (One-Time)

SSH into the GPU:
```bash
ssh gpu26@10.214.4.236 -p 22013
```

Upload the project (from your laptop, if first time):
```powershell
scp -P 22013 -r "C:\Users\Mayank Moyal\Desktop\Ghost Frame" gpu26@10.214.4.236:~/GhostFrame/
```

Run setup (installs everything — PyTorch, Ollama, models, ~15 min):
```bash
cd ~/GhostFrame/backend
bash setup.sh
```

---

## Step 2: Start the Cloud Backend

SSH into the GPU:
```bash
ssh gpu26@10.214.4.236 -p 22013
```

Start the server inside tmux (survives disconnects):
```bash
tmux new -s ghostframe
cd ~/GhostFrame/backend
ollama serve > /tmp/ollama.log 2>&1 &
python3 main.py
```

Wait for: `[Startup] All models loaded. Server ready!` (~2-3 min)

> **tmux tips:**
> - Detach: `Ctrl+B` then `D`
> - Reattach: `tmux attach -t ghostframe`

---

## Step 3: SSH Tunnel (Connect Laptop to GPU)

Open a **new terminal** on your laptop and keep it running:
```powershell
ssh -L 8000:localhost:8000 gpu26@10.214.4.236 -p 22013
```

Test: Open http://localhost:8000/health in your browser.

---

## Step 4: Start the Local Engine

In another terminal on your laptop:
```powershell
cd "C:\Users\Mayank Moyal\Desktop\Ghost Frame\local_engine"
python ghost_engine.py
```

---

## Step 5: Open the Panel & Start Streaming!

| URL | What |
|-----|------|
| http://localhost:8000/app/panel.html | **Control Panel** — Voice & text prompts (NOT on stream) |
| http://localhost:8000/app/ | **Dashboard** — Full analytics view |
| http://localhost:8000/app/overlay.html | **OBS Overlay** — Transparent prop layer |

### OBS Setup

1. **Window Capture** → Select the "Prop Engine" window from the local engine
2. **Browser Source** → URL: `http://localhost:8000/app/overlay.html` (1920×1080, transparent)
3. **Panel** → Open `http://localhost:8000/app/panel.html` in a separate browser window (NOT captured by OBS — viewers can't see it)

---

## Quick Cheat Sheet

### GPU side:
```bash
ssh gpu26@10.214.4.236 -p 22013
tmux attach -t ghostframe          # reconnect to running server
nvidia-smi                          # check VRAM
curl localhost:8000/health          # health check
```

### Laptop side:
```powershell
# Terminal 1 — SSH tunnel (keep open):
ssh -L 8000:localhost:8000 gpu26@10.214.4.236 -p 22013

# Terminal 2 — Local engine:
cd "C:\Users\Mayank Moyal\Desktop\Ghost Frame\local_engine"
python ghost_engine.py

# Terminal 3 — Open panel:
start http://localhost:8000/app/panel.html
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `Connection refused` on laptop | SSH tunnel not running — re-run the `-L 8000:...` command |
| `FileNotFoundError: GGUF` | Run `bash setup.sh` again on GPU |
| `Ollama connection refused` | `ollama serve > /tmp/ollama.log 2>&1 &` |
| Server dies on SSH disconnect | Use `tmux` (Step 2) |
| CUDA OOM | `nvidia-smi` — needs ~14GB free VRAM |
| Black images | Already fixed (VAE float32 cast) |
| Agent returns `agent_ok: false` | Warm up Ollama: `curl localhost:11434/api/generate -d '{"model":"command-r7b","prompt":"hi","stream":false}'` |
| Props too small | Adjust `scale_multiplier` in `local_engine/prop_config.py` |

---

## Key Files

| File | Purpose |
|------|---------|
| `backend/main.py` | Cloud server (FastAPI) |
| `backend/engine/zimage_turbo.py` | Image generation pipeline |
| `backend/agent/client.py` | Ollama agent (Command-R 7B) |
| `backend/setup.sh` | One-time GPU setup script |
| `frontend/panel.html` | Streamer control panel |
| `frontend/overlay.html` + `overlay.js` | OBS transparent overlay |
| `local_engine/ghost_engine.py` | Webcam + MediaPipe + prop compositing |
| `local_engine/prop_config.py` | Prop scaling & attachment profiles |
| `config.py` | Global configuration |
