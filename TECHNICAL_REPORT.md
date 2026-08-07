# Ghost Frame — Technical Preparation Document
### Project Presentation Q&A Guide for All Team Members

> **Purpose:** This document covers every technical detail of Ghost Frame so that any team member can confidently answer any question a judge might ask. Organized by the 5 evaluation criteria.

---

## SECTION 1: Problem Understanding & Clarity (5 Marks)

### 1.1 Problem Statement
**"How can livestreamers add real-time AI-generated visual props and backgrounds to their streams without expensive hardware, green screens, or manual editing?"**

### 1.2 Why This Problem Matters
- **Live streaming is a $90B+ industry** (Twitch, YouTube Live, Kick) with 8M+ active streamers
- Current prop/overlay solutions are **static** (pre-made images) or require **expensive motion capture**
- Streamers want **dynamic, interactive** content that responds to their chat/voice in real-time
- No existing tool lets a streamer say "give me a glowing sword" and have it appear tracked to their hand instantly

### 1.3 Key Objectives
1. **Voice-to-Prop:** Speak a prompt → AI generates an image → it appears on stream tracked to your body
2. **Real-time Tracking:** Props follow hand/head/body movements at 30fps using only a webcam
3. **Zero Green Screen:** AI-powered background removal (RVM) replaces backgrounds without physical setup
4. **Cloud-Local Split:** Heavy AI runs on cloud GPU, lightweight tracking runs locally — works on any laptop
5. **OBS Integration:** Plugs directly into OBS Studio (the #1 streaming software)

### 1.4 Target Users
- Twitch/YouTube streamers who want interactive visual effects
- VTubers and content creators
- Live event hosts and educators

### 1.5 Possible Judge Questions & Answers

**Q: Why not just use pre-made overlays?**
> Pre-made overlays are static. Ghost Frame generates **unique, context-aware** props on demand. A viewer can say "give the streamer a Viking axe" and it appears in seconds, tracked to their hand. No other tool does this.

**Q: How is this different from a green screen?**
> Green screens require physical setup, consistent lighting, and can't handle props. Ghost Frame uses **RVM (Robust Video Matting)** — a neural network that segments the person in real-time without any physical setup. Plus, it adds AI-generated props, which green screens can't do.

**Q: What's the real-world use case?**
> During a gaming stream, a viewer donates and requests "a flaming crown" — Ghost Frame generates it, removes its background, and places it above the streamer's head, all in ~6 seconds. The streamer never leaves their chair.

---

## SECTION 2: Technical Approach (10 Marks)

### 2.1 Architecture Overview

```
┌──────────────────────┐     SSH Tunnel     ┌──────────────────────┐
│   LOCAL LAPTOP       │◄══════════════════►│   CLOUD GPU          │
│                      │    Port 8000       │                      │
│  Ghost Engine (:8001)│                    │  Backend (:8000)     │
│  ├─ MediaPipe        │                    │  ├─ Z-Image-Turbo    │
│  ├─ RVM Matting      │                    │  ├─ Command-R 7B     │
│  └─ Prop Compositing │                    │  ├─ Whisper STT      │
│                      │                    │  └─ rembg            │
│  OBS Studio          │                    │                      │
│  └─ Window Capture   │                    │  Ollama (:11434)     │
└──────────────────────┘                    └──────────────────────┘
```

### 2.2 Why Two Machines?

| Concern | Cloud GPU | Local Laptop |
|---------|-----------|--------------|
| **What runs** | Image generation, AI agent, STT | Webcam, tracking, compositing |
| **Why** | Needs ~14GB VRAM | Needs <50ms latency for tracking |
| **VRAM** | 16GB (RTX T4/A100) | 2-4GB (RTX 5050) |
| **Latency tolerance** | 4-6s acceptable | Must be <33ms (30fps) |

**Design Decision:** Tracking MUST be local because even 100ms network latency would make props appear "floaty." Image generation can tolerate 4-6s latency since it's a one-time operation per prompt.

### 2.3 Tech Stack — Full Breakdown

| Component | Technology | Why This Choice |
|-----------|-----------|-----------------|
| **Image Generation** | Z-Image-Turbo (Diffusion, GGUF Q5_K_M) | 4-step inference = fast (~4s). GGUF quantization saves 40% VRAM vs FP16 |
| **Text Encoder** | Qwen3-4B (NF4 4-bit) | Small but powerful encoder for prompt understanding |
| **Agent/Classifier** | Command-R 7B via Ollama | Purpose-built for agentic JSON tasks. Classifies prop type, anchor point, and rewrites prompts |
| **Speech-to-Text** | Faster-Whisper (CTranslate2) | 10x faster than OpenAI Whisper, runs on CPU |
| **Hand Tracking** | MediaPipe Hands (21 landmarks) | Real-time, runs on CPU, sub-10ms per frame |
| **Pose Tracking** | MediaPipe Pose (33 landmarks) | Full body tracking for shields, body wear, etc. |
| **Background Removal (stream)** | RVM (Robust Video Matting) | Temporal consistency — no flickering unlike frame-by-frame methods |
| **Background Removal (props)** | rembg (U2-Net) | High-quality static image segmentation |
| **Backend Framework** | FastAPI + Uvicorn | Async Python, WebSocket support, high performance |
| **Local Engine** | FastAPI + OpenCV | Real-time frame processing and compositing |
| **Frontend** | Vanilla HTML/CSS/JS | Zero build step, served directly by FastAPI |
| **Streaming Integration** | OBS Studio (Browser Source + Window Capture) | Industry standard, supports transparent overlays |
| **Model Serving** | Ollama | Easy model management, GGUF support, API-compatible |

### 2.4 AI Agent Pipeline — Deep Dive

When a user says "give me a glowing blue sword, fantasy style":

```
Step 1: Speech → Text (Whisper)
  Input:  Audio blob (WebM)
  Output: "give me a glowing blue sword, fantasy style"
  Time:   ~1s

Step 2: Agent Classification (Command-R 7B)
  Input:  "give me a glowing blue sword, fantasy style"
  Output: {
    "type": "prop",
    "anchor_type": "hand_held",
    "prop_category": "hand_held", 
    "style": "fantasy",
    "is_safe": true,
    "rewritten_prompt": "A gleaming sapphire-blue longsword with fiery aura,
                         dramatic backlighting, intricate runes along the blade"
  }
  Time:   ~1-2s

Step 3: Image Generation (Z-Image-Turbo)
  Input:  Enhanced prompt + style suffix
  Output: 1024x1024 PNG image
  Time:   ~4s
  VRAM:   ~10.3GB peak

Step 4: Background Removal (rembg) — ASYNC
  Input:  sword.png (with background)
  Output: sword_nobg.png (transparent)
  Time:   ~6s (runs in background, doesn't block response)

Step 5: Prop Push to Local Engine
  Input:  Image URL + anchor_type
  Output: Prop loaded, tracked at 30fps
```

### 2.5 Anchor System — 8 Types

| Anchor Type | Body Part | Example Props | Tracking Method |
|-------------|-----------|---------------|-----------------|
| `hand_held` | Palm center | Sword, Wand, Staff | Hand landmark palm center + hand angle |
| `head_wear` | Above head | Crown, Helmet, Hat | Pose landmark head top |
| `body_wear` | Torso center | Cape, Armor, Vest | Pose landmark torso midpoint |
| `shield` | Forearm | Shield, Buckler | Wrist-to-elbow vector |
| `face_wear` | Eye center | Glasses, Mask | Eye landmark midpoint |
| `neck_wear` | Neck point | Necklace, Amulet | Pose landmark neck |
| `wrist_wear` | Wrist | Bracelet, Gauntlet | Hand landmark wrist |
| `ear_wear` | Ear | Earring, Ear cuff | Pose landmark ear |
| `background` | Full screen | Dungeon, Forest | RVM matting (replaces entire background) |

### 2.6 Scaling Algorithm

```python
# Body-proportional scaling ensures props look natural regardless of distance
body_px = measure_body_part(pose, profile.body_scale_ref)  # e.g., palm width in pixels
prop_px = measure_prop_part(image, profile.prop_scale_ref)  # e.g., handle width in pixels
raw_scale = (body_px / prop_px) * profile.scale_multiplier  # 3.5x for hand_held
smoothed_scale = one_euro_filter(raw_scale)  # Prevents jitter
```

**Why OneEuroFilter?** Raw MediaPipe landmarks jitter by 2-5px per frame. Without smoothing, props would vibrate visibly. The OneEuroFilter (from CHI 2012 paper) provides low-latency, low-jitter smoothing.

### 2.7 Possible Judge Questions & Answers

**Q: Why GGUF quantization instead of full precision?**
> GGUF Q5_K_M reduces the transformer model from ~10GB (FP16) to ~5GB while maintaining 95%+ quality. This lets us fit the image model + agent model + text encoder all in 16GB VRAM. Without quantization, we'd need a 24GB+ GPU.

**Q: Why Command-R 7B instead of GPT-4 or a larger model?**
> Three reasons: (1) It runs locally via Ollama — no API costs, no internet dependency, no latency. (2) Command-R is specifically designed for agentic JSON output — it reliably returns structured classifications. (3) 7B params is small enough to fit alongside the image model in VRAM (~3.4GB).

**Q: Why not use Stable Diffusion XL or DALL-E?**
> SDXL requires 25-50 inference steps (~15-30s). Z-Image-Turbo uses only 4 steps (~4s) — critical for a live streaming use case where users expect near-instant feedback. DALL-E requires API calls which add latency and cost.

**Q: Why separate local and cloud processing?**
> Tracking must be <33ms (30fps). Even on a fast network, round-trip latency is 50-200ms. By keeping MediaPipe local, we achieve ~5ms tracking latency. Image generation can tolerate 4-6s since it's a one-time cost per prompt.

**Q: How do you handle multiple props?**
> Currently we support one active prop at a time (matching the streamer use case — they hold one weapon/wear one hat). The architecture supports extension to multiple props via the prop_manager's slot system.

**Q: What happens if MediaPipe loses tracking?**
> The OneEuroFilter holds the last known position with decay, so the prop stays where it was rather than jumping or disappearing. When tracking resumes, the filter smoothly transitions back.

---

## SECTION 3: Implementation & Demonstration (15 Marks)

### 3.1 Complete File Inventory

| Module | Files | Lines of Code |
|--------|-------|---------------|
| Backend (Cloud) | 18 files | ~2,500 LoC |
| Frontend (Web UI) | 6 files | ~2,800 LoC |
| Local Engine | 9 files | ~1,800 LoC |
| Config + Docs | 5 files | ~500 LoC |
| **Total** | **38 files** | **~7,600 LoC** |

### 3.2 Key Implementation Details

#### Image Generation Pipeline (`engine/zimage_turbo.py`)
- Loads GGUF transformer via `GGUFQuantizationConfig`
- Text encoder (Qwen3-4B) loaded in NF4 4-bit with `BitsAndBytesConfig`
- VAE forced to float32 to prevent black images (common GGUF issue)
- 4-step inference with guidance_scale=0 (turbo mode)

#### Agent System (`agent/client.py`, `agent/router.py`)
- Structured JSON system prompt with explicit schema
- Fallback parsing: tries JSON → regex extraction → defaults
- Safety filter blocks NSFW/violent content
- Prompt rewriter enhances vague prompts ("sword" → "gleaming sapphire-blue longsword with fiery aura...")

#### Local Engine (`ghost_engine.py`, `live_equip.py`)
- 30fps render loop with OpenCV
- MediaPipe Hands (21 landmarks) + Pose (33 landmarks) running in parallel
- RVM (Robust Video Matting) for real-time background segmentation
- Alpha compositing with anti-aliased edges
- Rotation computed from hand/wrist angle vectors

#### Frontend (`panel.html`, `overlay.js`)
- **Panel:** Push-to-talk voice recording, text input, style selection, status monitoring
- **Overlay:** Transparent OBS Browser Source, dual WebSocket connections (local tracking + cloud events)
- **Dashboard:** Full analytics with pipeline steps, VRAM usage, latency metrics

### 3.3 Demo Script (Suggested Flow)

```
1. Show the Panel UI → explain it's "not visible to viewers"
2. Type: "a glowing blue lightsaber" → show generation (~6s)
3. Show the prop appearing tracked to your hand in real-time
4. Move your hand around → show 30fps tracking
5. Voice input: hold mic button, say "give me a golden crown"
6. Show it appear on your head
7. Type: "a dark enchanted forest background"
8. Show the background change behind you (RVM matting)
9. Show the dashboard metrics: latency, VRAM, agent classification
10. Clear props → show clean state
```

### 3.4 Possible Judge Questions & Answers

**Q: Can you show us the code for [specific component]?**
> Yes — have these files ready to show:
> - `main.py` — all endpoints, async rembg pattern
> - `live_equip.py` — prop compositing and scaling algorithm
> - `overlay.js` — WebSocket tracking at 30fps
> - `client.py` — Ollama agent system prompt and JSON parsing

**Q: How long did this take to build?**
> The core system was built over [X weeks]. The most challenging parts were: (1) getting GGUF quantized models to produce non-black images (VAE float32 fix), (2) achieving smooth prop tracking without jitter (OneEuroFilter), and (3) splitting the pipeline across two machines with minimal perceived latency.

**Q: What would you do differently?**
> (1) Use WebRTC instead of HTTP for prop pushing — would reduce the cloud-to-local push latency. (2) Pre-cache common props (sword, shield, crown) to skip generation entirely. (3) Add multi-prop support.

---

## SECTION 4: Results & Validation (10 Marks)

### 4.1 Performance Metrics (Measured)

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Image generation latency | **4.2s** | <10s | ✅ |
| Agent classification time | **1-2s** | <3s | ✅ |
| Total perceived latency | **~6s** | <15s | ✅ |
| Tracking FPS | **30 fps** | 30 fps | ✅ |
| Tracking latency | **~5ms** | <33ms | ✅ |
| Peak GPU VRAM (cloud) | **10.3 GB** | <16 GB | ✅ |
| Background removal (rembg) | **~6s** | N/A (async) | ✅ |
| Agent classification accuracy | **~90%** | >80% | ✅ |

### 4.2 VRAM Budget

| Component | VRAM Usage |
|-----------|-----------|
| Z-Image-Turbo (GGUF Q5_K_M) | ~10.3 GB |
| Command-R 7B (Ollama) | ~3.4 GB |
| Text Encoder (Qwen3-4B, NF4) | ~2.1 GB |
| **Total** | **~14 GB / 16 GB available** |

### 4.3 What We Validated
1. **End-to-end pipeline:** Voice → STT → Agent → Image Gen → OBS overlay → tracked prop
2. **All 9 anchor types:** hand_held, head_wear, body_wear, shield, face_wear, neck_wear, wrist_wear, ear_wear, background
3. **Cross-machine communication:** SSH tunnel, HTTP prop pushing, WebSocket events
4. **OBS integration:** Transparent overlay, window capture compositing
5. **Async performance:** rembg runs in background without blocking user response

### 4.4 Known Limitations (Be Honest)
1. **Single prop at a time** — architecture supports multi-prop but not implemented
2. **Prop scaling varies** — AI-generated images have inconsistent handle/body ratios
3. **Requires cloud GPU** — cannot run entirely on a consumer laptop (yet)
4. **No text-on-props** — AI occasionally generates text artifacts on props
5. **English only** — Whisper supports multilingual but agent prompt is English

### 4.5 Possible Judge Questions & Answers

**Q: How did you validate the system works?**
> We tested the full pipeline end-to-end: voice input → transcription → agent classification → image generation → background removal → prop tracking on OBS. We measured latency at each stage and confirmed 30fps tracking with ~6s total generation time.

**Q: What's the failure rate?**
> Agent classification succeeds ~90% of the time. When it fails, the router applies sensible defaults (type="prop", anchor="hand_held"). Image generation has ~99% success rate after the VAE float32 fix.

**Q: Can this scale to multiple users?**
> The current architecture is single-user (one streamer). Scaling would require: (1) GPU load balancing across multiple GPUs, (2) per-session prop state management, (3) queue-based generation. The FastAPI async architecture supports this.

---

## SECTION 5: Technical Understanding & Q&A (10 Marks)

### 5.1 Design Decisions — Why We Made Each Choice

| Decision | Alternatives Considered | Why We Chose This |
|----------|------------------------|-------------------|
| GGUF quantization | FP16, INT8, ONNX | GGUF Q5 gives best quality-per-VRAM ratio. Saves 40% VRAM vs FP16 |
| Ollama for LLM | vLLM, llama.cpp, API calls | Simple setup, automatic GGUF support, model management built-in |
| MediaPipe for tracking | OpenPose, MMPose, YOLO-Pose | Runs on CPU, <10ms latency, 21 hand landmarks (others only detect wrist) |
| RVM for background | rembg, MODNet, Background Matting V2 | Temporal consistency — uses video context, no flickering between frames |
| FastAPI for both servers | Flask, Express.js, gRPC | Native async, WebSocket support, auto-docs, same language as ML code |
| SSH tunnel for connection | ngrok, Cloudflare Tunnel, VPN | Zero-config, secure, works behind firewalls, no external dependencies |
| Async rembg | Synchronous blocking | Cuts perceived latency from 14s to 6s. User sees result immediately |

### 5.2 Technical Glossary (Know These Terms)

| Term | What It Is | Where We Use It |
|------|-----------|-----------------|
| **GGUF** | GPU-optimized quantized model format by llama.cpp | Z-Image-Turbo transformer weights |
| **NF4** | 4-bit NormalFloat quantization (QLoRA paper) | Qwen3-4B text encoder |
| **Q5_K_M** | 5-bit quantization with K-quants (medium) | Transformer model compression |
| **RVM** | Robust Video Matting — temporal-aware segmentation | Real-time background removal |
| **rembg** | Static image background removal using U2-Net | Removing backgrounds from generated props |
| **MediaPipe** | Google's real-time ML framework for hand/pose/face | 21 hand + 33 pose landmarks |
| **OneEuroFilter** | Low-latency jitter-reduction filter (CHI 2012) | Smoothing prop position/scale |
| **GGUF Q5_K_M** | 5-bit quantization with K-means clustering | Diffusion model compression |
| **Guidance Scale** | How closely the image follows the prompt | Set to 0 (turbo mode — no CFG needed) |
| **Ollama** | Local LLM server (like Docker for AI models) | Runs Command-R 7B agent |
| **WebSocket** | Persistent bidirectional connection | 30fps tracking data + prop events |
| **Alpha Compositing** | Blending transparent images onto a background | Prop overlay rendering |
| **Anchor Point** | Body landmark where a prop attaches | Palm, head, torso, etc. |

### 5.3 Common Hard Questions & Expert Answers

**Q: Why not use a single machine?**
> A single consumer laptop can't fit a 10GB image model + 3.4GB agent + tracking in <8GB VRAM. The cloud GPU provides the raw compute power, while the laptop handles the latency-critical webcam/tracking loop. This split mirrors production systems like Google Stadia (cloud rendering + local input).

**Q: How do you ensure the prop looks natural on the streamer?**
> Three mechanisms: (1) **Body-proportional scaling** — prop size is computed relative to the user's palm/shoulder width, not a fixed pixel size. (2) **Rotation tracking** — the prop rotates based on the hand/wrist angle vector. (3) **OneEuroFilter smoothing** — prevents jitter while maintaining responsiveness.

**Q: What's the bottleneck in your pipeline?**
> Image generation (~4s) is the primary bottleneck. The agent (~1-2s) and background removal (~6s, async) are secondary. Tracking is not a bottleneck (<5ms). To speed up generation, we could: use a smaller model, reduce resolution, or implement model caching.

**Q: How does the agent decide if something is a prop or background?**
> The Command-R 7B model is given a structured system prompt with clear classification rules: "If the user asks for an environment/scene/setting → background. If they ask for an object/weapon/clothing → prop." It then sub-classifies props into 8 anchor types based on what body part the item naturally attaches to.

**Q: What about safety? Can users generate inappropriate content?**
> Yes, we have a safety pipeline: (1) The agent returns an `is_safe` boolean. (2) `safety.py` checks this flag and blocks unsafe prompts with a clear error message. (3) The system prompt instructs the model to flag NSFW/violent/hateful content.

**Q: How does RVM differ from a green screen?**
> RVM (Robust Video Matting) uses a neural network trained on video sequences. Unlike frame-by-frame methods, it uses **temporal consistency** — it considers previous frames to avoid flickering at hair/edge boundaries. Green screens require physical setup and fail with similar-colored clothing. RVM works with any background and any clothing.

**Q: What happens if the internet goes down?**
> The local engine continues tracking and displaying the last loaded prop at 30fps — the stream is unaffected. Only new prop generation requires the cloud connection. The WebSocket auto-reconnects when connectivity resumes.

**Q: Can you explain the async rembg pattern?**
> Originally, the `/generate` endpoint waited for rembg (~6s) before responding. We changed it to: (1) Generate image (~4s), (2) Return HTTP response immediately with the original image, (3) Kick off rembg as a background `asyncio.create_task`. The user sees results in ~6s instead of ~14s. The background-removed version is pushed to the local engine when ready.

**Q: Why FastAPI instead of Flask?**
> FastAPI is natively async (ASGI), supports WebSockets out of the box, auto-generates API docs, and handles concurrent requests without threads. Flask is WSGI (synchronous) — it would block during image generation, preventing health checks and WebSocket connections from working.

**Q: How accurate is the hand tracking?**
> MediaPipe Hands detects 21 landmarks per hand with sub-pixel accuracy at 30fps. It handles partial occlusion (hand behind prop) and works in varying lighting. The main limitation is extreme motion blur during fast movements.

### 5.4 Division of Work (Prepare This)

> **Important:** Each team member should be able to explain ANY part of the project, but should be especially strong in their area. Prepare a rough division like:

| Member | Primary Area | Key Files to Know |
|--------|-------------|-------------------|
| Member 1 | Image Generation + Agent | `zimage_turbo.py`, `client.py`, `router.py` |
| Member 2 | Local Engine + Tracking | `ghost_engine.py`, `live_equip.py`, `tracker.py` |
| Member 3 | Frontend + OBS Integration | `panel.html`, `overlay.js`, `script.js` |
| Member 4 | System Architecture + Deployment | `main.py`, `setup.sh`, `config.py`, `START_HERE.md` |

### 5.5 One-Liner Explanations (Quick Reference)

- **"What is Ghost Frame?"** → "A real-time AI system that generates visual props from voice/text and tracks them to a streamer's body at 30fps."
- **"How does it work?"** → "You speak a prompt, our AI agent classifies it, a diffusion model generates the image, and MediaPipe tracks it to your hand/head/body in real-time."
- **"What makes it novel?"** → "No existing tool combines AI image generation + real-time body tracking + OBS integration in a single pipeline."
- **"What's the latency?"** → "~6 seconds from prompt to prop-on-screen, then 30fps tracking at <5ms latency."
- **"What GPU do you need?"** → "16GB VRAM cloud GPU for generation, any laptop with a webcam for tracking."
