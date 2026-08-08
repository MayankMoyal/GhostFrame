"""Ghost Frame — Unified Cloud Backend

Merges:
- Image generation (Z-Image-Turbo) from q8/backend
- Voice STT (faster-whisper) from newver/stt
- Agent routing (Ollama) from q8/backend/agent
- OBS overlay WebSocket broadcasting
- Event-driven prop hot-swap to Local Engine
- Background removal (rembg) for generated props

Endpoints:
    GET  /health            — Server health check
    POST /generate          — Text prompt → image generation
    POST /generate-voice    — Audio blob → STT → image generation
    POST /upload-prop       — Custom prop image upload
    POST /clear-props       — Clear all props from OBS overlay
    WS   /ws/anchor         — WebSocket for OBS overlay tracking + prop events
    GET  /outputs/{file}    — Static file serving for generated images
"""
import asyncio
import os
import sys
import tempfile
import traceback
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from time import time
from uuid import uuid4

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.zimage_turbo import generate_image, load_pipeline
from agent.router import run_agent
from agent.rewriter import get_rewritten_prompt
from agent.safety import UnsafePromptError, check_safety
from stt.transcriber import load_whisper_model, transcribe_audio

try:
    from config import (
        CLOUD_HOST, CLOUD_PORT, OUTPUT_DIR, LOCAL_API_URL,
        WHISPER_MODEL_SIZE, WHISPER_DEVICE, WHISPER_COMPUTE_TYPE,
    )
except ImportError:
    # Fallback defaults if config.py not accessible
    CLOUD_HOST = "0.0.0.0"
    CLOUD_PORT = 8000
    OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"
    LOCAL_API_URL = "http://localhost:8001"
    WHISPER_MODEL_SIZE = "base"
    WHISPER_DEVICE = "cpu"
    WHISPER_COMPUTE_TYPE = "int8"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Pipeline State ────────────────────────────────────────────────────────
pipeline = None
pipeline_lock = Lock()

# ── WebSocket Connections ─────────────────────────────────────────────────
# All connected OBS overlay clients
overlay_clients: list[WebSocket] = []


# ── Pydantic Models ───────────────────────────────────────────────────────
class GenerationRequest(BaseModel):
    prompt: str
    style: str = ""


# ── Helper Functions ──────────────────────────────────────────────────────
def build_prompt(prompt: str, style: str) -> str:
    clean_prompt = prompt.strip()
    clean_style = style.strip()
    if not clean_style:
        return clean_prompt
    return f"{clean_prompt}, {clean_style} style"


def create_output_path() -> Path:
    filename = f"generation_{int(time())}_{uuid4().hex[:8]}.png"
    return OUTPUT_DIR / filename


async def broadcast_to_overlays(message: dict):
    """Send a message to all connected OBS overlay WebSocket clients."""
    import json
    dead = []
    payload = json.dumps(message)
    for ws in list(overlay_clients):
        try:
            await ws.send_text(payload)
        except Exception:
            dead.append(ws)
    for ws in dead:
        if ws in overlay_clients:
            overlay_clients.remove(ws)


async def push_prop_to_local_engine(filename: str, anchor_type: str):
    """Event-driven: push new prop directly to the Local Engine (Port 8001)."""
    import requests as req
    try:
        image_url = f"{CLOUD_API_URL}/outputs/{filename}"

        # Route to the correct endpoint based on anchor type
        if anchor_type == "background":
            endpoint = f"{LOCAL_API_URL}/equip-background"
        else:
            endpoint = f"{LOCAL_API_URL}/equip"

        resp = await run_in_threadpool(
            req.post,
            endpoint,
            json={"image_url": image_url, "anchor_type": anchor_type, "prop_category": anchor_type if anchor_type != "background" else ""},
            timeout=10,
        )
        if resp.ok:
            print(f"[Event-Driven] {'Background' if anchor_type == 'background' else 'Prop'} pushed to Local Engine: {filename}")
        else:
            print(f"[Event-Driven] Local Engine rejected: {resp.status_code}")
    except Exception as exc:
        print(f"[Event-Driven] Could not reach Local Engine: {exc}")


def remove_background_from_image(image_path: Path) -> Path:
    """Remove background from a generated prop image using rembg."""
    try:
        from rembg import remove
        from PIL import Image
        import io

        with open(image_path, "rb") as f:
            input_data = f.read()

        output_data = remove(input_data)

        nobg_path = image_path.with_name(image_path.stem + "_nobg.png")
        with open(nobg_path, "wb") as f:
            f.write(output_data)

        print(f"[rembg] Background removed: {nobg_path.name}")
        return nobg_path
    except Exception as exc:
        print(f"[rembg] Failed: {exc}")
        return image_path


# ── Global URL for self-referencing ───────────────────────────────────────
CLOUD_API_URL = f"http://localhost:{CLOUD_PORT}"


# ── App Lifespan ──────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline

    # Load Z-Image-Turbo pipeline
    print("[Startup] Loading Z-Image-Turbo pipeline...")
    pipeline = load_pipeline()

    # Load Whisper STT model
    print("[Startup] Loading Whisper STT model...")
    load_whisper_model(
        model_size=WHISPER_MODEL_SIZE,
        device=WHISPER_DEVICE,
        compute_type=WHISPER_COMPUTE_TYPE,
    )

    print("[Startup] All models loaded. Server ready!")
    yield


# ── FastAPI App ───────────────────────────────────────────────────────────
app = FastAPI(title="Ghost Frame Cloud Backend", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve generated images
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")

# Serve frontend files (dashboard at /app, panel at /app/panel.html)
FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/app", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


# ── Exception Handlers ───────────────────────────────────────────────────
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"status": "error", "message": exc.detail},
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return JSONResponse(
        status_code=422,
        content={"status": "error", "message": "Invalid request body."},
    )


# ── Endpoints ─────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": pipeline is not None,
        "service": "ghost-frame-cloud",
    }


@app.post("/heartbeat")
async def heartbeat():
    return {
        "status": "ok",
        "model_loaded": pipeline is not None,
    }


@app.post("/generate")
async def generate(payload: GenerationRequest):
    """Text prompt → Agent → Z-Image-Turbo → Image."""
    if not payload.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")

    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model is still loading. Try again shortly.")

    # Agent step: rewrite + safety check + style/anchor tagging
    agent_result = await run_in_threadpool(run_agent, payload.prompt)

    try:
        check_safety(agent_result)
    except UnsafePromptError as exc:
        raise HTTPException(status_code=400, detail=f"Prompt rejected: {exc.reason}")

    rewritten_prompt = get_rewritten_prompt(agent_result, payload.prompt)

    output_path = create_output_path()
    final_prompt = build_prompt(rewritten_prompt, payload.style)

    def run_generation():
        with pipeline_lock:
            return generate_image(pipeline, final_prompt, output_path)

    try:
        metrics = await run_in_threadpool(run_generation)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    if not output_path.exists():
        raise HTTPException(
            status_code=500,
            detail="Inference finished, but the output image was not saved.",
        )

    anchor_type = agent_result.get("anchor_type", "background")

    # ── Async background removal + push ──────────────────────────────────
    # Return the response IMMEDIATELY with the original image.
    # rembg runs in the background and pushes the nobg version to Local Engine
    # when ready. This cuts ~6-8s off the perceived response time.
    async def _bg_remove_and_push():
        try:
            if anchor_type != "background":
                nobg_path = await run_in_threadpool(remove_background_from_image, output_path)
                
                # Apply orientation and dynamic pivot specifically for the OBS Overlay!
                from prop_processor import process_prop_image
                grip_x, grip_y = await run_in_threadpool(process_prop_image, str(nobg_path), anchor_type)
                
                push_filename = nobg_path.name
            else:
                push_filename = output_path.name
                grip_x, grip_y = 0.5, 0.5

            await broadcast_to_overlays({
                "type": "new_prop",
                "filename": push_filename,
                "anchor_type": anchor_type,
                "metrics": metrics,
                "agent": agent_result,
                "grip_x": grip_x,
                "grip_y": grip_y
            })
            await push_prop_to_local_engine(push_filename, anchor_type)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            print(f"[Background] rembg/push failed: {exc}")

    asyncio.create_task(_bg_remove_and_push())

    return {
        "status": "success",
        "filename": output_path.name,
        "filename_nobg": None,  # Will be ready asynchronously
        "metrics": metrics,
        "agent": {
            "original_prompt": payload.prompt,
            "final_prompt": final_prompt,
            "style_detected": agent_result.get("style", ""),
            "type": agent_result.get("type", "background"),
            "anchor_type": anchor_type,
            "prop_category": agent_result.get("prop_category", ""),
            "agent_ok": agent_result.get("agent_ok", False),
        },
    }


@app.post("/generate-voice")
async def generate_voice(
    audio: UploadFile = File(...),
    style: str = Form(""),
    remove_bg: str = Form("true"),
):
    """Audio blob → Whisper STT → Agent → Z-Image-Turbo → Image."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model is still loading. Try again shortly.")

    # Save the uploaded audio to a temp file
    suffix = Path(audio.filename or "audio.webm").suffix or ".webm"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await audio.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        # Step 1: Transcribe audio to text
        transcript = await run_in_threadpool(transcribe_audio, tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    if not transcript or not transcript.strip():
        raise HTTPException(status_code=400, detail="Could not transcribe any speech from the audio.")

    # Step 2: Run agent (rewrite + safety + anchor detection)
    agent_result = await run_in_threadpool(run_agent, transcript)

    try:
        check_safety(agent_result)
    except UnsafePromptError as exc:
        raise HTTPException(status_code=400, detail=f"Prompt rejected: {exc.reason}")

    rewritten_prompt = get_rewritten_prompt(agent_result, transcript)

    # Step 3: Generate image
    output_path = create_output_path()
    final_prompt = build_prompt(rewritten_prompt, style)

    def run_generation():
        with pipeline_lock:
            return generate_image(pipeline, final_prompt, output_path)

    try:
        metrics = await run_in_threadpool(run_generation)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    if not output_path.exists():
        raise HTTPException(
            status_code=500,
            detail="Inference finished, but the output image was not saved.",
        )

    anchor_type = agent_result.get("anchor_type", "background")

    # ── Async background removal + push ──────────────────────────────────
    # Return response IMMEDIATELY. rembg + push runs in background.
    async def _bg_remove_and_push():
        try:
            if remove_bg.lower() == "true" and anchor_type != "background":
                nobg_path = await run_in_threadpool(remove_background_from_image, output_path)
                from prop_processor import process_prop_image
                grip_x, grip_y = await run_in_threadpool(process_prop_image, str(nobg_path), anchor_type)
                push_filename = nobg_path.name
            else:
                push_filename = output_path.name
                grip_x, grip_y = 0.5, 0.5

            await broadcast_to_overlays({
                "type": "new_prop",
                "filename": push_filename,
                "anchor_type": anchor_type,
                "metrics": metrics,
                "agent": agent_result,
                "grip_x": grip_x,
                "grip_y": grip_y
            })
            await push_prop_to_local_engine(push_filename, anchor_type)
        except Exception as exc:
            print(f"[Background] rembg/push failed: {exc}")

    asyncio.create_task(_bg_remove_and_push())

    return {
        "status": "success",
        "transcript": transcript,
        "filename": output_path.name,
        "filename_nobg": None,  # Will be ready asynchronously
        "metrics": metrics,
        "agent": {
            "original_prompt": transcript,
            "final_prompt": final_prompt,
            "style_detected": agent_result.get("style", ""),
            "type": agent_result.get("type", "background"),
            "anchor_type": anchor_type,
            "prop_category": agent_result.get("prop_category", ""),
            "agent_ok": agent_result.get("agent_ok", False),
        },
    }


@app.post("/upload-prop")
async def upload_prop(
    file: UploadFile = File(...),
    anchor_type: str = Form("right_wrist"),
):
    """Upload a custom prop image and broadcast to OBS overlay."""
    content = await file.read()
    filename = f"custom_{int(time())}_{uuid4().hex[:8]}.png"
    save_path = OUTPUT_DIR / filename

    with open(save_path, "wb") as f:
        f.write(content)

    # ── Async background removal + push ──────────────────────────────────
    # Return response IMMEDIATELY. rembg + push runs in background.
    async def _bg_remove_and_push():
        try:
            if anchor_type != "background":
                nobg_path = await run_in_threadpool(remove_background_from_image, save_path)
                from prop_processor import process_prop_image
                grip_x, grip_y = await run_in_threadpool(process_prop_image, str(nobg_path), anchor_type)
                push_filename = nobg_path.name
            else:
                push_filename = save_path.name
                grip_x, grip_y = 0.5, 0.5

            await broadcast_to_overlays({
                "type": "new_prop",
                "filename": push_filename,
                "anchor_type": anchor_type,
                "metrics": {"latency_seconds": 0, "peak_vram_gb": 0},
                "agent": {"anchor_type": anchor_type, "type": "prop" if anchor_type != "background" else "background", "original_prompt": "Custom Upload"},
                "grip_x": grip_x,
                "grip_y": grip_y
            })
            await push_prop_to_local_engine(push_filename, anchor_type)
        except Exception as exc:
            print(f"[Background] upload-prop rembg/push failed: {exc}")

    asyncio.create_task(_bg_remove_and_push())

    return {
        "status": "success",
        "filename": filename,
        "anchor_type": anchor_type,
    }


@app.post("/clear-props")
async def clear_props():
    """Clear all props from the OBS overlay."""
    await broadcast_to_overlays({
        "type": "new_prop",
        "action": "clear",
    })

    # Also tell local engine to clear
    import requests as req
    try:
        await run_in_threadpool(
            req.post, f"{LOCAL_API_URL}/clear", timeout=5
        )
    except Exception:
        pass

    return {"status": "ok", "message": "Props cleared."}


@app.websocket("/ws/anchor")
async def ws_anchor(websocket: WebSocket):
    """WebSocket endpoint for OBS overlay clients.

    Receives anchor preference and broadcasts prop events.
    Tracking data is streamed directly from the Local Engine
    (Port 8001) for minimum latency — this endpoint only handles
    prop change events.
    """
    await websocket.accept()
    overlay_clients.append(websocket)
    print(f"[WebSocket] Overlay client connected. Total: {len(overlay_clients)}")

    try:
        while True:
            # Keep connection alive, receive anchor preference updates
            data = await websocket.receive_text()
            # Client may send anchor preferences — we don't need to process them here
            # since tracking data flows directly from local engine
    except WebSocketDisconnect:
        pass
    finally:
        if websocket in overlay_clients:
            overlay_clients.remove(websocket)
        print(f"[WebSocket] Overlay client disconnected. Total: {len(overlay_clients)}")


# ── Entry Point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=CLOUD_HOST, port=CLOUD_PORT)
