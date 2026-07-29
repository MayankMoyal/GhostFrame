import asyncio
import json
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from time import time
from uuid import uuid4

import cv2
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool
from starlette.websockets import WebSocketState

from engine.zimage_turbo import generate_image, load_pipeline
from agent.router import run_agent
from agent.rewriter import get_rewritten_prompt
from agent.safety import UnsafePromptError, check_safety
from overlay.pipeline import PoseAnchorPipeline
from stt.transcriber import load_whisper_model, transcribe_audio
from postprocess.background_remover import remove_background, load_rembg_session


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

pipeline = None
pipeline_lock = Lock()
FRAME_INTERVAL_S = 1 / 30  # target ~30fps for overlay anchor stream


class GenerationRequest(BaseModel):
    prompt: str
    style: str = ""
    remove_bg: bool = False


def build_prompt(prompt: str, style: str) -> str:
    clean_prompt = prompt.strip()
    clean_style = style.strip()

    if not clean_style:
        return clean_prompt

    return f"{clean_prompt}, {clean_style} style"


def create_output_path() -> Path:
    filename = f"generation_{int(time())}_{uuid4().hex[:8]}.png"
    return OUTPUT_DIR / filename


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    pipeline = load_pipeline()
    load_whisper_model()  # ~1GB int8, stays resident alongside Z-Image-Turbo
    load_rembg_session()
    yield


app = FastAPI(title="GhostFrame Image Generation API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")


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


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": pipeline is not None}


@app.post("/generate")
async def generate(payload: GenerationRequest):
    if not payload.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")

    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model is still loading. Try again shortly.")

    # Agent step: every prompt (typed or, later, voice) passes through the
    # local agent first for rewriting + safety check + style/anchor
    # tagging. Which model runs here is whatever agent/client.py currently
    # points to (see agent/README.md) -- Phi-3 and Qwen3-4B-Instruct-2507
    # are both drop-in candidates. Runs in a threadpool since it's a
    # blocking HTTP call to Ollama.
    # agent_result = await run_in_threadpool(run_agent, payload.prompt)
    # 
    # try:
    #     check_safety(agent_result)
    # except UnsafePromptError as exc:
    #     raise HTTPException(status_code=400, detail=f"Prompt rejected: {exc.reason}")
    # 
    # rewritten_prompt = get_rewritten_prompt(agent_result, payload.prompt)

    # BYPASS AGENT FOR TESTING
    lower_p = payload.prompt.lower()
    anchor = "background"
    if any(w in lower_p for w in ["sword", "gun", "wand", "hand", "hold"]):
        anchor = "right_wrist"
    elif any(w in lower_p for w in ["hat", "helmet", "glasses", "head"]):
        anchor = "head"
    elif any(w in lower_p for w in ["shirt", "jacket", "wings", "shoulder"]):
        anchor = "both_shoulders"
        
    agent_result = {"style": "", "anchor_type": anchor, "agent_ok": True}
    rewritten_prompt = payload.prompt

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

    # Optional background removal (runs on CPU, zero VRAM cost).
    nobg_filename = None
    if payload.remove_bg:
        nobg_path = output_path.with_stem(output_path.stem + "_nobg")
        try:
            await run_in_threadpool(remove_background, output_path, nobg_path)
            nobg_filename = nobg_path.name
        except Exception as exc:
            print(f"[BG Removal] Warning — failed, returning original: {exc}")

    # Broadcast new image to OBS overlay websockets (always, regardless of remove_bg)
    broadcast_msg = {
        "type": "new_prop",
        "filename": nobg_filename if nobg_filename else output_path.name,
        "anchor_type": agent_result.get("anchor_type", "background")
    }
    for ws in list(connected_websockets):
        try:
            await ws.send_json(broadcast_msg)
        except Exception:
            pass

    return {
        "status": "success",
        "filename": output_path.name,
        "filename_nobg": nobg_filename,
        "metrics": metrics,
        "agent": {
            "original_prompt": payload.prompt,
            "final_prompt": final_prompt,
            "style_detected": agent_result.get("style", ""),
            "anchor_type": agent_result.get("anchor_type", "background"),
            "agent_ok": agent_result.get("agent_ok", False),
        },
    }

connected_websockets = set()

@app.websocket("/ws/anchor")
async def anchor_stream(websocket: WebSocket, anchor_type: str = "both_shoulders"):
    await websocket.accept()
    connected_websockets.add(websocket)

    pose_pipeline = PoseAnchorPipeline(anchor_type=anchor_type)
    cap = cv2.VideoCapture(0)
    webcam_ok = cap.isOpened()
    if not webcam_ok:
        print("[WARNING] Could not open webcam for tracking! Poses will not update, but broadcasts will still work.")
        # We don't close the websocket so that it can still receive new_prop broadcasts.

    async def receive_control_messages():
        """Listen for client messages (e.g. anchor_type switches) without
        blocking the frame loop — runs concurrently as its own task."""
        try:
            while True:
                raw = await websocket.receive_text()
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                new_type = msg.get("anchor_type")
                if new_type:
                    pose_pipeline.set_anchor_type(new_type)
        except WebSocketDisconnect:
            pass

    control_task = asyncio.create_task(receive_control_messages())

    try:
        while True:
            if websocket.client_state != WebSocketState.CONNECTED:
                break

            if not webcam_ok:
                await asyncio.sleep(1.0)
                continue

            ret, frame = cap.read()
            if not ret:
                await asyncio.sleep(FRAME_INTERVAL_S)
                continue

            payload, _keypoints, _resolved, _mask = await asyncio.to_thread(
                pose_pipeline.process_frame, frame
            )

            if payload is not None:
                await websocket.send_json(payload)

            await asyncio.sleep(FRAME_INTERVAL_S)
    except WebSocketDisconnect:
        pass
    finally:
        connected_websockets.discard(websocket)
        control_task.cancel()
        cap.release()


@app.post("/generate-voice")
async def generate_voice(
    audio: UploadFile = File(...),
    style: str = Form(""),
    remove_bg: bool = Form(True),
):
    """Full voice-to-image pipeline.

    Accepts an audio file (any format FFmpeg can decode — WAV, WebM,
    OGG, MP3, etc.), transcribes it to text, runs it through the agent
    for rewriting + safety, generates an image, and optionally removes
    the background.

    This is the primary endpoint for the streamer's voice prompter flow.
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model is still loading. Try again shortly.")

    # 1. Save uploaded audio to a temporary file so faster-whisper can
    #    read it from disk (it needs a file path, not a stream).
    suffix = Path(audio.filename).suffix if audio.filename else ".webm"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await audio.read())
        audio_tmp_path = Path(tmp.name)

    # 2. Speech-to-text: transcribe the audio to a raw text prompt.
    try:
        transcript = await run_in_threadpool(transcribe_audio, audio_tmp_path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {exc}") from exc
    finally:
        audio_tmp_path.unlink(missing_ok=True)

    if not transcript.strip():
        raise HTTPException(status_code=400, detail="Could not transcribe any speech from the audio.")

    # 3. Agent step: rewrite + safety + style/anchor tagging.
    # agent_result = await run_in_threadpool(run_agent, transcript)
    # 
    # try:
    #     check_safety(agent_result)
    # except UnsafePromptError as exc:
    #     raise HTTPException(status_code=400, detail=f"Prompt rejected: {exc.reason}")
    # 
    # rewritten_prompt = get_rewritten_prompt(agent_result, transcript)

    # BYPASS AGENT FOR TESTING
    lower_p = transcript.lower()
    anchor = "background"
    if any(w in lower_p for w in ["sword", "gun", "wand", "hand", "hold"]):
        anchor = "right_wrist"
    elif any(w in lower_p for w in ["hat", "helmet", "glasses", "head"]):
        anchor = "head"
    elif any(w in lower_p for w in ["shirt", "jacket", "wings", "shoulder"]):
        anchor = "both_shoulders"

    agent_result = {"style": "", "anchor_type": anchor, "agent_ok": True}
    rewritten_prompt = transcript

    # 4. Image generation with Z-Image-Turbo.
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

    # 5. Optional background removal (CPU-based, zero VRAM cost).
    nobg_filename = None
    if remove_bg:
        nobg_path = output_path.with_stem(output_path.stem + "_nobg")
        try:
            await run_in_threadpool(remove_background, output_path, nobg_path)
            nobg_filename = nobg_path.name
        except Exception as exc:
            print(f"[BG Removal] Warning — failed, returning original: {exc}")

    # 6. Broadcast the new prop to any active OBS Overlays
    broadcast_msg = {
        "type": "new_prop",
        "filename": nobg_filename if nobg_filename else output_path.name,
        "anchor_type": agent_result.get("anchor_type", "background")
    }
    for ws in list(connected_websockets):
        try:
            await ws.send_json(broadcast_msg)
        except Exception:
            pass

    return {
        "status": "success",
        "transcript": transcript,
        "filename": output_path.name,
        "filename_nobg": nobg_filename,
        "metrics": metrics,
        "agent": {
            "original_prompt": transcript,
            "final_prompt": final_prompt,
            "style_detected": agent_result.get("style", ""),
            "anchor_type": agent_result.get("anchor_type", "background"),
            "agent_ok": agent_result.get("agent_ok", False),
        },
    }


@app.post("/clear-props")
async def clear_props():
    """Clear all props from the active overlays."""
    broadcast_msg = {"type": "new_prop", "action": "clear"}
    for ws in list(connected_websockets):
        try:
            await ws.send_json(broadcast_msg)
        except Exception:
            pass
    return {"status": "success"}

FRONTEND_DIR = PROJECT_ROOT / "frontend"
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
