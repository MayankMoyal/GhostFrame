import asyncio
import json
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from time import time
from uuid import uuid4
import os
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool
from starlette.websockets import WebSocketState

MOCK_AI = os.getenv("MOCK_AI", "0") == "1"

if not MOCK_AI:
    from engine.zimage_turbo import generate_image, load_pipeline
    from stt.transcriber import load_whisper_model, transcribe_audio
    from postprocess.background_remover import remove_background, load_rembg_session
    from postprocess.prop_alignment import align_prop
else:
    # Define stubs for type hinting / globals to prevent NameError
    generate_image = load_pipeline = load_whisper_model = transcribe_audio = None
    remove_background = load_rembg_session = align_prop = None

from agent.router import run_agent
from agent.rewriter import get_rewritten_prompt
from agent.safety import UnsafePromptError, check_safety


WEBCAM_INDEX = int(os.getenv("WEBCAM_INDEX", "1"))
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


async def _align_generated_prop(nobg_path: Path, anchor_type: str) -> dict:
    """Run prop_alignment.align_prop() on a background-removed PNG and
    overwrite it with the tight-cropped version.

    Raises HTTPException(500) if the generation had no visible content
    after background removal — that's a bad generation, not something
    to silently broadcast. Returns the grip fields for broadcast_msg.
    """
    rgba = Image.open(nobg_path).convert("RGBA")
    result = await run_in_threadpool(align_prop, rgba, anchor_type)

    if not result["valid"]:
        raise HTTPException(
            status_code=500,
            detail="Generation produced no visible content after background removal.",
        )

    result["image"].save(nobg_path)
    return {
        "grip_x": result["grip_x"],
        "grip_y": result["grip_y"],
        "intrinsic_angle_deg": result["intrinsic_angle_deg"],
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    if not MOCK_AI:
        pipeline = load_pipeline()
        load_whisper_model()  # ~1GB int8, stays resident alongside Z-Image-Turbo
        load_rembg_session()
    else:
        print("[INFO] MOCK_AI is enabled. Skipping heavy model loading.")
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
    return {"status": "ok", "model_loaded": MOCK_AI or pipeline is not None}


@app.post("/generate")
async def generate(payload: GenerationRequest):
    if not payload.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")

    if MOCK_AI:
        import asyncio
        from PIL import Image, ImageDraw
        await asyncio.sleep(1.5)
        
        output_path = create_output_path()
        img = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        draw.rectangle([100, 100, 412, 412], fill=(255, 0, 0, 200))
        nobg_path = output_path.with_stem(output_path.stem + "_nobg")
        img.save(nobg_path)
        
        return {
            "status": "success",
            "filename": nobg_path.name,
            "filename_nobg": nobg_path.name,
            "metrics": {"latency_seconds": 1.5, "peak_vram_gb": 0},
            "agent": {
                "original_prompt": payload.prompt,
                "final_prompt": "A mock red square, cinematic",
                "style_detected": payload.style,
                "anchor_type": "hand_held",
                "agent_ok": True,
            },
        }

    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model is still loading. Try again shortly.")

    # Agent step: every prompt (typed or, later, voice) passes through the
    # local agent first for rewriting + safety check + style/anchor
    # tagging. Which model runs here is whatever agent/client.py currently
    # points to (see agent/README.md) -- Phi-3 and Qwen3-4B-Instruct-2507
    # are both drop-in candidates. Runs in a threadpool since it's a
    # blocking HTTP call to Ollama.
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

    # Optional background removal (runs on CPU, zero VRAM cost).
    nobg_filename = None
    grip_fields = {}
    if payload.remove_bg:
        nobg_path = output_path.with_stem(output_path.stem + "_nobg")
        try:
            await run_in_threadpool(remove_background, output_path, nobg_path)
            nobg_filename = nobg_path.name
        except Exception as exc:
            print(f"[BG Removal] Warning — failed, returning original: {exc}")

        if nobg_filename:
            # Tight-crop to the object's real alpha content and compute an
            # accurate grip point (+ intrinsic tilt for held props) instead
            # of trusting overlay.js's static ZONE_GRIP fraction against a
            # padded canvas. Raises 500 if the crop is empty.
            grip_fields = await _align_generated_prop(nobg_path, agent_result.get("anchor_type", "background"))

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

# WebSocket tracking removed (Cloud GPU cannot access local webcam)


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
    if MOCK_AI:
        # Generate a dummy transparent image using PIL
        import asyncio
        from PIL import Image, ImageDraw
        await asyncio.sleep(1.5)  # simulate processing time
        
        output_path = create_output_path()
        img = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        draw.rectangle([100, 100, 412, 412], fill=(255, 0, 0, 200))
        nobg_path = output_path.with_stem(output_path.stem + "_nobg")
        img.save(nobg_path)
        
        return {
            "status": "success",
            "transcript": "Mock voice prompt testing",
            "filename": nobg_path.name,
            "filename_nobg": nobg_path.name,
            "metrics": {"latency_seconds": 1.5, "peak_vram_gb": 0},
            "agent": {
                "original_prompt": "Mock voice prompt testing",
                "final_prompt": "A mock red square, cinematic",
                "style_detected": style,
                "anchor_type": "hand_held",
                "agent_ok": True,
            },
        }

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
    agent_result = await run_in_threadpool(run_agent, transcript)
    
    try:
        check_safety(agent_result)
    except UnsafePromptError as exc:
        raise HTTPException(status_code=400, detail=f"Prompt rejected: {exc.reason}")
    
    rewritten_prompt = get_rewritten_prompt(agent_result, transcript)

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
    grip_fields = {}
    if remove_bg:
        nobg_path = output_path.with_stem(output_path.stem + "_nobg")
        try:
            await run_in_threadpool(remove_background, output_path, nobg_path)
            nobg_filename = nobg_path.name
        except Exception as exc:
            print(f"[BG Removal] Warning — failed, returning original: {exc}")

        if nobg_filename:
            grip_fields = await _align_generated_prop(nobg_path, agent_result.get("anchor_type", "background"))

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

@app.post("/upload-prop")
async def upload_prop(
    file: UploadFile = File(...),
    anchor_type: str = Form("right_wrist"),
):
    """Upload a custom image to be used as a prop immediately.
    
    The backend will save it, remove its background, align it, 
    and broadcast it to the overlay just like a generated image.
    """
    output_path = create_output_path()
    
    # Read the uploaded file
    file_bytes = await file.read()
    with open(output_path, "wb") as f:
        f.write(file_bytes)

    # Always attempt background removal and alignment on custom uploads
    nobg_path = output_path.with_stem(output_path.stem + "_nobg")
    nobg_filename = None
    grip_fields = {}
    
    if MOCK_AI:
        # Just use the original file as the nobg file
        import shutil
        shutil.copy(output_path, nobg_path)
        nobg_filename = nobg_path.name
        # Provide dummy grip fields
        grip_fields = {"grip_x": 0.5, "grip_y": 0.5, "intrinsic_angle_deg": 0}
    else:
        try:
            await run_in_threadpool(remove_background, output_path, nobg_path)
            nobg_filename = nobg_path.name
        except Exception as exc:
            print(f"[BG Removal] Warning — failed for uploaded prop: {exc}")
            
        if nobg_filename:
            grip_fields = await _align_generated_prop(nobg_path, anchor_type)

    return {
        "filename": output_path.name,
        "filename_nobg": nobg_filename,
        "anchor_type": anchor_type
    }

@app.post("/clear-props")
async def clear_props():
    """Clear all props from the active overlays."""
    return {"status": "success"}

FRONTEND_DIR = PROJECT_ROOT / "frontend"
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
