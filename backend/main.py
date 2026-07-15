from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from time import time
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from engine.zimage_turbo import generate_image, load_pipeline
from agent.router import run_agent
from agent.rewriter import get_rewritten_prompt
from agent.safety import UnsafePromptError, check_safety


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

pipeline = None
pipeline_lock = Lock()


class GenerationRequest(BaseModel):
    prompt: str
    style: str = ""


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
    # local Phi-3 agent first for rewriting + safety check + style/anchor
    # tagging. Runs in a threadpool since it's a blocking HTTP call to Ollama.
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

    return {
        "status": "success",
        "filename": output_path.name,
        "metrics": metrics,
        "agent": {
            "original_prompt": payload.prompt,
            "final_prompt": final_prompt,
            "style_detected": agent_result.get("style", ""),
            "anchor_type": agent_result.get("anchor_type", "background"),
            "agent_ok": agent_result.get("agent_ok", False),
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
