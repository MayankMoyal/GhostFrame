import os
import json
import subprocess
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI(title="GhostFrame Engineering Engine")

# Enable CORS cross-origin requests so Swati's frontend files can run safely from any local server context
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Application absolute pathing matrix mapping straight to your setup
BASE_DIR = "/teamspace/studios/this_studio/GhostFrame"
IDEOGRAM_DIR = os.path.join(BASE_DIR, "ideogram4")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# Ensure structural output paths exist safely
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Mount directory to automatically turn files into clean web-accessible URLs
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

class GenerationRequest(BaseModel):
    prompt: str
    style: str

def compile_asymmetric_prompt(user_prompt: str, style_modifier: str) -> str:
    """
    Formulates text vectors directly into the exact nested JSON architecture 
    required for Ideogram 4.0 Inference while applying frontend visual styles.
    """
    payload = {
        "high_level_description": f"{user_prompt}, {style_modifier} style",
        "compositional_deconstruction": {
            "background": f"{user_prompt}, high quality streaming asset, background frame layer",
            "elements": [
                {"type": "obj", "desc": f"stylized elements matched to {style_modifier} theme"}
            ]
        }
    }
    return json.dumps(payload)

@app.post("/generate")
async def process_generation(payload: GenerationRequest):
    if not payload.prompt.strip():
        raise HTTPException(status_code=400, detail="Null prompt telemetry received.")
    
    target_filename = "stream_output.png"
    target_filepath = os.path.join(OUTPUT_DIR, target_filename)
    
    # Structure the prompt parameters according to backend requirements
    final_prompt = compile_asymmetric_prompt(payload.prompt, payload.style)
    
    # Map the exact parameters that successfully generated your proof-of-concept image
    execution_cmd = [
        "python", "run_inference.py",
        "--prompt", final_prompt,
        "--no-magic-prompt",
        "--output", target_filepath,
        "--quantization", "nf4",
        "--sampler-preset", "V4_DEFAULT_20",
        "--seed", "42"
    ]
    
    print(f"[ENGINE] Launching Ideogram 4.0 Subprocess pipeline in directory: {IDEOGRAM_DIR}")
    
    try:
        # Run the conditional-only inference pipeline directly on your cloud hardware
        result = subprocess.run(
            execution_cmd,
            cwd=IDEOGRAM_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        
        if not os.path.exists(target_filepath):
            raise HTTPException(status_code=500, detail="Inference finished, but payload missing from filesystem.")
        
        # Calculate dynamic file modification Unix timestamp to bypass browser cache issues
        cache_buster = int(os.path.getmtime(target_filepath))
        
        return {
            "status": "success",
            "image_url": f"http://localhost:8000/outputs/{target_filename}?v={cache_buster}",
            "metadata": {
                "scene": "Stream Asset Container",
                "lighting": "Studio Dynamic Array"
            }
        }
        
    except subprocess.CalledProcessError as err:
        print(f"[ENGINE ERROR] Standard Error log from inference run:\n{err.stderr}")
        raise HTTPException(status_code=500, detail=f"Inference Engine Crash: {err.stderr[:250]}")
    except Exception as general_err:
        raise HTTPException(status_code=500, detail=str(general_err))

if __name__ == "__main__":
    import uvicorn
    # Serves the engine over port 8000
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 
