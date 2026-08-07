import os
from pathlib import Path
from time import time

import torch
from diffusers import ZImagePipeline, ZImageTransformer2DModel, GGUFQuantizationConfig
from transformers import AutoModel, BitsAndBytesConfig as TransformersBitsAndBytesConfig

_BACKEND_DIR = Path(__file__).resolve().parent.parent

GGUF_TRANSFORMER_PATH = os.environ.get(
    "ZIMAGE_GGUF_PATH",
    str(_BACKEND_DIR / "models" / "gguf" / "z_image_turbo-Q5_K_M.gguf"),
)

def load_pipeline():
    # Resolve to absolute path (diffusers requires this for local GGUF files)
    gguf_path = str(Path(GGUF_TRANSFORMER_PATH).resolve())

    if not Path(gguf_path).exists():
        raise FileNotFoundError(
            f"GGUF transformer not found at: {gguf_path}\n"
            f"Download it first with:  bash setup.sh"
        )

    print("Loading text encoder (Qwen3-4B) in 4-bit (NF4)...")
    # NF4 4-bit is faster and more stable than INT8. Using bfloat16 compute dtype.
    text_encoder_quant_config = TransformersBitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4"
    )
    text_encoder = AutoModel.from_pretrained(
        "Tongyi-MAI/Z-Image-Turbo",
        subfolder="text_encoder",
        quantization_config=text_encoder_quant_config,
        torch_dtype=torch.bfloat16,
    )

    print(f"Loading Z-Image-Turbo GGUF transformer from {gguf_path}...")
    # compute_dtype=bfloat16 prevents the NaNs that cause black images
    transformer = ZImageTransformer2DModel.from_single_file(
        gguf_path,
        quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
        torch_dtype=torch.bfloat16,
    )

    print("Loading Z-Image-Turbo pipeline into memory...")
    pipe = ZImagePipeline.from_pretrained(
        "Tongyi-MAI/Z-Image-Turbo",
        transformer=transformer,
        text_encoder=text_encoder,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )

    print("Routing the model to GPU...")
    pipe.to("cuda")
    
    # CRITICAL FIX FOR BLACK IMAGES:
    # The VAE must decode in float32 to prevent NaNs in the final pixels
    pipe.vae.to(torch.float32)

    print(f"[DIAGNOSTIC] VRAM allocated right after loading to GPU: {torch.cuda.memory_allocated() / (1024**3):.2f} GB")

    print("Running throwaway warmup generation...")
    pipe(
        prompt="a simple black square",
        height=576,
        width=1024,
        num_inference_steps=6,
        guidance_scale=0.0,
    )

    return pipe


def generate_image(pipe, prompt: str, output_path: Path) -> dict:
    print(f"generating the image for the prompt : {prompt}")

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start_time = time()

    image = pipe(
        prompt=prompt,
        height=576,
        width=1024,
        num_inference_steps=9,
        guidance_scale=0.0,
    ).images[0]

    torch.cuda.synchronize()
    end_time = time()
    latency = end_time - start_time
    peak_vram_use = torch.cuda.max_memory_allocated() / (1024**3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)

    print("\n=== Z IMAGE TURBO BASELINE METRICS ===")
    print(f"Latency = {latency:.4f} seconds")
    print(f"Peak VRAM USE : {peak_vram_use:.4f} GB")
    print("Image sucessfully saved")

    return {
        "latency_seconds": round(latency, 4),
        "peak_vram_gb": round(peak_vram_use, 4),
    }
