import os
from pathlib import Path
from time import time

import torch
from diffusers import ZImagePipeline

# ── Enable Ampere TF32 & cuDNN Benchmark ────────────────────────────────────
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True


def load_pipeline():
    print("Loading native unquantized Z-Image-Turbo (bfloat16) for RTX A6000...")

    # Load the official full-precision pipeline from Hugging Face
    pipe = ZImagePipeline.from_pretrained(
        "Tongyi-MAI/Z-Image-Turbo",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )

    print("Routing to GPU...")
    pipe.to("cuda")

    # Float32 VAE to prevent NaNs / black images
    pipe.vae.to(torch.float32)

    print("Applying memory optimizations...")
    # Slicing reduces memory pressure for the VAE without sacrificing quality
    pipe.enable_vae_slicing()
    # Skipping model CPU offload to keep weights on GPU 100% of the time for maximum speed.
    
    # Force memory layout to channels_last for faster convolution operations on Tensor Cores
    pipe.unet.to(memory_format=torch.channels_last) if hasattr(pipe, 'unet') else None
    if hasattr(pipe, 'transformer'):
        pipe.transformer.to(memory_format=torch.channels_last)
    pipe.vae.to(memory_format=torch.channels_last)

    # Note: torch.compile removed due to dynamic shape recompilation overhead (45s latency spikes on prompt changes).
    # Z-Image-Turbo will rely on native SDPA / Flash Attention and channels_last for optimization.

    print(f"[DIAGNOSTIC] VRAM allocated: {torch.cuda.memory_allocated() / (1024**3):.2f} GB")

    print("Running throwaway warmup generation...")
    pipe(
        prompt="a simple black square",
        height=576,
        width=1024,
        num_inference_steps=8, # Model is optimized for exactly 8 steps
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
        num_inference_steps=8, # Model is optimized for exactly 8 steps
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
    print("Image successfully saved")

    return {
        "latency_seconds": round(latency, 4),
        "peak_vram_gb": round(peak_vram_use, 4),
    }