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

    # ── END-TO-END INFERENCE OPTIMIZATION ─────────────────────────────────
    #
    # WHY torch.compile DOES NOT WORK WITH THIS PIPELINE:
    # ZImagePipeline._encode_prompt strips padding from token embeddings:
    #     embeddings_list.append(prompt_embeds[i][prompt_masks[i]])
    # This produces variable-length tensors (e.g. [23, 4096] vs [47, 4096])
    # that are passed as a Python list into the transformer. torch.compile
    # treats each unique shape as a new graph, causing a FULL recompilation
    # (30-60s) on every single new prompt. dynamic=True cannot help because
    # the inputs are lists of differently-shaped tensors, not padded batches.
    # Additionally, GGUF quantization + bitsandbytes NF4 layers use custom
    # CUDA kernels that cause graph breaks, making fullgraph=True impossible.
    #
    # WHAT WE DO INSTEAD (actual measured speedups):
    # 1. GGUF Q5_K_M quantization on transformer (already loaded above)
    # 2. NF4 4-bit quantization on text encoder (already loaded above)  
    # 3. BFloat16 compute dtype throughout the pipeline
    # 4. CUDA-level hardware optimizations below:

    # Enable cuDNN auto-tuner: benchmarks multiple convolution algorithms
    # and caches the fastest one for each input size. Since our image
    # dimensions are fixed (576x1024), this gives a permanent speedup
    # after the first generation.
    torch.backends.cudnn.benchmark = True

    # Enable TF32 on Ampere+ GPUs: uses Tensor Cores for 32-bit ops
    # with ~3x throughput at negligible precision loss (mantissa truncated
    # from 23 to 10 bits, which is invisible for image generation).
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Enable scaled_dot_product_attention (SDPA) optimizations.
    # This is built into PyTorch 2.x and automatically selects the fastest
    # attention backend (FlashAttention-2, Memory-Efficient, or Math).
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)

    print("[Optimization] Applied: GGUF Q5_K_M + NF4 + BF16 + cuDNN benchmark + TF32 + FlashSDP")
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
    print("Image successfully saved")

    return {
        "latency_seconds": round(latency, 4),
        "peak_vram_gb": round(peak_vram_use, 4),
    }
