from pathlib import Path
from time import time

import torch
from diffusers import ZImagePipeline
from torchao.quantization import quantize_, Int8WeightOnlyConfig


def load_pipeline():
    print("Loading Z-Image-Turbo model...into memory")
    pipe = ZImagePipeline.from_pretrained(
        "Tongyi-MAI/Z-Image-Turbo",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )

    print("Applying INT8 weight-only quantization...")
    quantize_(pipe.transformer, Int8WeightOnlyConfig())

    type_counts = {}
    for name, param in pipe.transformer.named_parameters():
        t = type(param.data).__name__
        type_counts[t] = type_counts.get(t, 0) + 1
    print(f"[DIAGNOSTIC] transformer parameter type breakdown: {type_counts}")

    # pipe.transformer.set_attention_backend("flash")  # disabled - flash-attn not installed

    print("routing the model to GPU")
    pipe.to("cuda")

    print(f"[DIAGNOSTIC] VRAM allocated right after loading to GPU: {torch.cuda.memory_allocated() / (1024**3):.2f} GB")

    # DISABLED for now (Option A) -- recompiling on every request because
    # the Phi-3/Qwen agent rewrites each prompt to a different length,
    # which changes the text-encoder output shape feeding the transformer's
    # cross-attention -> shape mismatch -> full recompile per call
    # (measured 5-10 min/generation with this on). Revisit as an isolated
    # experiment (Option B: pad/truncate the tokenizer to a fixed length
    # so the shape never changes, then re-enable).
    # print("Compiling transformer with torch.compile (this makes the FIRST run slow, one-time cost)...")
    # pipe.transformer = torch.compile(pipe.transformer, mode="default")

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
