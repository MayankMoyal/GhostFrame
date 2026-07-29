import os
from pathlib import Path
from time import time

import torch
from diffusers import ZImagePipeline, ZImageTransformer2DModel, GGUFQuantizationConfig
from transformers import AutoModel, BitsAndBytesConfig as TransformersBitsAndBytesConfig

# Local path preferred over streaming from HF on every restart. Download once:
#   hf download jayn7/Z-Image-Turbo-GGUF z_image_turbo-Q5_K_M.gguf \
#       --local-dir ./models/gguf
# and point this at the resulting file (override via env var if you want).
GGUF_TRANSFORMER_PATH = os.environ.get(
    "ZIMAGE_GGUF_PATH",
    "./models/gguf/z_image_turbo-Q5_K_M.gguf",
)


def load_pipeline():
    print("Loading & INT8-quantizing text encoder (Qwen3-4B)...")
    # bitsandbytes INT8 (LLM.int8()), not NF4 -- roughly ~4GB resident vs
    # NF4's ~2-2.5GB, but closer to fp16 quality. Still stays resident on
    # GPU with fused kernels (same reasoning as NF4: no CPU<->GPU transfer
    # per call, no per-layer dequant tax like GGUF). T4 (compute capability
    # 7.5) meets bitsandbytes' minimum for LLM.int8() (needs >= 7.5).
    text_encoder_quant_config = TransformersBitsAndBytesConfig(
        load_in_8bit=True,
    )
    text_encoder = AutoModel.from_pretrained(
        "Tongyi-MAI/Z-Image-Turbo",
        subfolder="text_encoder",
        quantization_config=text_encoder_quant_config,
        torch_dtype=torch.float16,
    )

    print(f"Loading Z-Image-Turbo GGUF Q5_K_M transformer from {GGUF_TRANSFORMER_PATH}...")
    transformer = ZImageTransformer2DModel.from_single_file(
        GGUF_TRANSFORMER_PATH,
        quantization_config=GGUFQuantizationConfig(compute_dtype=torch.float16),
        torch_dtype=torch.float16,
    )

    print("Loading Z-Image-Turbo pipeline with GGUF transformer + INT8 text encoder...into memory")
    pipe = ZImagePipeline.from_pretrained(
        "Tongyi-MAI/Z-Image-Turbo",
        transformer=transformer,
        text_encoder=text_encoder,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )

    type_counts = {}
    for name, param in pipe.transformer.named_parameters():
        t = type(param.data).__name__
        type_counts[t] = type_counts.get(t, 0) + 1
    print(f"[DIAGNOSTIC] transformer parameter type breakdown: {type_counts}")

    text_encoder_type_counts = {}
    for name, param in pipe.text_encoder.named_parameters():
        t = type(param.data).__name__
        text_encoder_type_counts[t] = text_encoder_type_counts.get(t, 0) + 1
    print(f"[DIAGNOSTIC] text_encoder parameter type breakdown: {text_encoder_type_counts}")

    # pipe.transformer.set_attention_backend("flash")  # disabled - flash-attn not installed

    print("routing the model to GPU")
    # NOTE: bitsandbytes-quantized modules load pre-placed on GPU during
    # from_pretrained; calling .to("cuda") on the whole pipe is safe (it's
    # effectively a no-op for the already-placed text encoder) but if you
    # ever see a device-mismatch error here, that's why -- don't try to
    # .to("cpu") a bnb 4-bit module later, it isn't supported the way a
    # normal fp16 module is (this is why we're not manually offloading it).
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
