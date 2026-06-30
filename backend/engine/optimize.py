import torch

def offload_encoder_to_cpu(text_encoder):
    """
    After text encoding is done, move encoder OFF the GPU to free VRAM.
    
    Why: Qwen encoder is ~8GB. Once it converts our text prompt to numbers,
    we don't need it on GPU anymore. So we move it to RAM (CPU) to free
    space for the image generation model.
    """
    print("Moving text encoder to CPU...")
    text_encoder.to("cpu")
    torch.cuda.empty_cache()  # Clear any leftover GPU memory
    print("✅ Encoder offloaded! VRAM freed.")

def load_encoder_to_gpu(text_encoder):
    """
    Move encoder back to GPU when we need to encode a new prompt.
    """
    print("Moving text encoder back to GPU...")
    text_encoder.to("cuda")
    print("✅ Encoder on GPU, ready to encode.")
