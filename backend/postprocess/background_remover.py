"""Background removal using rembg (CPU-based, zero VRAM cost).

Called after image generation is complete, so the GPU is idle at that
point.  Uses rembg's isnet-general-use model for faster processing.
A persistent session is loaded at startup.

Running on CPU keeps VRAM free for Z-Image-Turbo and Whisper.  For a
576×1024 image this typically takes under 2 seconds on CPU, which is
acceptable since it's a one-shot operation after a ~2s image generation.
"""

import time
from pathlib import Path

from PIL import Image
from rembg import remove, new_session

_session = None

def load_rembg_session():
    global _session
    if _session is None:
        _session = new_session("isnet-general-use")


def remove_background(input_path: Path, output_path: Path) -> Path:
    """Remove the background from an image and save with transparency.

    Parameters
    ----------
    input_path : Path
        Path to the input image (any format PIL can read).
    output_path : Path
        Path to save the output RGBA PNG with transparent background.

    Returns
    -------
    Path
        The output_path, for convenience.
    """
    print(f"[BG Removal] Processing: {input_path.name}")
    start_time = time.time()
    
    input_image = Image.open(input_path).convert("RGBA")
    original_size = input_image.size
    
    # Downscale the input image so its longest side is max 512px
    max_side = 512
    if max(original_size) > max_side:
        scale_factor = max_side / max(original_size)
        new_size = (int(original_size[0] * scale_factor), int(original_size[1] * scale_factor))
        downscaled_image = input_image.resize(new_size, Image.Resampling.LANCZOS)
    else:
        downscaled_image = input_image

    # Run rembg on the downscaled image to get an alpha mask
    mask = remove(downscaled_image, session=_session, only_mask=True)
    
    # Upscale the alpha mask back to the original image size using LANCZOS interpolation
    if mask.size != original_size:
        mask = mask.resize(original_size, Image.Resampling.LANCZOS)
    
    # Composite the original full-resolution image with the upscaled mask
    output_image = input_image.copy()
    output_image.putalpha(mask)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_image.save(output_path, format="PNG")
    
    duration = time.time() - start_time
    print(f"[BG Removal] ✅ Saved: {output_path.name} in {duration:.2f}s")
    
    return output_path
