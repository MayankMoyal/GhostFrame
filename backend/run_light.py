import sys
import time
from unittest.mock import MagicMock
from PIL import Image, ImageDraw

print("==================================================")
print("🚀 STARTING IN LIGHTWEIGHT MODE (DUMMY IMAGE GEN)")
print("🚫 Bypassing Z-Image-Turbo (using dummy images)")
print("==================================================")

# (STT is no longer mocked here so you can test audio transcription)

def fake_generate_image(pipeline, prompt, output_path):
    import shutil
    from pathlib import Path
    
    print(f"🎨 [DUMMY] Generating fake image for prompt: '{prompt}'")
    
    # Check if you've provided a real image to use as the dummy
    dummy_source = Path(__file__).parent / "dummy.png"
    if not dummy_source.exists():
        dummy_source = Path("dummy.jpg")
        
    if dummy_source.exists():
        print(f"   -> Found your existing image '{dummy_source}', using it!")
        shutil.copy(dummy_source, output_path)
    else:
        print("   -> No 'dummy.png' found, generating a solid color block instead.")
        time.sleep(1)  # Simulate a tiny bit of processing time
        
        # Create a solid color dummy image
        img = Image.new('RGB', (576, 1024), color=(45, 45, 45))
        d = ImageDraw.Draw(img)
        d.text((50, 500), f"DUMMY IMAGE\n{prompt[:30]}...", fill=(255, 255, 0))
        img.save(output_path)
    
    # Return fake metrics just like the real engine does
    return {"latency_seconds": 1.0, "peak_vram_gb": 0.0}

mock_engine = MagicMock()
mock_engine.load_pipeline.return_value = "DUMMY_PIPELINE_LOADED"
mock_engine.generate_image = fake_generate_image
sys.modules['engine.zimage_turbo'] = mock_engine

# 2. Now we can safely import your FastAPI app
import main
import uvicorn

if __name__ == "__main__":
    # 3. Start the server normally!
    uvicorn.run(main.app, host="0.0.0.0", port=8000)
