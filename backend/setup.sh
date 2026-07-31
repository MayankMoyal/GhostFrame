#!/usr/bin/env bash
set -e  # stop on first error

echo "=== Ghost Stream — Cloud Backend Setup ==="
echo ""

echo "=== 1. Python venv ==="
python3 -m venv venv
source venv/bin/activate

echo "=== 2. PyTorch with CUDA ==="
echo "    Checking your CUDA version..."
nvidia-smi | grep "CUDA Version"
echo "    Defaulting to cu121 below -- edit this line if nvidia-smi showed a different version"
pip install torch --index-url https://download.pytorch.org/whl/cu121

echo "=== 3. Python requirements (fastapi, diffusers, whisper, rembg, etc.) ==="
pip install -r requirements.txt

echo "=== 4. bitsandbytes GPU sanity check ==="
python -m bitsandbytes

echo "=== 5. Hugging Face CLI ==="
pip install -U "huggingface_hub[cli]"

echo "=== 6. Download GGUF transformer (Q5_K_M, ~5GB) ==="
mkdir -p models/gguf
hf download jayn7/Z-Image-Turbo-GGUF z_image_turbo-Q5_K_M.gguf --local-dir models/gguf
ls -lh models/gguf/

echo "=== 7. Install Ollama (system binary, not pip) ==="
curl -fsSL https://ollama.com/install.sh | sh

echo "=== 8. Pull the agent model ==="
ollama serve > /tmp/ollama.log 2>&1 &
sleep 3
ollama pull qwen3.5:4b

echo "=== 9. Verify Whisper will auto-download on first use ==="
echo "    (faster-whisper downloads the model automatically on first transcription)"
echo "    No manual download needed."

echo ""
echo "=== Setup complete ==="
echo ""
echo "Start the server with:"
echo "  source venv/bin/activate && python main.py"
echo ""
echo "Open the Web UI at:"
echo "  http://localhost:8000/app/"
