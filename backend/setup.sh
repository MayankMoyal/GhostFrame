#!/usr/bin/env bash
set -e  # stop on first error

echo "╔══════════════════════════════════════════════════════╗"
echo "║       Ghost Frame — Complete Backend Setup           ║"
echo "║  One script to install everything. Run once.         ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

# ── 1. System Dependencies ─────────────────────────────────
echo "=== [1/7] System Dependencies ==="
apt-get update -qq && apt-get install -y -qq ffmpeg > /dev/null 2>&1 || true
echo "    ✓ ffmpeg checked"

# ── 2. PyTorch with CUDA ───────────────────────────────────
echo "=== [2/7] PyTorch with CUDA ==="
if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "    ✓ PyTorch already installed with CUDA support"
else
    echo "    Installing PyTorch (cu121)..."
    pip install -q torch --index-url https://download.pytorch.org/whl/cu121
    echo "    ✓ PyTorch installed"
fi

# ── 3. Python Packages ─────────────────────────────────────
echo "=== [3/7] Python Packages ==="
pip install -q -r requirements.txt
pip install -q python-multipart "huggingface_hub[cli]"
# Pin numpy for scikit-learn compatibility
pip install -q "numpy<2" "scipy<1.13" "scikit-learn<1.4" 2>/dev/null || true
echo "    ✓ All Python packages installed"

# ── 4. Download Z-Image-Turbo GGUF Model (~5GB) ───────────
echo "=== [4/7] Download Z-Image-Turbo Model ==="
mkdir -p models/gguf
if [ -f "models/gguf/z_image_turbo-Q5_K_M.gguf" ]; then
    echo "    ✓ Model already downloaded"
else
    echo "    Downloading Q5_K_M (~5GB)..."
    # Try new 'hf' CLI first, fall back to 'huggingface-cli'
    hf download jayn7/Z-Image-Turbo-GGUF z_image_turbo-Q5_K_M.gguf --local-dir models/gguf 2>/dev/null \
        || huggingface-cli download jayn7/Z-Image-Turbo-GGUF z_image_turbo-Q5_K_M.gguf --local-dir models/gguf
    echo "    ✓ Model downloaded"
fi
ls -lh models/gguf/

# ── 5. Install Ollama ──────────────────────────────────────
echo "=== [5/7] Install Ollama ==="
if command -v ollama &> /dev/null; then
    echo "    ✓ Ollama already installed"
else
    echo "    Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
    echo "    ✓ Ollama installed"
fi

# ── 6. Start Ollama & Pull Agent Model ─────────────────────
echo "=== [6/7] Pull Agent Model (Command-R 7B) ==="
# Start Ollama server in background if not already running
if ! pgrep -x "ollama" > /dev/null; then
    ollama serve > /tmp/ollama.log 2>&1 &
    sleep 3
    echo "    ✓ Ollama server started"
else
    echo "    ✓ Ollama server already running"
fi

# Pull Command-R 7B (the active agent model)
if ollama list | grep -q "command-r:7b"; then
    echo "    ✓ command-r:7b already pulled"
else
    echo "    Pulling command-r:7b (~4.5GB)..."
    ollama pull command-r:7b
    echo "    ✓ command-r:7b ready"
fi

# ── 7. Verify Setup ───────────────────────────────────────
echo "=== [7/7] Verify Setup ==="
python3 -c "import torch; print(f'    PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python3 -c "import fastapi; print(f'    FastAPI {fastapi.__version__}')"
python3 -c "import diffusers; print(f'    Diffusers {diffusers.__version__}')"
ollama list | head -5

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║              ✓ Setup Complete!                       ║"
echo "╠══════════════════════════════════════════════════════╣"
echo "║  Start the server:                                   ║"
echo "║    python3 main.py                                   ║"
echo "║                                                      ║"
echo "║  Open the dashboard:                                 ║"
echo "║    http://localhost:8000/app/                         ║"
echo "║                                                      ║"
echo "║  Agent model: Command-R 7B (command-r:7b)            ║"
echo "║  Image model: Z-Image-Turbo Q5_K_M                   ║"
echo "╚══════════════════════════════════════════════════════╝"
