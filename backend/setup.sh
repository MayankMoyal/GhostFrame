#!/usr/bin/env bash
set -e  # stop on first error

echo "╔══════════════════════════════════════════════════════╗"
echo "║       Ghost Frame — Complete Backend Setup           ║"
echo "║  One script to install everything. Run once.         ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

# ── 1. System Dependencies ─────────────────────────────────
echo "=== [1/8] System Dependencies ==="
apt-get update -qq && apt-get install -y -qq ffmpeg > /dev/null 2>&1 || true
echo "    ✓ ffmpeg checked"

# ── 2. Python Environment ──────────────────────────────────
echo "=== [2/8] Python Environment ==="
if [ -d "venv" ]; then
    echo "    venv already exists, activating..."
    source venv/bin/activate
else
    python3 -m venv venv
    source venv/bin/activate
    echo "    ✓ venv created and activated"
fi

# ── 3. PyTorch with CUDA ───────────────────────────────────
echo "=== [3/8] PyTorch with CUDA ==="
if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "    ✓ PyTorch already installed with CUDA support"
else
    echo "    Installing PyTorch (cu121)..."
    pip install -q torch --index-url https://download.pytorch.org/whl/cu121
    echo "    ✓ PyTorch installed"
fi

# ── 4. Python Packages ─────────────────────────────────────
echo "=== [4/8] Python Packages ==="
pip install -q -r requirements.txt
pip install -q python-multipart "huggingface_hub[cli]"
# Pin numpy for scikit-learn compatibility
pip install -q "numpy<2" "scipy<1.13" "scikit-learn<1.4" 2>/dev/null || true
echo "    ✓ All Python packages installed"

# ── 5. Download Z-Image-Turbo GGUF Model (~5GB) ───────────
echo "=== [5/8] Download Z-Image-Turbo Model ==="
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

# ── 6. Install Ollama ──────────────────────────────────────
echo "=== [6/8] Install Ollama ==="
if command -v ollama &> /dev/null; then
    echo "    ✓ Ollama already installed"
else
    echo "    Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
    echo "    ✓ Ollama installed"
fi

# ── 7. Start Ollama & Pull Agent Model ─────────────────────
echo "=== [7/8] Pull Agent Model (Command-R 7B) ==="
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

# ── 8. Verify Setup ───────────────────────────────────────
echo "=== [8/8] Verify Setup ==="
python -c "import torch; print(f'    PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import fastapi; print(f'    FastAPI {fastapi.__version__}')"
python -c "import diffusers; print(f'    Diffusers {diffusers.__version__}')"
ollama list | head -5

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║              ✓ Setup Complete!                       ║"
echo "╠══════════════════════════════════════════════════════╣"
echo "║  Start the server:                                   ║"
echo "║    source venv/bin/activate && python main.py         ║"
echo "║                                                      ║"
echo "║  Open the dashboard:                                 ║"
echo "║    http://localhost:8000/app/                         ║"
echo "║                                                      ║"
echo "║  Agent model: Command-R 7B (command-r:7b)            ║"
echo "║  Image model: Z-Image-Turbo Q5_K_M                   ║"
echo "╚══════════════════════════════════════════════════════╝"
