# Ghost Frame — Agent Module

## Active Model
- **`client.py`** → **Command-R 7B** (`command-r7b` via Ollama)
  - Purpose-built for agentic JSON tasks
  - ~4.5GB VRAM (Q5)
  - Classifies: style, type (prop/background), anchor_type, prop_category
  - Rewrites prompts for optimal image generation

## Archived Models
- **`client_qwen_3.py`** → Qwen 3.5 4B (`qwen3.5:4b`)
  - Previous active model. Smaller but weaker classification.
  - To roll back: copy `client_qwen_3.py` → `client.py`

## How It Works
1. `router.py` calls `client.py` → `call_agent(prompt)`
2. The model returns structured JSON with classification + rewritten prompt
3. `router.py` validates the response and fills defaults for missing fields
4. `safety.py` checks the `is_safe` flag
5. `main.py` uses the result to route to background or prop rendering
