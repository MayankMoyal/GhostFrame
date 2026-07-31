"""
Agent client — MODEL: qwen3.5:0.8b (Alibaba Qwen3.5, 873M params, Q8_0, ~1GB via Ollama)

Was the active agent model. Superseded by qwen3.5:4b (see client.py).
Kept here as client_3.py for A/B rollback.
Replaces qwen3:4b-instruct-2507 (kept as client_2.py) as the active agent
model, for VRAM headroom on a 16GB T4 (see engine/zimage_turbo.py's move
to GGUF Q8) and lower per-request latency.

IMPORTANT — Qwen3.5 is a hybrid thinking/non-thinking model and defaults
to thinking mode ON in Ollama. Left alone, that means every call emits a
chain-of-thought block before the JSON answer, which would blow the
latency budget this switch is supposed to fix -- possibly ending up
SLOWER than qwen3:4b despite the 5x smaller parameter count. `"think":
false` below turns that off. If you ever see latency regress after this
switch, check that field first before assuming the model itself is slow.

UNVERIFIED, READ BEFORE TRUSTING IN PRODUCTION:
This is a ~5x parameter drop from qwen3:4b (4B -> 0.8B), not a lateral
move like Phi-3 -> Qwen3-4B was. `agent/safety.py`'s check_safety() is
the ONLY safety gate in this pipeline -- it trusts is_safe verbatim, and
router.py fails OPEN (is_safe=True) if the agent call errors or returns
malformed JSON. Small models are meaningfully weaker at nuanced safety
judgment than 4B-class models, and are also more likely to occasionally
break JSON formatting under load, which fails open here too. Before
shipping this as the live model:
  1. Run both client.py (this) and client_2.py (qwen3:4b) against the
     same batch of prompts, including deliberately borderline/adversarial
     ones (not just obviously-fine or obviously-unsafe), and diff the
     is_safe / safety_reason outputs.
  2. If 0.8B's safety judgment is noticeably worse, consider adding a
     lightweight keyword/regex pre-filter in safety.py as defense in
     depth, independent of whichever LLM is active -- right now there is
     no fallback layer if the LLM gets it wrong.
  3. Also spot-check rewritten_prompt/style/anchor_type richness -- these
     are lower stakes than is_safe but will likely be more generic at
     this size.

To roll back: rename client_2.py to client.py.
"""

import json

import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen3.5:0.8b"

SYSTEM_PROMPT = """You are an assistant embedded in a live AI image-generation pipeline for a livestreamer's OBS background.
Given the user's raw prompt, respond with ONLY a single JSON object, no other text, using exactly these keys:

{
  "rewritten_prompt": "an improved, more detailed version of the prompt for an image generation model, keeping the original subject/intent",
  "is_safe": true or false,
  "safety_reason": "short reason if is_safe is false, otherwise empty string",
  "style": "a short 1-2 word visual style tag, e.g. cyberpunk, fantasy, cozy, minimal",
  "anchor_type": "one of: background, prop_in_hand, head_shoulder_accessory, ambient_floating"
}

Rules:
- Set is_safe to false ONLY for prompts requesting sexual content, real identifiable people, hate symbols, or graphic violence/gore.
- Default anchor_type to "background" unless the prompt clearly describes an object meant to be held or worn.
- Output valid JSON only. No markdown, no code fences, no explanation text."""


def call_agent(user_prompt: str, timeout: int = 15) -> dict:
    """
    Sends the user's prompt to the local Ollama model and returns the
    parsed JSON response as a dict.

    Raises requests.RequestException on network/connection failure, and
    ValueError if the model did not return valid JSON. Callers should
    catch these and fall back gracefully (see agent/router.py).
    """
    full_prompt = f'{SYSTEM_PROMPT}\n\nUser prompt: "{user_prompt}"'

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": full_prompt,
            "format": "json",
            "stream": False,
            "think": False,  # see module docstring -- do not remove
        },
        timeout=timeout,
    )
    response.raise_for_status()

    raw_text = response.json().get("response", "")

    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Agent did not return valid JSON: {raw_text!r}") from exc

    return parsed
