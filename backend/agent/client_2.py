"""
Agent client — MODEL: qwen3:4b-instruct-2507 (Alibaba Qwen3, 4B, ~2.5GB via Ollama)

ACTIVE from 2026-03-02 until superseded by Qwen3.5-0.8B (see client.py).

Was the active agent model. Kept as client_2.py for A/B rollback.
Original candidate note: smaller footprint
(~3.2GB actual VRAM per `ollama ps`, vs Phi-3's ~3.8-3.9GB) and a newer
model generation.

Verified standalone via direct Ollama calls (agent alone, Z-Image
pipeline not loaded):
  - JSON output was well-formed and complete on both test prompts, with
    richer/more detailed rewrites than Phi-3 produced on the same kind
    of prompt.
  - style/anchor_type fields came back contextually correct (e.g.
    detected "cyberpunk" style from a cyberpunk prompt, not just a
    default).
  - Warm-state latency: ~1.4s total, ~1.0s eval for ~80 output tokens.
    (First call after a fresh pull took ~80s -- that's one-time model
    load cost, not representative of steady-state latency.)
  - `ollama ps` showed 100% GPU with no CPU split, standalone.

NOT YET verified: behavior with the Z-Image pipeline also loaded and
holding ~17GB VRAM at the same time -- that's the real test, since
that's the scenario that caused Phi-3 to split across CPU/GPU
originally. Re-check `ollama ps` after running this through the actual
/generate endpoint before trusting this in production.

To use this model: rename this file to client.py (agent/router.py
imports `from agent.client import call_agent`, so the active model is
whichever file is currently named client.py).
"""

import json

import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen3:4b-instruct-2507-q4_K_M"

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
