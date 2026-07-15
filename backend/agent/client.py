import json

import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "phi3"

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
    Sends the user's prompt to the local Ollama Phi-3 model and returns
    the parsed JSON response as a dict.

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
