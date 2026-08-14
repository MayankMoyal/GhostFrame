"""
Agent client — MODEL: command-r7b (Cohere Command-R, 7B params, Q5_K_M via Ollama)

Purpose-built by Cohere for agentic tasks, tool-calling, and structured JSON output.
This model excels at:
  - Strict JSON format compliance (no conversational filler)
  - Intent classification (background vs prop)
  - Granular body-part categorization for props
  - Prompt rewriting for image generation models
  - Safety judgment

VRAM: ~4.5GB Q4 — fits comfortably alongside Z-Image-Turbo (~10GB) on a 16GB T4.
Speed: ~25 tokens/sec on T4 — well under the 2-second response target.
"""

import json

import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "ghost-agent"


def call_agent(user_prompt: str, timeout: int = 30) -> dict:
    """
    Sends the user's prompt to the pre-baked Ollama ghost-agent (Command-R 7B)
    and returns the parsed JSON response as a dict.

    The response contains:
      - rewritten_prompt: enhanced prompt for image generation
      - is_safe: safety flag
      - safety_reason: explanation if unsafe
      - style: visual style tag
      - type: "background" or "prop"
      - anchor_type: where the item attaches on the body
      - prop_category: same as anchor_type for props, "" for backgrounds

    Raises:
      requests.RequestException on network/connection failure
      ValueError if the model did not return valid JSON
    """
    full_prompt = f'User prompt: "{user_prompt}"'

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": full_prompt,
            "format": "json",
            "stream": False,
            "keep_alive": -1,
        },
        timeout=timeout,
    )
    response.raise_for_status()

    raw_text = response.json().get("response", "")

    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Agent did not return valid JSON: {raw_text!r}") from exc

    # Normalize: ensure anchor_type matches prop_category for downstream compatibility
    if parsed.get("type") == "background":
        parsed["anchor_type"] = "background"
        parsed["prop_category"] = ""
    elif parsed.get("type") == "prop":
        # If prop_category is set but anchor_type isn't, sync them
        if parsed.get("prop_category") and not parsed.get("anchor_type"):
            parsed["anchor_type"] = parsed["prop_category"]
        elif parsed.get("anchor_type") and not parsed.get("prop_category"):
            parsed["prop_category"] = parsed["anchor_type"]

    return parsed
