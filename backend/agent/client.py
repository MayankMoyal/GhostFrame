import json

import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen3:4b"

SYSTEM_PROMPT = """You are an assistant embedded in a live AI image-generation pipeline for a livestreamer's OBS background.
Given the user's raw prompt, respond with ONLY a single JSON object, no other text, using exactly these keys:

{
  "rewritten_prompt": "an improved, more detailed version of the prompt for an image generation model, keeping the original subject/intent",
  "is_safe": true or false,
  "safety_reason": "short reason if is_safe is false, otherwise empty string",
  "style": "a short 1-2 word visual style tag, e.g. cyberpunk, fantasy, cozy, minimal",
  "anchor_type": "one of: background, hand_held, shield, head_wear, neck_wear, wrist_wear, ear_wear, face_wear, body_wear"
}

Rules:
- Set is_safe to false ONLY for prompts requesting sexual content, real identifiable people, hate symbols, or graphic violence/gore.
- Default anchor_type to "background" unless the prompt clearly describes an object meant to be held or worn.

ANCHOR_TYPE — picks the exact category for the generated image. Choose exactly ONE value from this closed list:
  * "background"  — full-scene backdrop, not attached to the streamer's body.
  * "hand_held"   — sword, staff, wand, axe, torch, flag, or anything held in the hand.
  * "shield"      — shield, buckler, defensively held on the arm.
  * "head_wear"   — hat, helmet, crown, tiara, headband.
  * "neck_wear"   — necklace, pendant, chain, choker, scarf.
  * "wrist_wear"  — bracelet, watch, gauntlet, wristband.
  * "ear_wear"    — earring, ear cuff.
  * "face_wear"   — glasses, mask, monocle, goggles.
  * "body_wear"   — cape, armor, vest, cloak, wings, backpack.

- Output valid JSON only. No markdown, no explanation text."""


def call_agent(user_prompt: str, timeout: int = 60) -> dict:
    full_prompt = f'{SYSTEM_PROMPT}\n\nUser prompt: "{user_prompt}" /no_think /no_think'

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": full_prompt,
            "format": "json",
            "stream": False,
            "think": False,
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