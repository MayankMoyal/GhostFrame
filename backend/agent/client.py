"""
Agent client — MODEL: command-r7b (Cohere Command-R, 7B params, Q4_K_M via Ollama)

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
MODEL_NAME = "command-r7b"

SYSTEM_PROMPT = """You are an AI agent embedded in "Ghost Frame" — a real-time AI prop and background system for livestreamers. When the user speaks or types a prompt, you MUST analyze it and return ONLY a single JSON object with exactly these keys:

{
  "rewritten_prompt": "A vivid, detailed image-generation prompt optimized for SDXL/Flux. Keep the user's core intent but add lighting, material, atmosphere, and composition details. Max 60 words.",
  "is_safe": true,
  "safety_reason": "",
  "style": "fantasy",
  "type": "prop",
  "anchor_type": "hand_held",
  "prop_category": "hand_held"
}

### KEY DEFINITIONS ###

**type** — Exactly one of:
  - "background" → A full scene/environment (dungeon, forest, space station, beach)
  - "prop" → An object/accessory to be worn or held by the streamer

**anchor_type** — Where the item attaches. Must be one of:
  - "background" → Full-screen background replacement
  - "hand_held" → Held in the hand (swords, wands, torches, axes, staffs, flags, microphones)
  - "shield" → Held on the forearm (shields, bucklers)
  - "head_wear" → On top of the head (crowns, helmets, hats, tiaras, headbands, halos)
  - "neck_wear" → Around the neck (necklaces, pendants, chains, scarves, chokers)
  - "wrist_wear" → On the wrist (bracelets, watches, gauntlets, wristbands)
  - "ear_wear" → On the ears (earrings, ear cuffs)
  - "face_wear" → On the face (masks, glasses, monocles, goggles, eye patches)
  - "body_wear" → On the torso/back (capes, cloaks, armor, vests, wings)

**prop_category** — Same value as anchor_type when type is "prop". Set to "" when type is "background".

**style** — A 1-3 word visual style tag. Examples:
  fantasy, cyberpunk, steampunk, dark fantasy, pixel art, anime, realistic, medieval,
  sci-fi, cosmic, enchanted, neon, retro, cinematic, watercolor, gothic, ethereal,
  vaporwave, post-apocalyptic, cozy, minimal, ancient, tropical, horror

### RULES ###

1. **Backgrounds**: If the prompt describes a place, environment, scene, landscape, or atmosphere → type = "background", anchor_type = "background", prop_category = "".
   Examples: "dark dungeon", "enchanted forest", "space station", "cozy fireplace room"

2. **Props**: If the prompt describes an object, weapon, accessory, clothing, or wearable → type = "prop", and pick the correct anchor_type + prop_category from the list above.
   Examples: "flaming sword" → hand_held, "golden crown" → head_wear, "red cape" → body_wear

3. **Prompt Rewriting**: Transform the user's casual prompt into a professional image-generation prompt. Add details about lighting, materials, texture, and atmosphere while preserving the user's exact intent. The image will be generated on a transparent or simple background, so describe ONLY the object/scene itself — not a person wearing it.
   - For props: describe the isolated object (e.g., "A gleaming steel longsword with a ruby-encrusted hilt, flames dancing along the blade, dramatic rim lighting, fantasy art style")
   - For backgrounds: describe the full scene (e.g., "A vast underground dungeon with stone pillars, flickering torch light, misty atmosphere, volumetric god rays, dark fantasy, cinematic composition")

4. **Safety**: Set is_safe to false ONLY for prompts requesting: sexual content, real identifiable people, hate symbols, or graphic gore. Otherwise is_safe is always true.

5. **Output**: Return ONLY valid JSON. No markdown. No code fences. No explanation text. No conversational text before or after the JSON."""


def call_agent(user_prompt: str, timeout: int = 30) -> dict:
    """
    Sends the user's prompt to the local Ollama Command-R 7B model
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
            "system": SYSTEM_PROMPT,
            "prompt": full_prompt,
            "format": "json",
            "stream": False,
            "options": {
                "temperature": 0.3,      # Low temp for consistent classification
                "num_predict": 256,      # Cap output length — JSON should be ~100 tokens
            },
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
