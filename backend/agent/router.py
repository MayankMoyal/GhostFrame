from agent.client import call_agent

DEFAULT_RESULT = {
    "rewritten_prompt": None,  # filled in with the original prompt if the agent fails
    "is_safe": True,
    "safety_reason": "",
    "style": "",
    "type": "background",
    "anchor_type": "background",
    "prop_category": "",
}


def run_agent(user_prompt: str) -> dict:
    """
    Single entry point for the agent layer. Calls the local Ollama model
    (Command-R 7B) once and returns a normalized dict with all agent fields.

    Returns a dict with:
      - rewritten_prompt: enhanced prompt for image generation
      - is_safe: safety flag
      - safety_reason: explanation if unsafe
      - style: visual style tag (e.g., "fantasy", "cyberpunk")
      - type: "background" or "prop"
      - anchor_type: where the item attaches (matches prop_config categories)
      - prop_category: same as anchor_type for props, "" for backgrounds
      - agent_ok: True if the agent call succeeded

    If the agent call fails for ANY reason (Ollama down, bad JSON, model
    not pulled, timeout, etc.), this falls back to safe defaults using
    the original untouched prompt -- so a broken agent never blocks
    image generation. agent_ok is set to False in that case so callers
    (and the frontend) can tell the difference.
    """
    try:
        result = call_agent(user_prompt)
    except Exception as exc:
        print(f"[agent] call failed, falling back to raw prompt: {exc}")
        fallback = dict(DEFAULT_RESULT)
        fallback["rewritten_prompt"] = user_prompt
        fallback["agent_ok"] = False
        return fallback

    # Normalize/validate keys — never trust an LLM's output shape completely.
    result.setdefault("rewritten_prompt", user_prompt)
    result.setdefault("is_safe", True)
    result.setdefault("safety_reason", "")
    result.setdefault("style", "")
    result.setdefault("type", "background")
    result.setdefault("anchor_type", "background")
    result.setdefault("prop_category", "")

    # Validate anchor_type against known categories
    VALID_ANCHORS = {
        "background", "hand_held", "shield", "head_wear", "neck_wear",
        "wrist_wear", "ear_wear", "face_wear", "body_wear",
    }
    if result["anchor_type"] not in VALID_ANCHORS:
        print(f"[agent] unknown anchor_type '{result['anchor_type']}', defaulting to 'background'")
        result["anchor_type"] = "background"
        result["type"] = "background"
        result["prop_category"] = ""

    result["agent_ok"] = True
    return result
