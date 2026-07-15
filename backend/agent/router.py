
from agent.client import call_agent

DEFAULT_RESULT = {
    "rewritten_prompt": None,  # filled in with the original prompt if the agent fails
    "is_safe": True,
    "safety_reason": "",
    "style": "",
    "anchor_type": "background",
}


def run_agent(user_prompt: str) -> dict:
    """
    Single entry point for the agent layer. Calls the local Phi-3 model
    once via Ollama and returns a normalized dict with all agent fields.

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

    # Normalize/validate keys in case the model omits one despite the
    # forced JSON format -- never trust an LLM's output shape completely.
    result.setdefault("rewritten_prompt", user_prompt)
    result.setdefault("is_safe", True)
    result.setdefault("safety_reason", "")
    result.setdefault("style", "")
    result.setdefault("anchor_type", "background")
    result["agent_ok"] = True

    return result
