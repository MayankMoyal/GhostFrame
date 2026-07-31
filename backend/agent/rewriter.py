
def get_rewritten_prompt(agent_result: dict, fallback_prompt: str) -> str:
    """
    Pulls the rewritten prompt out of the agent's result dict.
    Falls back to the original user prompt if it's missing or blank,
    so we never send an empty string into image generation.
    """
    rewritten = agent_result.get("rewritten_prompt")
    if not rewritten or not rewritten.strip():
        return fallback_prompt
    return rewritten.strip()
