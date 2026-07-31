def get_rewritten_prompt(agent_result: dict, original_prompt: str) -> str:
    """Extracts the rewritten prompt from the agent's JSON response, falling back to original if missing."""
    if not isinstance(agent_result, dict):
        return original_prompt
    return agent_result.get("rewritten_prompt", original_prompt)
