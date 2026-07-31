class UnsafePromptError(Exception):
    """Raised when the LLM flags a user prompt as unsafe."""
    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)

def check_safety(agent_result: dict):
    """Checks the agent's JSON response for safety flags and raises an error if unsafe."""
    if not isinstance(agent_result, dict):
        return
        
    is_safe = agent_result.get("is_safe", True)
    if not is_safe:
        reason = agent_result.get("safety_reason", "Prompt violated safety guidelines.")
        raise UnsafePromptError(reason)
