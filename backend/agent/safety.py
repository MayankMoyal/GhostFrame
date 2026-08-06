class UnsafePromptError(Exception):
    """Raised when the agent flags a prompt as unsafe to generate."""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


def check_safety(agent_result: dict) -> None:
    """
    Raises UnsafePromptError if the agent explicitly flagged this prompt
    as unsafe (is_safe is False). Does nothing otherwise.

    Note: this is fail-open by design -- if the agent call failed and we
    fell back to defaults (see router.py), is_safe defaults to True, so
    a broken agent does not block legitimate generations. It only blocks
    when the LLM actively said "no."
    """
    if agent_result.get("is_safe") is False:
        reason = agent_result.get("safety_reason") or "Prompt flagged as unsafe."
        raise UnsafePromptError(reason)
