from .client import call_agent

def run_agent(prompt: str) -> dict:
    """Routes the prompt to the Ollama client."""
    return call_agent(prompt)
