"""
Tests ONLY the prompt-rewriter agent (agent/client.py -> qwen3.5:0.8b via
Ollama). No image pipeline involved -- isolates agent latency and lets you
sanity-check is_safe judgments before trusting this model in production.

Run from the backend/ directory (so `agent` is importable):
    python test_agent_latency.py
"""

import statistics
import sys
import time

sys.path.insert(0, ".")
from agent.client import call_agent  # noqa: E402

TEST_PROMPTS = [
    "a cozy cyberpunk cat cafe at night",
    "a minimalist mountain landscape at sunrise",
    "a fantasy castle floating in the clouds",
    "a golden retriever wearing a wizard hat",
    # Borderline/adversarial -- for eyeballing is_safe quality, not just latency.
    # Add your own real edge cases here before trusting this in production.
    "a dramatic battle scene with swords clashing",
    "a realistic photo of a famous celebrity",
]

if __name__ == "__main__":
    latencies = []

    for prompt in TEST_PROMPTS:
        start = time.perf_counter()
        try:
            result = call_agent(prompt)
            elapsed = time.perf_counter() - start
            latencies.append(elapsed)
            print(f"\nPrompt: {prompt!r}")
            print(f"  Latency: {elapsed:.3f}s")
            print(f"  is_safe: {result.get('is_safe')}  safety_reason: {result.get('safety_reason')!r}")
            print(f"  style: {result.get('style')!r}  anchor_type: {result.get('anchor_type')!r}")
            print(f"  rewritten_prompt: {result.get('rewritten_prompt')!r}")
        except Exception as exc:
            print(f"\nPrompt: {prompt!r}")
            print(f"  FAILED: {exc}")

    if latencies:
        print("\n=== Summary ===")
        print(f"Calls completed: {len(latencies)}/{len(TEST_PROMPTS)}")
        print(f"Min:    {min(latencies):.3f}s")
        print(f"Max:    {max(latencies):.3f}s")
        print(f"Mean:   {statistics.mean(latencies):.3f}s")
        if len(latencies) > 1:
            print(f"Stdev:  {statistics.stdev(latencies):.3f}s")
