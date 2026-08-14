"""Ghost Frame — Agent Test Suite (Latency & Accuracy Benchmark)

Run this directly on the GPU server (or with SSH tunnel on port 11434) to verify:
1. JSON output correctness
2. Classification accuracy across anchor types
3. Per-prompt latency with tuned options (num_ctx: 2048, num_predict: 150, num_batch: 512)
"""
import sys
import time
from pathlib import Path

# Add backend directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.router import run_agent

TEST_PROMPTS = [
    # Props - Handheld
    ("a flaming demonic sword with glowing runes", "hand_held", "prop"),
    # Props - Headwear
    ("a golden royal crown with rubies and diamonds", "head_wear", "prop"),
    # Props - Facewear
    ("a cyberpunk neon visor with glowing holographic HUD", "face_wear", "prop"),
    # Props - Shield
    ("a heavy viking round shield made of oak with iron rim", "shield", "prop"),
    # Props - Bodywear (cape/armor)
    ("a crimson velvet superhero cape with gold embroidery", "body_wear", "prop"),
    # Backgrounds
    ("a dark underground fantasy dungeon with torches and stone pillars", "background", "background"),
    ("futuristic sci-fi spaceship bridge overlooking a neon nebula", "background", "background"),
]


def run_benchmark():
    print("=" * 70)
    print(" GHOST FRAME EXPO — AGENT BENCHMARK (Qwen3 4B)")
    print("=" * 70)

    total_time = 0.0
    passed = 0

    for idx, (prompt, expected_anchor, expected_type) in enumerate(TEST_PROMPTS, 1):
        print(f"\n[{idx}/{len(TEST_PROMPTS)}] Prompt: \"{prompt}\"")
        start = time.perf_counter()
        try:
            res = run_agent(prompt)
            elapsed = (time.perf_counter() - start) * 1000
            total_time += elapsed

            status = "✅ PASS" if res.get("agent_ok") else "❌ FAIL"
            cat_match = "✅" if res.get("anchor_type") == expected_anchor else f"⚠️ (got {res.get('anchor_type')}, expected {expected_anchor})"
            type_match = "✅" if res.get("type") == expected_type else f"⚠️ (got {res.get('type')}, expected {expected_type})"

            print(f"  Status:       {status} in {elapsed:.1f} ms")
            print(f"  Type:         {res.get('type')} {type_match}")
            print(f"  Anchor:       {res.get('anchor_type')} {cat_match}")
            print(f"  Style:        {res.get('style')}")
            print(f"  Rewritten:    {res.get('rewritten_prompt')}")

            if res.get("agent_ok"):
                passed += 1
        except Exception as e:
            print(f"  ❌ Error: {e}")

    avg_time = total_time / len(TEST_PROMPTS) if TEST_PROMPTS else 0
    print("\n" + "=" * 70)
    print(f" RESULTS: {passed}/{len(TEST_PROMPTS)} passed")
    print(f" Average Agent Latency: {avg_time:.1f} ms ({avg_time / 1000:.2f}s)")
    print("=" * 70)


if __name__ == "__main__":
    run_benchmark()
