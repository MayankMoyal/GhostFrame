"""
Tests the FULL pipeline end to end: agent rewrite -> safety check ->
Z-Image-Turbo generation, via the actual /generate HTTP endpoint.

Requires the server to already be running (python main.py) in another
terminal, on the default port 8000.

Run:
    python test_generate_latency.py
"""

import statistics
import time

import requests

BASE_URL = "http://localhost:8000"

TEST_PROMPTS = [
    {"prompt": "a cozy cyberpunk cat cafe at night", "style": ""},
    {"prompt": "a minimalist mountain landscape at sunrise", "style": "watercolor"},
    {"prompt": "a fantasy castle floating in the clouds", "style": ""},
    {"prompt": "a golden retriever wearing a wizard hat", "style": "cartoon"},
]

if __name__ == "__main__":
    print("Checking server health...")
    health = requests.get(f"{BASE_URL}/health", timeout=10).json()
    print(f"  {health}\n")
    if not health.get("model_loaded"):
        print("Model isn't loaded yet -- wait for startup to finish and retry.")
        raise SystemExit(1)

    round_trip_times = []
    backend_latencies = []
    peak_vram_readings = []

    for payload in TEST_PROMPTS:
        start = time.perf_counter()
        response = requests.post(f"{BASE_URL}/generate", json=payload, timeout=300)
        round_trip = time.perf_counter() - start

        print(f"Prompt: {payload['prompt']!r}  (style={payload['style']!r})")
        print(f"  HTTP status: {response.status_code}")
        print(f"  Client round-trip time: {round_trip:.3f}s")

        if response.status_code == 200:
            data = response.json()
            metrics = data.get("metrics", {})
            agent = data.get("agent", {})
            round_trip_times.append(round_trip)
            backend_latencies.append(metrics.get("latency_seconds", 0))
            peak_vram_readings.append(metrics.get("peak_vram_gb", 0))

            print(f"  Backend-reported inference latency: {metrics.get('latency_seconds')}s")
            print(f"  Backend-reported peak VRAM: {metrics.get('peak_vram_gb')}GB")
            print(f"  Agent ok: {agent.get('agent_ok')}  style_detected: {agent.get('style_detected')!r}")
            print(f"  Final prompt sent to model: {agent.get('final_prompt')!r}")
        else:
            print(f"  Error: {response.json()}")
        print()

    if round_trip_times:
        print("=== Summary ===")
        print(f"Requests completed: {len(round_trip_times)}/{len(TEST_PROMPTS)}")
        print(f"Client round-trip  -- mean: {statistics.mean(round_trip_times):.3f}s, "
              f"min: {min(round_trip_times):.3f}s, max: {max(round_trip_times):.3f}s")
        print(f"Backend inference  -- mean: {statistics.mean(backend_latencies):.3f}s")
        print(f"Peak VRAM observed -- max: {max(peak_vram_readings):.3f}GB")
        # Round-trip minus backend latency ~= agent call + HTTP/serialization
        # overhead. Large gap here points at the agent call, not the image
        # pipeline, as the thing to optimize next.
        overhead = statistics.mean(round_trip_times) - statistics.mean(backend_latencies)
        print(f"Overhead outside image inference (agent + network) -- mean: {overhead:.3f}s")
