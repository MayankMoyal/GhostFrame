# Agent model A/B testing

`agent/router.py` imports `from agent.client import call_agent` -- so
whichever file is named `client.py` is the model that actually runs.

Two candidates are provided:

- `client_1.py` -- Phi-3 (`phi3`, ~3.9GB). The original model. Already
  pulled and known-working on the Lightning AI instance.
- `client_2.py` -- Qwen3-4B-Instruct-2507 (`qwen3:4b-instruct-2507`,
  ~2.5GB). Candidate replacement: smaller, newer generation, documented
  strength on structured/JSON output. Not yet pulled or tested end-to-end
  -- see the checklist in its docstring before trusting it in production.

## To switch models

```bash
cd backend/agent
cp client_1.py client.py   # use Phi-3
# or
cp client_2.py client.py   # use Qwen3
```

(Using `cp` rather than `mv` keeps both candidates around for repeat
testing. Delete/overwrite the old `client.py` first if one already
exists.)

After switching, restart the backend (`python main.py`) so the new
`agent/client.py` gets picked up, then re-run the same `/generate` test
prompt against both to compare `agent_ok`, `final_prompt` quality, and
whatever latency you observe -- and confirm `ollama ps` shows the model
at 100% GPU, not split with CPU.
