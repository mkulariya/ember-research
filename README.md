# ember

A deep research agent built on a small open-weights model, made progressively smarter through post-training.

The agent core is one module. stdlib plus `openai`, nothing else — the whole control flow of an agent loop is readable end to end in a single sitting, which is the point.

## Status

Early. The agent core runs; the research layer, evaluation harness, and post-training pipeline are not built yet.

## Requirements

Python 3.11+, the `openai` package, and any OpenAI-compatible endpoint. Defaults target a local [Ollama](https://ollama.com) server.

```bash
pip install openai
ollama serve
ollama pull hf.co/LiquidAI/LFM2.5-1.2B-Thinking-GGUF
```

The `web_search` and `web_fetch` tools call [TinyFish](https://docs.tinyfish.ai), which needs an API key.
Both are free tier — no card, no minimum balance. Create one at
[agent.tinyfish.ai/api-keys](https://agent.tinyfish.ai/api-keys) and export it:

```bash
export TINYFISH_API_KEY=...        # or put it in .env, which is gitignored
```

Without the key the agent still runs; the two web tools return an error telling you to set it.

## Run

```bash
python3 run.py
python3 run.py --model <tag> --workspace ./workspace --debug
python3 run.py --yolo          # auto-approve tool confirmations (unattended runs)
```

Flags: `--model`, `--workspace`, `--api-base-url`, `--api-key`, `--yolo`, `--debug`.

Environment variables override the defaults in `ember/config.py`, all prefixed `EMBER_`:
`API_BASE_URL`, `API_KEY`, `MODEL`, `WORKSPACE`, `MAX_STEPS`, `CONTEXT_WINDOW`, `TOOL_TIMEOUT`,
`YOLO`, `COMPACT_THRESHOLD`, `COMPACT_KEEP_FRESH`, `LLM_MAX_RETRIES`, `LLM_RETRY_BASE_DELAY`, `LOG_LEVEL`.
They are read from a project-root `.env` at startup, which is gitignored.

In the REPL: `/help /quit /reset /memory /compact /session /history /sessions /tools /model /export`.

## Layout

```
run.py              REPL entry point
ember/
  core.py           the agent loop, tools, stores, LLM client, REPL
  config.py         non-secret defaults
  web_tools.py      web_search + web_fetch (TinyFish REST, stdlib urllib)
```

## Tools

`file`, `edit`, `exec`, `grep`, `memory`, `clarify`, `web_search`, `web_fetch`.

File and shell tools are confined to the workspace and prompt for confirmation before
changing anything. The two web tools are read-only and never prompt.
