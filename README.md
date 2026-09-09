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
ollama pull qwen2.5:7b
```

## Run

```bash
python3 run.py
python3 run.py --model qwen3:4b --workspace ./workspace --debug
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
```

## How it works

`Agent.run_turn()` is the loop everything else serves:

1. The user message is appended; history is compacted if it crosses
   `context_window * compact_threshold`.
2. Repair passes run **every step** — oversized tool results are truncated, orphaned
   tool calls and orphaned tool results are reconciled. Small models emit malformed
   tool-call sequences constantly, and these keep the history API-valid.
3. The system prompt is rebuilt from scratch each step and is not stored in the
   message history.
4. The model is called; tool calls are dispatched and their results appended; loop.
   Otherwise the final text is returned.

Three things worth knowing before changing anything:

- **Tool registration is a global side effect.** The `@tool(...)` decorator inserts into
  a module-level registry at import time. Importing a module that defines tools registers
  them. Duplicate names raise.
- **Behavior is configured from the workspace, not the code.** The system prompt
  interpolates `MEMORY.md` and `AGENTS.md` read from the workspace root. To change how the
  agent behaves, write `AGENTS.md` — don't patch the prompt template.
- **All file access is confined** to the workspace root, and state persists to SQLite there
  (`.ember.db`): a full-text-searchable memory store and an append-only, resumable session log.

Built-in tools: `file`, `edit`, `exec`, `grep`, `memory`, `clarify`. The mutating ones prompt
for confirmation unless `--yolo` is set.
