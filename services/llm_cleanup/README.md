# llm_cleanup

Standalone tool to tear down local LLM inference state and free RAM.

Use when you're switching between models / backends and want a clean slate, or any time inference performance feels sluggish from accumulated state.

**Standalone:** stdlib-only, no project deps. Safe to copy this file to any macOS machine and run.

## Default behavior

1. Kill any `llama-server` processes (SIGTERM → SIGKILL fallback)
2. Evict every model loaded in Ollama via `ollama stop <name>` — keeps the daemon up so the next launch is fast
3. Report free-RAM before/after and the delta
4. Show the top 10 memory consumers still running (so you can spot anything *else* eating RAM — usually browsers, editors, Spotlight indexer). Suppress with `--top 0`.

## Usage

```bash
# Most common: just clean up
python3 services/llm_cleanup/cleanup.py

# Or if you've chmod +x'd it:
./services/llm_cleanup/cleanup.py

# Heavier: also kill the ollama daemon (forces a full reload next launch)
python3 services/llm_cleanup/cleanup.py --include-ollama

# Nuclear: kill everything + flush macOS file cache (prompts for sudo)
python3 services/llm_cleanup/cleanup.py --include-ollama --include-opencode --include-dashboard --purge

# Preview without changing anything
python3 services/llm_cleanup/cleanup.py --dry-run
```

## Flags

| Flag | Effect |
|---|---|
| (no flags) | Kill llama-server + evict Ollama models |
| `--include-ollama` | Also kill `ollama serve` daemon |
| `--include-opencode` | Also kill any opencode TUI processes (opt-in — could surprise an active session) |
| `--include-dashboard` | Also kill the dashboard proxy on `:11435` |
| `--purge` | Run `sudo purge` to flush macOS file cache (prompts for password) |
| `--dry-run` | Show what would happen; don't change anything |
| `--quiet` | Single-line `key=value` output for scripting |
| `--top N` | After cleanup, show the top N memory consumers still running (default 10; `0` skips) |

## Why evict models instead of killing the daemon

`ollama stop <model>` releases the model's RAM/VRAM without killing `ollama serve`. The daemon stays warm, and the next launch only pays the model-load cost, not the daemon-startup cost. Pass `--include-ollama` if you want the full restart anyway.

## What it doesn't touch

- Models on disk (Ollama-pulled models, GGUF caches in `~/.cache/llama.cpp/`)
- Running benchmark scripts (`test_ollama.py`, `test_llama.py`) — those have their own lifecycle
- `launch_opencode_local.py` itself, or its config at `~/.config/opencode/opencode.json`

For just-our-backends cleanup invoked from within a launch workflow, see `launch_opencode_local.py --stop` at the repo root. `llm_cleanup` is the broader "reset for the next session" tool.
