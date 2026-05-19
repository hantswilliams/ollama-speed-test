# Privacy Filter (`services/privacy_filter/`)

OpenAI's open-weight PII/PHI **token classifier** from https://huggingface.co/openai/privacy-filter.

**This is not an Ollama model.** Custom bidirectional MoE architecture, safetensors only — runs via HuggingFace Transformers, not llama.cpp/Ollama. Lives in its own sibling service so it can stay separate from the generative stack.

---

## What it does

Takes text in, returns labeled spans for 8 PII categories:

- `account_number`
- `private_address`
- `private_email`
- `private_person`
- `private_phone`
- `private_url`
- `private_date`
- `secret`

Single forward pass, ~1.5B total params / 50M active (sparse MoE, 128 experts, top-4 routing). 128k token context. Apache 2.0.

---

## Setup

```bash
cd services/privacy_filter
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
python download.py
```

This snapshots the full repo into `models/privacy-filter/` (~3 GB, gitignored). The download script just pulls files — no model loading, no torch needed for the download itself.

---

## Running the tests

```bash
cd services/privacy_filter
source .venv/bin/activate
pytest tests/ -v -s
```

- `-v` prints each test name and PASS/FAIL/SKIP as it runs
- `-s` disables pytest's stdout capture so you see the `transformers` load progress (the first test waits ~30s for the 3 GB model to load — that's normal, not a hang)

The session-scoped fixture loads the model once, so subsequent tests are fast.

If you see "no output" or pytest seems to hang:

```bash
which pytest                        # confirm it's services/privacy_filter/.venv/bin/pytest
pytest --collect-only               # list discovered tests without running them
pip list | grep -E "pytest|transformers|torch"   # confirm deps are installed
```

Tests skip cleanly with a clear message if `models/privacy-filter/` is missing — run `python download.py` first.

## Status

Only the **download** step and **tests** exist right now. Inference, an API, and any integration with the dashboard proxy are deliberately deferred until we decide what to use this for.

Possible directions when we're ready:

| Direction | Shape |
|---|---|
| Standalone sanitizer | `POST /sanitize` on its own port; pass text in, get masked text + spans back |
| Pre-scrub for Ollama traffic | Middleware in `dashboard/proxy` that scrubs prompts before forwarding |
| Post-scrub generated output | Same proxy hook, but on the response side |
| Ad-hoc CLI | `python sanitize.py < input.txt > output.txt` |
