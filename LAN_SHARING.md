# Sharing Ollama on the LAN

Run your local Ollama model as a service that 2-3 trusted colleagues on the same office Wi-Fi can hit from their own machines. Think of this as the host-side counterpart to [launch_opencode_local.py](launch_opencode_local.py): instead of launching a client *against* your model, you bring up the service so others can launch their clients against it.

---

## Architecture

```
                  ┌──────────────────────────────────────────────┐
                  │  Your laptop                                 │
                  │                                              │
  colleague  ───► │  :11435  dashboard proxy  (0.0.0.0)          │
  colleague  ───► │     │                                        │
  you (local) ──► │     ▼                                        │
                  │  :11434  ollama serve     (127.0.0.1)        │
                  │     │                                        │
                  │     └──► qwen3-coder:30b-a3b-q4_K_M           │
                  │                                              │
                  │  :3030   Next.js dashboard (127.0.0.1)       │
                  └──────────────────────────────────────────────┘
                          ▲
                          │ SQLite (data/usage.db)
                          │ logs every request, per-IP
```

**Key design choices:**

- **Only the proxy is LAN-facing.** Ollama itself stays on `127.0.0.1`. There's exactly one entry point, and there's no way to accidentally bypass logging.
- **The dashboard already supports this.** It records `client_ip` on every request, so you'll see who is hitting your machine without any extra work.
- **No auth on the proxy.** Trusted office-mates only. Adding shared-key auth is ~20 lines and easy to add later if the trust model changes.
- **HTTP, not HTTPS.** Fine on a LAN. Don't do this on a coffee shop network.

---

## The new launcher: `launch_ollama_service.py`

A script that brings up "your machine as an inference server" with one command. Parallels `launch_opencode_local.py` but flips the perspective from client to host.

### What it does, in order

1. **Verify prereqs** — `ollama` installed (others' tools talk OpenAI-compatible, so nothing else is required server-side).
2. **Start `ollama serve` on `127.0.0.1`** if it's not already running. Ollama stays local; only the proxy reaches it.
3. **Preload the default model** (`qwen3-coder:30b-a3b-q4_K_M`) with a 1-token warm-up so the first colleague request isn't a 30-second cold load.
4. **Start the dashboard proxy on `0.0.0.0:11435`** if it's not already running. This is the new bit: the proxy currently binds to localhost — we'd add a `PROXY_HOST` env var and the launcher would set it to `0.0.0.0`.
5. **Detect the LAN IP** (`ipconfig getifaddr en0` for Wi-Fi, fall back to `en1` for Ethernet, fall back to the `.local` mDNS name).
6. **Print a "share this with your colleagues" block** with copy-pasteable URLs and example client configs.
7. **Optionally tail proxy logs** to a window so you can watch traffic in real time (or just exit and let the proxy keep running in the background).

### What it deliberately does NOT do

- Doesn't expose Ollama (`:11434`) on the LAN. Proxy-only.
- Doesn't expose the Next.js dashboard (`:3030`) on the LAN — that's your monitoring view, for you.
- Doesn't add auth, rate limiting, or TLS. Out of scope for trusted-office mode.
- Doesn't auto-start the Next.js dashboard. That's a separate `npm run dev` you start when you want to watch.

### CLI shape

```bash
python launch_ollama_service.py
python launch_ollama_service.py --model qwen3-coder:30b-a3b-q4_K_M
python launch_ollama_service.py --skip-preload
python launch_ollama_service.py --bind 0.0.0.0   # override; default is 0.0.0.0
```

---

## What colleagues do

Once you've run the launcher and shared your URL, colleagues point any OpenAI-compatible tool at it.

### OpenCode

In their `~/.config/opencode/opencode.json`:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "model": "remote/qwen3-coder:30b-a3b-q4_K_M",
  "provider": {
    "remote": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Hants's laptop",
      "options": {
        "baseURL": "http://<YOUR-IP>:11435/v1"
      },
      "models": {
        "qwen3-coder:30b-a3b-q4_K_M": {}
      }
    }
  }
}
```

### Plain `curl`

```bash
curl http://<YOUR-IP>:11435/api/generate -d '{
  "model": "qwen3-coder:30b-a3b-q4_K_M",
  "prompt": "say hi",
  "stream": false
}'
```

### Python (OpenAI SDK)

```python
from openai import OpenAI
client = OpenAI(base_url="http://<YOUR-IP>:11435/v1", api_key="anything")
resp = client.chat.completions.create(
    model="qwen3-coder:30b-a3b-q4_K_M",
    messages=[{"role": "user", "content": "say hi"}],
)
```

API key is unchecked — colleagues can pass any non-empty string.

---

## Required change to the proxy

The proxy at [dashboard/proxy/main.py](dashboard/proxy/main.py) currently hardcodes `host="127.0.0.1"` in its `uvicorn.run` call. To support LAN exposure cleanly, we'd add:

```python
import os
PROXY_HOST = os.environ.get("PROXY_HOST", "127.0.0.1")
# ...
uvicorn.run("main:app", host=PROXY_HOST, port=PROXY_PORT, ...)
```

Default stays `127.0.0.1` (safe). The launcher sets `PROXY_HOST=0.0.0.0` when starting it. Anyone running the proxy directly via `python main.py` still gets localhost-only behavior unless they explicitly opt in.

---

## Caveats and operational notes

### macOS firewall
On the first incoming connection, macOS will prompt to allow the proxy process to accept network connections. Click **Allow** once. To avoid the prompt later, pre-approve under *System Settings → Network → Firewall → Options → uvicorn / python*.

### Your LAN IP can change
DHCP leases rotate. If you don't want to re-share the URL whenever your address moves, use the mDNS hostname instead:

```bash
scutil --get LocalHostName    # prints something like "hants-mbp"
```

Colleagues then use `http://hants-mbp.local:11435/v1`. This survives reconnects on Bonjour-aware networks (every macOS, most modern routers).

### Concurrency ceiling
A single laptop running a 30B q4 model can serve roughly 1–2 concurrent generations before queueing latency becomes noticeable. Ollama processes requests serially for a single model; if three colleagues all hit you at once, requests 2 and 3 wait their turn. No proxy can fix this — it's a hardware/inference reality.

### Battery and thermals
Plug the laptop in. Inference is sustained heavy load; running 30B q4 at full tilt on battery is unpleasant on every axis (speed, battery life, fan noise).

### Stopping the service
- `pkill -f 'uvicorn'` — stops the proxy (use a more specific match if other uvicorn processes are running).
- `pkill -f 'ollama serve'` — stops Ollama.
- Or just close the launcher terminal if you used the foreground tail-logs mode.

---

## Security notes (for when "trusted office" stops being true)

The current setup has *no* authentication. If you ever want to:

- **Limit access to specific IPs** → add an `Allow`/`Deny` middleware in the proxy, or stick Caddy in front with a CIDR filter.
- **Add a shared API key** → middleware on the proxy that checks `Authorization: Bearer <key>` and 401s otherwise. About 15 lines.
- **Add per-user keys + usage tracking + rate limits** → swap the proxy for [LiteLLM proxy](https://docs.litellm.ai/docs/proxy/quick_start) and have it forward to Ollama. Heavier but real-deal.
- **Expose beyond the LAN** → don't open firewall ports. Use [Tailscale](https://tailscale.com) — each colleague installs the client, gets a private IP for your machine, works identically to LAN.

---

## TL;DR for getting going

1. Run `python launch_ollama_service.py` (after we build it).
2. Copy the printed URL.
3. Send it to two coworkers.
4. Watch their requests appear in your dashboard at `http://localhost:3030`, tagged with their IPs.
