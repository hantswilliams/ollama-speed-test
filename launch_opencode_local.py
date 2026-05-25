"""Launch OpenCode against a local model — Ollama or llama-server.

Two backends:

  --backend ollama (default)
    1. Start `ollama serve` if not already running.
    2. Preload the model via Ollama's API.
    3. Write opencode config pointing at the dashboard proxy (port 11435).

  --backend llama-server
    1. Start `llama-server` with `-hf <repo>` if not already running.
    2. Wait for /v1/models to respond, then detect the loaded model.
    3. Write opencode config pointing at llama-server (port 8080).
    Note: llama-server is left running on exit (opencode needs it).

Cleanup:
  --stop                       kill any running llama-server processes
  --stop --include-ollama      also stop `ollama serve`

Either non-stop path opens a new Terminal.app window running `opencode` in
the current working directory.
"""

import argparse
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import requests

DEFAULT_MODEL = "qwen3-coder:30b-a3b-q4_K_M"
OLLAMA_URL = "http://localhost:11434"
PROXY_URL = "http://localhost:11435"
# Use 127.0.0.1 (not localhost) so we always hit IPv4. On macOS, `localhost`
# resolves to IPv6 first; if anything else is bound to *:8080 over IPv6 it
# silently shadows llama-server's IPv4-only listener.
LLAMA_URL = "http://127.0.0.1:8080"
OPENCODE_CONFIG_DIR = Path.home() / ".config" / "opencode"
OPENCODE_CONFIG = OPENCODE_CONFIG_DIR / "opencode.json"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def have(cmd: str) -> bool:
    return shutil.which(cmd) is not None


PREREQ_HINTS = {
    "ollama": "https://ollama.com/download",
    "opencode": "curl -fsSL https://opencode.ai/install | bash   (or: npm install -g opencode-ai)",
    "osascript": "ships with macOS — you should not see this",
    "llama-server": "https://github.com/ggml-org/llama.cpp (brew install llama.cpp on macOS)",
}


def check_prereqs(required: tuple) -> None:
    missing = [c for c in required if not have(c)]
    if not missing:
        return
    lines = [f"Missing required command: {c}  ->  {PREREQ_HINTS[c]}" for c in missing]
    sys.exit("\n".join(lines))


def ollama_running() -> bool:
    try:
        return requests.get(f"{OLLAMA_URL}/api/tags", timeout=1).ok
    except requests.RequestException:
        return False


def proxy_running() -> bool:
    try:
        return requests.get(f"{PROXY_URL}/api/tags", timeout=1).ok
    except requests.RequestException:
        return False


def start_ollama() -> None:
    print("Starting `ollama serve` in the background...")
    subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    for _ in range(30):
        if ollama_running():
            print("  ollama is up.")
            return
        time.sleep(0.5)
    sys.exit("Ollama did not become reachable within 15s.")


def preload_model(model: str) -> None:
    print(f"Preloading `{model}` (first run can take a minute)...")
    r = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": model,
            "prompt": "hi",
            "stream": False,
            "keep_alive": "60m",
            "options": {"num_predict": 1},
        },
        timeout=600,
    )
    r.raise_for_status()
    print("  model resident.")


# Keys we own and will overwrite when writing config. Any other top-level keys
# present in the existing file (e.g. "mcp", "theme", "autoshare", "instructions")
# are preserved across backend switches so user-added settings don't get wiped.
_MANAGED_OPENCODE_KEYS = {"$schema", "model", "provider"}


def _write_opencode_config(config: dict, label: str) -> None:
    """Merge our managed keys into the existing config (if any), preserving
    user-added top-level settings like `mcp`. Back up the prior file if changed."""
    OPENCODE_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    existing_text = ""
    existing_obj = {}
    if OPENCODE_CONFIG.exists():
        existing_text = OPENCODE_CONFIG.read_text()
        try:
            existing_obj = json.loads(existing_text)
            if not isinstance(existing_obj, dict):
                existing_obj = {}
        except json.JSONDecodeError:
            print(f"  warning: existing {OPENCODE_CONFIG.name} is not valid JSON — treating as empty")
            existing_obj = {}

    merged = dict(existing_obj)
    for key in _MANAGED_OPENCODE_KEYS:
        if key in config:
            merged[key] = config[key]
    preserved = sorted(k for k in existing_obj if k not in _MANAGED_OPENCODE_KEYS)
    new_text = json.dumps(merged, indent=2)
    if existing_text.strip() == new_text.strip():
        print(f"OpenCode config already matches ({label}); leaving as-is.")
        return
    if existing_text:
        stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        backup = OPENCODE_CONFIG.with_name(f"opencode.json.bak.{stamp}")
        backup.write_text(existing_text)
        print(f"Backed up existing OpenCode config to {backup}")
    OPENCODE_CONFIG.write_text(new_text)
    note = f" (preserved user keys: {', '.join(preserved)})" if preserved else ""
    print(f"Wrote OpenCode config ({label}) to {OPENCODE_CONFIG}{note}")


def ensure_opencode_config(model: str, via_proxy: bool) -> None:
    base_url = f"{PROXY_URL}/v1" if via_proxy else f"http://127.0.0.1:11434/v1"
    label = f"Ollama via dashboard proxy → {model}" if via_proxy else f"Ollama direct → {model}"
    config = {
        "$schema": "https://opencode.ai/config.json",
        "model": f"ollama/{model}",
        "provider": {
            "ollama": {
                "npm": "@ai-sdk/openai-compatible",
                "name": label,
                "options": {"baseURL": base_url},
                "models": {model: {}},
            }
        },
    }
    _write_opencode_config(config, label)


# ---- llama-server backend -------------------------------------------------


def llama_server_running(base_url: str = LLAMA_URL) -> bool:
    """Stricter than a simple status check: verifies the response is the
    OpenAI-shaped JSON llama-server returns, so a generic web server squatting
    on the port can't fool us."""
    try:
        r = requests.get(f"{base_url}/v1/models", timeout=2)
        if r.status_code != 200:
            return False
        body = r.json()
    except (requests.RequestException, ValueError):
        return False
    # llama.cpp uses either "data" (OpenAI standard) or "models" key
    models = body.get("data") or body.get("models") or []
    return isinstance(models, list) and len(models) > 0


def detect_llama_model(base_url: str = LLAMA_URL) -> str:
    try:
        r = requests.get(f"{base_url}/v1/models", timeout=3)
        r.raise_for_status()
        body = r.json()
    except (requests.RequestException, ValueError):
        return "unknown"
    models = body.get("data") or body.get("models") or []
    if not models:
        return "unknown"
    raw = models[0].get("id") or models[0].get("name") or ""
    return Path(raw).name or raw or "unknown"


def port_from_url(url: str) -> int:
    parsed = urlparse(url)
    return parsed.port or (443 if parsed.scheme == "https" else 80)


def start_llama_server(
    hf_repo: str,
    port: int,
    spec_type: str | None,
    spec_draft_n_max: int | None,
    extra_args: list,
    bin_path: str,
    log_path: Path,
) -> subprocess.Popen:
    cmd = [bin_path, "-hf", hf_repo, "--port", str(port)]
    if spec_type:
        cmd += ["--spec-type", spec_type]
    if spec_draft_n_max is not None:
        cmd += ["--spec-draft-n-max", str(spec_draft_n_max)]
    cmd += extra_args
    print(f"Launching: {' '.join(cmd)}")
    print(f"llama-server log: {log_path}")
    log_file = log_path.open("w")
    return subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,  # detach so it outlives this script
    )


def wait_for_llama_ready(
    base_url: str,
    proc: subprocess.Popen,
    timeout: float,
    log_path: Path,
) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            tail = ""
            try:
                tail = "\n".join(log_path.read_text().splitlines()[-30:])
            except OSError:
                pass
            sys.exit(
                f"llama-server exited early with code {proc.returncode}.\n"
                f"Last log lines:\n{tail}"
            )
        if llama_server_running(base_url):
            return
        time.sleep(2.0)
    sys.exit(
        f"llama-server not ready within {timeout:.0f}s. See {log_path} for details."
    )


def ensure_opencode_config_llama(model_label: str, base_url: str) -> None:
    label = f"llama-server → {model_label}"
    config = {
        "$schema": "https://opencode.ai/config.json",
        "model": "llamacpp/local",
        "provider": {
            "llamacpp": {
                "npm": "@ai-sdk/openai-compatible",
                "name": label,
                "options": {"baseURL": f"{base_url}/v1"},
                "models": {"local": {}},
            }
        },
    }
    _write_opencode_config(config, label)


# ---- terminal launcher ----------------------------------------------------


# ---- stop helpers ---------------------------------------------------------


def _find_pids(pattern: str, exact: bool) -> list:
    """Use pgrep to find matching PIDs, excluding this script's own PID."""
    try:
        res = subprocess.run(
            ["pgrep", "-x" if exact else "-f", pattern],
            capture_output=True, text=True,
        )
    except FileNotFoundError:
        sys.exit("pgrep not found on PATH — can't run --stop without it.")
    my_pid = os.getpid()
    return [int(p) for p in res.stdout.split() if p.strip().isdigit() and int(p) != my_pid]


def _is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists but we can't signal it


def stop_backend(label: str, pattern: str, exact: bool, timeout: float = 10.0) -> int:
    """SIGTERM all matching processes, escalate to SIGKILL after timeout. Returns count terminated."""
    pids = _find_pids(pattern, exact=exact)
    if not pids:
        print(f"  no {label} processes running")
        return 0
    print(f"  stopping {len(pids)} {label} process(es): {pids}")
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except PermissionError:
            print(f"  no permission to signal PID {pid} — skipping")
    deadline = time.time() + timeout
    stragglers = []
    while time.time() < deadline:
        stragglers = [p for p in pids if _is_alive(p)]
        if not stragglers:
            print(f"  all {label} processes exited cleanly")
            return len(pids)
        time.sleep(0.5)
    for pid in stragglers:
        print(f"  SIGKILL {label} PID {pid} (did not respond to SIGTERM)")
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    return len(pids)


def run_stop(include_ollama: bool) -> None:
    print("Stopping local inference backends...")
    stopped = stop_backend("llama-server", "llama-server", exact=True)
    if include_ollama:
        # `ollama serve` runs as `ollama` with `serve` in argv; match the full line
        stopped += stop_backend("ollama serve", "ollama serve", exact=False)
    else:
        # Tell the user if ollama is still running so it isn't a surprise
        if _find_pids("ollama serve", exact=False):
            print("  (ollama serve is still running — pass --include-ollama to stop it too)")
    if stopped == 0:
        print("Nothing to stop.")


# ---- terminal launcher ----------------------------------------------------


def open_terminal_with_opencode(banner: str) -> None:
    cwd = os.getcwd()
    shell_cmd = " && ".join([
        f"cd {shlex.quote(cwd)}",
        f"echo {shlex.quote(banner)}",
        "opencode",
    ])
    escaped = shell_cmd.replace("\\", "\\\\").replace('"', '\\"')
    apple_script = (
        'tell application "Terminal"\n'
        f'    do script "{escaped}"\n'
        "    activate\n"
        "end tell"
    )
    subprocess.run(["osascript", "-e", apple_script], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--backend",
        choices=("ollama", "llama-server"),
        default="ollama",
        help="Which local inference server to use (default: ollama).",
    )
    # Ollama options
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model tag (backend=ollama only).")
    parser.add_argument("--skip-preload", action="store_true", help="Skip the Ollama model warm-up request.")
    parser.add_argument(
        "--via-proxy",
        action="store_true",
        help="Route opencode through the dashboard proxy on :11435 (so requests are logged). "
             "Off by default — opencode connects to Ollama directly.",
    )
    # llama-server options
    parser.add_argument("--hf", default=None, help="HuggingFace repo (or repo:quant) for llama-server -hf.")
    parser.add_argument("--url", default=LLAMA_URL, help="llama-server base URL (default http://localhost:8080).")
    parser.add_argument("--spec-type", default=None, help="Forwarded to llama-server --spec-type.")
    parser.add_argument("--spec-draft-n-max", type=int, default=None, help="Forwarded to llama-server --spec-draft-n-max.")
    parser.add_argument("--llama-bin", default="llama-server", help="Path to llama-server binary.")
    parser.add_argument("--llama-extra", action="append", default=[], help="Extra args for llama-server (repeatable).")
    parser.add_argument("--startup-timeout", type=float, default=1800.0, help="Seconds to wait for llama-server to become ready.")
    # Cleanup
    parser.add_argument(
        "--stop",
        action="store_true",
        help="Kill any running llama-server processes and exit. "
             "Add --include-ollama to also stop `ollama serve`.",
    )
    parser.add_argument(
        "--include-ollama",
        action="store_true",
        help="With --stop, also kill `ollama serve` (off by default since users often run it for other tools).",
    )
    args = parser.parse_args()

    if args.stop:
        run_stop(include_ollama=args.include_ollama)
        return
    if args.include_ollama:
        print("Note: --include-ollama only takes effect with --stop; ignoring.")

    if args.backend == "ollama":
        run_ollama_backend(args)
    else:
        run_llama_backend(args)


def run_ollama_backend(args) -> None:
    check_prereqs(("ollama", "opencode", "osascript"))

    if ollama_running():
        print("Ollama already running.")
    else:
        start_ollama()

    if not args.skip_preload:
        preload_model(args.model)

    if args.via_proxy:
        if proxy_running():
            print(f"Dashboard proxy reachable at {PROXY_URL} — traffic will be logged.")
        else:
            sys.exit(
                f"--via-proxy requested but nothing is responding at {PROXY_URL}.\n"
                f"  Start it first:  cd dashboard/proxy && python main.py\n"
                f"  Or omit --via-proxy to connect opencode directly to Ollama."
            )

    ensure_opencode_config(args.model, via_proxy=args.via_proxy)
    target = f"{PROXY_URL} (logging proxy) -> {OLLAMA_URL}" if args.via_proxy else "Ollama (direct)"
    banner = f"OpenCode -> {target} | model: {args.model}"
    open_terminal_with_opencode(banner)

    print(
        f"\nLaunched a new Terminal window running `opencode` against `{args.model}`.\n"
        f"Config: {OPENCODE_CONFIG}  |  Stop ollama (if script started it): pkill -f 'ollama serve'"
    )


def run_llama_backend(args) -> None:
    check_prereqs(("llama-server", "opencode", "osascript"))
    base_url = args.url.rstrip("/")

    if llama_server_running(base_url):
        print(f"llama-server already running at {base_url} — reusing it.")
    else:
        if not args.hf:
            sys.exit(
                "No llama-server running at {url} and no --hf repo given.\n"
                "  Either start llama-server yourself, or re-run with e.g.:\n"
                "    python launch_opencode_local.py --backend llama-server \\\n"
                "        --hf bartowski/DeepSeek-Coder-V2-Lite-Instruct-GGUF:Q4_K_M".format(url=base_url)
            )
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        log_path = OUTPUT_DIR / f"llama_server_opencode_{stamp}.log"
        proc = start_llama_server(
            hf_repo=args.hf,
            port=port_from_url(base_url),
            spec_type=args.spec_type,
            spec_draft_n_max=args.spec_draft_n_max,
            extra_args=args.llama_extra,
            bin_path=args.llama_bin,
            log_path=log_path,
        )
        print(f"Waiting for llama-server to become ready (up to {args.startup_timeout:.0f}s)...")
        wait_for_llama_ready(base_url, proc, args.startup_timeout, log_path)
        print("llama-server ready.")

    model_label = detect_llama_model(base_url)
    print(f"Model: {model_label}")

    ensure_opencode_config_llama(model_label, base_url)
    banner = f"OpenCode -> {base_url} (llama-server) | model: {model_label}"
    open_terminal_with_opencode(banner)

    print(
        f"\nLaunched a new Terminal window running `opencode` against llama-server ({model_label}).\n"
        f"Config: {OPENCODE_CONFIG}  |  Stop llama-server: pkill -f llama-server"
    )


if __name__ == "__main__":
    main()
