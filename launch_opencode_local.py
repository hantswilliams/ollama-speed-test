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
    Note: llama-server is left running on exit (opencode needs it). Stop
    it later with: pkill -f llama-server

Either path then opens a new Terminal.app window running `opencode` in the
current working directory.
"""

import argparse
import json
import os
import shlex
import shutil
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
LLAMA_URL = "http://localhost:8080"
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


def ensure_opencode_config(model: str) -> None:
    OPENCODE_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    if OPENCODE_CONFIG.exists():
        print(f"OpenCode config already at {OPENCODE_CONFIG} — leaving it alone.")
        print(f"  Make sure provider.ollama.options.baseURL is `{PROXY_URL}/v1` so the dashboard logs traffic.")
        return
    config = {
        "$schema": "https://opencode.ai/config.json",
        "model": f"ollama/{model}",
        "provider": {
            "ollama": {
                "npm": "@ai-sdk/openai-compatible",
                "name": "Ollama (via dashboard proxy)",
                "options": {
                    "baseURL": f"{PROXY_URL}/v1",
                },
                "models": {
                    model: {},
                },
            }
        },
    }
    OPENCODE_CONFIG.write_text(json.dumps(config, indent=2))
    print(f"Wrote starter OpenCode config to {OPENCODE_CONFIG}")


# ---- llama-server backend -------------------------------------------------


def llama_server_running(base_url: str = LLAMA_URL) -> bool:
    try:
        r = requests.get(f"{base_url}/v1/models", timeout=1)
        return r.status_code < 500
    except requests.RequestException:
        return False


def detect_llama_model(base_url: str = LLAMA_URL) -> str:
    try:
        r = requests.get(f"{base_url}/v1/models", timeout=3)
        r.raise_for_status()
        models = r.json().get("data", [])
        if models:
            raw = models[0].get("id", "")
            return Path(raw).name or raw or "unknown"
    except requests.RequestException:
        pass
    return "unknown"


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
    OPENCODE_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    config = {
        "$schema": "https://opencode.ai/config.json",
        "model": "llamacpp/local",
        "provider": {
            "llamacpp": {
                "npm": "@ai-sdk/openai-compatible",
                "name": f"llama-server: {model_label}",
                "options": {"baseURL": f"{base_url}/v1"},
                "models": {"local": {}},
            }
        },
    }
    new_text = json.dumps(config, indent=2)
    if OPENCODE_CONFIG.exists():
        existing = OPENCODE_CONFIG.read_text()
        if existing.strip() == new_text.strip():
            print(f"OpenCode config already pointed at llama-server at {base_url}.")
            return
        stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        backup = OPENCODE_CONFIG.with_name(f"opencode.json.bak.{stamp}")
        backup.write_text(existing)
        print(f"Backed up existing OpenCode config to {backup}")
    OPENCODE_CONFIG.write_text(new_text)
    print(f"Wrote llama-server OpenCode config to {OPENCODE_CONFIG} (model label: {model_label})")


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
    # llama-server options
    parser.add_argument("--hf", default=None, help="HuggingFace repo (or repo:quant) for llama-server -hf.")
    parser.add_argument("--url", default=LLAMA_URL, help="llama-server base URL (default http://localhost:8080).")
    parser.add_argument("--spec-type", default=None, help="Forwarded to llama-server --spec-type.")
    parser.add_argument("--spec-draft-n-max", type=int, default=None, help="Forwarded to llama-server --spec-draft-n-max.")
    parser.add_argument("--llama-bin", default="llama-server", help="Path to llama-server binary.")
    parser.add_argument("--llama-extra", action="append", default=[], help="Extra args for llama-server (repeatable).")
    parser.add_argument("--startup-timeout", type=float, default=1800.0, help="Seconds to wait for llama-server to become ready.")
    args = parser.parse_args()

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

    if proxy_running():
        print(f"Dashboard proxy reachable at {PROXY_URL} — traffic will be logged.")
    else:
        print(
            f"WARNING: dashboard proxy not reachable at {PROXY_URL}.\n"
            f"  OpenCode is configured to go through the proxy, so requests will fail until you start it:\n"
            f"    cd dashboard/proxy && python main.py"
        )

    ensure_opencode_config(args.model)
    banner = f"OpenCode -> {PROXY_URL} (logging proxy) -> {OLLAMA_URL} | model: {args.model}"
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
