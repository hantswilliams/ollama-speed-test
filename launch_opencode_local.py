"""Launch OpenCode against a local Ollama model.

Flow:
  1. Verify required CLIs are installed (ollama, opencode, osascript).
  2. Start `ollama serve` in the background if it's not already running.
  3. Preload the target model so the first turn isn't a cold load.
  4. Write a starter OpenCode config if none exists, pointing at Ollama's
     OpenAI-compatible endpoint.
  5. Open a new Terminal.app window running `opencode` in the current
     working directory.
"""

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

import requests

DEFAULT_MODEL = "qwen3-coder:30b-a3b-q4_K_M"
OLLAMA_URL = "http://localhost:11434"
PROXY_URL = "http://localhost:11435"
OPENCODE_CONFIG_DIR = Path.home() / ".config" / "opencode"
OPENCODE_CONFIG = OPENCODE_CONFIG_DIR / "opencode.json"


def have(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def check_prereqs() -> None:
    missing = [c for c in ("ollama", "opencode", "osascript") if not have(c)]
    if not missing:
        return
    hints = {
        "ollama": "https://ollama.com/download",
        "opencode": "curl -fsSL https://opencode.ai/install | bash   (or: npm install -g opencode-ai)",
        "osascript": "ships with macOS — you should not see this",
    }
    lines = [f"Missing required command: {c}  ->  {hints[c]}" for c in missing]
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


def open_terminal_with_opencode(model: str) -> None:
    cwd = os.getcwd()
    banner = f"OpenCode -> {PROXY_URL} (logging proxy) -> {OLLAMA_URL} | model: {model}"
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model tag to use in OpenCode.")
    parser.add_argument("--skip-preload", action="store_true", help="Skip the model warm-up request.")
    args = parser.parse_args()

    check_prereqs()

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
    open_terminal_with_opencode(args.model)

    print(
        f"\nLaunched a new Terminal window running `opencode` against `{args.model}`.\n"
        f"Config: {OPENCODE_CONFIG}  |  Stop ollama (if script started it): pkill -f 'ollama serve'"
    )


if __name__ == "__main__":
    main()
