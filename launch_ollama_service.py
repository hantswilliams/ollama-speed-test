"""Bring up your laptop as an Ollama inference service for colleagues on the LAN.

Flow:
  1. Verify `ollama` is installed.
  2. Start `ollama serve` on 127.0.0.1 if not already running.
  3. Preload the target model so the first colleague request isn't cold.
  4. Start the dashboard proxy on 0.0.0.0:11435 if not already running.
     The proxy stays local-only by default; this launcher overrides
     PROXY_HOST=0.0.0.0 so it accepts LAN traffic.
  5. Print a "share this URL" block (LAN IP + mDNS .local name).

Counterpart to launch_opencode_local.py: that one launches a client against
your model; this one exposes your model so other clients can hit it.
See LAN_SHARING.md for the full design and security notes.
"""

import argparse
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import requests

DEFAULT_MODEL = "qwen3-coder:30b-a3b-q4_K_M"
OLLAMA_URL = "http://localhost:11434"
PROXY_BIND_HOST = "0.0.0.0"
PROXY_PORT = 11435

REPO_ROOT = Path(__file__).resolve().parent
PROXY_DIR = REPO_ROOT / "dashboard" / "proxy"
PROXY_VENV_PY = PROXY_DIR / ".venv" / "bin" / "python"
PROXY_LOG = REPO_ROOT / "dashboard" / "data" / "proxy.log"


def have(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def check_prereqs() -> None:
    if not have("ollama"):
        sys.exit("Missing `ollama`. Install: https://ollama.com/download")
    if not PROXY_VENV_PY.exists():
        sys.exit(
            f"Dashboard proxy venv not found at {PROXY_VENV_PY}.\n"
            f"Set it up first:\n"
            f"  cd {PROXY_DIR}\n"
            f"  uv venv && source .venv/bin/activate && uv pip install -r requirements.txt"
        )


def ollama_running() -> bool:
    try:
        return requests.get(f"{OLLAMA_URL}/api/tags", timeout=1).ok
    except requests.RequestException:
        return False


def start_ollama() -> None:
    print("Starting `ollama serve` on 127.0.0.1...")
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


def proxy_running() -> bool:
    try:
        requests.get(f"http://127.0.0.1:{PROXY_PORT}/api/tags", timeout=1)
        return True
    except requests.RequestException:
        return False


def proxy_listening_on_lan() -> bool:
    """True if something is bound to 0.0.0.0 (or any non-loopback) on PROXY_PORT."""
    try:
        out = subprocess.run(
            ["lsof", "-nP", "-iTCP:" + str(PROXY_PORT), "-sTCP:LISTEN"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        return "*:" + str(PROXY_PORT) in out.stdout or "0.0.0.0:" + str(PROXY_PORT) in out.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def start_proxy() -> None:
    PROXY_LOG.parent.mkdir(parents=True, exist_ok=True)
    print(f"Starting dashboard proxy on {PROXY_BIND_HOST}:{PROXY_PORT} (logs: {PROXY_LOG})...")
    env = os.environ.copy()
    env["PROXY_HOST"] = PROXY_BIND_HOST
    env["PROXY_PORT"] = str(PROXY_PORT)
    log_handle = open(PROXY_LOG, "a")
    log_handle.write(f"\n--- launcher start {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
    log_handle.flush()
    subprocess.Popen(
        [str(PROXY_VENV_PY), "main.py"],
        cwd=str(PROXY_DIR),
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    for _ in range(40):
        if proxy_running():
            print("  proxy is up.")
            return
        time.sleep(0.5)
    sys.exit(f"Proxy did not become reachable within 20s. Check {PROXY_LOG}.")


def get_lan_ip() -> str | None:
    """Best-effort LAN IP — opens a UDP socket to a public address and reads
    the local end. Doesn't actually send packets."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        return ip if not ip.startswith("127.") else None
    except OSError:
        return None
    finally:
        s.close()


def get_mdns_name() -> str | None:
    try:
        out = subprocess.run(
            ["scutil", "--get", "LocalHostName"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        name = out.stdout.strip()
        return f"{name}.local" if name else None
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None


def print_share_block(model: str) -> None:
    lan_ip = get_lan_ip()
    mdns = get_mdns_name()

    print("\n" + "=" * 64)
    print("  Your machine is now serving Ollama on the LAN")
    print("=" * 64)

    urls = []
    if lan_ip:
        urls.append(("IP", f"http://{lan_ip}:{PROXY_PORT}"))
    if mdns:
        urls.append(("mDNS", f"http://{mdns}:{PROXY_PORT}"))
    if not urls:
        print("\n  WARNING: could not determine a LAN address. Run `ifconfig` manually.")
    else:
        print("\n  Share one of these with colleagues:\n")
        for label, url in urls:
            print(f"    [{label:>4}]  {url}")
            print(f"            OpenAI-compatible: {url}/v1")

    primary = (lan_ip and f"http://{lan_ip}:{PROXY_PORT}") or (mdns and f"http://{mdns}:{PROXY_PORT}") or "http://<your-ip>:11435"
    print("\n  Quick test from another machine on the network:")
    print(f"    curl {primary}/api/generate -d '{{\"model\":\"{model}\",\"prompt\":\"hi\",\"stream\":false}}'")
    print("\n  Their OpenCode config (~/.config/opencode/opencode.json):")
    print('    provider.remote.options.baseURL: ' + f"{primary}/v1")
    print("\n  Watch traffic at  http://localhost:3030  (dashboard logs every request, per-IP)")
    print("  Stop the proxy:   pkill -f 'main.py'")
    print("=" * 64 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model tag to preload and serve.")
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
        if proxy_listening_on_lan():
            print(f"Proxy already running on {PROXY_BIND_HOST}:{PROXY_PORT}.")
        else:
            print(
                f"WARNING: proxy is running on port {PROXY_PORT} but appears bound to localhost only.\n"
                f"  Stop it (pkill -f 'main.py') and rerun this script to re-bind to {PROXY_BIND_HOST}."
            )
    else:
        start_proxy()

    print_share_block(args.model)


if __name__ == "__main__":
    main()
