"""Benchmark a llama-server (llama.cpp) instance.

Counterpart to test_ollama.py. Uses the same prompts, the same SQLite DB
(`outputs/benchmarks.db`), and the same hardware capture — so rows from
both scripts can be compared directly. Writes CSVs to `outputs/llama_*.csv`.

Two modes:

1. Self-managed (pass --hf): the script launches llama-server, waits for
   it to become ready, runs the benchmarks, then shuts it down. Cleanup
   runs even on Ctrl+C / errors.

       python test_llama.py --hf ggml-org/Qwen3.6-27B-MTP-GGUF \
           --spec-type draft-mtp --spec-draft-n-max 2

       python test_llama.py --hf ggml-org/Qwen3.6-35B-A3B-MTP-GGUF \
           --spec-type draft-mtp --spec-draft-n-max 3

2. Externally managed: omit --hf and the script assumes llama-server is
   already running on --url (default http://localhost:8080).

By default the script asks the server which model it has loaded via
/v1/models and uses that as the row label and default filename suffix.
Pass --model-label to override.
"""

import argparse
import json
import os
import platform
import re
import signal
import sqlite3
import subprocess
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests

warnings.filterwarnings("ignore")

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = OUTPUT_DIR / "benchmarks.db"


def gather_hardware() -> dict:
    info = {
        "hostname": platform.node() or "",
        "os": platform.system() or "",
        "os_version": platform.release() or "",
        "cpu_count": os.cpu_count() or 0,
        "device_model": "",
        "chip": "",
        "memory_gb": None,
    }
    if info["os"] == "Darwin":
        try:
            sp = subprocess.run(
                ["system_profiler", "SPHardwareDataType", "-json"],
                capture_output=True, text=True, timeout=5,
            )
            block = json.loads(sp.stdout).get("SPHardwareDataType", [{}])[0]
            name = block.get("machine_name") or ""
            model_id = block.get("machine_model") or ""
            info["device_model"] = f"{name} ({model_id})" if name and model_id else (name or model_id)
            info["chip"] = block.get("chip_type") or block.get("cpu_type") or ""
            mem = block.get("physical_memory", "")
            m = re.match(r"([\d.]+)\s*GB", mem)
            if m:
                info["memory_gb"] = float(m.group(1))
        except Exception:
            pass
        try:
            ver = subprocess.run(["sw_vers", "-productVersion"], capture_output=True, text=True, timeout=2)
            if ver.stdout.strip():
                info["os_version"] = ver.stdout.strip()
        except Exception:
            pass
    elif info["os"] == "Linux":
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        info["memory_gb"] = round(int(line.split()[1]) / 1024 / 1024, 1)
                        break
        except Exception:
            pass
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        info["chip"] = line.split(":", 1)[1].strip()
                        break
        except Exception:
            pass
    return info


def ensure_db() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS benchmark_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                batch_id TEXT NOT NULL,
                batch_started_at TEXT NOT NULL,
                suffix TEXT,
                backend TEXT,
                model TEXT NOT NULL,
                category TEXT NOT NULL,
                prompt_name TEXT NOT NULL,
                run INTEGER NOT NULL,
                output_tokens INTEGER,
                output_tps REAL,
                prompt_tokens INTEGER,
                prompt_tps REAL,
                total_duration_sec REAL,
                load_duration_sec REAL,
                wall_time_sec REAL,
                hostname TEXT,
                device_model TEXT,
                chip TEXT,
                cpu_count INTEGER,
                memory_gb REAL,
                os TEXT,
                os_version TEXT
            )
        """)
        try:
            conn.execute("ALTER TABLE benchmark_runs ADD COLUMN backend TEXT")
        except sqlite3.OperationalError:
            pass
        conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_batch ON benchmark_runs(batch_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_model ON benchmark_runs(model)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_backend ON benchmark_runs(backend)")


def detect_loaded_model(base_url: str) -> str:
    """Ask llama-server which model it has loaded. Returns a short label."""
    try:
        r = requests.get(f"{base_url}/v1/models", timeout=3)
        r.raise_for_status()
        models = r.json().get("data", [])
        if models:
            raw = models[0].get("id", "")
            # llama-server often returns a file path; trim to basename
            return Path(raw).name or raw or "unknown"
    except requests.RequestException:
        pass
    return "unknown"


def is_server_up(base_url: str, timeout: float = 2.0) -> bool:
    try:
        r = requests.get(f"{base_url}/v1/models", timeout=timeout)
        return r.status_code < 500
    except requests.RequestException:
        return False


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
        start_new_session=True,  # own process group so we can kill cleanly
    )


def wait_for_ready(
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
            raise RuntimeError(
                f"llama-server exited early with code {proc.returncode}.\n"
                f"Last log lines:\n{tail}"
            )
        if is_server_up(base_url, timeout=2.0):
            return
        time.sleep(2.0)
    raise TimeoutError(
        f"llama-server not ready within {timeout:.0f}s. See {log_path} for details."
    )


def stop_llama_server(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    print("Stopping llama-server...")
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            return
        proc.wait(timeout=5)


parser = argparse.ArgumentParser(
    description="Benchmark a running llama-server instance on coding and general prompts.",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog=__doc__,
)
parser.add_argument("--url", default="http://localhost:8080", help="llama-server base URL.")
parser.add_argument(
    "--model-label",
    default=None,
    help="Override the model name written to CSV/DB. Defaults to whatever /v1/models reports.",
)
parser.add_argument(
    "--suffix",
    default="",
    help="Optional suffix appended to CSV filenames (e.g. 'mtp-spec2').",
)
parser.add_argument(
    "--num-predict",
    type=int,
    default=300,
    help="Max output tokens per generation (default 300, matches test_ollama.py).",
)
parser.add_argument(
    "--hf",
    default=None,
    help="If set, launch llama-server with `-hf <repo>` and shut it down when done.",
)
parser.add_argument(
    "--spec-type",
    default=None,
    help="Forwarded to llama-server as --spec-type when --hf is set.",
)
parser.add_argument(
    "--spec-draft-n-max",
    type=int,
    default=None,
    help="Forwarded to llama-server as --spec-draft-n-max when --hf is set.",
)
parser.add_argument(
    "--llama-bin",
    default="llama-server",
    help="Path to the llama-server binary (default: 'llama-server' on PATH).",
)
parser.add_argument(
    "--llama-extra",
    action="append",
    default=[],
    help="Extra args to forward to llama-server. Repeatable, one token per flag.",
)
parser.add_argument(
    "--startup-timeout",
    type=float,
    default=600.0,
    help="Seconds to wait for llama-server to become ready (default 600).",
)
args = parser.parse_args()

base_url = args.url.rstrip("/")
COMPLETION_URL = f"{base_url}/completion"

server_proc: subprocess.Popen | None = None
if args.hf:
    if is_server_up(base_url, timeout=1.0):
        sys.exit(
            f"Something is already responding at {base_url}. Stop it first or pass "
            "--url with a different port."
        )
    server_log = OUTPUT_DIR / f"llama_server_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.log"
    server_proc = start_llama_server(
        hf_repo=args.hf,
        port=port_from_url(base_url),
        spec_type=args.spec_type,
        spec_draft_n_max=args.spec_draft_n_max,
        extra_args=args.llama_extra,
        bin_path=args.llama_bin,
        log_path=server_log,
    )
    print(f"Waiting for llama-server to become ready (up to {args.startup_timeout:.0f}s)...")
    try:
        wait_for_ready(base_url, server_proc, args.startup_timeout, server_log)
    except BaseException:
        stop_llama_server(server_proc)
        raise
    print("llama-server ready.")
else:
    if not is_server_up(base_url, timeout=1.0):
        sys.exit(
            f"No llama-server responding at {base_url}.\n"
            "  - To have this script launch one for you, re-run with --hf, e.g.:\n"
            "      python test_llama.py --hf ggml-org/Qwen3.6-27B-MTP-GGUF \\\n"
            "          --spec-type draft-mtp --spec-draft-n-max 2\n"
            "  - Or start llama-server yourself in another terminal and re-run."
        )

try:
    model_label = args.model_label or detect_loaded_model(base_url)
    print(f"llama-server: {base_url}")
    print(f"Model: {model_label}")

    if args.suffix:
        suffix = f"_{args.suffix}"
    else:
        default_tag = re.sub(r"[^A-Za-z0-9._-]+", "-", model_label).strip("-")
        suffix = f"_{default_tag}" if default_tag and default_tag != "unknown" else ""

    prompts = [
        {
            "category": "coding",
            "name": "two_sum",
            "text": (
                "Write a Python function `two_sum(nums, target)` that returns the "
                "indices of the two numbers in `nums` that add up to `target`. "
                "Include type hints, a docstring, and handle the case where no "
                "solution exists."
            ),
        },
        {
            "category": "coding",
            "name": "binary_search_tree",
            "text": (
                "Implement a `BinarySearchTree` class in Python with "
                "`insert(value)`, `search(value)`, and `inorder_traversal()` "
                "methods. Include type hints and brief comments."
            ),
        },
        {
            "category": "coding",
            "name": "refactor_loop",
            "text": (
                "Refactor this Python code to be more idiomatic using "
                "comprehensions and built-ins, then briefly explain the changes:\n\n"
                "result = []\n"
                "for i in range(len(items)):\n"
                "    if items[i] is not None:\n"
                "        result.append(items[i].strip().lower())"
            ),
        },
        {
            "category": "general",
            "name": "data_storage",
            "text": (
                "Explain the difference between a data warehouse, a data lake, "
                "and a data lakehouse. Use clear language and provide examples."
            ),
        },
        {
            "category": "general",
            "name": "rest_vs_graphql",
            "text": (
                "Explain the key differences between REST and GraphQL APIs. "
                "When would you choose one over the other?"
            ),
        },
    ]

    runs_per_prompt = 3

    hardware = gather_hardware()
    batch_started_at = datetime.now(timezone.utc).isoformat()
    batch_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{hardware['hostname'] or 'unknown'}_llama"

    print(f"Hardware: {hardware['device_model']} | {hardware['chip']} | "
          f"{hardware['cpu_count']} cores | {hardware['memory_gb']} GB | "
          f"{hardware['os']} {hardware['os_version']}")
    print(f"Batch ID: {batch_id}\n")

    print("Warming up...")
    requests.post(
        COMPLETION_URL,
        json={"prompt": "hi", "n_predict": 1, "temperature": 0, "stream": False},
        timeout=120,
    ).raise_for_status()

    results = []

    for prompt_def in prompts:
        category = prompt_def["category"]
        name = prompt_def["name"]
        prompt_text = prompt_def["text"]

        print(f"  [{category}] {name}")

        for run in range(1, runs_per_prompt + 1):
            payload = {
                "prompt": prompt_text,
                "n_predict": args.num_predict,
                "temperature": 0,
                "stream": False,
            }

            start = time.time()
            response = requests.post(COMPLETION_URL, json=payload, timeout=600)
            wall_time = time.time() - start

            response.raise_for_status()
            data = response.json()

            output_tokens = data.get("tokens_predicted", 0)
            prompt_tokens = data.get("tokens_evaluated", 0)
            timings = data.get("timings", {})

            # llama-server returns ms; normalize to seconds for parity with Ollama.
            predicted_ms = timings.get("predicted_ms", 0.0)
            prompt_ms = timings.get("prompt_ms", 0.0)
            output_tps = timings.get("predicted_per_second") or (
                (output_tokens / predicted_ms * 1000.0) if predicted_ms > 0 else None
            )
            prompt_tps = timings.get("prompt_per_second") or (
                (prompt_tokens / prompt_ms * 1000.0) if prompt_ms > 0 else None
            )
            total_duration_sec = (predicted_ms + prompt_ms) / 1000.0

            results.append({
                "backend": "llama-server",
                "model": model_label,
                "category": category,
                "prompt_name": name,
                "run": run,
                "output_tokens": output_tokens,
                "output_tps": output_tps,
                "prompt_tokens": prompt_tokens,
                "prompt_tps": prompt_tps,
                "total_duration_sec": total_duration_sec,
                "load_duration_sec": 0.0,  # llama-server loads once at server start
                "wall_time_sec": wall_time,
                **hardware,
            })

            tps_str = f"{output_tps:.2f}" if output_tps else "?"
            print(
                f"    Run {run}: {tps_str} output tokens/sec | "
                f"{output_tokens} output tokens | {wall_time:.2f} sec wall time"
            )

    df = pd.DataFrame(results)

    summary = (
        df.groupby(["model", "category"])
        .agg(
            avg_output_tps=("output_tps", "mean"),
            min_output_tps=("output_tps", "min"),
            max_output_tps=("output_tps", "max"),
            avg_prompt_tps=("prompt_tps", "mean"),
            avg_total_duration_sec=("total_duration_sec", "mean"),
            avg_load_duration_sec=("load_duration_sec", "mean"),
            avg_wall_time_sec=("wall_time_sec", "mean"),
            avg_output_tokens=("output_tokens", "mean"),
        )
        .reset_index()
        .sort_values(["category", "avg_output_tps"], ascending=[True, False])
    )

    for col, val in hardware.items():
        summary[col] = val
    summary["backend"] = "llama-server"

    print("\nRaw results:")
    print(df)
    print("\nSummary by model and category:")
    print(summary)

    raw_path = OUTPUT_DIR / f"llama_benchmark_raw{suffix}.csv"
    summary_path = OUTPUT_DIR / f"llama_benchmark_summary{suffix}.csv"

    df.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)

    ensure_db()
    db_rows = df.assign(
        batch_id=batch_id,
        batch_started_at=batch_started_at,
        suffix=args.suffix or None,
    )[[
        "batch_id", "batch_started_at", "suffix", "backend",
        "model", "category", "prompt_name", "run",
        "output_tokens", "output_tps", "prompt_tokens", "prompt_tps",
        "total_duration_sec", "load_duration_sec", "wall_time_sec",
        "hostname", "device_model", "chip", "cpu_count", "memory_gb", "os", "os_version",
    ]]
    with sqlite3.connect(DB_PATH) as conn:
        db_rows.to_sql("benchmark_runs", conn, if_exists="append", index=False)

    print("\nSaved:")
    print(raw_path)
    print(summary_path)
    print(f"{DB_PATH}  (+{len(db_rows)} rows, batch_id={batch_id})")
finally:
    if server_proc is not None:
        stop_llama_server(server_proc)
