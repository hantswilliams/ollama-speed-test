import argparse
import json
import platform
import re
import sqlite3
import subprocess
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = OUTPUT_DIR / "benchmarks.db"

OLLAMA_URL = "http://localhost:11435/api/generate"


def gather_hardware() -> dict:
    """Capture host hardware info once per benchmark run. Best-effort, fails soft."""
    import os
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
        # Add backend column to any pre-existing DB created before this column existed
        try:
            conn.execute("ALTER TABLE benchmark_runs ADD COLUMN backend TEXT")
        except sqlite3.OperationalError:
            pass  # column already exists
        conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_batch ON benchmark_runs(batch_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_model ON benchmark_runs(model)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_backend ON benchmark_runs(backend)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_hostname ON benchmark_runs(hostname)")

DEFAULT_MODELS = [
    "gemma4:e4b",
    "qwen3-coder:30b-a3b-q4_K_M",
]

parser = argparse.ArgumentParser(description="Benchmark Ollama models on coding and general prompts.")
parser.add_argument(
    "--models",
    nargs="+",
    default=DEFAULT_MODELS,
    help="One or more Ollama model tags to benchmark. Defaults to all configured models.",
)
parser.add_argument(
    "--suffix",
    default="",
    help="Optional suffix appended to output CSV filenames (e.g. 'deepseek' -> ollama_benchmark_raw_deepseek.csv).",
)
parser.add_argument(
    "--no-visualize",
    action="store_true",
    help="Skip generating visualizations (default is to generate them).",
)
args = parser.parse_args()
models = args.models
suffix = f"_{args.suffix}" if args.suffix else ""

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
batch_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{hardware['hostname'] or 'unknown'}"

print(f"Hardware: {hardware['device_model']} | {hardware['chip']} | "
      f"{hardware['cpu_count']} cores | {hardware['memory_gb']} GB | "
      f"{hardware['os']} {hardware['os_version']}")
print(f"Batch ID: {batch_id}\n")

results = []

for model in models:
    print(f"\nBenchmarking: {model}")

    print("  warming up...")
    requests.post(
        OLLAMA_URL,
        json={
            "model": model,
            "prompt": "hi",
            "stream": False,
            "keep_alive": "30m",
            "options": {"num_predict": 1},
        },
    ).raise_for_status()

    for prompt_def in prompts:
        category = prompt_def["category"]
        name = prompt_def["name"]
        prompt_text = prompt_def["text"]

        print(f"  [{category}] {name}")

        for run in range(1, runs_per_prompt + 1):
            payload = {
                "model": model,
                "prompt": prompt_text,
                "stream": False,
                "keep_alive": "30m",
                "options": {
                    "temperature": 0,
                    "num_predict": 300,
                    "num_ctx": 4096,
                    "num_thread": 12,
                },
            }

            start = time.time()
            response = requests.post(OLLAMA_URL, json=payload)
            wall_time = time.time() - start

            response.raise_for_status()
            data = response.json()

            eval_count = data.get("eval_count", 0)
            eval_duration = data.get("eval_duration", 0)

            prompt_eval_count = data.get("prompt_eval_count", 0)
            prompt_eval_duration = data.get("prompt_eval_duration", 0)

            total_duration = data.get("total_duration", 0)
            load_duration = data.get("load_duration", 0)

            output_tps = (
                eval_count / eval_duration * 1_000_000_000
                if eval_duration > 0
                else None
            )

            prompt_tps = (
                prompt_eval_count / prompt_eval_duration * 1_000_000_000
                if prompt_eval_duration > 0
                else None
            )

            results.append({
                "backend": "ollama",
                "model": model,
                "category": category,
                "prompt_name": name,
                "run": run,
                "output_tokens": eval_count,
                "output_tps": output_tps,
                "prompt_tokens": prompt_eval_count,
                "prompt_tps": prompt_tps,
                "total_duration_sec": total_duration / 1_000_000_000,
                "load_duration_sec": load_duration / 1_000_000_000,
                "wall_time_sec": wall_time,
                **hardware,
            })

            print(
                f"    Run {run}: "
                f"{output_tps:.2f} output tokens/sec | "
                f"{eval_count} output tokens | "
                f"{wall_time:.2f} sec wall time"
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

# Hardware is constant for this run; carry it through the summary too
for col, val in hardware.items():
    summary[col] = val
summary["backend"] = "ollama"

print("\nRaw results:")
print(df)

print("\nSummary by model and category:")
print(summary)

raw_path = OUTPUT_DIR / f"ollama_benchmark_raw{suffix}.csv"
summary_path = OUTPUT_DIR / f"ollama_benchmark_summary{suffix}.csv"

df.to_csv(raw_path, index=False)
summary.to_csv(summary_path, index=False)

# Append raw rows to the cross-run SQLite store
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

# Only generate visualizations if not disabled
if not args.no_visualize:
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Set style for better-looking plots
        sns.set_style("whitegrid")
        plt.rcParams.update({'font.size': 10})
        
        # Create performance comparison chart
        plt.figure(figsize=(12, 8))
        
        # Bar chart of average output tokens/sec by model and category
        sns.barplot(data=summary, x='model', y='avg_output_tps', hue='category')
        plt.title('Average Output Tokens/Second by Model and Category')
        plt.xlabel('Model')
        plt.ylabel('Average Output Tokens/Second')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save the chart
        chart_path = OUTPUT_DIR / f"performance_comparison{suffix}.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization: {chart_path}")
        
        plt.figure(figsize=(12, 8))
        
        # Scatter plot of tokens vs performance
        sns.scatterplot(data=summary, x='avg_output_tokens', y='avg_output_tps', hue='model', s=100)
        plt.title('Performance vs Token Count')
        plt.xlabel('Average Output Tokens')
        plt.ylabel('Average Output Tokens/Second')
        plt.tight_layout()
        
        # Save the second chart
        scatter_path = OUTPUT_DIR / f"performance_vs_tokens{suffix}.png"
        plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization: {scatter_path}")
        
        # Create a summary table as text file for easy reading
        summary_txt = OUTPUT_DIR / f"performance_summary{suffix}.txt"
        with open(summary_txt, 'w') as f:
            f.write("Ollama Benchmark Performance Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(summary.to_string(index=False))
            f.write("\n\n")
            f.write("Best Performing Models:\n")
            best_models = summary.groupby('category')['avg_output_tps'].idxmax()
            for cat, idx in best_models.items():
                model_name = summary.loc[idx, 'model']
                tps = summary.loc[idx, 'avg_output_tps']
                f.write(f"  {cat}: {model_name} ({tps:.1f} tokens/sec)\n")
        
        print(f"Saved summary text file: {summary_txt}")
        
    except ImportError:
        print("Visualization libraries not available. Install with: uv pip install matplotlib seaborn")
    except Exception as e:
        print(f"Error generating visualizations: {e}")

