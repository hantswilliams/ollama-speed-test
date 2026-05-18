import argparse
import requests
import pandas as pd
import time
from pathlib import Path
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OLLAMA_URL = "http://localhost:11435/api/generate"

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

print("\nRaw results:")
print(df)

print("\nSummary by model and category:")
print(summary)

raw_path = OUTPUT_DIR / f"ollama_benchmark_raw{suffix}.csv"
summary_path = OUTPUT_DIR / f"ollama_benchmark_summary{suffix}.csv"

df.to_csv(raw_path, index=False)
summary.to_csv(summary_path, index=False)

print("\nSaved:")
print(raw_path)
print(summary_path)

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

