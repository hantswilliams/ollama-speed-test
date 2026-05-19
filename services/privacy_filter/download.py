"""Download openai/privacy-filter from Hugging Face into a local cache.

Pulls weights, tokenizer, and config into services/privacy_filter/models/privacy-filter/
so we have a local copy regardless of network access later.

Reads HF_TOKEN from the repo-root .env file (via python-dotenv). Anonymous
downloads hit aggressive HF rate limits, so a token is strongly recommended.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import snapshot_download

MODEL_ID = "openai/privacy-filter"
LOCAL_DIR = Path(__file__).resolve().parent / "models" / "privacy-filter"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    token = os.environ.get("HF_TOKEN") or None
    if token:
        print(f"Using HF_TOKEN from {PROJECT_ROOT / '.env'} (length {len(token)})")
    else:
        print("WARNING: no HF_TOKEN found — downloads will be rate-limited.")
        print(f"  Set HF_TOKEN in {PROJECT_ROOT / '.env'} and re-run.")

    LOCAL_DIR.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nDownloading {MODEL_ID} -> {LOCAL_DIR}")
    print("(first run pulls ~3 GB of weights; subsequent runs resume / no-op if files exist)\n")

    path = Path(snapshot_download(repo_id=MODEL_ID, local_dir=str(LOCAL_DIR), token=token))

    files = sorted(f for f in path.rglob("*") if f.is_file())
    total_mb = sum(f.stat().st_size for f in files) / 1024 / 1024

    print(f"\nDownloaded {len(files)} files, {total_mb:,.1f} MB total\n")
    for f in files:
        rel = f.relative_to(path)
        size_mb = f.stat().st_size / 1024 / 1024
        print(f"  {size_mb:>8.1f} MB   {rel}")


if __name__ == "__main__":
    main()
