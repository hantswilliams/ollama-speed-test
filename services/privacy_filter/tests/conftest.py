"""Shared pytest fixtures for privacy-filter tests.

The model is large (~3 GB) and slow to load, so we load it once per session.
All tests point at the local on-disk copy under services/privacy_filter/models/
so they run offline once download.py has been run.
"""

from pathlib import Path

import pytest

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "privacy-filter"


@pytest.fixture(scope="session")
def model_dir() -> Path:
    if not MODEL_DIR.exists() or not (MODEL_DIR / "config.json").exists():
        pytest.skip(
            f"Model not downloaded yet. Run `python download.py` first. "
            f"Expected at {MODEL_DIR}",
            allow_module_level=False,
        )
    return MODEL_DIR


@pytest.fixture(scope="session")
def tokenizer(model_dir: Path):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(model_dir))


@pytest.fixture(scope="session")
def model(model_dir: Path):
    from transformers import AutoModelForTokenClassification
    return AutoModelForTokenClassification.from_pretrained(str(model_dir))


@pytest.fixture(scope="session")
def classifier(model_dir: Path):
    """High-level pipeline; aggregates BIOES tokens into entity spans."""
    from transformers import pipeline
    return pipeline(
        task="token-classification",
        model=str(model_dir),
        aggregation_strategy="simple",
    )
