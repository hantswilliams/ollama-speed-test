"""Smoke tests: model files exist, tokenizer + model + pipeline all load."""

import json
from pathlib import Path


EXPECTED_CATEGORIES = {
    "account_number",
    "private_address",
    "private_date",
    "private_email",
    "private_person",
    "private_phone",
    "private_url",
    "secret",
}


def test_model_files_present(model_dir: Path) -> None:
    for required in ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"]:
        assert (model_dir / required).exists(), f"missing {required}"


def test_config_has_expected_labels(model_dir: Path) -> None:
    cfg = json.loads((model_dir / "config.json").read_text())
    assert cfg["model_type"] == "openai_privacy_filter"
    assert cfg["architectures"] == ["OpenAIPrivacyFilterForTokenClassification"]

    # BIOES expansion: 1 background + 8 categories * 4 boundary tags = 33 classes
    assert len(cfg["id2label"]) == 33
    assert cfg["id2label"]["0"] == "O"

    categories_in_config = {
        label.split("-", 1)[1]
        for label in cfg["id2label"].values()
        if label != "O"
    }
    assert categories_in_config == EXPECTED_CATEGORIES


def test_tokenizer_loads(tokenizer) -> None:
    encoded = tokenizer("hello world", return_tensors="pt")
    assert "input_ids" in encoded
    assert encoded["input_ids"].shape[0] == 1
    assert encoded["input_ids"].shape[1] >= 2


def test_model_loads(model) -> None:
    assert model.config.model_type == "openai_privacy_filter"
    assert model.config.num_hidden_layers == 8
    assert model.config.hidden_size == 640
    assert len(model.config.id2label) == 33


def test_pipeline_loads(classifier) -> None:
    assert classifier is not None
    assert callable(classifier)
