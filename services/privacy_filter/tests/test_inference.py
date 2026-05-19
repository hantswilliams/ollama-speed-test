"""End-to-end PII detection tests.

Each test runs a sample string through the high-level pipeline and asserts
the expected entity_group(s) appear in the output. Span boundaries are not
checked strictly — the simple aggregation strategy is enough to validate
the model is working without depending on Viterbi-specific boundary behavior.
"""

import pytest


def groups(output) -> set[str]:
    return {item["entity_group"] for item in output}


def test_harry_potter_example(classifier) -> None:
    """Exact example from the model card README."""
    text = "My name is Harry Potter and my email is harry.potter@hogwarts.edu."
    out = classifier(text)
    detected = groups(out)
    assert "private_person" in detected, f"expected private_person, got {detected}"
    assert "private_email" in detected, f"expected private_email, got {detected}"


def test_no_pii_returns_empty(classifier) -> None:
    text = "The quick brown fox jumps over the lazy dog."
    out = classifier(text)
    assert out == [], f"expected no entities, got {out}"


def test_multiple_categories_in_one_input(classifier) -> None:
    text = (
        "Hi, I'm Alice Johnson. You can reach me at alice.j@workmail.io "
        "or by phone at (415) 555-0123."
    )
    detected = groups(classifier(text))
    assert "private_person" in detected
    assert "private_email" in detected
    assert "private_phone" in detected


# Per-category sanity checks. Each input is a realistic-looking string
# that should obviously trip the corresponding detector.
PER_CATEGORY_CASES = [
    ("private_person", "My name is John Smith."),
    ("private_email", "Drop me a line at jane.doe@example.com anytime."),
    ("private_phone", "Call me at +1 (415) 555-0199 after 5pm."),
    ("private_address", "Please ship it to 1600 Pennsylvania Avenue NW, Washington, DC 20500."),
    ("account_number", "My credit card is 4532 1488 0343 6467 expiring next year."),
    ("private_url", "My personal site is https://johndoe.example/about-me."),
    ("private_date", "I was born on July 4, 1985 in San Francisco."),
    ("secret", "Use this API key: sk-proj-abc123XYZ456def789GHI012jkl345MNO678pqr901."),
]


@pytest.mark.parametrize("expected,text", PER_CATEGORY_CASES, ids=lambda x: x if isinstance(x, str) and x.startswith(("private", "account", "secret")) else None)
def test_detects_category(classifier, expected: str, text: str) -> None:
    detected = groups(classifier(text))
    assert expected in detected, (
        f"expected to detect {expected} in:\n  {text!r}\n"
        f"got: {sorted(detected)}"
    )


def test_output_shape(classifier) -> None:
    """Every entity has the documented schema."""
    out = classifier("Email me at test@example.com")
    assert len(out) >= 1
    entity = out[0]
    assert "entity_group" in entity
    assert "score" in entity
    assert "word" in entity
    assert "start" in entity
    assert "end" in entity
    assert 0.0 <= entity["score"] <= 1.0
