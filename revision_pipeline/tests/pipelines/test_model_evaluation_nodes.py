from __future__ import annotations

from urban_science_revision.pipelines.model_evaluation.nodes import (
    _classification_metrics,
    _token_f1,
    extract_reference,
    parse_mc_prediction,
)


def test_reference_and_mc_parsing_are_deterministic() -> None:
    answer, explanation = extract_reference("ANSWER:\nB\n\nEXPLANATION:\nBecause rail.")
    assert answer == "B"
    assert explanation == "Because rail."
    prompt = "Options:\nA. Bus\nB. Rail\nC. Car\nD. Walk"
    assert parse_mc_prediction("B", prompt) == ("B", "strict", True)
    assert parse_mc_prediction("The answer is B.", prompt) == (
        "B",
        "normalized_letter",
        False,
    )
    assert parse_mc_prediction("Rail", prompt) == ("B", "normalized_option_text", False)


def test_transparent_text_and_verification_metrics() -> None:
    assert _token_f1("regional transport authority", "transport authority") == 0.8
    metrics = _classification_metrics(
        ["correct", "incorrect", "incorrect"],
        ["correct", "incorrect", "correct"],
    )
    assert metrics["accuracy"] == 2 / 3
    assert metrics["confusion_matrix"]["incorrect"]["correct"] == 1
