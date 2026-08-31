from __future__ import annotations

import sys
from types import SimpleNamespace

from urban_science_revision.pipelines.model_evaluation.nodes import (
    _attach_bertscore,
    _classification_metrics,
    _token_f1,
    extract_reference,
    parse_mc_prediction,
)


def test_bertscore_skips_empty_predictions(monkeypatch) -> None:
    def fake_score(candidates, references, **kwargs):
        assert candidates == ["generated answer"]
        assert references == ["reference explanation"]
        return [0.7], [0.8], [0.75]

    monkeypatch.setitem(sys.modules, "bert_score", SimpleNamespace(score=fake_score))
    rows = [
        {
            "raw_prediction": "",
            "reference_answer": "reference",
            "reference_explanation": "explanation",
        },
        {
            "raw_prediction": "generated answer",
            "reference_answer": "reference",
            "reference_explanation": "explanation",
        },
    ]

    _attach_bertscore(rows, {"compute_bertscore": True})

    assert rows[0]["bertscore_f1"] == 0.0
    assert rows[1]["bertscore_precision"] == 0.7
    assert rows[1]["bertscore_recall"] == 0.8
    assert rows[1]["bertscore_f1"] == 0.75


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


def test_structured_sections_allow_inline_or_multiline_labels() -> None:
    inline = "ANSWER: B EXPLANATION: Because rail is correct."
    multiline = "ANSWER:\nB\n\nEXPLANATION:\nBecause rail is correct."
    prompt = "Options:\nA. Bus\nB. Rail\nC. Car\nD. Walk"

    assert extract_reference(inline) == ("B", "Because rail is correct.")
    assert extract_reference(multiline) == ("B", "Because rail is correct.")
    assert parse_mc_prediction(inline, prompt) == ("B", "strict_structured", True)
    assert parse_mc_prediction(multiline, prompt) == ("B", "strict_structured", True)
    assert extract_reference("unstructured output") == ("", "")
    verification = (
        "VERDICT: incorrect EXPLANATION: The candidate is wrong. "
        "CORRECT ANSWER: B"
    )
    assert extract_reference(verification) == (
        "incorrect",
        "The candidate is wrong.",
    )


def test_transparent_text_and_verification_metrics() -> None:
    assert _token_f1("regional transport authority", "transport authority") == 0.8
    metrics = _classification_metrics(
        ["correct", "incorrect", "incorrect"],
        ["correct", "incorrect", "correct"],
    )
    assert metrics["accuracy"] == 2 / 3
    assert metrics["confusion_matrix"]["incorrect"]["correct"] == 1
