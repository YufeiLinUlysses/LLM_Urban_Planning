from __future__ import annotations

from urban_science_revision.pipelines.model_training.nodes import _limit_rows, _training_rows


def test_training_rows_include_all_three_objectives() -> None:
    def partition(prompt: str) -> dict[str, object]:
        return {"split": "train", "records": [{"prompt": prompt, "target": f"{prompt}-target"}]}

    rows = _training_rows(
        {"source": partition("generation")},
        {"source": partition("verification")},
        {"source": partition("paraphrase")},
        "train",
        ["generation", "verification", "paraphrase"],
    )

    assert {row["prompt"] for row in rows} == {
        "generation",
        "verification",
        "paraphrase",
    }


def test_smoke_sample_limit_is_stable_and_exact() -> None:
    rows = [{"prompt": f"Prompt {index}", "target": f"Target {index}"} for index in range(500)]
    first = _limit_rows(rows, 200, 42)
    second = _limit_rows(list(reversed(rows)), 200, 42)
    assert len(first) == 200
    assert first == second
