from __future__ import annotations

from urban_science_revision.pipelines.model_training.nodes import _limit_rows


def test_smoke_sample_limit_is_stable_and_exact() -> None:
    rows = [{"prompt": f"Prompt {index}", "target": f"Target {index}"} for index in range(500)]
    first = _limit_rows(rows, 200, 42)
    second = _limit_rows(list(reversed(rows)), 200, 42)
    assert len(first) == 200
    assert first == second
