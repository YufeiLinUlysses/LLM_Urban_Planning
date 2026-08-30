"""Paired comparison of already generated evaluation artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def compare_evaluation_runs(parameters: dict[str, Any]) -> dict[str, Any]:
    """Compare per-example numeric metrics without running inference again."""

    import pandas as pd

    base_path = Path(parameters["base_predictions_path"])
    tuned_path = Path(parameters["fine_tuned_predictions_path"])
    if not base_path.is_file() or not tuned_path.is_file():
        raise FileNotFoundError("Both evaluation_comparison prediction files must exist locally")
    base = pd.read_parquet(base_path)
    tuned = pd.read_parquet(tuned_path)
    identity = ["example_id", "evaluation_task"]
    merged = base.merge(
        tuned, on=identity, suffixes=("_base", "_fine_tuned"), validate="one_to_one"
    )
    requested = parameters.get(
        "metrics",
        [
            "strict_correct",
            "normalized_correct",
            "format_compliant",
            "exact_match",
            "token_f1",
            "rouge1",
            "rouge2",
            "rougeL",
            "bertscore_f1",
            "nli_entailment_correct",
            "correct",
            "correction_exact_match",
            "answer_preserved",
            "field_preservation",
        ],
    )
    deltas: dict[str, Any] = {}
    for metric in requested:
        left = f"{metric}_base"
        right = f"{metric}_fine_tuned"
        if left not in merged or right not in merged:
            continue
        valid = merged[[left, right]].dropna().astype(float)
        if valid.empty:
            continue
        base_mean = float(valid[left].mean())
        tuned_mean = float(valid[right].mean())
        deltas[metric] = {
            "paired_count": len(valid),
            "base": base_mean,
            "fine_tuned": tuned_mean,
            "delta": tuned_mean - base_mean,
        }
    return {
        "base_predictions_path": str(base_path.resolve()),
        "fine_tuned_predictions_path": str(tuned_path.resolve()),
        "paired_prediction_count": len(merged),
        "metrics": deltas,
    }
