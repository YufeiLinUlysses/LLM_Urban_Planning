"""Recompute parser-dependent metrics from downloaded prediction artifacts."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

from urban_science_revision.pipelines.model_evaluation.nodes import (
    _aggregate,
    _classification_metrics,
    _extract_section,
    _normalize,
    _review_reason,
    _rouge_scores,
    _token_f1,
    parse_mc_prediction,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT = PROJECT_ROOT / "data" / "07_model_output" / "evaluations" / "revision_v3" / "t5_base"
RUNS = {
    "base": ROOT / "base" / "in_domain" / "t5-base-in-domain-v3",
    "fine_tuned": (
        ROOT
        / "fine_tuned"
        / "in_domain"
        / "t5-finetuned-in-domain-lr2e5-v3"
    ),
}


def _rescore(run_dir: Path) -> dict[str, object]:
    frame = pd.read_parquet(run_dir / "predictions.parquet")
    rows = frame.where(pd.notna(frame), None).to_dict(orient="records")
    answer_predictions = {
        (row["example_id"], row["evaluation_task"]): str(row.get("raw_prediction") or "")
        for row in rows
        if row["evaluation_task"] in {"mc_answer", "short_answer"}
    }

    for row in rows:
        task = row["evaluation_task"]
        prediction = str(row.get("raw_prediction") or "")
        reference_answer = str(row.get("reference_answer") or "")

        if task == "mc_answer":
            parsed, status, strict_format = parse_mc_prediction(
                prediction, str(row.get("prompt") or "")
            )
            row.update(
                parsed_answer=parsed,
                parser_status=status,
                strict_correct=bool(
                    strict_format and parsed == reference_answer.upper()
                ),
                normalized_correct=bool(parsed and parsed == reference_answer.upper()),
                format_compliant=strict_format,
            )
        elif task == "short_answer":
            parsed = _extract_section(prediction, "ANSWER") or prediction
            row.update(
                parsed_answer=parsed,
                exact_match=_normalize(parsed) == _normalize(reference_answer),
                token_f1=_token_f1(parsed, reference_answer),
                **_rouge_scores(parsed, reference_answer),
            )
        elif task in {"mc_explanation", "short_answer_explanation"}:
            answer_task = "mc_answer" if task == "mc_explanation" else "short_answer"
            full_prediction = answer_predictions[(row["example_id"], answer_task)]
            explanation = _extract_section(full_prediction, "EXPLANATION")
            parsed = _extract_section(full_prediction, "ANSWER") or full_prediction
            answer_correct = (
                parse_mc_prediction(full_prediction, str(row.get("prompt") or ""))[0]
                == reference_answer.upper()
                if task == "mc_explanation"
                else _normalize(parsed) == _normalize(reference_answer)
            )
            row.update(
                raw_prediction=explanation,
                parsed_answer=parsed,
                answer_explanation_consistent=bool(answer_correct and explanation),
                empty_response=not bool(explanation),
                response_length=len(explanation.split()),
                **_rouge_scores(
                    explanation, str(row.get("reference_explanation") or "")
                ),
            )
            # These semantic scores were computed against an erroneously empty
            # explanation and require a separate model-based metric pass.
            for key in (
                "bertscore_precision",
                "bertscore_recall",
                "bertscore_f1",
                "nli_entailment_score",
                "nli_entailment_correct",
            ):
                row[key] = math.nan
        elif task == "paraphrase":
            parsed = _extract_section(prediction, "ANSWER")
            row.update(
                parsed_answer=parsed,
                answer_preserved=_normalize(parsed) == _normalize(reference_answer),
                field_preservation=all(
                    label in prediction.upper() for label in ("ANSWER:", "EXPLANATION:")
                ),
            )

        row["review_reasons"] = _review_reason(row)

    overall = _aggregate(rows, ("evaluation_task",))
    grouped = _aggregate(rows, ("evaluation_task", "source_name", "Level", "region"))
    verification = [row for row in rows if row["evaluation_task"] == "answer_verification"]
    verification_metrics = _classification_metrics(
        [str(row["reference_answer"]) for row in verification],
        [str(row["parsed_answer"]) for row in verification],
    )
    metrics = {"overall": overall, "verification": verification_metrics}

    pd.DataFrame(rows).to_parquet(run_dir / "predictions_rescored.parquet", index=False)
    pd.DataFrame(grouped).to_parquet(run_dir / "grouped_metrics_rescored.parquet", index=False)
    (run_dir / "metrics_rescored.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=True), encoding="utf-8"
    )
    return metrics


def main() -> None:
    results = {label: _rescore(run_dir) for label, run_dir in RUNS.items()}
    tasks = sorted(
        {row["evaluation_task"] for result in results.values() for row in result["overall"]}
    )
    lookup = {
        label: {row["evaluation_task"]: row for row in result["overall"]}
        for label, result in results.items()
    }
    comparison = []
    for task in tasks:
        numeric = sorted(set(lookup["base"][task]) & set(lookup["fine_tuned"][task]))
        for metric in numeric:
            if metric in {"evaluation_task", "count", "Level"}:
                continue
            base = lookup["base"][task][metric]
            tuned = lookup["fine_tuned"][task][metric]
            if isinstance(base, (int, float)) and isinstance(tuned, (int, float)):
                comparison.append(
                    {
                        "evaluation_task": task,
                        "metric": metric,
                        "base": base,
                        "fine_tuned": tuned,
                        "delta": tuned - base,
                    }
                )
    output = ROOT / "in_domain_rescored_comparison.csv"
    pd.DataFrame(comparison).to_csv(output, index=False)
    print(pd.DataFrame(comparison).to_string(index=False))
    print(f"\nRescored comparison: {output}")


if __name__ == "__main__":
    main()
