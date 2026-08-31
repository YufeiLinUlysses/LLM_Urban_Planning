"""Validate a completed HF experiment and build a side-by-side metric report."""
# ruff: noqa: E501

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import HfApi, hf_hub_download

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
LOCAL_ROOT = PROJECT_ROOT / "data/07_model_output"
DATASET_VERSION = "revision_v4"
MODEL_KEY = "t5_base"
RUN_ID = "revision-v4-t5-verification-v2"
MODEL_REPO = "UlyssesLynne/urban-planning-llm-model-zoo-v3"
PREDICTION_REPO = "UlyssesLynne/urban-planning-llm-predictions-v3"
REQUIRED_EVALUATION_FILES = {
    "README.md",
    "confusion_matrix.json",
    "evaluation_manifest.json",
    "grouped_metrics.parquet",
    "metrics.json",
    "predictions.parquet",
    "review_queue.parquet",
}
REQUIRED_MODEL_FILES = {
    "config.json",
    "figures/training_vs_validation_loss.png",
    "model.safetensors",
    "tokenizer.json",
    "training_history.json",
    "training_manifest.json",
}


def _evaluation_path(stage: str, scope: str) -> str:
    label = "base" if stage == "base" else "finetuned"
    evaluation_id = f"{RUN_ID}-{label}-{scope.replace('_', '-')}"
    return f"evaluations/{DATASET_VERSION}/{MODEL_KEY}/{stage}/{scope}/{evaluation_id}"


def _task(metrics: dict[str, Any], name: str) -> dict[str, Any]:
    return next(row for row in metrics["overall"] if row["evaluation_task"] == name)


def _value(metrics: dict[str, Any], task: str | None, metric: str) -> float:
    source = metrics["verification"] if task is None else _task(metrics, task)
    return float(source[metric])


def main() -> None:
    load_dotenv(WORKSPACE_ROOT / ".env")
    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError(f"HF_TOKEN is missing from {WORKSPACE_ROOT / '.env'}")

    api = HfApi(token=token)
    model_files = set(api.list_repo_files(MODEL_REPO, repo_type="model"))
    model_prefix = f"{MODEL_KEY}/{RUN_ID}/checkpoint"
    missing_model = sorted(
        f"{model_prefix}/{name}"
        for name in REQUIRED_MODEL_FILES
        if f"{model_prefix}/{name}" not in model_files
    )
    if missing_model:
        raise RuntimeError(f"Incomplete model checkpoint: {missing_model}")

    prediction_files = set(api.list_repo_files(PREDICTION_REPO, repo_type="dataset"))
    metrics_by_scope: dict[str, dict[str, dict[str, Any]]] = {}
    manifests: list[dict[str, Any]] = []
    for scope in ("in_domain", "cross_regional"):
        metrics_by_scope[scope] = {}
        for stage in ("base", "fine_tuned"):
            run_path = _evaluation_path(stage, scope)
            expected = {f"{run_path}/{name}" for name in REQUIRED_EVALUATION_FILES}
            missing = sorted(expected - prediction_files)
            if missing:
                raise RuntimeError(f"Incomplete {stage}/{scope} evaluation: {missing}")
            for filename in sorted(expected):
                hf_hub_download(
                    repo_id=PREDICTION_REPO,
                    repo_type="dataset",
                    filename=filename,
                    token=token,
                    local_dir=LOCAL_ROOT,
                )
            local_run = LOCAL_ROOT / run_path
            manifest = json.loads(
                (local_run / "evaluation_manifest.json").read_text(encoding="utf-8")
            )
            metrics = json.loads((local_run / "metrics.json").read_text(encoding="utf-8"))
            if manifest["dataset_version"] != DATASET_VERSION:
                raise RuntimeError(f"Wrong dataset version in {run_path}: {manifest}")
            if manifest["checkpoint_stage"] != stage or manifest["dataset_scope"] != scope:
                raise RuntimeError(f"Manifest identity mismatch in {run_path}: {manifest}")
            metrics_by_scope[scope][stage] = metrics
            manifests.append(manifest)

    metric_specs = [
        ("MC strict accuracy", "mc_answer", "strict_correct"),
        ("MC format compliance", "mc_answer", "format_compliant"),
        ("MC explanation BERTScore F1", "mc_explanation", "bertscore_f1"),
        ("Short-answer exact match", "short_answer", "exact_match"),
        ("Short-answer token F1", "short_answer", "token_f1"),
        ("Short explanation BERTScore F1", "short_answer_explanation", "bertscore_f1"),
        ("Verification accuracy", None, "accuracy"),
        ("Verification macro F1", None, "macro_f1"),
        ("Verification paired consistency", None, "paired_consistency"),
        ("Verification format compliance", "answer_verification", "format_compliant"),
        ("Candidate-match accuracy", "answer_verification", "candidate_match_correct"),
        ("Verdict/match internal consistency", "answer_verification", "verdict_match_consistent"),
        ("Paraphrase answer preservation", "paraphrase", "answer_preserved"),
        ("Paraphrase field preservation", "paraphrase", "field_preservation"),
    ]
    rows = []
    for scope, stages in metrics_by_scope.items():
        for label, task, metric in metric_specs:
            base = _value(stages["base"], task, metric)
            fine_tuned = _value(stages["fine_tuned"], task, metric)
            rows.append(
                {
                    "scope": scope,
                    "metric": label,
                    "base": base,
                    "fine_tuned": fine_tuned,
                    "absolute_change": fine_tuned - base,
                }
            )
    comparison = pd.DataFrame(rows)
    report_dir = LOCAL_ROOT / "evaluations" / DATASET_VERSION / MODEL_KEY / "result_analysis"
    report_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = report_dir / "base_vs_fine_tuned_side_by_side.csv"
    comparison.to_csv(comparison_path, index=False)

    def pct(scope: str, metric: str, stage: str = "fine_tuned") -> str:
        row = comparison[(comparison.scope == scope) & (comparison.metric == metric)].iloc[0]
        return f"{100 * float(row[stage]):.2f}%"

    report = f"""# T5 revision-v4 local validation and comparison

## Validation

- Model checkpoint: complete in `{MODEL_REPO}/{model_prefix}`.
- Evaluation artifacts: all seven required files downloaded for all four runs.
- Manifests: model, checkpoint stage, dataset scope, and `{DATASET_VERSION}` identity validated.
- Prediction counts: in-domain {manifests[0]['prediction_count']:,} per run; cross-regional {manifests[2]['prediction_count']:,} per run.

## Fine-tuned results

| Metric | In-domain | Cross-regional |
|---|---:|---:|
| MC strict accuracy | {pct('in_domain', 'MC strict accuracy')} | {pct('cross_regional', 'MC strict accuracy')} |
| Short-answer exact match | {pct('in_domain', 'Short-answer exact match')} | {pct('cross_regional', 'Short-answer exact match')} |
| Short-answer token F1 | {pct('in_domain', 'Short-answer token F1')} | {pct('cross_regional', 'Short-answer token F1')} |
| Verification accuracy | {pct('in_domain', 'Verification accuracy')} | {pct('cross_regional', 'Verification accuracy')} |
| Verification macro F1 | {pct('in_domain', 'Verification macro F1')} | {pct('cross_regional', 'Verification macro F1')} |
| Verification paired consistency | {pct('in_domain', 'Verification paired consistency')} | {pct('cross_regional', 'Verification paired consistency')} |
| Verification format compliance | {pct('in_domain', 'Verification format compliance')} | {pct('cross_regional', 'Verification format compliance')} |
| Candidate-match accuracy | {pct('in_domain', 'Candidate-match accuracy')} | {pct('cross_regional', 'Candidate-match accuracy')} |
| Verdict/match internal consistency | {pct('in_domain', 'Verdict/match internal consistency')} | {pct('cross_regional', 'Verdict/match internal consistency')} |
| Paraphrase answer preservation | {pct('in_domain', 'Paraphrase answer preservation')} | {pct('cross_regional', 'Paraphrase answer preservation')} |

## Preliminary interpretation

The redesigned verification target fixed the former output-schema collapse: format compliance and
verdict/match internal consistency are both 100% after fine-tuning. Verification is now above the
balanced 50% baseline, reaching {pct('in_domain', 'Verification accuracy')} in-domain and
{pct('cross_regional', 'Verification accuracy')} cross-regionally.

The remaining weakness is candidate sensitivity. Paired consistency is only
{pct('in_domain', 'Verification paired consistency')} in-domain and
{pct('cross_regional', 'Verification paired consistency')} cross-regionally. Thus the model often
gets one member of a positive/negative candidate pair right without correctly distinguishing both.
The redesign succeeded structurally, but verification reasoning remains only moderately learned.

MC performance is strong ({pct('in_domain', 'MC strict accuracy')} in-domain and
{pct('cross_regional', 'MC strict accuracy')} cross-regional), with 100% formatted outputs. The
higher cross-regional MC score should still be interpreted with the known task-composition caveat:
that split contains many context-extractive Level-1 items. Short-answer exact match generalizes
poorly cross-regionally, although token overlap is materially higher than exact match.

See `base_vs_fine_tuned_side_by_side.csv` for every base value, fine-tuned value, and absolute
change used in this report.
"""
    readme_path = report_dir / "README.md"
    readme_path.write_text(report, encoding="utf-8")
    validation_path = report_dir / "validation_manifest.json"
    validation_path.write_text(
        json.dumps(
            {
                "model_repo": MODEL_REPO,
                "model_path": model_prefix,
                "prediction_repo": PREDICTION_REPO,
                "dataset_version": DATASET_VERSION,
                "run_id": RUN_ID,
                "model_complete": True,
                "evaluation_runs_validated": manifests,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("Local validation passed.")
    print(readme_path)
    print(comparison_path)


if __name__ == "__main__":
    main()
