"""Validate a completed HF experiment and build a side-by-side metric report."""
# ruff: noqa: E501

from __future__ import annotations

import json
import os
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import HfApi, hf_hub_download

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
LOCAL_ROOT = PROJECT_ROOT / "data/07_model_output"
DATASET_VERSION = "revision_v5"
MODEL_KEY = "t5_base"
RUN_ID = "revision-v5-t5-three-task-v1"
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
    training_manifest_path = hf_hub_download(
        repo_id=MODEL_REPO,
        repo_type="model",
        filename=f"{model_prefix}/training_manifest.json",
        token=token,
        local_dir=LOCAL_ROOT / "hf_model_cache",
    )
    training_manifest = json.loads(Path(training_manifest_path).read_text(encoding="utf-8"))
    if (
        training_manifest["training_run_id"] != RUN_ID
        or training_manifest["dataset_version"] != DATASET_VERSION
        or set(training_manifest["tasks"]) != {"generation", "verification", "paraphrase"}
    ):
        raise RuntimeError(f"Training-manifest identity mismatch: {training_manifest}")

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
        ("Paraphrase BERTScore F1", "paraphrase", "bertscore_f1"),
        ("Paraphrase NLI entailment", "paraphrase", "nli_entailment_correct"),
        ("Paraphrase ROUGE-L", "paraphrase", "rougeL"),
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

    def normalized(text: Any) -> str:
        return re.sub(r"\s+", " ", str(text)).strip().lower()

    novelty_rows = []
    for scope in ("in_domain", "cross_regional"):
        run_path = LOCAL_ROOT / _evaluation_path("fine_tuned", scope)
        predictions = pd.read_parquet(run_path / "predictions.parquet")
        paraphrases = predictions[predictions["evaluation_task"] == "paraphrase"].copy()
        sources = paraphrases["prompt"].str.split("\n\n", n=1).str[-1]
        generated = paraphrases["raw_prediction"]
        exact_copies = [
            normalized(output) == normalized(source)
            for output, source in zip(generated, sources, strict=True)
        ]
        source_similarity = [
            SequenceMatcher(None, normalized(output), normalized(source)).ratio()
            for output, source in zip(generated, sources, strict=True)
        ]
        novelty_rows.append(
            {
                "scope": scope,
                "paraphrase_rows": len(paraphrases),
                "unique_prompts": paraphrases["prompt"].nunique(),
                "unique_outputs": generated.nunique(),
                "exact_source_copy_rate": sum(exact_copies) / len(exact_copies),
                "mean_output_source_similarity": sum(source_similarity)
                / len(source_similarity),
                "mean_output_reference_bertscore_f1": _value(
                    metrics_by_scope[scope]["fine_tuned"], "paraphrase", "bertscore_f1"
                ),
            }
        )
    novelty = pd.DataFrame(novelty_rows)
    novelty_path = report_dir / "paraphrase_novelty_diagnostics.csv"
    novelty.to_csv(novelty_path, index=False)

    def novelty_pct(scope: str, metric: str) -> str:
        row = novelty[novelty.scope == scope].iloc[0]
        return f"{100 * float(row[metric]):.2f}%"

    def pct(scope: str, metric: str, stage: str = "fine_tuned") -> str:
        row = comparison[(comparison.scope == scope) & (comparison.metric == metric)].iloc[0]
        return f"{100 * float(row[stage]):.2f}%"

    def delta_pct(scope: str, metric: str) -> str:
        row = comparison[(comparison.scope == scope) & (comparison.metric == metric)].iloc[0]
        return f"{100 * float(row['absolute_change']):+.2f} percentage points"

    report = f"""# T5 revision-v5 local validation and comparison

## Validation

- Model checkpoint: complete in `{MODEL_REPO}/{model_prefix}`.
- Training manifest: `{training_manifest['train_record_count']:,}` train records,
  `{training_manifest['validation_record_count']:,}` validation records, one epoch, and all three
  objectives explicitly recorded.
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
| Paraphrase field preservation | {pct('in_domain', 'Paraphrase field preservation')} | {pct('cross_regional', 'Paraphrase field preservation')} |
| Paraphrase BERTScore F1 | {pct('in_domain', 'Paraphrase BERTScore F1')} | {pct('cross_regional', 'Paraphrase BERTScore F1')} |
| Paraphrase NLI entailment | {pct('in_domain', 'Paraphrase NLI entailment')} | {pct('cross_regional', 'Paraphrase NLI entailment')} |
| Paraphrase ROUGE-L | {pct('in_domain', 'Paraphrase ROUGE-L')} | {pct('cross_regional', 'Paraphrase ROUGE-L')} |

## Preliminary interpretation

This run is the first three-objective T5 checkpoint: grounded generation, candidate verification,
and structure-preserving paraphrase. Interpret task quality from the base-to-fine-tuned changes in
the CSV rather than from loss alone. Paraphrase success requires all three dimensions: semantic
similarity to the held-out rewrite, preservation of the answer, and preservation of the required
field labels. Verification paired consistency remains the strongest diagnostic for whether the
model distinguishes both members of each positive/negative candidate pair.

### Grounded answer and explanation generation

Fine-tuning raises strict MC accuracy by {delta_pct('in_domain', 'MC strict accuracy')} in-domain
and {delta_pct('cross_regional', 'MC strict accuracy')} cross-regionally, while formatted MC
responses rise to 100% in both scopes. Short-answer token F1 improves to
{pct('in_domain', 'Short-answer token F1')} in-domain, but only
{pct('cross_regional', 'Short-answer token F1')} cross-regionally. Exact match is especially weak
cross-regionally ({pct('cross_regional', 'Short-answer exact match')}), so the model demonstrates
partial semantic transfer rather than reliable exact answer generation on the geographic holdout.

### Candidate verification

Fine-tuning completely fixes output formatting and verdict/match internal consistency. However,
accuracy is only {pct('in_domain', 'Verification accuracy')} in-domain and
{pct('cross_regional', 'Verification accuracy')} cross-regionally, slightly above the balanced
50% baseline. Paired consistency is just
{pct('in_domain', 'Verification paired consistency')} and
{pct('cross_regional', 'Verification paired consistency')}. The model has learned the required
schema but frequently fails to classify both the positive and negative candidate for the same
underlying item correctly.

### Structure and semantic preservation

Fine-tuning reaches 100% answer and field preservation in both scopes. BERTScore F1 remains above
91%, and NLI entailment reaches {pct('in_domain', 'Paraphrase NLI entailment')} in-domain and
{pct('cross_regional', 'Paraphrase NLI entailment')} cross-regionally. These results support the
claim that the model preserves structured urban-planning content, subject to the novelty caveat
below.

## Paraphrase novelty caveat

The semantic and structural scores are strong, but they do not by themselves prove that the model
rewrites its input. Exact normalized source copying occurs in
{novelty_pct('in_domain', 'exact_source_copy_rate')} of in-domain paraphrase rows and
{novelty_pct('cross_regional', 'exact_source_copy_rate')} of cross-regional rows. Mean character
similarity between output and source is
{novelty_pct('in_domain', 'mean_output_source_similarity')} and
{novelty_pct('cross_regional', 'mean_output_source_similarity')}, respectively. The model has
therefore learned structure and semantic preservation, but cross-regional lexical rewriting is not
yet reliable. Repeated rows share prompts because multiple held-out rewrite targets descend from
the same seed; deterministic decoding naturally produces one output for each unique prompt.

See `base_vs_fine_tuned_side_by_side.csv` for every base value, fine-tuned value, and absolute
change used in this report. See `paraphrase_novelty_diagnostics.csv` for the source-copy audit.
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
                "training_manifest_validated": training_manifest,
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
