"""Build cluster-aware paraphrase metrics from published prediction artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
DATASET_VERSION = "revision_v5"
PREDICTION_REPO = "UlyssesLynne/urban-planning-llm-predictions-v3"
MODEL_RUNS = {
    "t5_base": "revision-v5-t5-three-task-v1",
    "qwen25_7b": "revision-v5-qwen25-7b-three-task-chat-v2",
    "qwen25_14b": "revision-v5-qwen25-14b-three-task-chat-v2",
    "llama31_8b": "revision-v5-llama31-8b-three-task-chat-v2",
    "llama31_70b": "revision-v5-llama31-70b-three-task-chat-v2",
}
METRICS = (
    "answer_preserved",
    "field_preservation",
    "bertscore_f1",
    "nli_entailment_correct",
    "rouge1",
    "rouge2",
    "rougeL",
)


def _evaluation_path(model_key: str, run_id: str, stage: str, scope: str) -> str:
    label = "base" if stage == "base" else "finetuned"
    evaluation_id = f"{run_id}-{label}-{scope.replace('_', '-')}"
    return (
        f"evaluations/{DATASET_VERSION}/{model_key}/{stage}/{scope}/"
        f"{evaluation_id}/predictions.parquet"
    )


def _prompt_id(prompt: str) -> str:
    normalized = " ".join(str(prompt).split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def _cluster_summary(frame: pd.DataFrame, metric: str, cluster: str) -> dict[str, Any]:
    usable = frame[[cluster, metric]].dropna()
    cluster_means = usable.groupby(cluster, sort=True)[metric].mean()
    return {
        "comparison_count": int(len(usable)),
        "cluster_count": int(len(cluster_means)),
        "mean": float(cluster_means.mean()),
        "cluster_values": cluster_means.to_numpy(dtype=float),
    }


def _bootstrap_interval(
    values: np.ndarray, replicates: int, rng: np.random.Generator
) -> tuple[float, float]:
    if not len(values):
        return float("nan"), float("nan")
    draws = rng.choice(values, size=(replicates, len(values)), replace=True).mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(low), float(high)


def analyze_frame(
    frame: pd.DataFrame, replicates: int, random_seed: int
) -> list[dict[str, Any]]:
    """Return row-, seed-, and prompt-level summaries for paraphrase rows."""

    paraphrase = frame.loc[frame["evaluation_task"] == "paraphrase"].copy()
    if paraphrase.empty:
        raise ValueError("Prediction artifact contains no paraphrase rows")
    paraphrase["prompt_id"] = paraphrase["prompt"].map(_prompt_id)
    rng = np.random.default_rng(random_seed)
    results: list[dict[str, Any]] = []
    for metric in METRICS:
        if metric not in paraphrase:
            continue
        values = pd.to_numeric(paraphrase[metric], errors="coerce").dropna()
        seed = _cluster_summary(paraphrase, metric, "seed_id")
        prompt = _cluster_summary(paraphrase, metric, "prompt_id")
        seed_low, seed_high = _bootstrap_interval(
            seed.pop("cluster_values"), replicates, rng
        )
        prompt_low, prompt_high = _bootstrap_interval(
            prompt.pop("cluster_values"), replicates, rng
        )
        results.append(
            {
                "metric": metric,
                "comparison_count": int(len(values)),
                "seed_count": seed["cluster_count"],
                "prompt_count": prompt["cluster_count"],
                "row_micro": float(values.mean()),
                "seed_macro": seed["mean"],
                "seed_macro_ci95_low": seed_low,
                "seed_macro_ci95_high": seed_high,
                "prompt_macro": prompt["mean"],
                "prompt_macro_ci95_low": prompt_low,
                "prompt_macro_ci95_high": prompt_high,
            }
        )
    return results


def _markdown_table(results: list[dict[str, Any]]) -> str:
    lines = [
        "| Model | Stage | Scope | Metric | Comparisons | Seeds | Prompts | "
        "Row micro | Seed macro (95% CI) | Prompt macro (95% CI) |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in results:
        lines.append(
            "| {model_key} | {stage} | {scope} | {metric} | {comparison_count:,} | "
            "{seed_count:,} | {prompt_count:,} | {row_micro:.3f} | "
            "{seed_macro:.3f} ({seed_macro_ci95_low:.3f}–{seed_macro_ci95_high:.3f}) | "
            "{prompt_macro:.3f} "
            "({prompt_macro_ci95_low:.3f}–{prompt_macro_ci95_high:.3f}) |".format(**row)
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--random-seed", type=int, default=20260909)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT
        / "data/07_model_output/evaluations/revision_v5/paraphrase_cluster_analysis",
    )
    args = parser.parse_args()
    load_dotenv(WORKSPACE_ROOT / ".env")
    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError(f"HF_TOKEN is missing from {WORKSPACE_ROOT / '.env'}")

    cache = PROJECT_ROOT / "data/07_model_output"
    results: list[dict[str, Any]] = []
    for model_key, run_id in MODEL_RUNS.items():
        for stage in ("base", "fine_tuned"):
            for scope in ("in_domain", "cross_regional"):
                remote_path = _evaluation_path(model_key, run_id, stage, scope)
                local_path = hf_hub_download(
                    repo_id=PREDICTION_REPO,
                    repo_type="dataset",
                    filename=remote_path,
                    token=token,
                    local_dir=cache,
                )
                rows = analyze_frame(
                    pd.read_parquet(local_path),
                    replicates=args.bootstrap_replicates,
                    random_seed=args.random_seed,
                )
                for row in rows:
                    row.update(
                        {
                            "model_key": model_key,
                            "run_id": run_id,
                            "stage": stage,
                            "scope": scope,
                        }
                    )
                results.extend(rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "paraphrase_cluster_metrics.json"
    report_path = args.output_dir / "README.md"
    json_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    report = f"""# Cluster-aware paraphrase evaluation

This analysis distinguishes generated-reference comparisons from independent evaluation units.
`row_micro` reproduces the existing row-weighted result. `seed_macro` first averages within each
seed family, while `prompt_macro` first averages across all references attached to an identical
normalized prompt. Confidence intervals are percentile bootstrap intervals over the corresponding
clusters ({args.bootstrap_replicates:,} replicates; random seed {args.random_seed}).

{_markdown_table(results)}

## Reporting recommendation

Use prompt-macro as the primary paraphrase estimate, report the number of unique prompts, and retain
row-micro as a reproducibility check. Use seed-macro as a sensitivity analysis. Do not describe the
reference-comparison count as the number of independent prompts.
"""
    report_path.write_text(report, encoding="utf-8")
    print(report_path)
    print(json_path)


if __name__ == "__main__":
    main()
