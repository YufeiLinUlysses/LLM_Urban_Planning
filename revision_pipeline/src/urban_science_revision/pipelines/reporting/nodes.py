"""Plot only persisted training logs and evaluation predictions (no hard-coded scores)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def _pyplot() -> Any:
    # Colab exports its notebook-only inline backend to child processes. The
    # isolated uv environment does not include that backend, and Matplotlib
    # validates MPLBACKEND during import, before matplotlib.use() can replace it.
    os.environ["MPLBACKEND"] = "Agg"
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def render_training_history(history: list[dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    """Render Figure 3 diagnostics against the real Trainer global step."""

    import pandas as pd

    output_dir.mkdir(parents=True, exist_ok=True)
    train = [row for row in history if "loss" in row and "eval_loss" not in row]
    validation = [row for row in history if "eval_loss" in row]
    if not train:
        # Very short smoke runs can finish before logging_steps is reached. Transformers still
        # records the aggregate train_loss in its final summary, which is sufficient for a
        # one-point smoke-test diagnostic.
        train = [
            {**row, "loss": row["train_loss"]}
            for row in history
            if "train_loss" in row and "step" in row
        ]
    if not train:
        raise ValueError("Training history contains no per-step or aggregate training loss")

    plt = _pyplot()
    fig, axis = plt.subplots(figsize=(7.2, 4.6))
    axis.plot(
        [row["step"] for row in train],
        [row["loss"] for row in train],
        marker="o",
        markersize=3,
        linewidth=1.4,
        label=f"Train loss (n={len(train)})",
    )
    if validation:
        axis.plot(
            [row["step"] for row in validation],
            [row["eval_loss"] for row in validation],
            marker="s",
            markersize=5,
            linewidth=1.6,
            label=f"Validation loss (n={len(validation)})",
        )
    axis.set(xlabel="Global training step", ylabel="Loss", title="Training and validation loss")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    loss_path = output_dir / "training_vs_validation_loss.png"
    fig.savefig(loss_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    learning_rate = [row for row in history if "learning_rate" in row and "step" in row]
    figure_paths = [str(loss_path.resolve())]
    if learning_rate:
        fig, axis = plt.subplots(figsize=(7.2, 4.2))
        axis.plot(
            [row["step"] for row in learning_rate],
            [row["learning_rate"] for row in learning_rate],
            linewidth=1.5,
        )
        axis.set(
            xlabel="Global training step",
            ylabel="Learning rate",
            title="Learning-rate schedule",
        )
        axis.grid(alpha=0.25)
        fig.tight_layout()
        lr_path = output_dir / "learning_rate.png"
        fig.savefig(lr_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        figure_paths.append(str(lr_path.resolve()))

    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
    summary = {
        "training_log_points": len(train),
        "validation_log_points": len(validation),
        "training_steps": [row["step"] for row in train],
        "validation_steps": [row["step"] for row in validation],
        "figures": figure_paths,
    }
    (output_dir / "figure_manifest.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def render_training_figures_from_receipt(
    receipt: dict[str, Any], parameters: dict[str, Any]
) -> dict[str, Any]:
    history_path = parameters.get("training_history_path")
    if history_path:
        history_file = Path(history_path)
    else:
        history_file = Path(receipt["local_checkpoint"]) / "training_history.json"
    if not history_file.is_file():
        raise FileNotFoundError(f"Training history not found: {history_file.resolve()}")
    output_dir = parameters.get("output_dir")
    destination = Path(output_dir) if output_dir else history_file.parent / "figures"
    result = render_training_history(
        json.loads(history_file.read_text(encoding="utf-8")), destination
    )
    return {"training_history": str(history_file.resolve()), **result}


def _prediction_path(
    root: Path, model: str, stage: str, scope: str, run_ids: dict[str, Any]
) -> Path | None:
    run_id = run_ids.get(model, {}).get(stage, {}).get(scope)
    stage_dir = root / "evaluations" / model / stage / scope
    if run_id:
        candidate = stage_dir / str(run_id) / "predictions.parquet"
        return candidate if candidate.is_file() else None
    candidates = sorted(stage_dir.glob("*/predictions.parquet"))
    return candidates[-1] if candidates else None


def _mean(frame: Any, task: str, metric: str) -> float | None:
    selected = frame.loc[frame["evaluation_task"] == task, metric].dropna()
    return float(selected.astype(float).mean()) if len(selected) else None


def _paired_bars(
    rows: list[dict[str, Any]], metric: str, title: str, ylabel: str, path: Path
) -> None:
    import numpy as np

    plt = _pyplot()
    labels = [row["model_label"] for row in rows]
    base = [100 * row[f"base_{metric}"] for row in rows]
    tuned = [100 * row[f"fine_tuned_{metric}"] for row in rows]
    x = np.arange(len(labels))
    width = 0.36
    fig, axis = plt.subplots(figsize=(max(7.2, len(labels) * 1.45), 4.8))
    axis.bar(x - width / 2, base, width, label="Base")
    axis.bar(x + width / 2, tuned, width, label="Fine-tuned")
    axis.set(title=title, ylabel=ylabel, xticks=x, xticklabels=labels, ylim=(0, 100))
    axis.tick_params(axis="x", rotation=18)
    axis.grid(axis="y", alpha=0.2)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_evaluation_figures(parameters: dict[str, Any]) -> dict[str, Any]:
    """Build Figure 4 and diagnostics from the latest paired base/tuned runs."""

    import pandas as pd

    root = Path(parameters["artifact_root"])
    output_dir = Path(parameters["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    scope = parameters.get("dataset_scope", "in_domain")
    labels = parameters.get("model_labels", {})
    run_ids = parameters.get("run_ids", {})
    rows: list[dict[str, Any]] = []
    skipped: dict[str, str] = {}
    sources: list[str] = []
    for model in parameters["model_order"]:
        paths = {
            stage: _prediction_path(root, model, stage, scope, run_ids)
            for stage in ("base", "fine_tuned")
        }
        if not all(paths.values()):
            skipped[model] = "missing paired base/fine_tuned predictions"
            continue
        frames = {stage: pd.read_parquet(path) for stage, path in paths.items()}
        row: dict[str, Any] = {"model_key": model, "model_label": labels.get(model, model)}
        valid = True
        for stage, frame in frames.items():
            values = {
                "mc_accuracy": _mean(frame, "mc_answer", "normalized_correct"),
                "mc_strict_accuracy": _mean(frame, "mc_answer", "strict_correct"),
                "mc_format_compliance": _mean(frame, "mc_answer", "format_compliant"),
                "sa_bertscore_f1": _mean(frame, "short_answer", "bertscore_f1"),
            }
            if values["mc_accuracy"] is None or values["sa_bertscore_f1"] is None:
                skipped[model] = "required MC or short-answer BERTScore metric is absent"
                valid = False
                break
            row.update({f"{stage}_{key}": value for key, value in values.items()})
            sources.append(str(paths[stage].resolve()))
        if valid:
            rows.append(row)
    if not rows:
        raise FileNotFoundError(
            "No complete base/fine_tuned evaluation pairs were found. Run evaluate_model "
            f"for both stages under {root.resolve()} first. Details: {skipped}"
        )

    figures = {
        "mc_accuracy": output_dir / "figure_4_mc_accuracy.png",
        "sa_bertscore": output_dir / "figure_4_sa_paraphrasing_bertscore.png",
        "mc_diagnostics": output_dir / "mc_strict_normalized_format.png",
    }
    _paired_bars(
        rows,
        "mc_accuracy",
        "Multiple-choice answer accuracy",
        "Accuracy (%)",
        figures["mc_accuracy"],
    )
    _paired_bars(
        rows,
        "sa_bertscore_f1",
        "Short-answer semantic similarity",
        "BERTScore F1 (%)",
        figures["sa_bertscore"],
    )

    diagnostic_rows = []
    for row in rows:
        for metric in ("mc_strict_accuracy", "mc_accuracy", "mc_format_compliance"):
            for stage in ("base", "fine_tuned"):
                diagnostic_rows.append(
                    {
                        "model_label": row["model_label"],
                        "stage": stage,
                        "metric": metric,
                        "value": row[f"{stage}_{metric}"],
                    }
                )
    diagnostic = pd.DataFrame(diagnostic_rows)
    plt = _pyplot()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7), sharey=True)
    titles = {
        "mc_strict_accuracy": "Strict accuracy",
        "mc_accuracy": "Normalized accuracy",
        "mc_format_compliance": "Format compliance",
    }
    for axis, metric in zip(axes, titles, strict=True):
        subset = diagnostic[diagnostic["metric"] == metric]
        pivot = subset.pivot(index="model_label", columns="stage", values="value") * 100
        pivot.plot.bar(ax=axis, legend=False)
        axis.set(
            title=titles[metric],
            xlabel="",
            ylabel="Percent" if metric == "mc_strict_accuracy" else "",
            ylim=(0, 100),
        )
        axis.tick_params(axis="x", rotation=25)
        axis.grid(axis="y", alpha=0.2)
    axes[-1].legend(title="Stage")
    fig.tight_layout()
    fig.savefig(figures["mc_diagnostics"], dpi=300, bbox_inches="tight")
    plt.close(fig)

    metrics_path = output_dir / "paper_figure_metrics.csv"
    pd.DataFrame(rows).to_csv(metrics_path, index=False)
    manifest = {
        "dataset_scope": scope,
        "models_included": [row["model_key"] for row in rows],
        "models_skipped": skipped,
        "source_predictions": sorted(set(sources)),
        "metrics_table": str(metrics_path.resolve()),
        "figures": {key: str(path.resolve()) for key, path in figures.items()},
        "metric_definitions": {
            "mc_accuracy": "mean normalized_correct on mc_answer rows",
            "sa_bertscore_f1": "mean bertscore_f1 on short_answer rows",
            "mc_diagnostics": "strict accuracy, normalized accuracy, and format compliance",
        },
    }
    return manifest
