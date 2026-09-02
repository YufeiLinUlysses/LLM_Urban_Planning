import json

import pandas as pd

from urban_science_revision.pipelines.reporting.nodes import (
    build_evaluation_figures,
    render_training_figures_from_receipt,
)


def test_render_training_figures_uses_actual_steps(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    history = [
        {"loss": 2.0, "learning_rate": 0.001, "step": 5},
        {"eval_loss": 1.7, "step": 10},
        {"loss": 1.5, "learning_rate": 0.0005, "step": 10},
    ]
    (checkpoint / "training_history.json").write_text(json.dumps(history), encoding="utf-8")

    result = render_training_figures_from_receipt(
        {"local_checkpoint": str(checkpoint)},
        {"training_history_path": None, "output_dir": None},
    )

    assert result["training_steps"] == [5, 10]
    assert result["validation_steps"] == [10]
    assert (checkpoint / "figures" / "training_vs_validation_loss.png").is_file()
    assert (checkpoint / "figures" / "learning_rate.png").is_file()


def test_render_training_figures_accepts_short_smoke_summary(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    history = [
        {"eval_loss": 4.1, "step": 4},
        {"train_loss": 7.2, "train_runtime": 8.0, "step": 4},
    ]
    (checkpoint / "training_history.json").write_text(json.dumps(history), encoding="utf-8")

    result = render_training_figures_from_receipt(
        {"local_checkpoint": str(checkpoint)},
        {"training_history_path": None, "output_dir": None},
    )

    assert result["training_steps"] == [4]
    assert result["validation_steps"] == [4]
    assert (checkpoint / "figures" / "training_vs_validation_loss.png").is_file()


def test_build_evaluation_figures_from_prediction_artifacts(tmp_path):
    for stage, adjustment in (("base", 0.0), ("fine_tuned", 0.1)):
        run = tmp_path / "evaluations" / "qwen25_7b" / stage / "in_domain" / "run-1"
        run.mkdir(parents=True)
        pd.DataFrame(
            [
                {
                    "evaluation_task": "mc_answer",
                    "normalized_correct": 0.8 + adjustment,
                    "strict_correct": 0.7 + adjustment,
                    "format_compliant": 0.75 + adjustment,
                    "bertscore_f1": None,
                },
                {
                    "evaluation_task": "short_answer",
                    "normalized_correct": None,
                    "strict_correct": None,
                    "format_compliant": None,
                    "bertscore_f1": 0.82 + adjustment,
                },
            ]
        ).to_parquet(run / "predictions.parquet", index=False)

    output = tmp_path / "figures"
    result = build_evaluation_figures(
        {
            "artifact_root": str(tmp_path),
            "output_dir": str(output),
            "dataset_scope": "in_domain",
            "model_order": ["qwen25_7b", "missing_model"],
            "model_labels": {"qwen25_7b": "Qwen2.5-7B"},
            "run_ids": {},
        }
    )

    assert result["models_included"] == ["qwen25_7b"]
    assert "missing_model" in result["models_skipped"]
    assert (output / "figure_4_mc_accuracy.png").is_file()
    assert (output / "figure_4_sa_paraphrasing_bertscore.png").is_file()
    assert (output / "mc_strict_normalized_format.png").is_file()
