"""Build the reusable, sectioned Colab runner notebook."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "urban_science_colab_runner.ipynb"


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


cells = [
    markdown(
        """# Urban Science: reusable Colab training and evaluation runner

Run the sections from top to bottom on a fresh GPU runtime. For later models or runs, change only
Section 1. Checkpoint and evaluation artifacts are written to temporary Colab storage and uploaded
to Hugging Face; this notebook does not mount or use Google Drive.
"""
    ),
    markdown("## 0. Confirm the GPU"),
    code(
        """import subprocess

subprocess.run(["nvidia-smi"], check=True)
"""
    ),
    markdown("## 1. Run configuration — edit this cell only"),
    code(
        """from dataclasses import dataclass

@dataclass(frozen=True)
class RunConfig:
    git_url: str = "https://github.com/YufeiLinUlysses/LLM_Urban_Planning.git"
    git_branch: str = "revision"
    project_dir: str = "/content/LLM_Urban_Planning/revision_pipeline"

    dataset_repo: str = "UlyssesLynne/urban_planning_llm"
    dataset_revision: str = "revision-v4"
    dataset_folder: str = "data/revision_v2"  # Folder retained for repository compatibility.
    dataset_version: str = "revision_v4"

    model_key: str = "t5_base"  # t5_base, qwen25_7b, qwen25_14b, llama31_8b, llama31_70b
    run_id: str = "revision-v4-t5-verification-v2"
    model_repo: str = "UlyssesLynne/urban-planning-llm-model-zoo-v3"
    prediction_repo: str = "UlyssesLynne/urban-planning-llm-predictions-v3"
    artifact_root: str = "/content/urban_science_artifacts"

    train_batch_size: int = 4
    eval_batch_size: int = 8
    gradient_accumulation_steps: int = 2
    epochs: int = 1
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    smoke_test: bool = False

CFG = RunConfig()
CFG
"""
    ),
    markdown("## 2. Read the Hugging Face token from Colab Secrets"),
    code(
        """import os
from getpass import getpass

try:
    from google.colab import userdata
    token = userdata.get("HF_TOKEN")
except Exception:
    token = os.environ.get("HF_TOKEN")

if not token:
    token = getpass("HF token (input is hidden): ")
os.environ["HF_TOKEN"] = token
print("HF_TOKEN is available; its value was not printed.")
"""
    ),
    markdown("## 3. Clone/update the project and install the locked environment"),
    code(
        """from pathlib import Path
import re
import shutil
import subprocess

def raw_url(value: str) -> str:
    # Undo Markdown-link formatting introduced by rich-text copy/paste.
    match = re.fullmatch(r"\\[[^]]+\\]\\((https?://[^)]+)\\)", value.strip())
    return match.group(1) if match else value.strip()

repo_root = Path(CFG.project_dir).parent
if not repo_root.exists():
    clone_command = [
        "git", "clone", "--branch", CFG.git_branch, "--single-branch",
        raw_url(CFG.git_url), str(repo_root),
    ]
    subprocess.run(clone_command, check=True)
else:
    subprocess.run(["git", "-C", str(repo_root), "pull", "--ff-only"], check=True)

if not shutil.which("uv"):
    subprocess.run(
        ["python", "-m", "pip", "install", "-q", "uv", "huggingface_hub"],
        check=True,
    )
subprocess.run(["uv", "sync", "--frozen"], cwd=CFG.project_dir, check=True)
print("Environment ready:", CFG.project_dir)
"""
    ),
    markdown("## 4. Shared command helper"),
    code(
        """import os
import subprocess

def run_project(*args: str) -> None:
    env = os.environ.copy()
    env.update({
        "MPLBACKEND": "Agg",
        "KEDRO_DISABLE_TELEMETRY": "1",
        "PYTHONUNBUFFERED": "1",
        "HF_HUB_DISABLE_PROGRESS_BARS": "0",
    })
    command = ["uv", "run", *args]
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, cwd=CFG.project_dir, env=env, check=True)

def kedro_params(values: dict) -> str:
    def encode(value):
        if isinstance(value, bool):
            return str(value).lower()
        return str(value)
    return ",".join(f"{key}={encode(value)}" for key, value in values.items())
"""
    ),
    markdown(
        """## 5. Download the versioned task-view data from Hugging Face

This fetches only generation and verification JSON files. It deliberately fails if the configured
revision does not exist, preventing accidental v3 training with a v4 run ID.
"""
    ),
    code(
        """from pathlib import Path
import shutil
from huggingface_hub import snapshot_download

snapshot = Path(snapshot_download(
    repo_id=CFG.dataset_repo,
    repo_type="dataset",
    revision=CFG.dataset_revision,
    token=os.environ["HF_TOKEN"],
    allow_patterns=[
        f"{CFG.dataset_folder}/generation/*.json",
        f"{CFG.dataset_folder}/verification/*.json",
        f"{CFG.dataset_folder}/manifest.json",
    ],
))

project = Path(CFG.project_dir)
for task in ("generation", "verification"):
    source = snapshot / CFG.dataset_folder / task
    destination = project / "data/05_model_input" / task
    if not source.is_dir() or not list(source.glob("*.json")):
        raise FileNotFoundError(f"No {task} JSON files found at {source}")
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)
    print(task, "files:", len(list(destination.glob("*.json"))))
"""
    ),
    markdown("## 6. Recreate and audit the deterministic experiment split"),
    code(
        """run_project(
    "kedro", "run", "--pipelines=prepare_experiment_data",
    "--params=" + kedro_params({"experiment_data.dataset_version": CFG.dataset_version}),
)
"""
    ),
    markdown("## 7. Pre-training checks"),
    code(
        """import json
from pathlib import Path

run_project("pytest")

reporting = Path(CFG.project_dir) / "data/08_reporting/experiment"
manifest = json.loads((reporting / "split_manifest.json").read_text())
audit = json.loads((reporting / "leakage_audit.json").read_text())
print("Dataset:", manifest.get("dataset_version"))
print("Leakage audit passed:", audit.get("passed"))
if manifest.get("dataset_version") != CFG.dataset_version:
    raise RuntimeError("The prepared dataset version does not match CFG.dataset_version")
if not audit.get("passed"):
    raise RuntimeError("Leakage audit failed; do not train")
"""
    ),
    markdown("## 8. Optional smoke training (safe to skip after the first successful setup)"),
    code(
        """if CFG.smoke_test:
    params = {
        "training.model_key": CFG.model_key,
        "training.run_id": CFG.run_id + "-smoke",
        "training.artifact_root": CFG.artifact_root,
        "training.model_repo_id": CFG.model_repo,
        "training.publish_to_hf": False,
        "training.max_train_samples": 200,
        "training.max_validation_samples": 50,
        "training.per_device_train_batch_size": CFG.train_batch_size,
        "training.per_device_eval_batch_size": CFG.train_batch_size,
        "training.gradient_accumulation_steps": CFG.gradient_accumulation_steps,
        "training.eval_steps": 25,
        "training.save_steps": 25,
    }
    run_project("kedro", "run", "--env=colab", "--pipelines=train_model",
                "--params=" + kedro_params(params))
else:
    print("Smoke test skipped. Set CFG.smoke_test=True to enable it.")
"""
    ),
    markdown("## 9. Full training and publication"),
    code(
        """train_params = {
    "training.model_key": CFG.model_key,
    "training.run_id": CFG.run_id,
    "training.artifact_root": CFG.artifact_root,
    "training.model_repo_id": CFG.model_repo,
    "training.publish_to_hf": True,
    "training.per_device_train_batch_size": CFG.train_batch_size,
    "training.per_device_eval_batch_size": CFG.train_batch_size,
    "training.gradient_accumulation_steps": CFG.gradient_accumulation_steps,
    "training.num_train_epochs": CFG.epochs,
    "training.learning_rate": CFG.learning_rate,
    "training.warmup_ratio": CFG.warmup_ratio,
    "training.weight_decay": CFG.weight_decay,
    "training.logging_steps": 10,
    "training.eval_steps": 100,
    "training.save_steps": 100,
    "training.save_total_limit": 2,
}
run_project("kedro", "run", "--env=colab", "--pipelines=train_model",
            "--params=" + kedro_params(train_params))
"""
    ),
    markdown("## 10. Display the saved loss graph"),
    code(
        """from IPython.display import Image, display
from pathlib import Path

loss_graph = (Path(CFG.artifact_root) / "models" / CFG.model_key / CFG.run_id /
              "checkpoint/figures/training_vs_validation_loss.png")
print(loss_graph)
if not loss_graph.is_file():
    raise FileNotFoundError("Loss graph missing; inspect the training receipt before evaluation")
display(Image(filename=str(loss_graph)))
"""
    ),
    markdown("## 11. Evaluation helper"),
    code(
        """def evaluate(stage: str, scope: str) -> None:
    label = "base" if stage == "base" else "finetuned"
    params = {
        "evaluation.model_key": CFG.model_key,
        "evaluation.checkpoint_stage": stage,
        "evaluation.dataset_scope": scope,
        "evaluation.run_id": f"{CFG.run_id}-{label}-{scope.replace('_', '-')}",
        "evaluation.artifact_root": CFG.artifact_root,
        "evaluation.prediction_repo_id": CFG.prediction_repo,
        "evaluation.publish_to_hf": True,
        "evaluation.batch_size": CFG.eval_batch_size,
    }
    if stage == "fine_tuned":
        params.update({
            "evaluation.checkpoint_uri": CFG.model_repo,
            "evaluation.checkpoint_subfolder": f"{CFG.model_key}/{CFG.run_id}/checkpoint",
        })
    run_project("kedro", "run", "--env=colab", "--pipelines=evaluate_model",
                "--params=" + kedro_params(params))
"""
    ),
    markdown("## 12A. Base model — in-domain"),
    code('evaluate("base", "in_domain")\n'),
    markdown("## 12B. Fine-tuned model — in-domain"),
    code('evaluate("fine_tuned", "in_domain")\n'),
    markdown("## 12C. Base model — cross-regional"),
    code('evaluate("base", "cross_regional")\n'),
    markdown("## 12D. Fine-tuned model — cross-regional"),
    code('evaluate("fine_tuned", "cross_regional")\n'),
    markdown("## 13. Verify the model and all four evaluation runs on Hugging Face"),
    code(
        """from huggingface_hub import HfApi

api = HfApi(token=os.environ["HF_TOKEN"])
model_files = set(api.list_repo_files(CFG.model_repo, repo_type="model"))
model_prefix = f"{CFG.model_key}/{CFG.run_id}/checkpoint"
model_required = {"config.json", "training_manifest.json", "training_history.json"}
if CFG.model_key == "t5_base":
    model_required |= {"model.safetensors", "tokenizer.json"}

missing_model = [f"{model_prefix}/{name}" for name in model_required
                 if f"{model_prefix}/{name}" not in model_files]
print("Model complete:", not missing_model, missing_model)

prediction_files = set(api.list_repo_files(CFG.prediction_repo, repo_type="dataset"))
required = {"evaluation_manifest.json", "metrics.json", "predictions.parquet",
            "grouped_metrics.parquet", "review_queue.parquet"}
missing_evaluations = []
for stage, label in (("base", "base"), ("fine_tuned", "finetuned")):
    for scope in ("in_domain", "cross_regional"):
        run_id = f"{CFG.run_id}-{label}-{scope.replace('_', '-')}"
        prefix = f"evaluations/{CFG.dataset_version}/{CFG.model_key}/{stage}/{scope}/{run_id}"
        missing = [
            f"{prefix}/{name}"
            for name in required
            if f"{prefix}/{name}" not in prediction_files
        ]
        print(stage, scope, "complete:", not missing)
        missing_evaluations.extend(missing)

if missing_model or missing_evaluations:
    raise RuntimeError("Hugging Face verification failed; do not discard the runtime yet")
print("Everything required is safely stored on Hugging Face.")
"""
    ),
    markdown(
        """## 14. End the runtime

After Section 13 succeeds, use **Runtime → Disconnect and delete runtime**. Colab-local artifacts
can then be discarded because the selected checkpoint, loss history/figures, raw predictions, and
metrics have been verified on Hugging Face.
"""
    ),
]

notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"name": "Urban Science reusable runner", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
print(OUTPUT)
