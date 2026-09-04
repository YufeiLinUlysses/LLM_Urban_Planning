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
    dataset_revision: str = "revision-v5"
    dataset_folder: str = "data/revision_v2"  # Folder retained for repository compatibility.
    dataset_version: str = "revision_v5"

    model_key: str = "llama31_8b"  # t5_base, qwen25_7b, qwen25_14b, llama31_8b, llama31_70b
    run_id: str = "revision-v5-llama31-8b-three-task-v1"
    model_repo: str = "UlyssesLynne/urban-planning-llm-model-zoo-v3"
    prediction_repo: str = "UlyssesLynne/urban-planning-llm-predictions-v3"
    artifact_root: str = "/content/urban_science_artifacts"

    train_batch_size: int = 8
    eval_batch_size: int = 32
    gradient_accumulation_steps: int = 2
    epochs: int = 1
    learning_rate: float = 1e-4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    logging_steps: int = 10
    eval_steps: int = 50
    save_steps: int = 50
    save_total_limit: int = 2
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.001
    smoke_test: bool = False
    run_semantic_audit: bool = False  # Slow on CPU; deterministic leakage checks always run.
    email_notifications: bool = True
    auto_shutdown_after_verification: bool = True

CFG = RunConfig()
CFG
"""
    ),
    markdown(
        "## 2. Read credentials from Colab Secrets\n\n"
        "Add `HF_TOKEN`, `GMAIL_ADDRESS`, and `GMAIL_TOKEN` in the Colab Secrets panel. "
        "`GMAIL_TOKEN` must be a Gmail app password. `NOTIFY_EMAIL` is optional and defaults "
        "to `GMAIL_ADDRESS`."
    ),
    code(
        """import os
from getpass import getpass


def read_secret(name: str) -> str | None:
    try:
        from google.colab import userdata

        return userdata.get(name)
    except Exception:
        return os.environ.get(name)


token = read_secret("HF_TOKEN")
if not token:
    token = getpass("HF token (input is hidden): ")
os.environ["HF_TOKEN"] = token

if CFG.email_notifications:
    gmail_address = read_secret("GMAIL_ADDRESS")
    gmail_token = read_secret("GMAIL_TOKEN")
    notify_email = read_secret("NOTIFY_EMAIL") or gmail_address
    if not gmail_address or not gmail_token or not notify_email:
        raise RuntimeError(
            "Email notifications require GMAIL_ADDRESS and GMAIL_TOKEN in Colab Secrets"
        )
    os.environ["GMAIL_ADDRESS"] = gmail_address
    os.environ["GMAIL_TOKEN"] = gmail_token.replace(" ", "")
    os.environ["NOTIFY_EMAIL"] = notify_email

print("Required credentials are available; their values were not printed.")
"""
    ),
    markdown("## 3. Clone/update the project and install the locked environment"),
    code(
        """import re
import shutil
import subprocess
from pathlib import Path


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
import shutil
import smtplib
import subprocess
from datetime import UTC, datetime
from email.message import EmailMessage
from pathlib import Path


def show_resources() -> None:
    memory = Path("/proc/meminfo").read_text(encoding="utf-8").splitlines()[:2]
    disk = shutil.disk_usage("/content")
    print("System memory:", " | ".join(memory), flush=True)
    print(
        f"/content disk: {disk.used / 2**30:.1f} GiB used, "
        f"{disk.free / 2**30:.1f} GiB free",
        flush=True,
    )
    subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.used,memory.free,utilization.gpu", "--format=csv"],
        check=False,
    )


def send_email(subject: str, body: str, required: bool = False) -> bool:
    if not CFG.email_notifications:
        print("Email notifications are disabled.", flush=True)
        if required:
            raise RuntimeError("Required email notification is disabled; runtime retained")
        return False
    message = EmailMessage()
    message["From"] = os.environ["GMAIL_ADDRESS"]
    message["To"] = os.environ["NOTIFY_EMAIL"]
    message["Subject"] = subject
    message.set_content(
        f"{body}\\n\\nUTC time: {datetime.now(UTC).isoformat(timespec='seconds')}"
    )
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=30) as smtp:
            smtp.login(os.environ["GMAIL_ADDRESS"], os.environ["GMAIL_TOKEN"])
            smtp.send_message(message)
    except Exception as exc:
        print(f"Email notification failed: {type(exc).__name__}: {exc}", flush=True)
        if required:
            raise RuntimeError("Required email notification failed; runtime retained") from exc
        return False
    print("Email notification sent.", flush=True)
    return True

def run_project(*args: str) -> None:
    env = os.environ.copy()
    env.update({
        "MPLBACKEND": "Agg",
        "KEDRO_DISABLE_TELEMETRY": "1",
        "PYTHONUNBUFFERED": "1",
        "HF_HUB_DISABLE_PROGRESS_BARS": "0",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })
    command = ["uv", "run", *args]
    show_resources()
    print("Running:", " ".join(command), flush=True)
    process = subprocess.Popen(
        command,
        cwd=CFG.project_dir,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
    return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)

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

This fetches generation, verification, and structure-preserving paraphrase JSON files. It
deliberately fails if the configured revision does not exist, preventing mixed-version training.
"""
    ),
    code(
        """import shutil
from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.errors import RevisionNotFoundError

try:
    snapshot = Path(snapshot_download(
        repo_id=CFG.dataset_repo,
        repo_type="dataset",
        revision=CFG.dataset_revision,
        token=os.environ["HF_TOKEN"],
        allow_patterns=[
            f"{CFG.dataset_folder}/generation/*.json",
            f"{CFG.dataset_folder}/verification/*.json",
            f"{CFG.dataset_folder}/paraphrase/*.json",
            f"{CFG.dataset_folder}/manifest.json",
        ],
    ))
except RevisionNotFoundError as error:
    raise RuntimeError(
        f"Dataset branch {CFG.dataset_revision!r} does not exist in {CFG.dataset_repo}. "
        "Publish the prepared v5 dataset once from the local project before running Colab: "
        "uv run kedro run --pipelines=prepare_huggingface_release, followed by "
        "uv run kedro run --pipelines=publish_huggingface. Do not use an older revision "
        "for a v5 training run."
    ) from error

project = Path(CFG.project_dir)
for task in ("generation", "verification", "paraphrase"):
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
    markdown(
        "## 6. Recreate and audit the deterministic experiment split\n\n"
        "The seed, concept-group, and exact-prompt leakage checks always run. Set "
        "`CFG.run_semantic_audit=True` only when you intentionally want the slower "
        "sentence-embedding review. Paraphrase descendants inherit the already-audited "
        "concept split."
    ),
    code(
        """run_project(
    "kedro", "run", "--pipelines=prepare_experiment_data",
    "--params=" + kedro_params({
        "experiment_data.dataset_version": CFG.dataset_version,
        "experiment_data.semantic_audit.enabled": CFG.run_semantic_audit,
    }),
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
        "training.per_device_eval_batch_size": CFG.eval_batch_size,
        "training.gradient_accumulation_steps": CFG.gradient_accumulation_steps,
        "training.logging_steps": 1,
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
    "training.per_device_eval_batch_size": CFG.eval_batch_size,
    "training.gradient_accumulation_steps": CFG.gradient_accumulation_steps,
    "training.num_train_epochs": CFG.epochs,
    "training.learning_rate": CFG.learning_rate,
    "training.warmup_ratio": CFG.warmup_ratio,
    "training.weight_decay": CFG.weight_decay,
    "training.logging_steps": CFG.logging_steps,
    "training.eval_steps": CFG.eval_steps,
    "training.save_steps": CFG.save_steps,
    "training.save_total_limit": CFG.save_total_limit,
    "training.early_stopping_patience": CFG.early_stopping_patience,
    "training.early_stopping_threshold": CFG.early_stopping_threshold,
}
try:
    run_project("kedro", "run", "--env=colab", "--pipelines=train_model",
                "--params=" + kedro_params(train_params))
except BaseException as exc:
    send_email(
        f"FAILED: training {CFG.model_key} / {CFG.run_id}",
        f"Training did not complete. Error: {type(exc).__name__}: {exc}",
    )
    raise
else:
    send_email(
        f"COMPLETE: training {CFG.model_key} / {CFG.run_id}",
        "Training completed and the selected checkpoint was published to Hugging Face.\\n"
        f"Model repository: {CFG.model_repo}\\n"
        f"Checkpoint: {CFG.model_key}/{CFG.run_id}/checkpoint",
    )
"""
    ),
    markdown("## 10. Verify and display the Hugging Face loss graph"),
    code(
        """import json
import os
from pathlib import Path

from huggingface_hub import hf_hub_download
from IPython.display import Image, display

checkpoint_prefix = f"{CFG.model_key}/{CFG.run_id}/checkpoint"
preview_root = Path("/content/hf_artifact_preview")

manifest_path = Path(hf_hub_download(
    repo_id=CFG.model_repo,
    repo_type="model",
    filename=f"{checkpoint_prefix}/training_manifest.json",
    token=os.environ["HF_TOKEN"],
    local_dir=preview_root,
))
loss_graph = Path(hf_hub_download(
    repo_id=CFG.model_repo,
    repo_type="model",
    filename=f"{checkpoint_prefix}/figures/training_vs_validation_loss.png",
    token=os.environ["HF_TOKEN"],
    local_dir=preview_root,
))

manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
expected = {
    "training_run_id": CFG.run_id,
    "model_key": CFG.model_key,
    "dataset_version": CFG.dataset_version,
}
actual = {key: manifest.get(key) for key in expected}
if actual != expected or not manifest.get("published"):
    raise RuntimeError(
        f"Hugging Face training manifest mismatch: expected={expected}, actual={actual}"
    )

print("Verified Hugging Face checkpoint:", f"{CFG.model_repo}/{checkpoint_prefix}")
print("Training records:", manifest.get("train_record_count"))
print("Validation records:", manifest.get("validation_record_count"))
print("Validation metrics:", manifest.get("validation_metrics"))
print("Hub graph downloaded to:", loss_graph)
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
    evaluation_id = params["evaluation.run_id"]
    try:
        run_project("kedro", "run", "--env=colab", "--pipelines=evaluate_model",
                    "--params=" + kedro_params(params))
    except BaseException as exc:
        send_email(
            f"FAILED: evaluation {CFG.model_key} / {stage} / {scope}",
            f"Evaluation {evaluation_id} did not complete. "
            f"Error: {type(exc).__name__}: {exc}",
        )
        raise
    else:
        send_email(
            f"COMPLETE: evaluation {CFG.model_key} / {stage} / {scope}",
            f"Evaluation {evaluation_id} completed and was published to Hugging Face.\\n"
            f"Prediction repository: {CFG.prediction_repo}",
        )
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
model_required = {"training_manifest.json", "training_history.json"}
if CFG.model_key == "t5_base":
    model_required |= {"config.json", "model.safetensors", "tokenizer.json"}
else:
    model_required |= {"adapter_config.json", "adapter_model.safetensors"}

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

completion_lines = [
    f"Model: {CFG.model_key}",
    f"Run: {CFG.run_id}",
    f"Dataset: {CFG.dataset_version}",
    f"Model repository: {CFG.model_repo}",
    f"Prediction repository: {CFG.prediction_repo}",
    "Verified evaluations: base/fine-tuned × in-domain/cross-regional",
]
send_email(
    f"VERIFIED: all experiments complete for {CFG.model_key} / {CFG.run_id}",
    "\\n".join(completion_lines),
    required=CFG.auto_shutdown_after_verification,
)

if CFG.auto_shutdown_after_verification:
    import time

    from google.colab import runtime

    print("Email sent and Hugging Face artifacts verified. Releasing runtime.", flush=True)
    time.sleep(2)
    runtime.unassign()
"""
    ),
    markdown(
        """## 14. Manual shutdown fallback

When `CFG.auto_shutdown_after_verification=True`, Section 13 sends the final required email and
releases the runtime automatically. If automatic shutdown is disabled, use **Runtime → Disconnect
and delete runtime** after Section 13 succeeds.
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
