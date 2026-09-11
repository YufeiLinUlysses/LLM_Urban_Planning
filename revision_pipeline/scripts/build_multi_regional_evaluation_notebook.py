"""Build the Colab notebook for all multi-regional model evaluations."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

OUTPUT = Path("notebooks/urban_science_multi_regional_evaluations.ipynb")


def markdown(source: str) -> dict[str, object]:
    return {"cell_type": "markdown", "metadata": {}, "source": dedent(source).strip()}


def code(source: str) -> dict[str, object]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(source).strip() + "\n",
    }


cells = [
    markdown(
        """
        # Urban Science: multi-regional evaluation runner

        Evaluation only: ten external datasets, five model families, and both base and
        fine-tuned checkpoints. Completed runs are detected on Hugging Face and skipped.
        """
    ),
    markdown("## 1. Configuration — edit this cell only"),
    code(
        '''
        from dataclasses import dataclass, field

        @dataclass
        class RunConfig:
            git_url: str = "https://github.com/YufeiLinUlysses/LLM_Urban_Planning.git"
            git_branch: str = "revision"
            project_dir: str = "/content/LLM_Urban_Planning/revision_pipeline"
            model_repo: str = "UlyssesLynne/urban-planning-llm-model-zoo-v3"
            prediction_repo: str = "UlyssesLynne/urban-planning-llm-predictions-v3"
            dataset_version: str = "multi_regional_external_v1"
            artifact_root: str = "/content/urban_science_artifacts"
            model_runs: dict[str, str] = field(default_factory=lambda: {
                "t5_base": "revision-v5-t5-three-task-v1",
                "qwen25_7b": "revision-v5-qwen25-7b-three-task-chat-v2",
                "qwen25_14b": "revision-v5-qwen25-14b-three-task-chat-v2",
                "llama31_8b": "revision-v5-llama31-8b-three-task-chat-v2",
                "llama31_70b": "revision-v5-llama31-70b-three-task-chat-v2",
            })
            evaluation_batch_sizes: dict[str, int] = field(default_factory=lambda: {
                "t5_base": 32, "qwen25_7b": 32, "qwen25_14b": 16,
                "llama31_8b": 32, "llama31_70b": 8,
            })
            selected_models: tuple[str, ...] = (
                "t5_base", "qwen25_7b", "qwen25_14b", "llama31_8b", "llama31_70b"
            )
            use_chat_template: bool = True
            maximum_empty_response_rate: float = 0.05
            skip_completed: bool = True
            auto_shutdown_after_verification: bool = False

        CFG = RunConfig()
        CFG
        '''
    ),
    markdown("## 2. Read Hugging Face and optional Gmail credentials"),
    code(
        '''
        import os
        from google.colab import userdata

        def secret(name: str, required: bool = True) -> str | None:
            try:
                value = userdata.get(name)
            except Exception:
                value = None
            if required and not value:
                raise RuntimeError(f"Add {name} to Colab Secrets and enable notebook access")
            return value

        os.environ["HF_TOKEN"] = secret("HF_TOKEN")
        GMAIL_ADDRESS = secret("GMAIL_ADDRESS", required=False)
        GMAIL_TOKEN = secret("GMAIL_TOKEN", required=False)
        NOTIFY_EMAIL = secret("NOTIFY_EMAIL", required=False) or GMAIL_ADDRESS
        print("HF token loaded; email enabled:", bool(GMAIL_ADDRESS and GMAIL_TOKEN))
        '''
    ),
    markdown("## 3. Clone/update the project and install dependencies"),
    code(
        '''
        from pathlib import Path
        import shutil
        import subprocess

        project = Path(CFG.project_dir)
        repo = project.parent
        if repo.exists() and not (repo / ".git").is_dir():
            raise RuntimeError(f"{repo} exists but is not a Git clone; remove or rename it")
        if not repo.exists():
            subprocess.run([
                "git", "clone", "--branch", CFG.git_branch, "--single-branch",
                CFG.git_url, str(repo),
            ], check=True)
        else:
            subprocess.run(["git", "-C", str(repo), "pull", "--ff-only"], check=True)
        if not shutil.which("uv"):
            subprocess.run(["python", "-m", "pip", "install", "-q", "uv"], check=True)
        subprocess.run(["uv", "sync", "--frozen"], cwd=project, check=True)
        print("Environment ready:", project)
        '''
    ),
    markdown("## 4. Shared execution, email, and Hugging Face helpers"),
    code(
        '''
        import gc
        import smtplib
        import subprocess
        from email.message import EmailMessage
        from huggingface_hub import HfApi

        def run_project(*args: str) -> None:
            command = list(args)
            print("Running:", " ".join(command), flush=True)
            subprocess.run(command, cwd=CFG.project_dir, check=True)

        def kedro_params(values: dict) -> str:
            return ",".join(f"{key}={str(value).lower() if isinstance(value, bool) else value}"
                            for key, value in values.items())

        def send_email(subject: str, body: str) -> None:
            if not (GMAIL_ADDRESS and GMAIL_TOKEN and NOTIFY_EMAIL):
                print("Email skipped (Gmail secrets not configured):", subject)
                return
            message = EmailMessage()
            message["From"] = GMAIL_ADDRESS
            message["To"] = NOTIFY_EMAIL
            message["Subject"] = subject
            message.set_content(body)
            with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
                smtp.login(GMAIL_ADDRESS, GMAIL_TOKEN)
                smtp.send_message(message)

        api = HfApi(token=os.environ["HF_TOKEN"])
        REQUIRED_RESULTS = {
            "evaluation_manifest.json", "metrics.json", "predictions.parquet",
            "grouped_metrics.parquet", "review_queue.parquet",
        }

        def evaluation_location(model_key: str, training_run: str, stage: str):
            label = "base" if stage == "base" else "finetuned"
            run_id = f"{training_run}-{label}-multi-regional"
            prefix = (f"evaluations/{CFG.dataset_version}/{model_key}/{stage}/"
                      f"multi_regional/{run_id}")
            return run_id, prefix

        def is_complete(model_key: str, training_run: str, stage: str) -> bool:
            files = set(api.list_repo_files(CFG.prediction_repo, repo_type="dataset"))
            _, prefix = evaluation_location(model_key, training_run, stage)
            return all(f"{prefix}/{name}" in files for name in REQUIRED_RESULTS)

        def release_memory() -> None:
            gc.collect()
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
        '''
    ),
    markdown("## 5. Prepare and validate the ten-dataset benchmark"),
    code(
        '''
        run_project("uv", "run", "kedro", "run", "--env=colab",
                    "--pipelines=prepare_multi_regional_data")

        import json
        report = json.loads((Path(CFG.project_dir) /
            "data/08_reporting/multi_regional_external_v1/quality.json").read_text())
        stats = json.loads((Path(CFG.project_dir) /
            "data/08_reporting/multi_regional_external_v1/statistics.json").read_text())
        print("Quality passed:", report["passed"])
        print("Accepted/rejected:", report["accepted_record_count"],
              report["rejected_record_count"])
        print("Generation/verification/paraphrase:", stats["generation_record_count"],
              stats["verification_record_count"], stats["paraphrase_record_count"])
        assert report["passed"] and report["accepted_record_count"] == 350
        assert stats["generation_record_count"] == 350
        assert stats["verification_record_count"] == 700
        '''
    ),
    markdown("## 6. Evaluation function"),
    code(
        '''
        def evaluate(model_key: str, stage: str) -> None:
            training_run = CFG.model_runs[model_key]
            run_id, prefix = evaluation_location(model_key, training_run, stage)
            if CFG.skip_completed and is_complete(model_key, training_run, stage):
                print("SKIP (already complete):", prefix)
                return
            params = {
                "evaluation.model_key": model_key,
                "evaluation.checkpoint_stage": stage,
                "evaluation.dataset_scope": "multi_regional",
                "evaluation.run_id": run_id,
                "evaluation.artifact_root": CFG.artifact_root,
                "evaluation.prediction_repo_id": CFG.prediction_repo,
                "evaluation.publish_to_hf": True,
                "evaluation.batch_size": CFG.evaluation_batch_sizes[model_key],
                "evaluation.use_chat_template": CFG.use_chat_template,
                "evaluation.maximum_empty_response_rate": CFG.maximum_empty_response_rate,
                "evaluation.evaluate_paraphrase": False,
            }
            if stage == "fine_tuned":
                params.update({
                    "evaluation.checkpoint_uri": CFG.model_repo,
                    "evaluation.checkpoint_subfolder":
                        f"{model_key}/{training_run}/checkpoint",
                })
            try:
                run_project("uv", "run", "kedro", "run", "--env=colab",
                            "--pipelines=evaluate_multi_regional_model",
                            "--params=" + kedro_params(params))
                if not is_complete(model_key, training_run, stage):
                    raise RuntimeError(f"Published files are incomplete at {prefix}")
            except BaseException as exc:
                send_email(f"FAILED: {model_key} {stage} multi-regional",
                           f"{type(exc).__name__}: {exc}")
                raise
            else:
                send_email(f"COMPLETE: {model_key} {stage} multi-regional",
                           f"Verified on Hugging Face at {prefix}")
            finally:
                release_memory()
        '''
    ),
    markdown("## 7. Run every base and fine-tuned evaluation"),
    code(
        '''
        for model_key in CFG.selected_models:
            if model_key not in CFG.model_runs:
                raise KeyError(f"No checkpoint run configured for {model_key}")
            for stage in ("base", "fine_tuned"):
                print("=" * 80)
                print(model_key, stage, "multi_regional")
                evaluate(model_key, stage)
        '''
    ),
    markdown("## 8. Verify all ten result bundles and optionally release Colab"),
    code(
        '''
        missing = []
        for model_key in CFG.selected_models:
            training_run = CFG.model_runs[model_key]
            for stage in ("base", "fine_tuned"):
                complete = is_complete(model_key, training_run, stage)
                _, prefix = evaluation_location(model_key, training_run, stage)
                print(model_key, stage, "complete:", complete)
                if not complete:
                    missing.append(prefix)
        if missing:
            raise RuntimeError("Incomplete Hugging Face evaluations:\\n" + "\\n".join(missing))
        send_email("VERIFIED: all multi-regional evaluations complete",
                   "All selected base and fine-tuned model evaluations are on Hugging Face.")
        print("All selected evaluations are safely stored on Hugging Face.")

        if CFG.auto_shutdown_after_verification:
            from google.colab import runtime
            runtime.unassign()
        '''
    ),
]

notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"name": "Urban Science multi-regional evaluations", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
print(OUTPUT)
