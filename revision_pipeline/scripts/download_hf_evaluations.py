"""Download completed T5 evaluation runs from the private Hugging Face repo."""

from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi, hf_hub_download


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
LOCAL_ROOT = PROJECT_ROOT / "data" / "07_model_output"
REPO_ID = "UlyssesLynne/urban-planning-llm-predictions-v3"
RUNS = {
    "in_domain_base": (
        "evaluations/revision_v3/t5_base/base/in_domain/"
        "t5-base-in-domain-v3"
    ),
    "in_domain_fine_tuned": (
        "evaluations/revision_v3/t5_base/fine_tuned/in_domain/"
        "t5-finetuned-in-domain-lr2e5-v3"
    ),
    "cross_regional_base": (
        "evaluations/revision_v3/t5_base/base/cross_regional/"
        "t5-base-cross-regional-v3"
    ),
    "cross_regional_fine_tuned": (
        "evaluations/revision_v3/t5_base/fine_tuned/cross_regional/"
        "t5-finetuned-cross-regional-lr2e5-v3"
    ),
}
REQUIRED_FILES = {
    "README.md",
    "confusion_matrix.json",
    "evaluation_manifest.json",
    "grouped_metrics.parquet",
    "metrics.json",
    "predictions.parquet",
    "review_queue.parquet",
}


def main() -> None:
    load_dotenv(WORKSPACE_ROOT / ".env")
    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError(f"HF_TOKEN is missing from {WORKSPACE_ROOT / '.env'}")

    api = HfApi(token=token)
    repo_files = set(api.list_repo_files(repo_id=REPO_ID, repo_type="dataset"))

    for label, run_path in RUNS.items():
        expected = {f"{run_path}/{name}" for name in REQUIRED_FILES}
        missing = sorted(expected - repo_files)
        if missing:
            raise RuntimeError(f"Incomplete {label} run; missing: {missing}")

        print(f"Downloading {label}: {run_path}")
        for filename in sorted(expected):
            hf_hub_download(
                repo_id=REPO_ID,
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
        print(
            json.dumps(
                {
                    "label": label,
                    "local_path": str(local_run),
                    "manifest": manifest,
                    "metrics": metrics,
                },
                indent=2,
            )
        )

    print(f"All configured evaluation runs downloaded under: {LOCAL_ROOT / 'evaluations'}")


if __name__ == "__main__":
    main()
