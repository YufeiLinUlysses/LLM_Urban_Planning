"""Prepare and explicitly publish a versioned Hugging Face dataset release."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from huggingface_hub import HfApi, create_branch

PartitionValue = dict[str, Any] | Callable[[], dict[str, Any]]


def _load_hf_token() -> str:
    for candidate in (Path.cwd() / ".env", Path.cwd().parent / ".env"):
        if candidate.is_file():
            load_dotenv(candidate, override=False)
            break
    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required for Hugging Face repository operations")
    return token


def initialize_experiment_repositories(parameters: dict[str, Any]) -> dict[str, Any]:
    """Create the model-zoo and prediction repositories before GPU work begins."""

    token = _load_hf_token()
    api = HfApi(token=token)
    private = bool(parameters.get("private", True))
    repositories = [
        (parameters["model_repo_id"], "model"),
        (parameters["prediction_repo_id"], "dataset"),
    ]
    created = []
    for repo_id, repo_type in repositories:
        repo = api.create_repo(
            repo_id=repo_id,
            repo_type=repo_type,
            private=private,
            exist_ok=True,
        )
        created.append({"repo_id": repo_id, "repo_type": repo_type, "url": str(repo)})
    return {
        "created_at": datetime.now(UTC).isoformat(),
        "private": private,
        "repositories": created,
    }


def _materialize(partitions: Mapping[str, PartitionValue]) -> dict[str, dict[str, Any]]:
    return {
        partition_id: value() if callable(value) else value
        for partition_id, value in sorted(partitions.items())
    }


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _dataset_card(version: str) -> str:
    return f"""---
configs:
  - config_name: revision_v2_canonical
    data_files:
      - split: train
        path: data/revision_v2/canonical/*.json
  - config_name: revision_v2_generation
    data_files:
      - split: train
        path: data/revision_v2/generation/*.json
  - config_name: revision_v2_verification
    data_files:
      - split: train
        path: data/revision_v2/verification/*.json
  - config_name: legacy_v1
    data_files:
      - split: train
        path: "*_final.json"
---

# Urban Planning LLM Instruction Dataset

This repository contains dataset-grounded urban instruction data. Release `{version}` adds
deterministic seed lineage, leakage-safe prompt/target separation, answer-generation and
answer-verification task views, structural validation, and duplicate auditing.

## Configurations

- `revision_v2_canonical`: complete records and augmentation provenance.
- `revision_v2_generation`: flat correct-answer generation prompts and targets.
- `revision_v2_verification`: flat positive and negative candidate-verification prompts.
- `legacy_v1`: the previously published final JSON files, when present.

The `train` name is only the Hub's container for this unsplit release. Before model training,
split records by `concept_group_id`; all task formats and descendants of one source fact must
remain in the same partition.
Never perform a row-level random split over augmented variants.

Negative candidates are not used as incorrect generation targets. They are used only in the
verification view, where the target explicitly identifies them as incorrect and supplies the
correct answer and explanation.
"""


def prepare_release_bundle(
    canonical: Mapping[str, PartitionValue],
    generation: Mapping[str, PartitionValue],
    verification: Mapping[str, PartitionValue],
    statistics: dict[str, Any],
    quality_report: dict[str, Any],
    duplicate_report: dict[str, Any],
    rejected_samples: list[dict[str, Any]],
    augmentation_manifest: dict[str, Any],
    parameters: dict[str, Any],
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    str,
    dict[str, Any],
]:
    """Build the local Hub-ready bundle after enforcing the publication gate."""

    if not quality_report.get("passed", False):
        raise ValueError("Quality-control gate failed; refusing to prepare a Hub release")
    canonical_data = {
        partition_id: dataset["records"]
        for partition_id, dataset in _materialize(canonical).items()
    }
    generation_data = {
        partition_id: dataset["records"]
        for partition_id, dataset in _materialize(generation).items()
    }
    verification_data = {
        partition_id: dataset["records"]
        for partition_id, dataset in _materialize(verification).items()
    }
    quality = {
        "augmentation_statistics": statistics,
        "quality_control_report": quality_report,
        "duplicate_audit": duplicate_report,
        "rejected_samples": {"records": rejected_samples},
    }
    manifest = {
        "dataset_version": parameters["dataset_version"],
        "created_at": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(),
        "repository": parameters["repo_id"],
        "revision": parameters["revision"],
        "augmentation_configuration_hash": augmentation_manifest["augmentation_configuration_hash"],
        "augmentation_manifest": augmentation_manifest,
        "record_counts": {
            "canonical": statistics["canonical_record_count"],
            "generation": statistics["generation_record_count"],
            "verification": statistics["verification_record_count"],
        },
        "split_policy": "Group all downstream partitions by concept_group_id",
    }
    return (
        canonical_data,
        generation_data,
        verification_data,
        quality,
        _dataset_card(parameters["dataset_version"]),
        manifest,
    )


def publish_release(manifest: dict[str, Any], parameters: dict[str, Any]) -> dict[str, Any]:
    """Upload a prepared release. This is the only externally mutating node."""

    token = _load_hf_token()
    release_dir = Path(parameters["release_dir"]).resolve()
    if not release_dir.is_dir():
        raise FileNotFoundError(f"Prepared Hugging Face release is missing: {release_dir}")
    expected_manifest = release_dir / "data" / "revision_v2" / "manifest.json"
    if not expected_manifest.is_file():
        raise FileNotFoundError(f"Release manifest is missing: {expected_manifest}")

    api = HfApi(token=token)
    api.create_repo(
        repo_id=parameters["repo_id"],
        repo_type=parameters["repo_type"],
        exist_ok=True,
    )
    try:
        create_branch(
            repo_id=parameters["repo_id"],
            branch=parameters["revision"],
            repo_type=parameters["repo_type"],
            token=token,
            exist_ok=True,
        )
    except TypeError:
        # Compatibility with huggingface_hub releases that do not expose exist_ok.
        refs = api.list_repo_refs(parameters["repo_id"], repo_type=parameters["repo_type"])
        if parameters["revision"] not in {branch.name for branch in refs.branches}:
            create_branch(
                repo_id=parameters["repo_id"],
                branch=parameters["revision"],
                repo_type=parameters["repo_type"],
                token=token,
            )

    commit = api.upload_folder(
        folder_path=str(release_dir),
        repo_id=parameters["repo_id"],
        repo_type=parameters["repo_type"],
        revision=parameters["revision"],
        path_in_repo=parameters.get("path_in_repo", ""),
        commit_message=parameters["commit_message"],
    )
    return {
        "published_at": datetime.now(UTC).isoformat(),
        "repo_id": parameters["repo_id"],
        "revision": parameters["revision"],
        "dataset_version": manifest["dataset_version"],
        "commit_url": str(commit),
    }
