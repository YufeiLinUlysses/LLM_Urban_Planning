"""Nodes for preparing independent multi-regional evaluation seeds."""

from __future__ import annotations

import copy
import hashlib
from collections import defaultdict
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

PartitionValue = dict[str, Any] | Callable[[], dict[str, Any]]

PERSPECTIVES = (
    "provenance",
    "spatial_coverage",
    "temporal_coverage",
    "measurement_resolution",
    "variable_definitions",
    "file_structure_and_documentation",
    "collection_methodology",
    "data_accessibility_and_availability",
    "usage_context",
    "dataset_limitations",
)

LEVEL2_PERSPECTIVES = (
    ("spatial_coverage", "measurement_resolution"),
    ("temporal_coverage", "collection_methodology"),
    ("variable_definitions", "file_structure_and_documentation"),
    ("data_accessibility_and_availability", "usage_context"),
    ("measurement_resolution", "dataset_limitations"),
)

LEVEL3_PERSPECTIVES = (
    ("usage_context", "dataset_limitations"),
    ("spatial_coverage", "dataset_limitations"),
    ("temporal_coverage", "measurement_resolution"),
    ("variable_definitions", "collection_methodology"),
    ("provenance", "data_accessibility_and_availability"),
)


def _materialize(partitions: Mapping[str, PartitionValue]) -> dict[str, dict[str, Any]]:
    return {
        partition_id: value() if callable(value) else value
        for partition_id, value in sorted(partitions.items())
    }


def _profile_by_source(source_profiles: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        f"{profile['dataset_id']}_instructions": profile
        for profile in source_profiles["datasets"]
    }


def _perspectives(seed: dict[str, Any]) -> list[str]:
    number = int(str(seed["seed_id"]).rsplit("__", 1)[-1])
    level = int(seed["Level"])
    if level == 1:
        return [PERSPECTIVES[number - 1]]
    if level == 2:
        return list(LEVEL2_PERSPECTIVES[number - 1])
    return list(LEVEL3_PERSPECTIVES[number - 1])


def _incorrect_candidates(seeds: list[dict[str, Any]]) -> dict[str, str]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for seed in seeds:
        groups[(str(seed["task_type"]), int(seed["Level"]))].append(seed)

    result: dict[str, str] = {}
    for grouped in groups.values():
        for index, seed in enumerate(grouped):
            if seed["task_type"] == "multiple_choice":
                result[seed["seed_id"]] = next(
                    value
                    for key, value in seed["Selections"].items()
                    if key != seed["Output"]
                )
                continue
            correct = str(seed["Output"]).strip()
            alternatives = [
                str(candidate["Output"]).strip()
                for candidate in grouped[index + 1 :] + grouped[:index]
                if str(candidate["Output"]).strip() != correct
            ]
            if not alternatives:
                raise ValueError(f"No negative candidate available for {seed['seed_id']}")
            result[seed["seed_id"]] = alternatives[0]
    return result


def assemble_original_canonical_datasets(
    partitions: Mapping[str, PartitionValue], source_profiles: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Create original-only canonical records and attach external benchmark metadata."""

    profiles = _profile_by_source(source_profiles)
    output: dict[str, Any] = {}
    counts: dict[str, int] = {}
    for partition_id, normalized in _materialize(partitions).items():
        source_name = Path(partition_id).stem
        profile = profiles.get(source_name)
        if profile is None:
            raise ValueError(f"No source profile found for {source_name}")
        seeds = normalized["seeds"]
        negatives = _incorrect_candidates(seeds)
        records = []
        for seed in seeds:
            correct_text = (
                seed["Selections"][seed["Output"]]
                if seed["task_type"] == "multiple_choice"
                else seed["Output"]
            )
            record = {
                "seed_id": seed["seed_id"],
                "concept_group_id": seed["concept_group_id"],
                "variant_id": f"{seed['seed_id']}__original",
                "source_name": source_name,
                "task_type": seed["task_type"],
                "Level": seed["Level"],
                "origin": "original",
                "Fact": seed["Fact"].strip(),
                "Question": seed["Question"].strip(),
                "correct_answer": seed["Output"].strip(),
                "correct_answer_text": str(correct_text).strip(),
                "incorrect_candidate": negatives[seed["seed_id"]],
                "Explanation": seed["Explanation"].strip(),
                "lineage": {"seed_id": seed["seed_id"]},
                "benchmark": "multi_regional_external_v1",
                "evaluation_scope": "multi_regional",
                "evaluation_only": True,
                "country": profile["country"],
                "region": profile["region"],
                "perspectives": _perspectives(seed),
                "evidence_urls": profile["source_urls"],
                "source_review_status": profile["review_status"],
            }
            if seed["task_type"] == "multiple_choice":
                record["Selections"] = copy.deepcopy(seed["Selections"])
            records.append(record)
        output[partition_id] = {
            "schema_version": "2.1",
            "source_name": source_name,
            "records": records,
        }
        counts[source_name] = len(records)

    report = {
        "benchmark": "multi_regional_external_v1",
        "evaluation_only": True,
        "augmentation_applied": False,
        "dataset_count": len(output),
        "record_count": sum(counts.values()),
        "counts": counts,
        "configuration_hash": hashlib.sha256(
            b"multi_regional_external_v1:original_only"
        ).hexdigest(),
    }
    return output, report
