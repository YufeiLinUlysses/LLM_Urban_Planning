"""Nodes for deterministic seed-level experimental partitioning."""

from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping
from typing import Any

PartitionValue = dict[str, Any] | Callable[[], dict[str, Any]]


def _materialize(partitions: Mapping[str, PartitionValue]) -> dict[str, dict[str, Any]]:
    return {
        key.removesuffix(".json"): value() if callable(value) else value
        for key, value in sorted(partitions.items())
    }


def _records(dataset: dict[str, Any]) -> list[dict[str, Any]]:
    records = dataset.get("records")
    if not isinstance(records, list):
        raise ValueError("Every task-view dataset must contain a records list")
    return records


def _is_holdout(source: str, holdout_patterns: list[str]) -> bool:
    lowered = source.lower()
    return any(re.search(pattern.lower(), lowered) for pattern in holdout_patterns)


def _region(source: str, region_patterns: Mapping[str, str]) -> str:
    lowered = source.lower()
    for region, pattern in region_patterns.items():
        if re.search(pattern.lower(), lowered):
            return region
    return "unknown"


def _stable_order(seed_id: str, random_seed: int) -> str:
    return hashlib.sha256(f"{random_seed}:{seed_id}".encode()).hexdigest()


def _assign_seed_splits(
    datasets: Mapping[str, dict[str, Any]], parameters: dict[str, Any]
) -> dict[str, str]:
    ratios = parameters["split_ratios"]
    if abs(sum(float(value) for value in ratios.values()) - 1.0) > 1e-9:
        raise ValueError("experiment_data.split_ratios must sum to 1")

    strata: dict[tuple[str, str, int], set[str]] = defaultdict(set)
    for source, dataset in datasets.items():
        if _is_holdout(source, parameters["holdout_source_patterns"]):
            continue
        for record in _records(dataset):
            strata[(source, str(record["task_type"]), int(record["Level"]))].add(
                str(record["seed_id"])
            )

    assignment: dict[str, str] = {}
    residual: list[str] = []
    random_seed = int(parameters["random_seed"])
    split_names = ("train", "validation", "test")
    for stratum, seed_ids in sorted(strata.items()):
        ordered = sorted(seed_ids, key=lambda item: _stable_order(item, random_seed))
        count = len(ordered)
        cursor = 0
        for split in split_names:
            allocation = int(count * float(ratios[split]))
            for seed_id in ordered[cursor : cursor + allocation]:
                previous = assignment.setdefault(seed_id, split)
                if previous != split:
                    raise ValueError(f"Seed {seed_id!r} was assigned inconsistently")
            cursor += allocation
        residual.extend(
            sorted(
                ordered[cursor:],
                key=lambda item: _stable_order(f"{stratum}:{item}", random_seed),
            )
        )

    total = sum(len(seed_ids) for seed_ids in strata.values())
    targets = {
        "train": round(total * float(ratios["train"])),
        "validation": round(total * float(ratios["validation"])),
    }
    targets["test"] = total - targets["train"] - targets["validation"]
    counts = Counter(assignment.values())
    for seed_id in residual:
        deficits = {split: targets[split] - counts[split] for split in split_names}
        split = max(split_names, key=lambda name: (deficits[name], -split_names.index(name)))
        if deficits[split] <= 0:
            raise ValueError("Unable to satisfy requested global split ratios")
        assignment[seed_id] = split
        counts[split] += 1
    return assignment


def _partition_view(
    datasets: Mapping[str, dict[str, Any]],
    assignment: Mapping[str, str],
    parameters: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for source, dataset in datasets.items():
        holdout = _is_holdout(source, parameters["holdout_source_patterns"])
        region = _region(source, parameters["region_patterns"]) if holdout else None
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for original in _records(dataset):
            record = dict(original)
            split = "cross_regional" if holdout else assignment[str(record["seed_id"])]
            record["split"] = split
            record["evaluation_scope"] = "cross_regional" if holdout else "in_domain"
            if region is not None:
                record["region"] = region
            grouped[split].append(record)
        for split, records in grouped.items():
            output[f"{split}__{source}"] = {
                "schema_version": "2.1",
                "source_name": source,
                "split": split,
                "region": region,
                "records": records,
            }
    return output


def _normalize_prompt(text: str) -> str:
    return re.sub(r"\s+", " ", text.casefold()).strip()


def _leakage_audit(partitions: Mapping[str, dict[str, Any]]) -> dict[str, Any]:
    seeds_by_split: dict[str, set[str]] = defaultdict(set)
    prompts: dict[str, set[str]] = defaultdict(set)
    for dataset in partitions.values():
        split = str(dataset["split"])
        if split == "cross_regional":
            continue
        for record in _records(dataset):
            seeds_by_split[split].add(str(record["seed_id"]))
            prompts[split].add(_normalize_prompt(str(record["prompt"])))

    seed_overlaps: dict[str, list[str]] = {}
    prompt_overlaps: dict[str, int] = {}
    names = sorted(seeds_by_split)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            key = f"{left}__{right}"
            overlap = sorted(seeds_by_split[left] & seeds_by_split[right])
            seed_overlaps[key] = overlap
            prompt_overlaps[key] = len(prompts[left] & prompts[right])
    passed = not any(seed_overlaps.values()) and not any(prompt_overlaps.values())
    return {
        "passed": passed,
        "seed_overlap": seed_overlaps,
        "exact_prompt_overlap_count": prompt_overlaps,
        "note": "Near-duplicate audit is performed by the upstream augmentation pipeline.",
    }


def prepare_experiment_partitions(
    generation_partitions: Mapping[str, PartitionValue],
    verification_partitions: Mapping[str, PartitionValue],
    parameters: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Split training sources by seed and combine France/Japan as one holdout."""

    generation = _materialize(generation_partitions)
    verification = _materialize(verification_partitions)
    if set(generation) != set(verification):
        raise ValueError("Generation and verification sources do not match")

    assignment = _assign_seed_splits(generation, parameters)
    generation_output = _partition_view(generation, assignment, parameters)
    verification_output = _partition_view(verification, assignment, parameters)
    audit = _leakage_audit(generation_output)
    if parameters.get("fail_on_leakage", True) and not audit["passed"]:
        raise ValueError(f"Cross-split leakage audit failed: {audit}")

    split_counts = Counter(assignment.values())
    record_counts = Counter(
        dataset["split"] for dataset in generation_output.values() for _ in dataset["records"]
    )
    manifest = {
        "dataset_version": parameters["dataset_version"],
        "random_seed": parameters["random_seed"],
        "split_ratios": parameters["split_ratios"],
        "split_unit": "seed_id",
        "holdout_policy": "France and Japan are combined as cross_regional and never trained",
        "seed_counts": dict(split_counts),
        "generation_record_counts": dict(record_counts),
        "seed_assignments": assignment,
        "sources": sorted(generation),
    }
    return generation_output, verification_output, manifest, audit
