"""Nodes for deterministic concept-level experimental partitioning."""

from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping
from itertools import combinations
from typing import Any

PartitionValue = dict[str, Any] | Callable[[], dict[str, Any]]


def _materialize(partitions: Mapping[str, PartitionValue]) -> dict[str, dict[str, Any]]:
    return {
        key.removesuffix(".json"): value() if callable(value) else value
        for key, value in sorted(partitions.items())
    }


def _records(dataset: dict[str, Any] | list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Read local wrapped datasets and flat Hugging Face release partitions."""

    records = dataset if isinstance(dataset, list) else dataset.get("records")
    if not isinstance(records, list):
        raise ValueError(
            "Every task-view dataset must be a record list or contain a records list"
        )
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


def _concept_group(record: Mapping[str, Any]) -> str:
    group_id = record.get("concept_group_id")
    if not group_id:
        raise ValueError(
            "Task-view record is missing concept_group_id. Re-run data_augmentation and "
            "publish/download the updated dataset before preparing experiment splits."
        )
    return str(group_id)


def _concept_alias_lookup(parameters: Mapping[str, Any]) -> dict[str, str]:
    """Build a validated reviewed-alias-to-canonical-concept lookup."""

    configured = parameters.get("concept_group_aliases", {})
    if not isinstance(configured, Mapping):
        raise ValueError("experiment_data.concept_group_aliases must be a mapping")

    lookup: dict[str, str] = {}
    for canonical, aliases in configured.items():
        canonical_id = str(canonical)
        if not isinstance(aliases, list):
            raise ValueError(
                f"Aliases for canonical concept {canonical_id!r} must be a list"
            )
        for member in [canonical_id, *(str(alias) for alias in aliases)]:
            previous = lookup.setdefault(member, canonical_id)
            if previous != canonical_id:
                raise ValueError(
                    f"Concept {member!r} is assigned to both {previous!r} and "
                    f"{canonical_id!r}"
                )
    return lookup


def _resolved_concept_group(
    record: Mapping[str, Any], alias_lookup: Mapping[str, str]
) -> str:
    original = _concept_group(record)
    return alias_lookup.get(original, original)


def _assign_concept_splits(
    datasets: Mapping[str, dict[str, Any]], parameters: dict[str, Any]
) -> dict[str, str]:
    ratios = parameters["split_ratios"]
    if abs(sum(float(value) for value in ratios.values()) - 1.0) > 1e-9:
        raise ValueError("experiment_data.split_ratios must sum to 1")

    # A concept may have MCQ and short-answer variants at different levels. Stratifying
    # by task type would put the same concept into competing strata, so source is the
    # only safe deterministic stratum for the indivisible concept-level split unit.
    strata: dict[str, set[str]] = defaultdict(set)
    alias_lookup = _concept_alias_lookup(parameters)
    for source, dataset in datasets.items():
        if _is_holdout(source, parameters["holdout_source_patterns"]):
            continue
        for record in _records(dataset):
            strata[source].add(_resolved_concept_group(record, alias_lookup))

    assignment: dict[str, str] = {}
    residual: list[str] = []
    random_seed = int(parameters["random_seed"])
    split_names = ("train", "validation", "test")
    for stratum, group_ids in sorted(strata.items()):
        ordered = sorted(group_ids, key=lambda item: _stable_order(item, random_seed))
        count = len(ordered)
        cursor = 0
        for split in split_names:
            allocation = int(count * float(ratios[split]))
            for group_id in ordered[cursor : cursor + allocation]:
                previous = assignment.setdefault(group_id, split)
                if previous != split:
                    raise ValueError(f"Concept group {group_id!r} was assigned inconsistently")
            cursor += allocation
        residual.extend(
            sorted(
                ordered[cursor:],
                key=lambda item: _stable_order(f"{stratum}:{item}", random_seed),
            )
        )

    total = sum(len(group_ids) for group_ids in strata.values())
    targets = {
        "train": round(total * float(ratios["train"])),
        "validation": round(total * float(ratios["validation"])),
    }
    targets["test"] = total - targets["train"] - targets["validation"]
    counts = Counter(assignment.values())
    for group_id in residual:
        deficits = {split: targets[split] - counts[split] for split in split_names}
        split = max(split_names, key=lambda name: (deficits[name], -split_names.index(name)))
        if deficits[split] <= 0:
            raise ValueError("Unable to satisfy requested global split ratios")
        assignment[group_id] = split
        counts[split] += 1
    return assignment


def _partition_view(
    datasets: Mapping[str, dict[str, Any]],
    assignment: Mapping[str, str],
    parameters: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    alias_lookup = _concept_alias_lookup(parameters)
    for source, dataset in datasets.items():
        holdout = _is_holdout(source, parameters["holdout_source_patterns"])
        region = _region(source, parameters["region_patterns"]) if holdout else None
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for original in _records(dataset):
            record = dict(original)
            source_concept = _concept_group(record)
            resolved_concept = _resolved_concept_group(record, alias_lookup)
            split = "cross_regional" if holdout else assignment[resolved_concept]
            if resolved_concept != source_concept:
                record["source_concept_group_id"] = source_concept
            record["concept_group_id"] = resolved_concept
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


def _semantic_cross_split_audit(
    partitions: Mapping[str, dict[str, Any]],
    parameters: dict[str, Any],
    encoder: Any | None = None,
) -> dict[str, Any]:
    """Rank semantically similar prompts that occur in different data splits."""

    config = parameters.get("semantic_audit", {})
    if not config.get("enabled", False):
        return {"enabled": False, "review_required": False, "split_pairs": {}}

    from sklearn.neighbors import NearestNeighbors

    if encoder is None:
        from sentence_transformers import SentenceTransformer

        encoder = SentenceTransformer(config["model_id"])

    rows_by_split: dict[str, list[dict[str, str]]] = defaultdict(list)
    for dataset in partitions.values():
        split = str(dataset["split"])
        if split == "cross_regional":
            continue
        for record in _records(dataset):
            rows_by_split[split].append(
                {
                    "example_id": str(record["example_id"]),
                    "seed_id": str(record["seed_id"]),
                    "concept_group_id": _concept_group(record),
                    "source_name": str(record["source_name"]),
                    "prompt": str(record["prompt"]),
                }
            )

    embeddings = {
        split: encoder.encode(
            [row["prompt"] for row in rows],
            batch_size=int(config.get("batch_size", 64)),
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        for split, rows in rows_by_split.items()
    }
    threshold = float(config.get("similarity_threshold", 0.90))
    neighbor_limit = int(config.get("nearest_neighbors_per_record", 10))
    candidate_limit = int(config.get("max_candidates_per_split_pair", 1000))
    top_n = int(config.get("human_review_top_n", 50))
    boundary_n = int(config.get("human_review_boundary_n", 50))
    split_results: dict[str, Any] = {}
    review_queue: list[dict[str, Any]] = []

    for left, right in combinations(sorted(rows_by_split), 2):
        right_rows = rows_by_split[right]
        neighbors = min(neighbor_limit, len(right_rows))
        if not rows_by_split[left] or not right_rows or neighbors == 0:
            continue
        model = NearestNeighbors(metric="cosine", n_neighbors=neighbors, n_jobs=-1)
        model.fit(embeddings[right])
        distances, indices = model.kneighbors(embeddings[left])
        candidates_by_concept_pair: dict[tuple[str, str], dict[str, Any]] = {}
        raw_candidate_count = 0
        for left_index, (row_distances, row_indices) in enumerate(
            zip(distances, indices, strict=True)
        ):
            for distance, right_index in zip(row_distances, row_indices, strict=True):
                similarity = 1.0 - float(distance)
                if similarity < threshold:
                    continue
                raw_candidate_count += 1
                left_row = rows_by_split[left][left_index]
                right_row = right_rows[int(right_index)]
                concept_key = (
                    left_row["concept_group_id"],
                    right_row["concept_group_id"],
                )
                candidate = {
                        "split_pair": f"{left}__{right}",
                        "left_split": left,
                        "right_split": right,
                        "left_example_id": left_row["example_id"],
                        "right_example_id": right_row["example_id"],
                        "left_seed_id": left_row["seed_id"],
                        "right_seed_id": right_row["seed_id"],
                        "left_concept_group_id": left_row["concept_group_id"],
                        "right_concept_group_id": right_row["concept_group_id"],
                        "left_source_name": left_row["source_name"],
                        "right_source_name": right_row["source_name"],
                        "left_prompt": left_row["prompt"],
                        "right_prompt": right_row["prompt"],
                        "similarity": round(similarity, 6),
                        "supporting_record_pair_count": 1,
                        "human_decision": "pending",
                    }
                existing = candidates_by_concept_pair.get(concept_key)
                if existing is None:
                    candidates_by_concept_pair[concept_key] = candidate
                else:
                    existing["supporting_record_pair_count"] += 1
                    if candidate["similarity"] > existing["similarity"]:
                        support = existing["supporting_record_pair_count"]
                        candidates_by_concept_pair[concept_key] = candidate
                        candidates_by_concept_pair[concept_key][
                            "supporting_record_pair_count"
                        ] = support
        candidates = list(candidates_by_concept_pair.values())
        candidates.sort(key=lambda row: row["similarity"], reverse=True)
        total = len(candidates)
        retained = candidates[:candidate_limit]
        high_similarity = retained[:top_n]
        # The lowest-scoring flagged items test whether the chosen threshold is sensible.
        boundary = list(reversed(retained[-boundary_n:])) if boundary_n else []
        selected_keys: set[tuple[str, str]] = set()
        selected = []
        for candidate in high_similarity + boundary:
            key = (candidate["left_example_id"], candidate["right_example_id"])
            if key not in selected_keys:
                selected_keys.add(key)
                selected.append(candidate)
        review_queue.extend(selected)
        split_results[f"{left}__{right}"] = {
            "flagged_record_pair_count": raw_candidate_count,
            "flagged_candidate_count": total,
            "retained_candidate_count": len(retained),
            "human_review_count": len(selected),
            "maximum_similarity": retained[0]["similarity"] if retained else None,
            "candidates": retained,
        }

    return {
        "enabled": True,
        "model_id": config["model_id"],
        "similarity_threshold": threshold,
        "comparison_scope": "cross-split prompts only; cross-regional holdout excluded",
        "review_required": bool(review_queue),
        "human_review_policy": (
            "Candidates are deduplicated by concept pair. Review the highest-similarity "
            "concept pairs and deterministic boundary cases closest to the threshold; "
            "review all when the flagged set is small."
        ),
        "human_review_queue": review_queue,
        "split_pairs": split_results,
    }


def _leakage_audit(
    partitions: Mapping[str, dict[str, Any]], parameters: dict[str, Any]
) -> dict[str, Any]:
    seeds_by_split: dict[str, set[str]] = defaultdict(set)
    concepts_by_split: dict[str, set[str]] = defaultdict(set)
    prompts: dict[str, set[str]] = defaultdict(set)
    for dataset in partitions.values():
        split = str(dataset["split"])
        if split == "cross_regional":
            continue
        for record in _records(dataset):
            seeds_by_split[split].add(str(record["seed_id"]))
            concepts_by_split[split].add(_concept_group(record))
            prompts[split].add(_normalize_prompt(str(record["prompt"])))

    seed_overlaps: dict[str, list[str]] = {}
    concept_overlaps: dict[str, list[str]] = {}
    prompt_overlaps: dict[str, int] = {}
    names = sorted(seeds_by_split)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            key = f"{left}__{right}"
            overlap = sorted(seeds_by_split[left] & seeds_by_split[right])
            seed_overlaps[key] = overlap
            concept_overlaps[key] = sorted(concepts_by_split[left] & concepts_by_split[right])
            prompt_overlaps[key] = len(prompts[left] & prompts[right])
    passed = (
        not any(concept_overlaps.values())
        and not any(seed_overlaps.values())
        and not any(prompt_overlaps.values())
    )
    return {
        "passed": passed,
        "seed_overlap": seed_overlaps,
        "concept_group_overlap": concept_overlaps,
        "exact_prompt_overlap_count": prompt_overlaps,
        "semantic_cross_split_audit": _semantic_cross_split_audit(partitions, parameters),
        "note": (
            "passed covers deterministic concept-group, seed, and exact-prompt leakage. "
            "Semantic candidates require human adjudication and do not automatically fail "
            "the pipeline."
        ),
    }


def prepare_experiment_partitions(
    generation_partitions: Mapping[str, PartitionValue],
    verification_partitions: Mapping[str, PartitionValue],
    paraphrase_partitions: Mapping[str, PartitionValue],
    parameters: dict[str, Any],
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    """Split training sources by seed and combine France/Japan as one holdout."""

    generation = _materialize(generation_partitions)
    verification = _materialize(verification_partitions)
    paraphrase = _materialize(paraphrase_partitions)
    if set(generation) != set(verification) or set(generation) != set(paraphrase):
        raise ValueError("Generation, verification, and paraphrase sources do not match")

    assignment = _assign_concept_splits(generation, parameters)
    generation_output = _partition_view(generation, assignment, parameters)
    verification_output = _partition_view(verification, assignment, parameters)
    paraphrase_output = _partition_view(paraphrase, assignment, parameters)
    audit = _leakage_audit(generation_output, parameters)
    if parameters.get("fail_on_leakage", True) and not audit["passed"]:
        raise ValueError(f"Cross-split leakage audit failed: {audit}")

    split_counts = Counter(assignment.values())
    record_counts = Counter(
        dataset["split"] for dataset in generation_output.values() for _ in dataset["records"]
    )
    paraphrase_record_counts = Counter(
        dataset["split"] for dataset in paraphrase_output.values() for _ in dataset["records"]
    )
    manifest = {
        "dataset_version": parameters["dataset_version"],
        "random_seed": parameters["random_seed"],
        "split_ratios": parameters["split_ratios"],
        "split_unit": "concept_group_id",
        "holdout_policy": "France and Japan are combined as cross_regional and never trained",
        "concept_group_counts": dict(split_counts),
        "generation_record_counts": dict(record_counts),
        "paraphrase_record_counts": dict(paraphrase_record_counts),
        "concept_group_assignments": assignment,
        "concept_group_aliases": parameters.get("concept_group_aliases", {}),
        "sources": sorted(generation),
    }
    return generation_output, verification_output, paraphrase_output, manifest, audit
