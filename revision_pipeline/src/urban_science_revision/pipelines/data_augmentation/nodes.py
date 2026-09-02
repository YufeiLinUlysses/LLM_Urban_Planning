"""Kedro nodes for normalization, augmentation, assembly, validation, and task views."""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import random
import re
from collections import Counter
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from .llm import OpenAIAugmenter
from .prompts import (
    completed_instruction,
    fact_explanation_augmentation,
    generation_prompt,
    generation_target,
    mcq_answer_augmentation,
    mcq_explanation_augmentation,
    mcq_question_augmentation,
    negative_answer_augmentation,
    negative_verification_target,
    positive_verification_target,
    short_question_augmentation,
    structure_preserving_paraphrase_prompt,
    verification_prompt,
)

PartitionValue = dict[str, Any] | Callable[[], dict[str, Any]]


def _materialize(partitions: Mapping[str, PartitionValue]) -> dict[str, dict[str, Any]]:
    return {
        partition_id: value() if callable(value) else value
        for partition_id, value in sorted(partitions.items())
    }


def _slug(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()
    return value or "dataset"


def _normalise_text(value: Any) -> str:
    text = str(value or "").casefold()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def _stable_seed(random_seed: int, value: str) -> int:
    digest = hashlib.sha256(f"{random_seed}:{value}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _concept_group_id(source_name: str, fact: Any) -> str:
    """Identify all task-format variants derived from the same source fact."""

    normalized_fact = _normalise_text(fact)
    digest = hashlib.sha256(normalized_fact.encode("utf-8")).hexdigest()[:16]
    return f"{_slug(source_name)}__concept__{digest}"


def _level_config(parameters: dict[str, Any], level: int, task_type: str) -> dict[str, int]:
    levels = parameters["levels"]
    level_entry = levels.get(level, levels.get(str(level)))
    if level_entry is None or task_type not in level_entry:
        raise ValueError(f"No augmentation configuration for Level {level} {task_type}")
    return level_entry[task_type]


def normalize_seed_datasets(
    partitions: Mapping[str, PartitionValue], parameters: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate source structure and assign deterministic seed lineage identifiers."""

    del parameters  # Reserved for schema-policy evolution without adding node inputs.
    normalized: dict[str, Any] = {}
    errors: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()

    for partition_id, dataset in _materialize(partitions).items():
        source_name = Path(partition_id).stem
        instructions = dataset.get("instructions")
        if not isinstance(instructions, list):
            errors.append({"source": source_name, "error": "Missing instructions list"})
            continue

        seeds: list[dict[str, Any]] = []
        counters: Counter[tuple[str, int]] = Counter()
        for instruction_index, instruction_group in enumerate(instructions):
            category = instruction_group.get("Instruction information", {}).get("Categories", "")
            task_definition = instruction_group.get("Task_Definition", "")
            examples = instruction_group.get("Positive Example", [])
            if not isinstance(examples, list):
                errors.append(
                    {
                        "source": source_name,
                        "instruction_index": instruction_index,
                        "error": "Positive Example must be a list",
                    }
                )
                continue

            for example_index, example in enumerate(examples):
                task_type = "multiple_choice" if "Selections" in example else "short_answer"
                required = {"Fact", "Question", "Output", "Explanation", "Level"}
                missing = sorted(required - set(example))
                location = {
                    "source": source_name,
                    "instruction_index": instruction_index,
                    "example_index": example_index,
                }
                if missing:
                    errors.append({**location, "error": f"Missing fields: {missing}"})
                    continue
                try:
                    level = int(example["Level"])
                except (TypeError, ValueError):
                    errors.append({**location, "error": "Level must be an integer"})
                    continue
                if level not in {1, 2, 3}:
                    errors.append({**location, "error": f"Unsupported Level {level}"})
                    continue
                if task_type == "multiple_choice":
                    selections = example.get("Selections")
                    if not isinstance(selections, dict) or len(selections) < 2:
                        errors.append({**location, "error": "MCQ Selections must be a mapping"})
                        continue
                    if example["Output"] not in selections:
                        errors.append({**location, "error": "MCQ Output is not a selection key"})
                        continue
                    if level == 3:
                        errors.append({**location, "error": "Level 3 is short-answer only"})
                        continue

                counters[(task_type, level)] += 1
                seed_id = (
                    f"{_slug(source_name)}__{task_type}__l{level}__"
                    f"{counters[(task_type, level)]:04d}"
                )
                seed = copy.deepcopy(example)
                seed.update(
                    {
                        "seed_id": seed_id,
                        "concept_group_id": _concept_group_id(source_name, example["Fact"]),
                        "source_name": source_name,
                        "task_type": task_type,
                        "Level": level,
                        "source_instruction_index": instruction_index,
                        "source_example_index": example_index,
                        "category": category,
                        "task_definition": task_definition,
                    }
                )
                seeds.append(seed)
                counts[f"{source_name}:{task_type}:L{level}"] += 1

        normalized[source_name] = {
            "schema_version": "2.0",
            "source_name": source_name,
            "source_document": copy.deepcopy(dataset),
            "seeds": seeds,
        }

    report = {
        "valid": not errors,
        "source_count": len(normalized),
        "seed_count": sum(len(item["seeds"]) for item in normalized.values()),
        "counts": dict(sorted(counts.items())),
        "errors": errors,
    }
    if errors:
        preview = json.dumps(errors[:5], ensure_ascii=False)
        raise ValueError(f"Source validation failed with {len(errors)} error(s): {preview}")
    return normalized, report


def _choice_permutations(
    seed: dict[str, Any], count: int, random_seed: int
) -> list[dict[str, Any]]:
    selections = seed["Selections"]
    labels = list(selections)
    correct_text = selections[seed["Output"]]
    value_permutations = list(itertools.permutations(selections.values()))
    rng = random.Random(_stable_seed(random_seed, seed["seed_id"]))
    rng.shuffle(value_permutations)
    result: list[dict[str, Any]] = []
    for values in value_permutations[:count]:
        new_selections = dict(zip(labels, values, strict=True))
        correct_key = next(key for key, value in new_selections.items() if value == correct_text)
        result.append({"Selections": new_selections, "Output": correct_key})
    if len(result) != count:
        raise ValueError(f"Cannot create {count} unique permutations for {seed['seed_id']}")
    return result


def generate_augmentation_components(
    partitions: Mapping[str, PartitionValue],
    parameters: dict[str, Any],
    augmenter: OpenAIAugmenter | None = None,
) -> dict[str, Any]:
    """Generate and cache paraphrase components for every validated seed."""

    generator = augmenter or OpenAIAugmenter()
    random_seed = int(parameters["random_seed"])
    output: dict[str, Any] = {}
    for partition_id, normalized in _materialize(partitions).items():
        components: list[dict[str, Any]] = []
        for seed in normalized["seeds"]:
            config = _level_config(parameters, seed["Level"], seed["task_type"])
            if seed["task_type"] == "short_answer":
                fact_count = int(config["fact_explanation_paraphrases"])
                question_count = int(config["question_paraphrases"])
                fact_system, fact_user = fact_explanation_augmentation(seed, fact_count)
                question_system, question_user = short_question_augmentation(seed, question_count)
                negative_system, negative_user = negative_answer_augmentation(seed, 10)
                generated = {
                    "facts_and_explanations": generator.fact_explanations(
                        fact_system, fact_user, fact_count
                    ),
                    "questions": generator.questions(
                        question_system, question_user, question_count, temperature=0.2
                    ),
                    "negative_candidates": generator.negative_answers(
                        negative_system, negative_user, 10
                    ),
                }
            else:
                question_count = int(config["question_paraphrases"])
                answer_count = int(config["correct_answer_paraphrases"])
                explanation_count = int(config["explanation_paraphrases"])
                permutation_count = int(config["choice_permutations"])
                question_system, question_user = mcq_question_augmentation(seed, question_count)
                answer_system, answer_user = mcq_answer_augmentation(seed, answer_count)
                explanation_system, explanation_user = mcq_explanation_augmentation(
                    seed, explanation_count
                )
                generated = {
                    "questions": generator.questions(
                        question_system, question_user, question_count, temperature=0.7
                    ),
                    "correct_answers": generator.answers(answer_system, answer_user, answer_count),
                    "explanations": generator.explanations(
                        explanation_system, explanation_user, explanation_count
                    ),
                    "choice_permutations": _choice_permutations(
                        seed, permutation_count, random_seed
                    ),
                }
            components.append({"seed": seed, "components": generated})

        output[partition_id] = {
            "schema_version": "2.0",
            "source_name": normalized["source_name"],
            "generator_model": "gpt-5.1",
            "random_seed": random_seed,
            "seed_components": components,
        }
    return output


def _pick_candidate(candidates: list[str], random_seed: int, variant_id: str) -> str:
    if not candidates:
        raise ValueError(f"No negative candidate available for {variant_id}")
    rng = random.Random(_stable_seed(random_seed, variant_id))
    return candidates[rng.randrange(len(candidates))]


def _canonical_record(
    *,
    seed: dict[str, Any],
    variant_id: str,
    fact: str,
    question: str,
    explanation: str,
    correct_answer: str,
    correct_answer_text: str,
    incorrect_candidate: str,
    selections: dict[str, str] | None,
    origin: str,
    lineage: dict[str, Any],
) -> dict[str, Any]:
    concept_group_id = seed.get("concept_group_id") or _concept_group_id(
        seed["source_name"], seed["Fact"]
    )
    record = {
        "seed_id": seed["seed_id"],
        "concept_group_id": concept_group_id,
        "variant_id": variant_id,
        "source_name": seed["source_name"],
        "task_type": seed["task_type"],
        "Level": seed["Level"],
        "origin": origin,
        "Fact": fact.strip(),
        "Question": question.strip(),
        "correct_answer": correct_answer.strip(),
        "correct_answer_text": correct_answer_text.strip(),
        "incorrect_candidate": incorrect_candidate.strip(),
        "Explanation": explanation.strip(),
        "lineage": lineage,
    }
    if selections is not None:
        record["Selections"] = copy.deepcopy(selections)
    return record


def assemble_canonical_datasets(
    partitions: Mapping[str, PartitionValue], parameters: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Assemble original and synthetic records without mutating shared MCQ options."""

    random_seed = int(parameters["random_seed"])
    output: dict[str, Any] = {}
    counts: Counter[str] = Counter()

    for partition_id, augmented in _materialize(partitions).items():
        records: list[dict[str, Any]] = []
        for seed_bundle in augmented["seed_components"]:
            seed = seed_bundle["seed"]
            components = seed_bundle["components"]
            seed_id = seed["seed_id"]

            if seed["task_type"] == "short_answer":
                original_id = f"{seed_id}__original"
                original_negative = _pick_candidate(
                    components["negative_candidates"], random_seed, original_id
                )
                records.append(
                    _canonical_record(
                        seed=seed,
                        variant_id=original_id,
                        fact=seed["Fact"],
                        question=seed["Question"],
                        explanation=seed["Explanation"],
                        correct_answer=seed["Output"],
                        correct_answer_text=seed["Output"],
                        incorrect_candidate=original_negative,
                        selections=None,
                        origin="original",
                        lineage={"seed_id": seed_id},
                    )
                )
                variant_number = 0
                for fact_index, fact_exp in enumerate(components["facts_and_explanations"]):
                    for question_index, question in enumerate(components["questions"]):
                        variant_number += 1
                        variant_id = f"{seed_id}__aug_{variant_number:05d}"
                        negative = _pick_candidate(
                            components["negative_candidates"], random_seed, variant_id
                        )
                        records.append(
                            _canonical_record(
                                seed=seed,
                                variant_id=variant_id,
                                fact=fact_exp["Fact"],
                                question=question,
                                explanation=fact_exp["Explanation"],
                                correct_answer=seed["Output"],
                                correct_answer_text=seed["Output"],
                                incorrect_candidate=negative,
                                selections=None,
                                origin="augmented",
                                lineage={
                                    "seed_id": seed_id,
                                    "fact_explanation_variant": fact_index,
                                    "question_variant": question_index,
                                },
                            )
                        )
            else:
                original_id = f"{seed_id}__original"
                original_selections = copy.deepcopy(seed["Selections"])
                original_correct_text = original_selections[seed["Output"]]
                original_wrong = [
                    text for key, text in original_selections.items() if key != seed["Output"]
                ]
                records.append(
                    _canonical_record(
                        seed=seed,
                        variant_id=original_id,
                        fact=seed["Fact"],
                        question=seed["Question"],
                        explanation=seed["Explanation"],
                        correct_answer=seed["Output"],
                        correct_answer_text=original_correct_text,
                        incorrect_candidate=_pick_candidate(
                            original_wrong, random_seed, original_id
                        ),
                        selections=original_selections,
                        origin="original",
                        lineage={"seed_id": seed_id},
                    )
                )
                variant_number = 0
                product = itertools.product(
                    enumerate(components["questions"]),
                    enumerate(components["explanations"]),
                    enumerate(components["choice_permutations"]),
                    enumerate(components["correct_answers"]),
                )
                for (
                    (question_index, question),
                    (explanation_index, explanation),
                    (permutation_index, permutation),
                    (answer_index, answer_text),
                ) in product:
                    variant_number += 1
                    variant_id = f"{seed_id}__aug_{variant_number:05d}"
                    selections = copy.deepcopy(permutation["Selections"])
                    correct_key = permutation["Output"]
                    selections[correct_key] = answer_text
                    wrong_candidates = [
                        text for key, text in selections.items() if key != correct_key
                    ]
                    records.append(
                        _canonical_record(
                            seed=seed,
                            variant_id=variant_id,
                            fact=seed["Fact"],
                            question=question,
                            explanation=explanation,
                            correct_answer=correct_key,
                            correct_answer_text=answer_text,
                            incorrect_candidate=_pick_candidate(
                                wrong_candidates, random_seed, variant_id
                            ),
                            selections=selections,
                            origin="augmented",
                            lineage={
                                "seed_id": seed_id,
                                "question_variant": question_index,
                                "explanation_variant": explanation_index,
                                "choice_permutation": permutation_index,
                                "correct_answer_variant": answer_index,
                            },
                        )
                    )

        for record in records:
            counts[
                f"{record['source_name']}:{record['task_type']}:L{record['Level']}:{record['origin']}"
            ] += 1
        output[partition_id] = {
            "schema_version": "2.0",
            "source_name": augmented["source_name"],
            "records": records,
        }

    report = {
        "record_count": sum(len(value["records"]) for value in output.values()),
        "counts": dict(sorted(counts.items())),
    }
    return output, report


def _exact_key(record: dict[str, Any]) -> str:
    content = {
        "Fact": _normalise_text(record["Fact"]),
        "Question": _normalise_text(record["Question"]),
        "Selections": {
            key: _normalise_text(value)
            for key, value in sorted(record.get("Selections", {}).items())
        },
        "correct_answer": _normalise_text(record["correct_answer"]),
        "incorrect_candidate": _normalise_text(record["incorrect_candidate"]),
        "Explanation": _normalise_text(record["Explanation"]),
    }
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()


def _record_errors(record: dict[str, Any]) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    required = [
        "seed_id",
        "variant_id",
        "Fact",
        "Question",
        "correct_answer",
        "incorrect_candidate",
        "Explanation",
    ]
    for field in required:
        if not str(record.get(field, "")).strip():
            errors.append(f"empty_{field}")

    correct_text = record.get("correct_answer_text", record.get("correct_answer", ""))
    normal_correct = _normalise_text(correct_text)
    normal_incorrect = _normalise_text(record.get("incorrect_candidate"))
    if normal_incorrect == normal_correct:
        errors.append("negative_candidate_equals_correct_answer")
    else:
        correct_tokens = set(normal_correct.split())
        incorrect_tokens = set(normal_incorrect.split())
        token_jaccard = len(correct_tokens & incorrect_tokens) / max(
            1, len(correct_tokens | incorrect_tokens)
        )
        sequence_similarity = SequenceMatcher(None, normal_correct, normal_incorrect).ratio()
        correct_numbers = re.findall(r"[-+]?(?:\d+(?:\.\d+)?|\.\d+)", normal_correct)
        incorrect_numbers = re.findall(r"[-+]?(?:\d+(?:\.\d+)?|\.\d+)", normal_incorrect)
        has_distinct_numeric_value = bool(correct_numbers or incorrect_numbers) and (
            correct_numbers != incorrect_numbers
        )
        if not has_distinct_numeric_value and (sequence_similarity >= 0.9 or token_jaccard >= 0.8):
            errors.append("negative_candidate_too_similar_to_correct_answer")
    if record.get("task_type") == "multiple_choice":
        selections = record.get("Selections")
        output = record.get("correct_answer")
        if not isinstance(selections, dict) or output not in selections:
            errors.append("invalid_mcq_output")
        elif _normalise_text(selections[output]) != _normalise_text(correct_text):
            errors.append("mcq_output_text_mismatch")
    else:
        fact = _normalise_text(record.get("Fact"))
        answer = _normalise_text(correct_text)
        answer_tokens = set(answer.split())
        coverage = len(answer_tokens & set(fact.split())) / max(1, len(answer_tokens))
        if answer not in fact and coverage < 0.7:
            warnings.append("weak_lexical_answer_grounding")

    prompt = generation_prompt(record)
    if "\nANSWER:" in prompt.upper() or "\nEXPLANATION:" in prompt.upper():
        errors.append("target_marker_in_prompt")
    return errors, warnings


def _near_duplicate_audit(records: list[dict[str, Any]]) -> dict[str, Any]:
    if len(records) < 2:
        return {"threshold": 0.92, "pairs": [], "pair_count": 0}
    texts = [f"{item['Fact']} {item['Question']}" for item in records]
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        analyzer="word",
        ngram_range=(1, 2),
        min_df=1,
        max_features=50_000,
    )
    matrix = vectorizer.fit_transform(texts)
    neighbor_count = min(8, len(records))
    model = NearestNeighbors(metric="cosine", n_neighbors=neighbor_count, n_jobs=-1)
    model.fit(matrix)
    distances, indices = model.kneighbors(matrix)
    pairs: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for left, (row_distances, row_indices) in enumerate(zip(distances, indices, strict=True)):
        for distance, right in zip(row_distances[1:], row_indices[1:], strict=True):
            pair = tuple(sorted((left, int(right))))
            if pair in seen or records[left]["seed_id"] == records[right]["seed_id"]:
                continue
            similarity = 1.0 - float(distance)
            if similarity >= 0.92:
                seen.add(pair)
                pairs.append(
                    {
                        "left_variant_id": records[left]["variant_id"],
                        "right_variant_id": records[right]["variant_id"],
                        "left_seed_id": records[left]["seed_id"],
                        "right_seed_id": records[right]["seed_id"],
                        "similarity": round(similarity, 6),
                    }
                )
                if len(pairs) >= 5_000:
                    return {
                        "threshold": 0.92,
                        "pairs": pairs,
                        "pair_count": len(pairs),
                        "truncated": True,
                    }
    return {"threshold": 0.92, "pairs": pairs, "pair_count": len(pairs), "truncated": False}


def validate_and_audit_datasets(
    partitions: Mapping[str, PartitionValue],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """Reject structural failures, deduplicate exact copies, and audit near duplicates."""

    materialized = _materialize(partitions)
    validated: dict[str, Any] = {}
    rejected: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    exact_seen: dict[str, str] = {}
    all_valid_records: list[dict[str, Any]] = []

    for partition_id, dataset in materialized.items():
        accepted: list[dict[str, Any]] = []
        for record in dataset["records"]:
            errors, record_warnings = _record_errors(record)
            exact_key = _exact_key(record)
            if exact_key in exact_seen:
                errors.append("exact_duplicate")
            if errors:
                rejected.append(
                    {
                        "variant_id": record.get("variant_id"),
                        "seed_id": record.get("seed_id"),
                        "source_name": record.get("source_name"),
                        "reasons": errors,
                        "duplicate_of": exact_seen.get(exact_key),
                    }
                )
                continue
            exact_seen[exact_key] = record["variant_id"]
            accepted.append(record)
            all_valid_records.append(record)
            if record_warnings:
                warnings.append(
                    {
                        "variant_id": record["variant_id"],
                        "seed_id": record["seed_id"],
                        "warnings": record_warnings,
                    }
                )
        validated[partition_id] = {
            "schema_version": "2.0",
            "source_name": dataset["source_name"],
            "records": accepted,
        }

    near_audit = _near_duplicate_audit(all_valid_records)
    duplicate_report = {
        "exact_duplicate_count": sum("exact_duplicate" in item["reasons"] for item in rejected),
        "near_duplicates": near_audit,
    }
    quality_report = {
        "passed": not rejected,
        "input_record_count": sum(len(item["records"]) for item in materialized.values()),
        "accepted_record_count": len(all_valid_records),
        "rejected_record_count": len(rejected),
        "warning_count": len(warnings),
        "warnings": warnings,
    }
    return validated, quality_report, duplicate_report, rejected


def materialize_task_views(
    partitions: Mapping[str, PartitionValue],
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    """Create canonical, generation, verification, and paraphrase task views."""

    final: dict[str, Any] = {}
    generation: dict[str, Any] = {}
    verification: dict[str, Any] = {}
    paraphrase: dict[str, Any] = {}
    counts: Counter[str] = Counter()

    for partition_id, dataset in _materialize(partitions).items():
        final_records: list[dict[str, Any]] = []
        generation_records: list[dict[str, Any]] = []
        verification_records: list[dict[str, Any]] = []
        paraphrase_records: list[dict[str, Any]] = []
        originals = {
            str(record["seed_id"]): record
            for record in dataset["records"]
            if record.get("origin") == "original"
        }
        for record in dataset["records"]:
            generation_view = {
                "example_id": f"{record['variant_id']}__generation",
                "seed_id": record["seed_id"],
                "concept_group_id": record["concept_group_id"],
                "variant_id": record["variant_id"],
                "source_name": record["source_name"],
                "task_type": record["task_type"],
                "Level": record["Level"],
                "prompt": generation_prompt(record),
                "target": generation_target(record),
            }
            positive_verification = {
                "example_id": f"{record['variant_id']}__verify_correct",
                "seed_id": record["seed_id"],
                "concept_group_id": record["concept_group_id"],
                "variant_id": record["variant_id"],
                "source_name": record["source_name"],
                "task_type": "answer_verification",
                "base_task_type": record["task_type"],
                "Level": record["Level"],
                "candidate_polarity": "positive",
                "prompt": verification_prompt(record, record["correct_answer_text"]),
                "target": positive_verification_target(record),
            }
            negative_verification = {
                "example_id": f"{record['variant_id']}__verify_incorrect",
                "seed_id": record["seed_id"],
                "concept_group_id": record["concept_group_id"],
                "variant_id": record["variant_id"],
                "source_name": record["source_name"],
                "task_type": "answer_verification",
                "base_task_type": record["task_type"],
                "Level": record["Level"],
                "candidate_polarity": "negative",
                "prompt": verification_prompt(record, record["incorrect_candidate"]),
                "target": negative_verification_target(record),
            }
            enriched = copy.deepcopy(record)
            enriched["tasks"] = {
                "generation": generation_view,
                "positive_verification": positive_verification,
                "negative_verification": negative_verification,
            }
            final_records.append(enriched)
            generation_records.append(generation_view)
            verification_records.extend([positive_verification, negative_verification])
            original = originals.get(str(record["seed_id"]))
            preserves_answer = bool(
                original
                and str(record.get("correct_answer", ""))
                == str(original.get("correct_answer", ""))
            )
            if record.get("origin") != "original" and preserves_answer:
                paraphrase_records.append(
                    {
                        "example_id": f"{record['variant_id']}__paraphrase",
                        "seed_id": record["seed_id"],
                        "concept_group_id": record["concept_group_id"],
                        "variant_id": record["variant_id"],
                        "source_variant_id": original["variant_id"],
                        "source_name": record["source_name"],
                        "task_type": "structure_preserving_paraphrase",
                        "base_task_type": record["task_type"],
                        "Level": record["Level"],
                        "prompt": structure_preserving_paraphrase_prompt(
                            completed_instruction(original)
                        ),
                        "target": completed_instruction(record),
                    }
                )
            counts[f"{record['source_name']}:canonical"] += 1
            counts[f"{record['source_name']}:generation"] += 1
            counts[f"{record['source_name']}:verification_positive"] += 1
            counts[f"{record['source_name']}:verification_negative"] += 1
            counts[f"{record['task_type']}:L{record['Level']}"] += 1

        for record in paraphrase_records:
            counts[f"{record['source_name']}:paraphrase"] += 1

        metadata = {
            "schema_version": "2.1",
            "source_name": dataset["source_name"],
            "split_policy": "Downstream splits must group by concept_group_id",
        }
        final[partition_id] = {**metadata, "records": final_records}
        generation[partition_id] = {**metadata, "records": generation_records}
        verification[partition_id] = {**metadata, "records": verification_records}
        paraphrase[partition_id] = {**metadata, "records": paraphrase_records}

    statistics = {
        "canonical_record_count": sum(len(dataset["records"]) for dataset in final.values()),
        "generation_record_count": sum(len(dataset["records"]) for dataset in generation.values()),
        "verification_record_count": sum(
            len(dataset["records"]) for dataset in verification.values()
        ),
        "paraphrase_record_count": sum(
            len(dataset["records"]) for dataset in paraphrase.values()
        ),
        "counts": dict(sorted(counts.items())),
    }
    return final, generation, verification, paraphrase, statistics


def build_augmentation_manifest(
    parameters: dict[str, Any],
    source_report: dict[str, Any],
    assembly_report: dict[str, Any],
    statistics: dict[str, Any],
    quality_report: dict[str, Any],
    duplicate_report: dict[str, Any],
) -> dict[str, Any]:
    """Record the exact data-building policy and result needed for reproducibility."""

    serialized_config = json.dumps(parameters, sort_keys=True, ensure_ascii=False)
    return {
        "schema_version": "2.0",
        "created_at": datetime.now(UTC).isoformat(),
        "generator_model": "gpt-5.1",
        "random_seed": parameters["random_seed"],
        "augmentation_levels": parameters["levels"],
        "augmentation_configuration_hash": hashlib.sha256(
            serialized_config.encode("utf-8")
        ).hexdigest(),
        "source_validation": source_report,
        "assembly": assembly_report,
        "statistics": statistics,
        "quality_summary": {
            key: quality_report[key]
            for key in (
                "passed",
                "input_record_count",
                "accepted_record_count",
                "rejected_record_count",
                "warning_count",
            )
        },
        "duplicate_summary": {
            "exact_duplicate_count": duplicate_report["exact_duplicate_count"],
            "near_duplicate_count": duplicate_report["near_duplicates"]["pair_count"],
        },
        "downstream_split_requirement": (
            "Group all records by concept_group_id before assigning splits"
        ),
    }
