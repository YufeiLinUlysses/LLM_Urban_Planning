"""Build evaluation-only multi-regional seed files from reviewed source profiles.

The source profile stores ten independently evidenced Level-1 concepts per
dataset. Five Level-2 and five Level-3 concepts are derived deterministically
from those facts. Level 1 and 2 receive both MC and short-answer task views;
Level 3 remains short-answer only, matching the existing instruction schema.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

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

LEVEL2_PAIRS = (
    ("spatial_coverage", "measurement_resolution"),
    ("temporal_coverage", "collection_methodology"),
    ("variable_definitions", "file_structure_and_documentation"),
    ("data_accessibility_and_availability", "usage_context"),
    ("measurement_resolution", "dataset_limitations"),
)

LEVEL3_PAIRS = (
    ("usage_context", "dataset_limitations", "planning_suitability"),
    ("spatial_coverage", "dataset_limitations", "representativeness"),
    ("temporal_coverage", "measurement_resolution", "temporal_inference"),
    ("variable_definitions", "collection_methodology", "measurement_interpretation"),
    (
        "provenance",
        "data_accessibility_and_availability",
        "reproducibility_and_governance",
    ),
)

LEVEL1_EXPLANATION_TEMPLATES = {
    "provenance": "{answer} is responsible for publishing or maintaining the dataset.",
    "spatial_coverage": "The data cover the following geographic area: {answer}.",
    "temporal_coverage": "The available records have this time coverage: {answer}.",
    "measurement_resolution": "The data use this reporting resolution: {answer}.",
    "variable_definitions": "Each record describes: {answer}.",
    "file_structure_and_documentation": "The dataset uses this file organization: {answer}.",
    "collection_methodology": "The observations were gathered using this method: {answer}.",
    "data_accessibility_and_availability": "The stated access terms are: {answer}.",
    "usage_context": "The data support this use: {answer}.",
    "dataset_limitations": "The stated limitation is: {answer}.",
}


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")


def _concept_id(dataset_id: str, level: int, number: int) -> str:
    return f"{dataset_id}__l{level}__{number:02d}"


def _group_id(dataset_id: str, fact: str) -> str:
    digest = hashlib.sha256(" ".join(fact.casefold().split()).encode()).hexdigest()[:16]
    return f"{dataset_id}__concept__{digest}"


def _join_facts(left: dict[str, Any], right: dict[str, Any]) -> str:
    return f"{left['fact']} {right['fact']}"


def _evidence(item: dict[str, Any], profile: dict[str, Any]) -> list[str]:
    return item.get("evidence_urls", profile["source_urls"])


def _level2_concepts(profile: dict[str, Any]) -> list[dict[str, Any]]:
    perspectives = profile["perspectives"]
    concepts = []
    for number, (left_key, right_key) in enumerate(LEVEL2_PAIRS, 1):
        left, right = perspectives[left_key], perspectives[right_key]
        concepts.append(
            {
                "concept_id": _concept_id(profile["dataset_id"], 2, number),
                "Level": 2,
                "perspectives": [left_key, right_key],
                "Fact": _join_facts(left, right),
                "Question": (
                    f"How should the documented {left_key.replace('_', ' ')} and "
                    f"{right_key.replace('_', ' ')} be considered together?"
                ),
                "Output": f"{left['answer']} {right['answer']}",
                "Explanation": (
                    "The conclusion combines two independently documented properties of "
                    f"{profile['dataset_name']}: {left['answer']} {right['answer']}"
                ),
                "evidence_urls": sorted(set(_evidence(left, profile) + _evidence(right, profile))),
                "inference_type": "multi_attribute_synthesis",
            }
        )
    return concepts


def _level3_concepts(profile: dict[str, Any]) -> list[dict[str, Any]]:
    perspectives = profile["perspectives"]
    question_stems = {
        "planning_suitability": (
            "What planning use is supported, and what limitation must qualify that use?"
        ),
        "representativeness": (
            "What limits the geographic or population representativeness of this dataset?"
        ),
        "temporal_inference": (
            "What temporal inference is justified, and what temporal inference should be avoided?"
        ),
        "measurement_interpretation": (
            "How does the collection method affect interpretation of the recorded variables?"
        ),
        "reproducibility_and_governance": (
            "What supports reproducible reuse, and what access or governance condition remains?"
        ),
    }
    concepts = []
    for number, (left_key, right_key, reasoning) in enumerate(LEVEL3_PAIRS, 1):
        left, right = perspectives[left_key], perspectives[right_key]
        concepts.append(
            {
                "concept_id": _concept_id(profile["dataset_id"], 3, number),
                "Level": 3,
                "perspectives": [left_key, right_key],
                "Fact": _join_facts(left, right),
                "Question": question_stems[reasoning],
                "Output": f"{left['answer']} However, {right['answer']}",
                "Explanation": (
                    "This interpretation is limited to the supplied documentation and "
                    "does not infer unreported coverage, causality, or performance."
                ),
                "evidence_urls": sorted(set(_evidence(left, profile) + _evidence(right, profile))),
                "inference_type": reasoning,
            }
        )
    return concepts


def _level1_concepts(profile: dict[str, Any]) -> list[dict[str, Any]]:
    concepts = []
    for number, perspective in enumerate(PERSPECTIVES, 1):
        item = profile["perspectives"][perspective]
        concepts.append(
            {
                "concept_id": _concept_id(profile["dataset_id"], 1, number),
                "Level": 1,
                "perspectives": [perspective],
                "Fact": item["fact"],
                "Question": item["question"],
                "Output": item["answer"],
                "Explanation": item.get(
                    "explanation",
                    LEVEL1_EXPLANATION_TEMPLATES[perspective].format(
                        answer=item["answer"].rstrip(".")
                    ),
                ),
                "evidence_urls": item.get("evidence_urls", profile["source_urls"]),
                "inference_type": item.get("inference_type", "direct"),
            }
        )
    return concepts


def _mc_view(
    concept: dict[str, Any], profile: dict[str, Any], all_profiles: list[dict[str, Any]]
) -> dict[str, Any]:
    correct = concept["Output"]
    candidates = []
    for other in all_profiles:
        if other["dataset_id"] == profile["dataset_id"]:
            continue
        answers = [other["perspectives"][key]["answer"] for key in concept["perspectives"]]
        candidates.append(" ".join(answers))
    distractors = []
    for candidate in candidates:
        if candidate != correct and candidate not in distractors:
            distractors.append(candidate)
        if len(distractors) == 3:
            break
    while len(distractors) < 3:
        distractors.append(
            "The supplied documentation does not support this alternative conclusion."
        )
    choices = distractors[:3]
    correct_index = int(hashlib.sha256(concept["concept_id"].encode()).hexdigest(), 16) % 4
    choices.insert(correct_index, correct)
    labels = "ABCD"
    return {
        "Fact": concept["Fact"],
        "Question": concept["Question"],
        "Selections": dict(zip(labels, choices, strict=True)),
        "Output": labels[correct_index],
        "Explanation": concept["Explanation"],
        "Level": concept["Level"],
    }


def _short_view(concept: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    return {
        "Fact": concept["Fact"],
        "Question": concept["Question"],
        "Output": concept["Output"],
        "Explanation": concept["Explanation"],
        "Level": concept["Level"],
    }


def _validate_profile(profile: dict[str, Any]) -> None:
    missing = set(PERSPECTIVES) - set(profile["perspectives"])
    extra = set(profile["perspectives"]) - set(PERSPECTIVES)
    if missing or extra:
        raise ValueError(
            f"{profile['dataset_id']} perspective mismatch: missing={missing}, extra={extra}"
        )
    for perspective, item in profile["perspectives"].items():
        required = {"fact", "question", "answer"}
        absent = required - set(item)
        if absent:
            raise ValueError(f"{profile['dataset_id']}:{perspective} missing {absent}")
        if not item.get("evidence_urls", profile.get("source_urls")):
            raise ValueError(f"{profile['dataset_id']}:{perspective} has no evidence URL")


def build(source: Path, output_dir: Path) -> dict[str, Any]:
    manifest = json.loads(source.read_text(encoding="utf-8"))
    profiles = manifest["datasets"]
    for profile in profiles:
        _validate_profile(profile)

    output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "benchmark": manifest["benchmark"],
        "dataset_version": "multi_regional_external_v1",
        "datasets": {},
    }
    for profile in profiles:
        concepts = _level1_concepts(profile) + _level2_concepts(profile) + _level3_concepts(profile)
        mc = [_mc_view(item, profile, profiles) for item in concepts if item["Level"] < 3]
        short = [_short_view(item, profile) for item in concepts]
        document = {
            "dataset_name": profile["dataset_name"],
            "dataset_link": profile["source_urls"][0],
            "dataset_description": profile["description"],
            "instructions": [
                {
                    "Instruction information": {"Categories": "Multiple Selection"},
                    "Task_Definition": "Select one supported option and provide its letter.",
                    "Positive Example": mc,
                },
                {
                    "Instruction information": {"Categories": "Question answering"},
                    "Task_Definition": "Answer using only the supplied dataset context.",
                    "Positive Example": short,
                },
            ],
        }
        destination = output_dir / f"{profile['dataset_id']}_instructions.json"
        destination.write_text(
            json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        summary["datasets"][profile["dataset_id"]] = {
            "concepts": len(concepts),
            "mc_views": len(mc),
            "short_answer_views": len(short),
            "task_views": len(mc) + len(short),
            "review_status": profile["review_status"],
        }

    summary["independent_concepts"] = sum(
        entry["concepts"] for entry in summary["datasets"].values()
    )
    summary["task_views"] = sum(entry["task_views"] for entry in summary["datasets"].values())
    (output_dir.parent / "seed_manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/01_raw/multi_regional_external_v1/source_profiles.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/01_raw/multi_regional_external_v1/seeds"),
    )
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
