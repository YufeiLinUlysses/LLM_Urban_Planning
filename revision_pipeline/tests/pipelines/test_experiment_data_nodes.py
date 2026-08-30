from __future__ import annotations

from urban_science_revision.pipelines.experiment_data.nodes import (
    prepare_experiment_partitions,
)


def _dataset(source: str, seeds: int = 10) -> dict:
    return {
        "source_name": source,
        "records": [
            {
                "example_id": f"{source}-{index}-variant",
                "seed_id": f"{source}-{index}",
                "source_name": source,
                "task_type": "short_answer",
                "Level": 1,
                "prompt": f"Question {index}",
                "target": f"ANSWER:\nAnswer {index}\n\nEXPLANATION:\nReason {index}",
            }
            for index in range(seeds)
        ],
    }


def _parameters() -> dict:
    return {
        "dataset_version": "test-v1",
        "random_seed": 42,
        "split_ratios": {"train": 0.8, "validation": 0.1, "test": 0.1},
        "holdout_source_patterns": ["france", "japan"],
        "region_patterns": {"france": "france", "japan": "japan"},
        "fail_on_leakage": True,
    }


def test_seed_split_and_combined_cross_regional_holdout() -> None:
    generation = {
        "highd": _dataset("highd"),
        "france": _dataset("france", 2),
        "japan": _dataset("japan", 2),
    }
    verification = {key: value for key, value in generation.items()}
    gen, ver, manifest, audit = prepare_experiment_partitions(
        generation, verification, _parameters()
    )

    assert manifest["seed_counts"] == {"train": 8, "validation": 1, "test": 1}
    assert audit["passed"] is True
    holdouts = [value for value in gen.values() if value["split"] == "cross_regional"]
    assert {value["region"] for value in holdouts} == {"france", "japan"}
    assert all(
        row["evaluation_scope"] == "cross_regional" for item in holdouts for row in item["records"]
    )
    assert set(gen) == set(ver)
