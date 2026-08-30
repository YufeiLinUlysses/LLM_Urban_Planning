from __future__ import annotations

from urban_science_revision.pipelines.experiment_data.nodes import (
    _semantic_cross_split_audit,
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
        "semantic_audit": {"enabled": False},
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


def test_huggingface_flat_list_partitions_are_supported() -> None:
    generation = {"highd": _dataset("highd")["records"]}
    verification = {"highd": _dataset("highd")["records"]}

    gen, ver, manifest, audit = prepare_experiment_partitions(
        generation, verification, _parameters()
    )

    assert manifest["seed_counts"] == {"train": 8, "validation": 1, "test": 1}
    assert audit["passed"] is True
    assert set(gen) == set(ver)


class _FakeEncoder:
    def encode(self, prompts, **kwargs):
        del kwargs
        vectors = {
            "same meaning alpha": [1.0, 0.0],
            "same meaning beta": [0.99, 0.01],
            "unrelated": [0.0, 1.0],
        }
        return [vectors[prompt] for prompt in prompts]


def test_semantic_audit_compares_only_different_splits() -> None:
    partitions = {
        "train__source": {
            "split": "train",
            "records": [
                {
                    "example_id": "train-1",
                    "seed_id": "seed-1",
                    "source_name": "source",
                    "prompt": "same meaning alpha",
                }
            ],
        },
        "test__source": {
            "split": "test",
            "records": [
                {
                    "example_id": "test-1",
                    "seed_id": "seed-2",
                    "source_name": "source",
                    "prompt": "same meaning beta",
                },
                {
                    "example_id": "test-2",
                    "seed_id": "seed-3",
                    "source_name": "source",
                    "prompt": "unrelated",
                },
            ],
        },
    }
    parameters = {
        "semantic_audit": {
            "enabled": True,
            "model_id": "fake",
            "similarity_threshold": 0.9,
            "nearest_neighbors_per_record": 2,
            "human_review_top_n": 10,
            "human_review_boundary_n": 10,
            "max_candidates_per_split_pair": 100,
        }
    }

    audit = _semantic_cross_split_audit(partitions, parameters, _FakeEncoder())

    assert audit["review_required"] is True
    assert audit["split_pairs"]["test__train"]["flagged_candidate_count"] == 1
    candidate = audit["human_review_queue"][0]
    assert {candidate["left_seed_id"], candidate["right_seed_id"]} == {"seed-1", "seed-2"}
