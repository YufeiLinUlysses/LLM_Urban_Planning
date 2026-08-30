from __future__ import annotations

import pytest

from urban_science_revision.pipelines.huggingface.nodes import prepare_release_bundle


def _partition() -> dict:
    return {
        "sample": {
            "schema_version": "2.0",
            "source_name": "sample",
            "records": [{"seed_id": "sample__short_answer__l1__0001"}],
        }
    }


def _parameters() -> dict:
    return {
        "repo_id": "owner/dataset",
        "repo_type": "dataset",
        "revision": "revision-v2",
        "path_in_repo": "",
        "dataset_version": "2.0.0",
        "release_dir": "data/07_model_output/huggingface_release",
        "commit_message": "Publish v2",
    }


def test_prepare_release_builds_named_configs_and_manifest() -> None:
    outputs = prepare_release_bundle(
        _partition(),
        _partition(),
        _partition(),
        {
            "canonical_record_count": 1,
            "generation_record_count": 1,
            "verification_record_count": 2,
        },
        {"passed": True},
        {"exact_duplicate_count": 0},
        [],
        {"augmentation_configuration_hash": "abc123"},
        _parameters(),
    )
    readme = outputs[4]
    manifest = outputs[5]
    assert isinstance(outputs[0]["sample"], list)
    assert isinstance(outputs[1]["sample"], list)
    assert isinstance(outputs[2]["sample"], list)
    assert "revision_v2_generation" in readme
    assert "split records by `concept_group_id`" in readme
    assert manifest["dataset_version"] == "2.0.0"
    assert manifest["record_counts"]["verification"] == 2
    assert manifest["augmentation_configuration_hash"] == "abc123"


def test_prepare_release_refuses_failed_quality_gate() -> None:
    with pytest.raises(ValueError, match="Quality-control gate failed"):
        prepare_release_bundle(
            _partition(),
            _partition(),
            _partition(),
            {
                "canonical_record_count": 1,
                "generation_record_count": 1,
                "verification_record_count": 2,
            },
            {"passed": False},
            {},
            [],
            {"augmentation_configuration_hash": "abc123"},
            _parameters(),
        )
