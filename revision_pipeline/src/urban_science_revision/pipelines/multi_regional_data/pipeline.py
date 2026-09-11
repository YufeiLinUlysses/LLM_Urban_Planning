"""Kedro definition for evaluation-only multi-regional data preparation."""

from kedro.pipeline import Pipeline, node, pipeline

from urban_science_revision.pipelines.data_augmentation.nodes import (
    materialize_task_views,
    normalize_seed_datasets,
    validate_and_audit_datasets,
)

from .nodes import assemble_original_canonical_datasets


def create_pipeline(**kwargs: object) -> Pipeline:
    del kwargs
    return pipeline(
        [
            node(
                normalize_seed_datasets,
                inputs=[
                    "multi_regional_seed_datasets",
                    "params:multi_regional_preparation",
                ],
                outputs=[
                    "multi_regional_normalized_datasets",
                    "multi_regional_source_validation_report",
                ],
                name="normalize_multi_regional_seeds",
            ),
            node(
                assemble_original_canonical_datasets,
                inputs=[
                    "multi_regional_normalized_datasets",
                    "multi_regional_source_profiles",
                ],
                outputs=[
                    "multi_regional_canonical_datasets",
                    "multi_regional_assembly_report",
                ],
                name="assemble_multi_regional_originals",
            ),
            node(
                validate_and_audit_datasets,
                inputs="multi_regional_canonical_datasets",
                outputs=[
                    "multi_regional_validated_datasets",
                    "multi_regional_quality_report",
                    "multi_regional_duplicate_audit",
                    "multi_regional_rejected_samples",
                ],
                name="validate_multi_regional_records",
            ),
            node(
                materialize_task_views,
                inputs="multi_regional_validated_datasets",
                outputs=[
                    "multi_regional_final_datasets",
                    "multi_regional_generation_datasets",
                    "multi_regional_verification_datasets",
                    "multi_regional_paraphrase_datasets",
                    "multi_regional_statistics",
                ],
                name="materialize_multi_regional_task_views",
            ),
        ]
    )
