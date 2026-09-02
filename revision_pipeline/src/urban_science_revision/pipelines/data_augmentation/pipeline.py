"""Kedro pipeline definition for instruction augmentation."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    assemble_canonical_datasets,
    build_augmentation_manifest,
    generate_augmentation_components,
    materialize_task_views,
    normalize_seed_datasets,
    validate_and_audit_datasets,
)


def create_validation_pipeline(**kwargs: object) -> Pipeline:
    del kwargs
    return pipeline(
        [
            node(
                normalize_seed_datasets,
                inputs=["seed_instruction_datasets", "params:augmentation"],
                outputs=["normalized_seed_datasets", "source_validation_report"],
                name="normalize_and_validate_seeds",
            )
        ]
    )


def create_pipeline(**kwargs: object) -> Pipeline:
    del kwargs
    return pipeline(
        [
            *create_validation_pipeline().nodes,
            node(
                generate_augmentation_components,
                inputs=["normalized_seed_datasets", "params:augmentation"],
                outputs="augmented_instruction_components",
                name="generate_augmentation_components",
            ),
            node(
                assemble_canonical_datasets,
                inputs=["augmented_instruction_components", "params:augmentation"],
                outputs=["canonical_instruction_datasets", "assembly_report"],
                name="assemble_canonical_records",
            ),
            node(
                validate_and_audit_datasets,
                inputs="canonical_instruction_datasets",
                outputs=[
                    "validated_instruction_datasets",
                    "quality_control_report",
                    "duplicate_audit",
                    "rejected_samples",
                ],
                name="validate_and_audit_records",
            ),
            node(
                materialize_task_views,
                inputs="validated_instruction_datasets",
                outputs=[
                    "final_instruction_datasets",
                    "generation_instruction_datasets",
                    "verification_instruction_datasets",
                    "paraphrase_instruction_datasets",
                    "augmentation_statistics",
                ],
                name="materialize_leakage_safe_task_views",
            ),
            node(
                build_augmentation_manifest,
                inputs=[
                    "params:augmentation",
                    "source_validation_report",
                    "assembly_report",
                    "augmentation_statistics",
                    "quality_control_report",
                    "duplicate_audit",
                ],
                outputs="augmentation_manifest",
                name="build_augmentation_manifest",
            ),
        ]
    )
