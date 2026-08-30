"""Kedro definitions for local release preparation and opt-in Hub publication."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import prepare_release_bundle, publish_release


def create_prepare_pipeline(**kwargs: object) -> Pipeline:
    del kwargs
    return pipeline(
        [
            node(
                prepare_release_bundle,
                inputs=[
                    "final_instruction_datasets",
                    "generation_instruction_datasets",
                    "verification_instruction_datasets",
                    "augmentation_statistics",
                    "quality_control_report",
                    "duplicate_audit",
                    "rejected_samples",
                    "augmentation_manifest",
                    "params:huggingface",
                ],
                outputs=[
                    "hf_release_canonical",
                    "hf_release_generation",
                    "hf_release_verification",
                    "hf_release_quality",
                    "hf_release_readme",
                    "hf_release_manifest",
                ],
                name="prepare_huggingface_release_bundle",
            )
        ]
    )


def create_publish_pipeline(**kwargs: object) -> Pipeline:
    del kwargs
    return pipeline(
        [
            node(
                publish_release,
                inputs=["hf_release_manifest", "params:huggingface"],
                outputs="hf_publish_receipt",
                name="publish_huggingface_release",
            )
        ]
    )
