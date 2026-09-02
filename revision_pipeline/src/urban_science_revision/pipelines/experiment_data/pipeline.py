"""Kedro definition for leakage-safe experimental partitions."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import prepare_experiment_partitions


def create_pipeline(**kwargs: object) -> Pipeline:
    del kwargs
    return pipeline(
        [
            node(
                prepare_experiment_partitions,
                inputs=[
                    "generation_instruction_datasets",
                    "verification_instruction_datasets",
                    "paraphrase_instruction_datasets",
                    "params:experiment_data",
                ],
                outputs=[
                    "experiment_generation_partitions",
                    "experiment_verification_partitions",
                    "experiment_paraphrase_partitions",
                    "experiment_split_manifest",
                    "experiment_leakage_audit",
                ],
                name="prepare_leakage_safe_experiment_partitions",
            )
        ]
    )
