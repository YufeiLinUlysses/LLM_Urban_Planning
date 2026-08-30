"""Kedro definition for model training and Hugging Face publication."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import train_and_publish_model


def create_pipeline(**kwargs: object) -> Pipeline:
    del kwargs
    return pipeline(
        [
            node(
                train_and_publish_model,
                inputs=[
                    "experiment_generation_partitions",
                    "experiment_verification_partitions",
                    "experiment_split_manifest",
                    "params:models",
                    "params:training",
                ],
                outputs="model_training_receipt",
                name="train_validate_save_and_publish_model",
            )
        ]
    )
