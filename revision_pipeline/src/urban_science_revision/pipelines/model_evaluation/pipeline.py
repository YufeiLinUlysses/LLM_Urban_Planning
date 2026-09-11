"""Kedro definition for inference, scoring, and direct publication."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import evaluate_and_publish_model


def create_pipeline(**kwargs: object) -> Pipeline:
    del kwargs
    return pipeline(
        [
            node(
                evaluate_and_publish_model,
                inputs=[
                    "experiment_generation_partitions",
                    "experiment_verification_partitions",
                    "experiment_paraphrase_partitions",
                    "experiment_split_manifest",
                    "params:models",
                    "params:evaluation",
                ],
                outputs="model_evaluation_receipt",
                name="evaluate_score_save_and_publish_model",
            )
        ]
    )


def create_multi_regional_pipeline(**kwargs: object) -> Pipeline:
    """Evaluate only the isolated ten-dataset multi-regional benchmark."""

    del kwargs
    return pipeline(
        [
            node(
                evaluate_and_publish_model,
                inputs=[
                    "multi_regional_generation_datasets",
                    "multi_regional_verification_datasets",
                    "multi_regional_paraphrase_datasets",
                    "multi_regional_evaluation_manifest",
                    "params:models",
                    "params:evaluation",
                ],
                outputs="model_evaluation_receipt",
                name="evaluate_multi_regional_model",
            )
        ]
    )
