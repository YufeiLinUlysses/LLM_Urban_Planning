"""Kedro definition for evaluation comparison."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import compare_evaluation_runs


def create_pipeline(**kwargs: object) -> Pipeline:
    del kwargs
    return pipeline(
        [
            node(
                compare_evaluation_runs,
                inputs="params:evaluation_comparison",
                outputs="evaluation_comparison_report",
                name="compare_base_and_fine_tuned_evaluations",
            )
        ]
    )
