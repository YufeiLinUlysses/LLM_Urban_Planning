"""Kedro definitions for figure generation from persisted artifacts."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import build_evaluation_figures, render_training_figures_from_receipt


def create_training_figures_pipeline(**kwargs: object) -> Pipeline:
    del kwargs
    return pipeline(
        [
            node(
                render_training_figures_from_receipt,
                inputs=["model_training_receipt", "params:training_reporting"],
                outputs="training_figure_receipt",
                name="render_training_diagnostics_from_history",
            )
        ]
    )


def create_evaluation_figures_pipeline(**kwargs: object) -> Pipeline:
    del kwargs
    return pipeline(
        [
            node(
                build_evaluation_figures,
                inputs="params:paper_figures",
                outputs="paper_figure_manifest",
                name="build_manuscript_figures_from_predictions",
            )
        ]
    )
