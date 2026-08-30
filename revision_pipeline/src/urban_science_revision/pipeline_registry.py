"""Register project pipelines."""

from kedro.pipeline import Pipeline

from urban_science_revision.pipelines.data_augmentation.pipeline import (
    create_pipeline as create_augmentation_pipeline,
)
from urban_science_revision.pipelines.data_augmentation.pipeline import (
    create_validation_pipeline,
)
from urban_science_revision.pipelines.evaluation_comparison.pipeline import (
    create_pipeline as create_comparison_pipeline,
)
from urban_science_revision.pipelines.experiment_data.pipeline import (
    create_pipeline as create_experiment_data_pipeline,
)
from urban_science_revision.pipelines.huggingface.pipeline import (
    create_prepare_pipeline,
    create_publish_pipeline,
)
from urban_science_revision.pipelines.model_evaluation.pipeline import (
    create_pipeline as create_evaluation_pipeline,
)
from urban_science_revision.pipelines.model_training.pipeline import (
    create_pipeline as create_training_pipeline,
)
from urban_science_revision.pipelines.reporting.pipeline import (
    create_evaluation_figures_pipeline,
    create_training_figures_pipeline,
)


def register_pipelines() -> dict[str, Pipeline]:
    augmentation = create_augmentation_pipeline()
    prepare = create_prepare_pipeline()
    publish = create_publish_pipeline()
    experiment_data = create_experiment_data_pipeline()
    training = create_training_pipeline()
    evaluation = create_evaluation_pipeline()
    comparison = create_comparison_pipeline()
    training_figures = create_training_figures_pipeline()
    evaluation_figures = create_evaluation_figures_pipeline()
    return {
        "__default__": augmentation,
        "validate_inputs": create_validation_pipeline(),
        "data_augmentation": augmentation,
        "prepare_huggingface_release": prepare,
        "augmentation_and_prepare_release": augmentation + prepare,
        "publish_huggingface": publish,
        "prepare_experiment_data": experiment_data,
        "train_model": training,
        "evaluate_model": evaluation,
        "compare_evaluations": comparison,
        "render_training_figures": training_figures,
        "build_paper_figures": evaluation_figures,
    }
