# Urban Science Revision Pipeline

This isolated Kedro project replaces the legacy one-file augmentation workflow while leaving
the paper and existing data directories untouched.

## Data stages

- `../instruction_dataset`: curated seed instruction datasets (read-only input).
- `data/02_intermediate/augmented`: reusable OpenAI-generated augmentation components.
- `data/03_primary/validated`: validated canonical augmented records.
- `data/05_model_input/final`: canonical, generation, and verification instruction datasets.
- `data/07_model_output/huggingface_release`: a Hub-ready release bundle.
- `data/08_reporting`: validation, duplicate, rejection, and count reports.

Every augmented descendant retains a deterministic `seed_id`. Downstream train/validation/test
splits must group on that field; row-level random splitting is unsafe.

## Setup

```powershell
uv sync
```

Set `OPENAI_API_KEY` before augmentation and `HF_TOKEN` before publishing. The pipeline checks
the environment and then the existing parent-level `.env`; secrets are never copied into Kedro
configuration or release artifacts.

## Pipelines

```powershell
# Validate and normalize the source files without API calls
uv run kedro run --pipelines validate_inputs

# Generate, assemble, validate, and materialize all instruction views
uv run kedro run --pipelines data_augmentation

# Build a local Hugging Face release bundle without uploading it
uv run kedro run --pipelines prepare_huggingface_release

# Explicit external mutation: upload the prepared release
uv run kedro run --pipelines publish_huggingface
```

The publishing pipeline is intentionally excluded from the default pipeline.
# Model training and evaluation

The revision workflow provides three experiment pipelines in addition to augmentation:

```text
prepare_experiment_data -> train_model -> evaluate_model
```

`prepare_experiment_data` performs a deterministic 80/10/10 split by `seed_id` for the
training-source datasets. France and Japan bypass this split and are combined under the
`cross_regional` evaluation scope while retaining a `region` field.

`train_model` trains on generation and verification train records, uses only validation records
for checkpoint selection, saves the selected checkpoint locally, and publishes it to
`UlyssesLynne/urban-planning-llm-model-zoo` when `training.publish_to_hf=true`.

`evaluate_model` runs the same suite for base and fine-tuned checkpoints against either
`in_domain` or `cross_regional`. It saves all raw predictions, parsed outputs, metrics, grouped
metrics, and the human-review queue locally before uploading the run directory directly to
`UlyssesLynne/urban-planning-llm-predictions`.

Typical Colab commands are:

```bash
kedro run --pipeline=prepare_experiment_data
kedro run --pipeline=evaluate_model --params=evaluation.model_key:llama31_8b,evaluation.checkpoint_stage:base,evaluation.dataset_scope:in_domain
kedro run --pipeline=train_model --params=training.model_key:llama31_8b
kedro run --pipeline=evaluate_model --params=evaluation.model_key:llama31_8b,evaluation.checkpoint_stage:fine_tuned,evaluation.checkpoint_uri:UlyssesLynne/urban-planning-llm-model-zoo,evaluation.checkpoint_subfolder:llama31_8b/RUN_ID/checkpoint,evaluation.dataset_scope:cross_regional
```

Set `HF_TOKEN` through a Colab secret or environment variable. Override
`training.artifact_root` and `evaluation.artifact_root` with a mounted Google Drive path to
make GPU runs resumable across sessions. Set either `publish_to_hf` flag to `false` for local
smoke tests.
