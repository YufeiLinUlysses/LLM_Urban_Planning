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

Every augmented descendant retains deterministic `seed_id` and `concept_group_id` lineage.
Downstream train/validation/test splits group on `concept_group_id`, keeping MCQ,
short-answer, verification, and augmented variants of the same normalized source fact together.

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

`prepare_experiment_data` performs a deterministic 80/10/10 split by `concept_group_id` for the
training-source datasets. France and Japan bypass this split and are combined under the
`cross_regional` evaluation scope while retaining a `region` field.

The same pipeline audits cross-split leakage in four layers: concept-group overlap, seed
overlap, exact normalized prompt overlap, and sentence-embedding similarity. Semantic
candidates above the configured threshold are ranked into a human-review queue in
`data/08_reporting/experiment/leakage_audit.json`; they are not automatically deleted because
similar urban terminology does not necessarily imply leakage. Human-confirmed duplicates can be
recorded in `experiment_data.concept_group_aliases` and are resolved before splitting. Review
candidates are deduplicated by concept pair so augmented variants do not require repeating the
same adjudication.

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

### Reusable Colab runner

Use `notebooks/urban_science_colab_runner.ipynb` for GPU runs. Its workflow is split into
independent environment, data, audit, smoke-test, training, graph, evaluation, and Hugging Face
verification sections. Change only the `RunConfig` cell when switching dataset revisions, models,
or run IDs. The default workflow uses Colab-local storage and verifies all required artifacts on
Hugging Face before the runtime is discarded; Google Drive is not required.

Regenerate the notebook after changing its template with:

```powershell
uv run python scripts/build_colab_notebook.py
```

## Reproducible figures

`train_model` writes `training_history.json`, a CSV export, and 300-dpi training/validation-loss
and learning-rate figures inside the final adapter's `checkpoint/figures` directory. Because the
figures are created before model publication, they are included in the Hugging Face model upload.
To regenerate the plots for an existing run without retraining:

```bash
kedro run --pipeline=render_training_figures
```

After running both `base` and `fine_tuned` evaluation for each desired model, build manuscript
figures directly from the persisted prediction Parquets:

```bash
kedro run --pipeline=build_paper_figures
```

Outputs are stored under `data/08_reporting/evaluations/paper_figures`: multiple-choice accuracy,
short-answer BERTScore F1, the strict-versus-normalized MC formatting diagnostic, a CSV containing
the exact plotted values, and a manifest recording every source prediction file. Configure model
order, exact run IDs, artifact roots, and output paths under `paper_figures` in
`conf/base/parameters_reporting.yml` or a Colab environment override.
