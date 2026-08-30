"""Architecture-neutral supervised fine-tuning and model publication."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PartitionValue = dict[str, Any] | Callable[[], dict[str, Any]]


def _materialize_records(
    partitions: Mapping[str, PartitionValue], split: str
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for value in partitions.values():
        dataset = value() if callable(value) else value
        if dataset.get("split") == split:
            rows.extend(dataset.get("records", []))
    return rows


def _manifest_hash(manifest: dict[str, Any]) -> str:
    encoded = json.dumps(manifest, sort_keys=True, ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def _training_rows(
    generation: Mapping[str, PartitionValue],
    verification: Mapping[str, PartitionValue],
    split: str,
    tasks: list[str],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if "generation" in tasks:
        rows.extend(
            {"prompt": str(row["prompt"]), "target": str(row["target"])}
            for row in _materialize_records(generation, split)
        )
    if "verification" in tasks:
        rows.extend(
            {"prompt": str(row["prompt"]), "target": str(row["target"])}
            for row in _materialize_records(verification, split)
        )
    if not rows:
        raise ValueError(f"No {split} rows found for tasks {tasks}")
    return rows


def _limit_rows(
    rows: list[dict[str, str]], limit: int | None, random_seed: int
) -> list[dict[str, str]]:
    """Select a stable smoke-test subset independent of partition file order."""

    if limit is None or limit >= len(rows):
        return rows
    if limit <= 0:
        raise ValueError("Sample limits must be positive")
    return sorted(
        rows,
        key=lambda row: hashlib.sha256(
            f"{random_seed}:{row['prompt']}:{row['target']}".encode()
        ).hexdigest(),
    )[:limit]


def _publish_folder(local_dir: Path, path_in_repo: str, parameters: dict[str, Any]) -> str:
    from huggingface_hub import HfApi

    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required when training.publish_to_hf is true")
    api = HfApi(token=token)
    repo_id = parameters["model_repo_id"]
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    commit = api.upload_folder(
        folder_path=str(local_dir),
        repo_id=repo_id,
        repo_type="model",
        path_in_repo=path_in_repo,
        commit_message=f"Publish trained model {path_in_repo}",
    )
    return str(commit)


def train_and_publish_model(
    generation_partitions: Mapping[str, PartitionValue],
    verification_partitions: Mapping[str, PartitionValue],
    split_manifest: dict[str, Any],
    model_registry: dict[str, Any],
    parameters: dict[str, Any],
) -> dict[str, Any]:
    """Fine-tune one configured model, save it locally, and optionally publish it."""

    try:
        import torch
        from datasets import Dataset
        from transformers import (
            AutoModelForCausalLM,
            AutoModelForSeq2SeqLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            DataCollatorForSeq2Seq,
            Trainer,
            TrainingArguments,
            default_data_collator,
        )
    except ImportError as exc:  # pragma: no cover - exercised in Colab/GPU environments
        raise RuntimeError(
            "Training requires the ML dependencies; run `pip install -e .`."
        ) from exc

    model_key = parameters["model_key"]
    if model_key not in model_registry:
        raise KeyError(f"Unknown model_key {model_key!r}; choose from {sorted(model_registry)}")
    spec = model_registry[model_key]
    family = spec["family"]
    tasks = list(parameters.get("tasks", ["generation", "verification"]))
    train_rows = _training_rows(generation_partitions, verification_partitions, "train", tasks)
    validation_rows = _training_rows(
        generation_partitions, verification_partitions, "validation", tasks
    )
    random_seed = int(parameters.get("random_seed", 42))
    train_rows = _limit_rows(train_rows, parameters.get("max_train_samples"), random_seed)
    validation_rows = _limit_rows(
        validation_rows, parameters.get("max_validation_samples"), random_seed
    )

    run_id = parameters.get("run_id") or datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    local_dir = Path(parameters["artifact_root"]) / "models" / model_key / run_id
    checkpoint_dir = local_dir / "checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(spec["model_id"], revision=spec.get("revision"))

    quantization = None
    if family == "causal_lm" and spec.get("load_in_4bit", False):
        if not torch.cuda.is_available():
            raise RuntimeError("4-bit QLoRA training requires a CUDA runtime")
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    if family == "causal_lm":
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            spec["model_id"],
            revision=spec.get("revision"),
            quantization_config=quantization,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        if spec.get("use_lora", True):
            from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

            if quantization is not None:
                model = prepare_model_for_kbit_training(model)
            model = get_peft_model(
                model,
                LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=int(spec.get("lora_r", 16)),
                    lora_alpha=int(spec.get("lora_alpha", 32)),
                    lora_dropout=float(spec.get("lora_dropout", 0.05)),
                    target_modules=spec.get(
                        "target_modules",
                        [
                            "q_proj",
                            "k_proj",
                            "v_proj",
                            "o_proj",
                            "gate_proj",
                            "up_proj",
                            "down_proj",
                        ],
                    ),
                    bias="none",
                ),
            )

        max_source = int(parameters["max_source_length"])
        max_target = int(parameters["max_target_length"])

        def tokenize(batch: dict[str, list[str]]) -> dict[str, Any]:
            all_ids: list[list[int]] = []
            all_labels: list[list[int]] = []
            all_masks: list[list[int]] = []
            for prompt, target in zip(batch["prompt"], batch["target"], strict=True):
                prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"][:max_source]
                target_ids = tokenizer(target, add_special_tokens=False)["input_ids"][:max_target]
                if tokenizer.eos_token_id is not None:
                    target_ids.append(tokenizer.eos_token_id)
                ids = (prompt_ids + target_ids)[: max_source + max_target]
                labels = ([-100] * len(prompt_ids) + target_ids)[: len(ids)]
                padding = max_source + max_target - len(ids)
                all_ids.append(ids + [tokenizer.pad_token_id] * padding)
                all_labels.append(labels + [-100] * padding)
                all_masks.append([1] * len(ids) + [0] * padding)
            return {"input_ids": all_ids, "labels": all_labels, "attention_mask": all_masks}

        data_collator = default_data_collator
    elif family == "seq2seq":
        model = AutoModelForSeq2SeqLM.from_pretrained(
            spec["model_id"], revision=spec.get("revision")
        )
        max_source = int(parameters["max_source_length"])
        max_target = int(parameters["max_target_length"])

        def tokenize(batch: dict[str, list[str]]) -> dict[str, Any]:
            encoded = tokenizer(batch["prompt"], truncation=True, max_length=max_source)
            labels = tokenizer(text_target=batch["target"], truncation=True, max_length=max_target)[
                "input_ids"
            ]
            encoded["labels"] = labels
            return encoded

        data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    else:
        raise ValueError(f"Unsupported model family {family!r}")

    train_dataset = Dataset.from_list(train_rows).map(
        tokenize, batched=True, remove_columns=["prompt", "target"]
    )
    validation_dataset = Dataset.from_list(validation_rows).map(
        tokenize, batched=True, remove_columns=["prompt", "target"]
    )
    args = TrainingArguments(
        output_dir=str(local_dir / "trainer"),
        per_device_train_batch_size=int(parameters["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(parameters["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(parameters["gradient_accumulation_steps"]),
        num_train_epochs=float(parameters["num_train_epochs"]),
        learning_rate=float(parameters["learning_rate"]),
        warmup_ratio=float(parameters.get("warmup_ratio", 0.03)),
        weight_decay=float(parameters.get("weight_decay", 0.0)),
        logging_steps=int(parameters.get("logging_steps", 10)),
        eval_strategy="steps",
        eval_steps=int(parameters.get("eval_steps", 200)),
        save_strategy="steps",
        save_steps=int(parameters.get("save_steps", 200)),
        save_total_limit=int(parameters.get("save_total_limit", 2)),
        load_best_model_at_end=True,
        report_to="none",
        seed=int(parameters.get("random_seed", 42)),
        bf16=bool(parameters.get("bf16", True) and torch.cuda.is_available()),
        fp16=bool(parameters.get("fp16", False) and torch.cuda.is_available()),
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
    )
    train_result = trainer.train(resume_from_checkpoint=parameters.get("resume_from_checkpoint"))
    validation_metrics = trainer.evaluate()
    trainer.save_model(str(checkpoint_dir))
    tokenizer.save_pretrained(str(checkpoint_dir))

    path_in_repo = f"{model_key}/{run_id}/checkpoint"
    receipt = {
        "training_run_id": run_id,
        "model_key": model_key,
        "family": family,
        "base_model_id": spec["model_id"],
        "dataset_version": split_manifest["dataset_version"],
        "split_manifest_hash": _manifest_hash(split_manifest),
        "tasks": tasks,
        "train_record_count": len(train_rows),
        "validation_record_count": len(validation_rows),
        "local_checkpoint": str(checkpoint_dir.resolve()),
        "hf_repo_id": parameters["model_repo_id"],
        "hf_path": path_in_repo,
        "train_metrics": train_result.metrics,
        "validation_metrics": validation_metrics,
        "published": False,
    }
    (checkpoint_dir / "training_history.json").write_text(
        json.dumps(trainer.state.log_history, indent=2, default=str), encoding="utf-8"
    )
    if parameters.get("publish_to_hf", True):
        receipt["published"] = True
        (checkpoint_dir / "training_manifest.json").write_text(
            json.dumps(receipt, indent=2, default=str), encoding="utf-8"
        )
        receipt["commit"] = _publish_folder(checkpoint_dir, path_in_repo, parameters)
    (checkpoint_dir / "training_manifest.json").write_text(
        json.dumps(receipt, indent=2, default=str), encoding="utf-8"
    )
    (local_dir / "training_manifest.json").write_text(
        json.dumps(receipt, indent=2, default=str), encoding="utf-8"
    )
    return receipt
