"""Unified model inference, transparent scoring, and Hugging Face publication."""

from __future__ import annotations

import json
import math
import os
import re
import statistics
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PartitionValue = dict[str, Any] | Callable[[], dict[str, Any]]


def _materialize_records(
    partitions: Mapping[str, PartitionValue], scope: str
) -> list[dict[str, Any]]:
    wanted_split = "test" if scope == "in_domain" else "cross_regional"
    rows: list[dict[str, Any]] = []
    for value in partitions.values():
        dataset = value() if callable(value) else value
        if dataset.get("split") == wanted_split:
            rows.extend(dict(row) for row in dataset.get("records", []))
    if not rows:
        raise ValueError(f"No evaluation rows found for scope {scope!r}")
    return rows


def _extract_section(text: str, name: str) -> str:
    pattern = rf"(?:^|\n){re.escape(name)}:\s*\n?(.+?)(?=\n[A-Z][A-Z _-]*:\s*\n?|\Z)"
    match = re.search(pattern, text.strip(), flags=re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else ""


def extract_reference(target: str) -> tuple[str, str]:
    """Extract answer/verdict and explanation from a structured target."""

    answer = _extract_section(target, "ANSWER") or _extract_section(target, "VERDICT")
    explanation = _extract_section(target, "EXPLANATION")
    return answer.splitlines()[0].strip(), explanation


def parse_mc_prediction(text: str, prompt: str) -> tuple[str, str, bool]:
    """Return parsed option, parser status, and strict-format compliance."""

    stripped = text.strip()
    strict = bool(re.fullmatch(r"[A-D]", stripped, flags=re.IGNORECASE))
    if strict:
        return stripped.upper(), "strict", True
    match = re.search(r"(?:answer|option|choice)\s*(?:is|:)?\s*\(?([A-D])\)?\b", text, re.I)
    if match:
        return match.group(1).upper(), "normalized_letter", False
    options = dict(re.findall(r"^([A-D])\.\s*(.+)$", prompt, flags=re.MULTILINE))
    normalized_output = re.sub(r"\s+", " ", text.casefold()).strip(" .")
    matches = [
        key
        for key, value in options.items()
        if re.sub(r"\s+", " ", value.casefold()).strip(" .") in normalized_output
    ]
    if len(matches) == 1:
        return matches[0], "normalized_option_text", False
    return "", "unparseable", False


def _normalize(text: str) -> str:
    return re.sub(r"[^\w]+", " ", text.casefold()).strip()


def _token_f1(prediction: str, reference: str) -> float:
    predicted = _normalize(prediction).split()
    expected = _normalize(reference).split()
    if not predicted or not expected:
        return float(predicted == expected)
    common = sum((Counter(predicted) & Counter(expected)).values())
    if not common:
        return 0.0
    precision = common / len(predicted)
    recall = common / len(expected)
    return 2 * precision * recall / (precision + recall)


def _rouge_scores(prediction: str, reference: str) -> dict[str, float]:
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        return {"rouge1": math.nan, "rouge2": math.nan, "rougeL": math.nan}
    scores = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True).score(
        reference, prediction
    )
    return {name: float(score.fmeasure) for name, score in scores.items()}


def _classification_metrics(labels: list[str], predictions: list[str]) -> dict[str, Any]:
    classes = ["correct", "incorrect"]
    confusion = {
        actual: {predicted: 0 for predicted in classes + ["unparseable"]} for actual in classes
    }
    for actual, predicted in zip(labels, predictions, strict=True):
        confusion[actual][predicted if predicted in classes else "unparseable"] += 1
    result: dict[str, Any] = {"confusion_matrix": confusion}
    f1_values = []
    recalls = []
    for label in classes:
        tp = confusion[label][label]
        fp = sum(confusion[other][label] for other in classes if other != label)
        fn = sum(value for predicted, value in confusion[label].items() if predicted != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        result[f"{label}_precision"] = precision
        result[f"{label}_recall"] = recall
        result[f"{label}_f1"] = f1
        f1_values.append(f1)
        recalls.append(recall)
    result["macro_f1"] = statistics.fmean(f1_values)
    result["balanced_accuracy"] = statistics.fmean(recalls)
    result["accuracy"] = sum(a == p for a, p in zip(labels, predictions, strict=True)) / len(labels)
    return result


def _load_model(model_spec: dict[str, Any], parameters: dict[str, Any]) -> tuple[Any, Any]:
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )

    stage = parameters["checkpoint_stage"]
    model_source = model_spec["model_id"] if stage == "base" else parameters.get("checkpoint_uri")
    if not model_source:
        raise ValueError("evaluation.checkpoint_uri is required for a fine_tuned evaluation")
    subfolder = parameters.get("checkpoint_subfolder") if stage == "fine_tuned" else None
    tokenizer_source = model_spec["model_id"] if stage == "fine_tuned" else model_source
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    load_kwargs: dict[str, Any] = {}
    if torch.cuda.is_available():
        load_kwargs["device_map"] = "auto"
        load_kwargs["torch_dtype"] = torch.bfloat16
        if model_spec["family"] == "causal_lm" and model_spec.get("load_in_4bit", False):
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
    model_class = (
        AutoModelForCausalLM if model_spec["family"] == "causal_lm" else AutoModelForSeq2SeqLM
    )
    if stage == "base":
        model = model_class.from_pretrained(model_source, **load_kwargs)
    else:
        from peft import PeftConfig, PeftModel

        try:
            peft_config = PeftConfig.from_pretrained(model_source, subfolder=subfolder)
        except (OSError, ValueError):
            model = model_class.from_pretrained(model_source, **load_kwargs)
        else:
            base = model_class.from_pretrained(peft_config.base_model_name_or_path, **load_kwargs)
            model = PeftModel.from_pretrained(base, model_source, subfolder=subfolder)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer


def _generate(
    model: Any, tokenizer: Any, prompts: list[str], parameters: dict[str, Any]
) -> tuple[list[str], list[float]]:
    import torch

    predictions: list[str] = []
    latencies: list[float] = []
    batch_size = int(parameters["batch_size"])
    tokenizer.padding_side = "left"
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        encoded = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=int(parameters["max_input_tokens"]),
        )
        device = next(model.parameters()).device
        encoded = {key: value.to(device) for key, value in encoded.items()}
        began = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=int(parameters["max_new_tokens"]),
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        elapsed = (time.perf_counter() - began) / len(batch)
        if getattr(model.config, "is_encoder_decoder", False):
            decoded_ids = generated
        else:
            decoded_ids = generated[:, encoded["input_ids"].shape[1] :]
        predictions.extend(tokenizer.batch_decode(decoded_ids, skip_special_tokens=True))
        latencies.extend([elapsed] * len(batch))
    return [item.strip() for item in predictions], latencies


def _answer_rows(
    records: list[dict[str, Any]], predictions: list[str], latencies: list[float]
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for record, prediction, latency in zip(records, predictions, latencies, strict=True):
        reference_answer, reference_explanation = extract_reference(str(record["target"]))
        task_type = str(record["task_type"])
        row = {
            **{
                key: record.get(key)
                for key in [
                    "example_id",
                    "seed_id",
                    "source_name",
                    "Level",
                    "region",
                    "evaluation_scope",
                ]
            },
            "evaluation_task": "mc_answer" if task_type == "multiple_choice" else "short_answer",
            "prompt": record["prompt"],
            "reference_answer": reference_answer,
            "reference_explanation": reference_explanation,
            "raw_prediction": prediction,
            "latency_seconds": latency,
            "empty_response": not bool(prediction.strip()),
            "response_length": len(prediction.split()),
        }
        if task_type == "multiple_choice":
            parsed, status, strict_format = parse_mc_prediction(prediction, str(record["prompt"]))
            row.update(
                parsed_answer=parsed,
                parser_status=status,
                strict_correct=bool(strict_format and parsed == reference_answer.upper()),
                normalized_correct=bool(parsed and parsed == reference_answer.upper()),
                format_compliant=strict_format,
            )
        else:
            predicted_answer = _extract_section(prediction, "ANSWER") or prediction
            row.update(
                parsed_answer=predicted_answer,
                exact_match=_normalize(predicted_answer) == _normalize(reference_answer),
                token_f1=_token_f1(predicted_answer, reference_answer),
                **_rouge_scores(predicted_answer, reference_answer),
            )
        output.append(row)
        generated_explanation = _extract_section(prediction, "EXPLANATION")
        output.append(
            {
                **{
                    key: record.get(key)
                    for key in [
                        "example_id",
                        "seed_id",
                        "source_name",
                        "Level",
                        "region",
                        "evaluation_scope",
                    ]
                },
                "evaluation_task": (
                    "mc_explanation"
                    if task_type == "multiple_choice"
                    else "short_answer_explanation"
                ),
                "prompt": record["prompt"],
                "reference_answer": reference_answer,
                "reference_explanation": reference_explanation,
                "raw_prediction": generated_explanation,
                "parsed_answer": row.get("parsed_answer", ""),
                "answer_explanation_consistent": bool(
                    row.get("normalized_correct", row.get("exact_match", False))
                    and generated_explanation
                ),
                "empty_response": not bool(generated_explanation),
                "response_length": len(generated_explanation.split()),
                "latency_seconds": latency,
                **_rouge_scores(generated_explanation, reference_explanation),
            }
        )
    return output


def _verification_rows(
    records: list[dict[str, Any]], predictions: list[str], latencies: list[float]
) -> list[dict[str, Any]]:
    output = []
    for record, prediction, latency in zip(records, predictions, latencies, strict=True):
        reference, explanation = extract_reference(str(record["target"]))
        match = re.search(r"\b(correct|incorrect)\b", prediction, re.I)
        parsed = match.group(1).casefold() if match else "unparseable"
        actual = reference.casefold()
        correction = _extract_section(prediction, "CORRECT ANSWER")
        expected_correction = _extract_section(str(record["target"]), "CORRECT ANSWER")
        output.append(
            {
                **{
                    key: record.get(key)
                    for key in [
                        "example_id",
                        "seed_id",
                        "source_name",
                        "Level",
                        "region",
                        "evaluation_scope",
                        "candidate_polarity",
                    ]
                },
                "evaluation_task": "answer_verification",
                "prompt": record["prompt"],
                "reference_answer": actual,
                "reference_explanation": explanation,
                "raw_prediction": prediction,
                "parsed_answer": parsed,
                "correct": parsed == actual,
                "correction": correction,
                "correction_exact_match": (
                    _normalize(correction) == _normalize(expected_correction)
                    if actual == "incorrect"
                    else None
                ),
                "latency_seconds": latency,
                "empty_response": not bool(prediction.strip()),
                "response_length": len(prediction.split()),
            }
        )
    return output


def _paraphrase_prompt(record: dict[str, Any]) -> str:
    return (
        "Paraphrase the completed instruction below while preserving every fact, answer, "
        "option-answer relationship, and explanation. Preserve the field labels.\n\n"
        f"{record['prompt']}\n\n{record['target']}"
    )


def _paraphrase_rows(
    records: list[dict[str, Any]], predictions: list[str], latencies: list[float]
) -> list[dict[str, Any]]:
    output = []
    for record, prediction, latency in zip(records, predictions, latencies, strict=True):
        reference = f"{record['prompt']}\n\n{record['target']}"
        answer, explanation = extract_reference(str(record["target"]))
        parsed_answer = _extract_section(prediction, "ANSWER")
        output.append(
            {
                **{
                    key: record.get(key)
                    for key in [
                        "example_id",
                        "seed_id",
                        "source_name",
                        "Level",
                        "region",
                        "evaluation_scope",
                    ]
                },
                "evaluation_task": "paraphrase",
                "prompt": _paraphrase_prompt(record),
                "reference_answer": answer,
                "reference_explanation": explanation,
                "raw_prediction": prediction,
                "parsed_answer": parsed_answer,
                "answer_preserved": _normalize(parsed_answer) == _normalize(answer),
                "field_preservation": all(
                    label in prediction.upper() for label in ["ANSWER:", "EXPLANATION:"]
                ),
                "latency_seconds": latency,
                **_rouge_scores(prediction, reference),
            }
        )
    return output


def _attach_bertscore(rows: list[dict[str, Any]], parameters: dict[str, Any]) -> None:
    if not parameters.get("compute_bertscore", True) or not rows:
        return
    from bert_score import score

    candidates = [str(row["raw_prediction"]) for row in rows]
    references = [
        str(row.get("reference_answer", "")) + " " + str(row.get("reference_explanation", ""))
        for row in rows
    ]
    precision, recall, f1 = score(
        candidates,
        references,
        model_type=parameters.get("bertscore_model_type"),
        lang=parameters.get("bertscore_language", "en"),
        device=parameters.get("metric_device"),
        verbose=False,
    )
    for row, p_value, r_value, f_value in zip(rows, precision, recall, f1, strict=True):
        row["bertscore_precision"] = float(p_value)
        row["bertscore_recall"] = float(r_value)
        row["bertscore_f1"] = float(f_value)


def _attach_nli_scores(rows: list[dict[str, Any]], parameters: dict[str, Any]) -> None:
    """Attach the notebook's BERT-family entailment classification score."""

    if not parameters.get("compute_nli_accuracy", True) or not rows:
        return
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_id = parameters.get("nli_model_id", "roberta-large-mnli")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    classifier = AutoModelForSequenceClassification.from_pretrained(model_id)
    device = parameters.get("metric_device") or ("cuda" if torch.cuda.is_available() else "cpu")
    classifier.to(device)
    classifier.eval()
    entailment_id = classifier.config.label2id.get("ENTAILMENT", 2)
    batch_size = int(parameters.get("metric_batch_size", parameters["batch_size"]))
    threshold = float(parameters.get("nli_entailment_threshold", 0.5))
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        references = [
            f"{row.get('reference_answer', '')} {row.get('reference_explanation', '')}".strip()
            for row in batch
        ]
        candidates = [str(row.get("raw_prediction", "")) for row in batch]
        encoded = tokenizer(
            references,
            candidates,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)
        with torch.inference_mode():
            probabilities = classifier(**encoded).logits.softmax(dim=-1)[:, entailment_id]
        for row, probability in zip(batch, probabilities, strict=True):
            value = float(probability)
            row["nli_entailment_score"] = value
            row["nli_entailment_correct"] = value >= threshold


def _aggregate(rows: list[dict[str, Any]], keys: tuple[str, ...] = ()) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(key) for key in keys)].append(row)
    output = []
    for values, members in groups.items():
        result = {key: value for key, value in zip(keys, values, strict=True)}
        result["count"] = len(members)
        numeric_keys = sorted(
            {
                key
                for member in members
                for key, value in member.items()
                if isinstance(value, (bool, int, float))
                and not isinstance(value, str)
                and not (isinstance(value, float) and math.isnan(value))
            }
        )
        for key in numeric_keys:
            values_to_average = [
                float(member[key])
                for member in members
                if key in member
                and member[key] is not None
                and not (isinstance(member[key], float) and math.isnan(member[key]))
            ]
            if values_to_average:
                result[key] = statistics.fmean(values_to_average)
        output.append(result)
    return output


def _review_reason(row: dict[str, Any]) -> list[str]:
    reasons = []
    if row.get("empty_response"):
        reasons.append("empty_response")
    if row.get("parser_status") == "unparseable":
        reasons.append("unparseable")
    if row.get("strict_correct") is False and row.get("normalized_correct") is True:
        reasons.append("correct_meaning_invalid_format")
    if row.get("normalized_correct") is False or row.get("correct") is False:
        reasons.append("incorrect_answer")
    if row.get("response_length", 0) > 200:
        reasons.append("overgeneration")
    return reasons


def evaluate_and_publish_model(
    generation_partitions: Mapping[str, PartitionValue],
    verification_partitions: Mapping[str, PartitionValue],
    split_manifest: dict[str, Any],
    model_registry: dict[str, Any],
    parameters: dict[str, Any],
) -> dict[str, Any]:
    """Run the complete suite, persist every prediction, then upload the run."""

    import pandas as pd

    model_key = parameters["model_key"]
    if model_key not in model_registry:
        raise KeyError(f"Unknown model_key {model_key!r}")
    scope = parameters["dataset_scope"]
    generation = _materialize_records(generation_partitions, scope)
    verification = _materialize_records(verification_partitions, scope)
    model, tokenizer = _load_model(model_registry[model_key], parameters)

    answer_predictions, answer_latency = _generate(
        model, tokenizer, [str(row["prompt"]) for row in generation], parameters
    )
    rows = _answer_rows(generation, answer_predictions, answer_latency)

    verification_predictions, verification_latency = _generate(
        model, tokenizer, [str(row["prompt"]) for row in verification], parameters
    )
    verification_rows = _verification_rows(
        verification, verification_predictions, verification_latency
    )
    rows.extend(verification_rows)

    if parameters.get("evaluate_paraphrase", True):
        paraphrase_predictions, paraphrase_latency = _generate(
            model, tokenizer, [_paraphrase_prompt(row) for row in generation], parameters
        )
        rows.extend(_paraphrase_rows(generation, paraphrase_predictions, paraphrase_latency))

    _attach_bertscore(rows, parameters)
    _attach_nli_scores(rows, parameters)
    run_id = parameters.get("run_id") or datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    stage = parameters["checkpoint_stage"]
    local_dir = (
        Path(parameters["artifact_root"]) / "evaluations" / model_key / stage / scope / run_id
    )
    local_dir.mkdir(parents=True, exist_ok=True)

    for row in rows:
        row["model_key"] = model_key
        row["checkpoint_stage"] = stage
        row["evaluation_run_id"] = run_id
        row["review_reasons"] = _review_reason(row)
    review_rows = [row for row in rows if row["review_reasons"]]
    grouped = _aggregate(rows, ("evaluation_task", "source_name", "Level", "region"))
    overall = _aggregate(rows, ("evaluation_task",))
    labels = [str(row["reference_answer"]) for row in verification_rows]
    predictions = [str(row["parsed_answer"]) for row in verification_rows]
    verification_metrics = _classification_metrics(labels, predictions)
    metrics = {"overall": overall, "verification": verification_metrics}
    manifest = {
        "evaluation_run_id": run_id,
        "model_key": model_key,
        "base_model_id": model_registry[model_key]["model_id"],
        "checkpoint_stage": stage,
        "checkpoint_uri": parameters.get("checkpoint_uri"),
        "checkpoint_subfolder": parameters.get("checkpoint_subfolder"),
        "dataset_scope": scope,
        "dataset_version": split_manifest["dataset_version"],
        "generation_parameters": {
            "do_sample": False,
            "max_input_tokens": parameters["max_input_tokens"],
            "max_new_tokens": parameters["max_new_tokens"],
        },
        "prediction_count": len(rows),
    }
    pd.DataFrame(rows).to_parquet(local_dir / "predictions.parquet", index=False)
    pd.DataFrame(grouped).to_parquet(local_dir / "grouped_metrics.parquet", index=False)
    pd.DataFrame(review_rows).to_parquet(local_dir / "review_queue.parquet", index=False)
    (local_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (local_dir / "confusion_matrix.json").write_text(
        json.dumps(verification_metrics["confusion_matrix"], indent=2), encoding="utf-8"
    )
    (local_dir / "evaluation_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    (local_dir / "README.md").write_text(
        f"# Evaluation {run_id}\n\nModel: `{model_key}`  \nStage: `{stage}`  \nScope: `{scope}`\n",
        encoding="utf-8",
    )

    published = False
    commit = None
    hf_path = (
        f"evaluations/{split_manifest['dataset_version']}/{model_key}/{stage}/{scope}/{run_id}"
    )
    if parameters.get("publish_to_hf", True):
        from huggingface_hub import HfApi

        token = os.getenv("HF_TOKEN")
        if not token:
            raise RuntimeError("HF_TOKEN is required when evaluation.publish_to_hf is true")
        api = HfApi(token=token)
        api.create_repo(
            repo_id=parameters["prediction_repo_id"], repo_type="dataset", exist_ok=True
        )
        commit = str(
            api.upload_folder(
                folder_path=str(local_dir),
                repo_id=parameters["prediction_repo_id"],
                repo_type="dataset",
                path_in_repo=hf_path,
                commit_message=f"Publish evaluation {model_key} {stage} {scope} {run_id}",
            )
        )
        published = True
    return {
        **manifest,
        "local_dir": str(local_dir.resolve()),
        "prediction_repo_id": parameters["prediction_repo_id"],
        "hf_path": hf_path,
        "published": published,
        "commit": commit,
    }
