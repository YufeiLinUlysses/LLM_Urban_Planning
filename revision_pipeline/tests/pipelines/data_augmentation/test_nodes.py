from __future__ import annotations

from typing import Any

from urban_science_revision.pipelines.data_augmentation.nodes import (
    _record_errors,
    assemble_canonical_datasets,
    generate_augmentation_components,
    materialize_task_views,
    normalize_seed_datasets,
    validate_and_audit_datasets,
)


class FakeAugmenter:
    def fact_explanations(self, system: str, user: str, count: int) -> list[dict[str, str]]:
        return [
            {
                "Fact": f"Supported answer appears in fact variant {index}.",
                "Explanation": f"Explanation variant {index}.",
            }
            for index in range(count)
        ]

    def questions(self, system: str, user: str, count: int, *, temperature: float) -> list[str]:
        return [f"Question variant {index}?" for index in range(count)]

    def answers(self, system: str, user: str, count: int) -> list[str]:
        return [f"Correct answer variant {index}" for index in range(count)]

    def explanations(self, system: str, user: str, count: int) -> list[str]:
        return [f"MCQ explanation variant {index}." for index in range(count)]

    def negative_answers(self, system: str, user: str, count: int = 10) -> list[str]:
        return [f"Plausible wrong answer {index}" for index in range(count)]


def _parameters() -> dict[str, Any]:
    return {
        "random_seed": 2025,
        "levels": {
            1: {
                "short_answer": {
                    "fact_explanation_paraphrases": 5,
                    "question_paraphrases": 10,
                },
                "multiple_choice": {
                    "question_paraphrases": 5,
                    "correct_answer_paraphrases": 2,
                    "explanation_paraphrases": 2,
                    "choice_permutations": 5,
                },
            },
            2: {
                "short_answer": {
                    "fact_explanation_paraphrases": 4,
                    "question_paraphrases": 10,
                },
                "multiple_choice": {
                    "question_paraphrases": 5,
                    "correct_answer_paraphrases": 2,
                    "explanation_paraphrases": 2,
                    "choice_permutations": 3,
                },
            },
            3: {
                "short_answer": {
                    "fact_explanation_paraphrases": 4,
                    "question_paraphrases": 10,
                }
            },
        },
    }


def _source_dataset() -> dict[str, Any]:
    return {
        "instructions": [
            {
                "Instruction information": {"Categories": "Multiple Selection"},
                "Task_Definition": "Choose A, B, C, or D.",
                "Positive Example": [
                    {
                        "Fact": "The correct mode is rail.",
                        "Question": "Which mode is correct?",
                        "Selections": {
                            "A": "Bus",
                            "B": "Rail",
                            "C": "Car",
                            "D": "Walk",
                        },
                        "Output": "B",
                        "Explanation": "Rail is stated in the fact.",
                        "Level": 2,
                    }
                ],
                "Negative Example": [],
            },
            {
                "Instruction information": {"Categories": "Question answering"},
                "Task_Definition": "Answer from the fact.",
                "Positive Example": [
                    {
                        "Fact": "The supported answer appears in this fact.",
                        "Question": "What appears in the fact?",
                        "Output": "Supported answer",
                        "Explanation": "The fact states the supported answer.",
                        "Level": 1,
                    }
                ],
                "Negative Example": [],
            },
        ]
    }


def _build_canonical() -> dict[str, Any]:
    normalized, report = normalize_seed_datasets({"sample": _source_dataset()}, _parameters())
    assert report["valid"] is True
    components = generate_augmentation_components(
        normalized, _parameters(), augmenter=FakeAugmenter()
    )
    canonical, _ = assemble_canonical_datasets(components, _parameters())
    return canonical


def test_level_specific_expansion_and_lineage() -> None:
    canonical = _build_canonical()["sample"]["records"]
    short_answers = [record for record in canonical if record["task_type"] == "short_answer"]
    multiple_choice = [record for record in canonical if record["task_type"] == "multiple_choice"]

    assert len(short_answers) == 51  # original + 5 facts * 10 questions
    assert len(multiple_choice) == 61  # original + 5 questions * 2 exp * 3 swaps * 2 answers
    assert all(record["seed_id"] in record["variant_id"] for record in canonical)


def test_mcq_outputs_follow_each_permutation_without_shared_mutation() -> None:
    canonical = _build_canonical()["sample"]["records"]
    augmented_mcq = [
        record
        for record in canonical
        if record["task_type"] == "multiple_choice" and record["origin"] == "augmented"
    ]

    for record in augmented_mcq:
        key = record["correct_answer"]
        assert record["Selections"][key] == record["correct_answer_text"]
        assert record["incorrect_candidate"] != record["correct_answer_text"]

    first = augmented_mcq[0]
    second = augmented_mcq[1]
    first["Selections"][first["correct_answer"]] = "mutated only here"
    assert second["Selections"][second["correct_answer"]] != "mutated only here"


def test_task_views_keep_targets_out_of_prompts_and_wrong_answers_out_of_sft() -> None:
    validated, quality, _, rejected = validate_and_audit_datasets(_build_canonical())
    assert quality["passed"] is True
    assert not [item for item in rejected if "exact_duplicate" not in item["reasons"]]

    final, generation, verification, paraphrase, statistics = materialize_task_views(validated)
    assert statistics["generation_record_count"] == len(final["sample"]["records"])
    assert statistics["verification_record_count"] == 2 * len(final["sample"]["records"])
    assert statistics["paraphrase_record_count"] == len(paraphrase["sample"]["records"])
    assert paraphrase["sample"]["records"]

    for record in generation["sample"]["records"]:
        assert "\nANSWER:" not in record["prompt"].upper()
        assert "\nEXPLANATION:" not in record["prompt"].upper()
        assert record["target"].startswith("ANSWER:")

    negative_views = [
        record
        for record in verification["sample"]["records"]
        if record["candidate_polarity"] == "negative"
    ]
    assert negative_views
    assert all(record["target"].startswith("VERDICT:\nIncorrect") for record in negative_views)
    assert all(
        "CANDIDATE MATCHES CONTEXT:\nNo" in record["target"]
        for record in negative_views
    )
    assert all("EXPLANATION:" not in record["target"] for record in negative_views)

    for record in paraphrase["sample"]["records"]:
        assert record["prompt"].startswith("Paraphrase the completed instruction")
        assert "DATASET CONTEXT:" in record["target"].upper()
        assert "ANSWER:" in record["target"].upper()
        assert "EXPLANATION:" in record["target"].upper()


def test_seed_ids_are_stable() -> None:
    first, _ = normalize_seed_datasets({"sample": _source_dataset()}, _parameters())
    second, _ = normalize_seed_datasets({"sample": _source_dataset()}, _parameters())
    assert [seed["seed_id"] for seed in first["sample"]["seeds"]] == [
        seed["seed_id"] for seed in second["sample"]["seeds"]
    ]


def test_same_fact_across_task_formats_shares_concept_group() -> None:
    source = _source_dataset()
    source["instructions"][1]["Positive Example"][0]["Fact"] = "The correct mode is rail."
    normalized, _ = normalize_seed_datasets({"sample": source}, _parameters())

    seeds = normalized["sample"]["seeds"]
    assert seeds[0]["seed_id"] != seeds[1]["seed_id"]
    assert seeds[0]["concept_group_id"] == seeds[1]["concept_group_id"]


def test_distinct_numeric_negative_candidates_are_not_rejected_as_paraphrases() -> None:
    base_record = {
        "seed_id": "sample__short_answer__l1__0001",
        "variant_id": "sample__short_answer__l1__0001__original",
        "Fact": "The collection began in 2009.",
        "Question": "When did collection begin?",
        "correct_answer": "Since 2009.",
        "correct_answer_text": "Since 2009.",
        "incorrect_candidate": "Since 2010.",
        "Explanation": "The fact states 2009.",
        "task_type": "short_answer",
    }
    errors, _ = _record_errors(base_record)
    assert "negative_candidate_too_similar_to_correct_answer" not in errors

    frequency_record = {
        **base_record,
        "correct_answer": "10 Hz.",
        "correct_answer_text": "10 Hz.",
        "incorrect_candidate": "100 Hz.",
    }
    errors, _ = _record_errors(frequency_record)
    assert "negative_candidate_too_similar_to_correct_answer" not in errors
