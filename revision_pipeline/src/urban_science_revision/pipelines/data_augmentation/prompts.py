"""Prompt builders for augmentation and leakage-safe model-input views."""

import json
from typing import Any


def fact_explanation_augmentation(seed: dict[str, Any], count: int) -> tuple[str, str]:
    system = (
        "You create high-quality instruction-data variants. Paraphrase the fact and "
        "explanation without changing meaning. Every fact must still contain or directly "
        "support the given answer, remain relevant to the question, and introduce no new "
        f"claims. Return exactly {count} distinct pairs using the supplied schema."
    )
    user = (
        f"Question:\n{seed['Question']}\n\n"
        f"Fact:\n{seed['Fact']}\n\n"
        f"Correct answer:\n{seed['Output']}\n\n"
        f"Explanation:\n{seed['Explanation']}"
    )
    return system, user


def short_question_augmentation(seed: dict[str, Any], count: int) -> tuple[str, str]:
    system = (
        "Paraphrase only the question. Preserve its meaning, reasoning level, and answer. "
        "Each version must remain answerable from the supporting fact and test the same "
        f"concept. Return exactly {count} distinct questions using the supplied schema."
    )
    user = (
        f"Original question:\n{seed['Question']}\n\n"
        f"Supporting fact:\n{seed['Fact']}\n\n"
        f"Correct answer:\n{seed['Output']}\n\n"
        f"Explanation:\n{seed['Explanation']}"
    )
    return system, user


def negative_answer_augmentation(seed: dict[str, Any], count: int = 10) -> tuple[str, str]:
    system = (
        "Generate plausible but clearly incorrect candidate answers for an answer-verification "
        "task. Candidates must not equal, entail, or paraphrase the correct answer. They must be "
        "topically credible rather than random or nonsensical. "
        f"Return exactly {count} distinct candidates using the supplied schema."
    )
    user = (
        f"Fact:\n{seed['Fact']}\n\n"
        f"Question:\n{seed['Question']}\n\n"
        f"Correct answer:\n{seed['Output']}\n\n"
        f"Explanation:\n{seed['Explanation']}"
    )
    return system, user


def mcq_question_augmentation(seed: dict[str, Any], count: int) -> tuple[str, str]:
    system = (
        "Paraphrase only the multiple-choice question stem. Preserve meaning, difficulty, "
        "available options, and the correct answer. Do not mention the answer in meta-language. "
        f"Return exactly {count} distinct questions using the supplied schema."
    )
    user = (
        f"Original question:\n{seed['Question']}\n\n"
        f"Options:\n{json.dumps(seed['Selections'], ensure_ascii=False)}\n\n"
        f"Correct option:\n{seed['Output']}\n\n"
        f"Fact:\n{seed['Fact']}\n\n"
        f"Explanation:\n{seed['Explanation']}"
    )
    return system, user


def mcq_answer_augmentation(seed: dict[str, Any], count: int) -> tuple[str, str]:
    correct_text = seed["Selections"][seed["Output"]]
    system = (
        "Paraphrase only the correct option text. Preserve exact semantic and numerical meaning, "
        "introduce no new information, and do not add an option label. "
        f"Return exactly {count} distinct answers using the supplied schema."
    )
    user = (
        f"Question for context:\n{seed['Question']}\n\n"
        f"Correct answer text:\n{correct_text}\n\n"
        f"Explanation for meaning reference:\n{seed['Explanation']}"
    )
    return system, user


def mcq_explanation_augmentation(seed: dict[str, Any], count: int) -> tuple[str, str]:
    correct_text = seed["Selections"][seed["Output"]]
    system = (
        "Paraphrase only the explanation. Preserve exact meaning, keep it consistent with the "
        "question and correct answer, and introduce no new facts or numerical changes. "
        f"Return exactly {count} distinct explanations using the supplied schema."
    )
    user = (
        f"Question:\n{seed['Question']}\n\n"
        f"Options:\n{json.dumps(seed['Selections'], ensure_ascii=False)}\n\n"
        f"Correct answer text:\n{correct_text}\n\n"
        f"Explanation:\n{seed['Explanation']}"
    )
    return system, user


def generation_prompt(record: dict[str, Any]) -> str:
    parts = [f"Dataset context:\n{record['Fact']}", f"Question:\n{record['Question']}"]
    if record["task_type"] == "multiple_choice":
        options = "\n".join(f"{key}. {value}" for key, value in record["Selections"].items())
        parts.append(f"Options:\n{options}")
        parts.append("Please provide the correct option and a concise explanation.")
    else:
        parts.append("Please provide a concise answer grounded in the dataset information.")
    return "\n\n".join(parts)


def generation_target(record: dict[str, Any]) -> str:
    return f"ANSWER:\n{record['correct_answer']}\n\nEXPLANATION:\n{record['Explanation']}"


def verification_prompt(record: dict[str, Any], candidate: str) -> str:
    parts = [
        f"Dataset context:\n{record['Fact']}",
        f"Question:\n{record['Question']}",
    ]
    if record["task_type"] == "multiple_choice":
        options = "\n".join(f"{key}. {value}" for key, value in record["Selections"].items())
        parts.append(f"Options:\n{options}")
    parts.extend(
        [
            f"Candidate answer:\n{candidate}",
            (
                "Compare the candidate answer directly with the dataset context.\n\n"
                "Return exactly:\n"
                "VERDICT: Correct or Incorrect\n"
                "CANDIDATE MATCHES CONTEXT: Yes or No"
            ),
        ]
    )
    return "\n\n".join(parts)


def positive_verification_target(record: dict[str, Any]) -> str:
    del record
    return "VERDICT:\nCorrect\n\nCANDIDATE MATCHES CONTEXT:\nYes"


def negative_verification_target(record: dict[str, Any]) -> str:
    del record
    return "VERDICT:\nIncorrect\n\nCANDIDATE MATCHES CONTEXT:\nNo"
