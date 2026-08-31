from urban_science_revision.pipelines.data_augmentation.prompts import (
    generation_prompt,
    generation_target,
    negative_verification_target,
    positive_verification_target,
    verification_prompt,
)


def test_prompt_target_contract() -> None:
    record = {
        "task_type": "short_answer",
        "Fact": "Hourly observations are provided.",
        "Question": "What is the temporal resolution?",
        "correct_answer": "Hourly",
        "correct_answer_text": "Hourly",
        "incorrect_candidate": "Monthly",
        "Explanation": "The fact explicitly says hourly.",
    }
    prompt = generation_prompt(record)
    target = generation_target(record)
    negative_prompt = verification_prompt(record, record["incorrect_candidate"])
    negative_target = negative_verification_target(record)
    positive_target = positive_verification_target(record)

    assert "Explanation" not in prompt
    assert "ANSWER:" not in prompt
    assert "Hourly" in target
    assert "Monthly" in negative_prompt
    assert "CANDIDATE MATCHES CONTEXT: Yes or No" in negative_prompt
    assert "VERDICT:\nIncorrect" in negative_target
    assert "CANDIDATE MATCHES CONTEXT:\nNo" in negative_target
    assert "VERDICT:\nCorrect" in positive_target
    assert "CANDIDATE MATCHES CONTEXT:\nYes" in positive_target
    assert "EXPLANATION:" not in negative_target
    assert "CORRECT ANSWER:" not in negative_target
