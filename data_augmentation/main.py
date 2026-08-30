# Level 1 both positive and negative
# 500 short-answer
# 1000 multiple-choice
# Level 2 both positive and negative
# 200 short-answer
# 300 multiple-choice
# Level 3 both positive and negative
# 200 short-answer
# 300 multiple-choice

# Short answer
# 1. Paraphrase facts and explanation to align each other 5 time
# 2. Paraphrase questions 10 times
# 3. Negative examples: change answers incorrect based on facts or simply use answers I don't know or from other datasets
# Multiple choice
# 1. Paraphrase questions 5 times
# 2. Paraphrase answer choices 2 times
# 3. Paraphrase explanations 2 times
# 4. Swap answer choices 5 times
# 5. Negative examples: change answers to the wrong ones, or simply no answer at all

from typing import List
import os
import json
import random
from typing import List, Tuple, Dict

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field, RootModel
import glob

rseed = random.Random(2025)

load_dotenv()  # reads .env in the current working directory
api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI()


class ParaphraseItem(BaseModel):
    Fact: str = Field(
        ...,
        description="A paraphrased fact that preserves meaning and embeds the answer",
    )
    Explanation: str = Field(
        ...,
        description="Short rationale consistent with the fact, question, and answer",
    )


class ParaphraseList(BaseModel):
    items: List[ParaphraseItem]


class QuestionParaphraseList(BaseModel):
    questions: List[str]


class MCQParaphraseList(BaseModel):
    questions: List[str] = Field(
        ...,
        description="Paraphrased multiple-choice question stems"
    )


class AnswerParaphraseList(BaseModel):
    answers: List[str] = Field(
        ...,
        description="Paraphrased versions of the correct answer text"
    )


class ExplanationParaphraseList(BaseModel):
    explanations: List[str] = Field(
        ...,
        description="Paraphrased versions of the explanation text"
    )


class NegativeAnswerList(BaseModel):
    answers: List[str] = Field(
        ...,
        description="A list of plausible but incorrect answers."
    )


def paraphrase_facts_and_explanations(
    instruction: dict,
    temperature: float = 0.2,
    num_paraphrases: int = 5,
):
    """
    Paraphrase a single instruction's fact and explanation into:
    [{\"Fact\":..., \"Explanation\":...}, ...]

    Args:
        instruction (dict): Keys include Fact, Question, Output (answer), Explanation.
        temperature (float): LLM temperature.
        num_paraphrases (int): How many paraphrase pairs to return.
    """
    fact = instruction.get("Fact", "")
    question = instruction.get("Question", "")
    answer = instruction.get("Output", "")
    explanation = instruction.get("Explanation", "")

    system_prompt = (
        "Paraphrase the fact and explanation without changing meaning. "
        "Each paraphrased fact must still support the given answer and stay relevant to the question. "
        f"Create exactly {num_paraphrases} paraphrases. "
        "Return only JSON matching the provided schema."
    )

    user_prompt = (
        f"Question:\n{question}\n\n"
        f"Fact:\n{fact}\n\n"
        f"Answer:\n{answer}\n\n"
        f"Explanation:\n{explanation}\n"
    )

    completion = client.responses.parse(
        model="gpt-5.1",
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        text_format=ParaphraseList,
    )

    paraphrase_list: ParaphraseList = completion.output_parsed
    return [item.model_dump() for item in paraphrase_list.items]


def paraphrase_questions(
    instruction: dict,
    temperature: float = 0.2,
    num_paraphrases: int = 10,
) -> List[str]:
    """
    Paraphrase a single instruction's question into multiple variants:

    Returns:
        List[str]: A list of paraphrased questions (length == num_paraphrases).
    """
    fact = instruction.get("Fact", "")
    question = instruction.get("Question", "")
    answer = instruction.get("Output", "")
    explanation = instruction.get("Explanation", "")

    system_prompt = (
        "You are an expert data annotation assistant. "
        "Paraphrase the question into multiple variants that:\n"
        "- Preserve the original meaning and difficulty level.\n"
        "- Remain consistent with the given fact, answer, and explanation.\n"
        "- Stay focused on the same underlying concept being tested.\n"
        f"Create exactly {num_paraphrases} paraphrased questions.\n"
        "Return only JSON matching the provided schema."
    )

    user_prompt = (
        f"Original question:\n{question}\n\n"
        f"Supporting fact:\n{fact}\n\n"
        f"Correct answer:\n{answer}\n\n"
        f"Explanation of the answer:\n{explanation}\n"
    )

    completion = client.responses.parse(
        model="gpt-5.1",
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        text_format=QuestionParaphraseList,
    )

    q_list: QuestionParaphraseList = completion.output_parsed
    return q_list.questions


def generate_negative_examples(instruction: dict,
                               num_negatives: int = 10,
                               temperature: float = 0.7) -> List[str]:
    """
    Generate a list of wrong answers (negative examples).

    Args:
        instruction (dict): A dict with keys Fact, Question, Output, Explanation.
        num_negatives (int): How many wrong answers to generate.
        temperature (float): LLM temperature.

    Returns:
        List[str]: A list of `num_negatives` wrong answers as plain strings.
    """
    fact = instruction["Fact"]
    question = instruction["Question"]
    correct_answer = instruction["Output"]
    explanation = instruction.get("Explanation", "")

    system_prompt = (
        "You are an expert data augmentation assistant. "
        "Given a question, fact, correct answer, and explanation, generate WRONG answers. "
        "Each wrong answer must be plausible but clearly incorrect, and must not match "
        "or be nearly identical to the correct answer and may not necessarily related to the fact or question, be creative. "
        f"Return exactly {num_negatives} wrong answers as JSON with the schema:\n"
        '{ "answers": ["...", "...", ...] }.'
    )

    user_prompt = (
        f"Fact:\n{fact}\n\n"
        f"Question:\n{question}\n\n"
        f"Correct Answer:\n{correct_answer}\n\n"
        f"Explanation:\n{explanation}\n"
    )

    completion = client.responses.parse(
        model="gpt-5.1",
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        text_format=NegativeAnswerList,  # Pydantic model, not List[str]
    )

    negative_list: NegativeAnswerList = completion.output_parsed
    return negative_list.answers


def short_answer_augmentation_one_instruction(instruction: dict, num_p: int = 5) -> Tuple[List[dict], List[str], List[str]]:
    """
    Perform data augmentation for a single short-answer instruction.

    Returns:
        paraphrased_facts_and_explanations: List of dicts with keys Fact and Explanation.
        paraphrased_questions: List of paraphrased questions (str).
        negative_answers: List of wrong answers (str).
    """
    paraphrased_facts_and_explanations = paraphrase_facts_and_explanations(
        instruction, num_paraphrases=num_p)
    paraphrased_questions = paraphrase_questions(instruction)
    negative_answers = generate_negative_examples(instruction)

    return paraphrased_facts_and_explanations, paraphrased_questions, negative_answers


def short_answer_augmentation(instructions: List[dict], source_name: str) -> dict:
    """
    Perform augmentation for a list of instructions.

    Returns:
        {
            "facts_and_explanations": List[List[dict]],
            "questions": List[List[str]],
            "negative_answers": List[List[str]]
        }
    """

    all_facts_exps: List[List[dict]] = []
    all_questions: List[List[str]] = []
    all_negative_answers: List[List[str]] = []

    for i in range(len(instructions)):
        instr = instructions[i]
        if i >= 15 and (source_name == "tlc_trip_instructions.json" or source_name == "urban_flow_prediction_survey_instructions.json"):
            facts_exps, questions, negatives = short_answer_augmentation_one_instruction(
                instr, num_p=10)
        else:
            facts_exps, questions, negatives = short_answer_augmentation_one_instruction(
                instr)
        all_facts_exps.append(facts_exps)
        all_questions.append(questions)
        all_negative_answers.append(negatives)

    return {
        "facts_and_explanations": all_facts_exps,
        "questions": all_questions,
        "negative_answers": all_negative_answers,
    }


def paraphrase_mcq_questions(
    instruction: dict,
    temperature: float = 0.7,
    num_paraphrases: int = 5,
) -> List[str]:
    """
    Paraphrase a multiple-choice question stem while preserving:
    - Meaning
    - Correct answer
    - Choice structure (A/B/C/D)

    Returns:
        List[str]: Exactly 5 paraphrased question stems.
    """

    question = instruction.get("Question", "")
    selections = instruction.get("Selections", {})
    answer = instruction.get("Output", "")
    explanation = instruction.get("Explanation", "")
    fact = instruction.get("Fact", "")

    system_prompt = (
        "You are an expert dataset annotation assistant. "
        "Paraphrase ONLY the multiple-choice question stem. "
        "Do NOT change the answer, choices, or difficulty. "
        "The paraphrased questions must still be answerable "
        "using the same options and correct answer.\n\n"
        f"Create exactly {num_paraphrases} paraphrased versions.\n"
        "Return only JSON matching the given schema."
    )

    user_prompt = (
        f"Original Question:\n{question}\n\n"
        f"Choices:\n{selections}\n\n"
        f"Correct Answer:\n{answer}\n\n"
        f"Supporting Fact:\n{fact}\n\n"
        f"Explanation:\n{explanation}\n"
    )

    completion = client.responses.parse(
        model="gpt-5.1",
        temperature=temperature,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        text_format=MCQParaphraseList,
    )

    parsed: MCQParaphraseList = completion.output_parsed
    return parsed.questions


def paraphrase_mcq_correct_answer(
    instruction: dict,
    temperature: float = 0.2,
    num_paraphrases: int = 2,
) -> List[str]:
    """
    1. Use Output to get the correct option text from Selections
    2. Paraphrase ONLY that text (exactly 2 times)

    Returns:
        List[str]: Two paraphrases of the correct option text.
    """
    selections = instruction.get("Selections", {})
    output_key = instruction.get("Output", "")
    explanation = instruction.get("Explanation", "")
    question = instruction.get("Question", "")

    # --- Step 1: deterministically get correct option text ---
    if output_key not in selections:
        raise ValueError("Output key not found in Selections")

    correct_text = selections[output_key]

    # --- Prompts ---
    system_prompt = (
        "You are an expert dataset annotation assistant. "
        "Paraphrase ONLY the provided answer text. "
        "Preserve the exact meaning and numerical interpretation. "
        "Do NOT introduce new information. "
        f"Create exactly {num_paraphrases} paraphrases. "
        "Return only JSON matching the given schema."
    )

    user_prompt = (
        f"Question (for context only):\n{question}\n\n"
        f"Correct answer text to paraphrase:\n{correct_text}\n\n"
        f"Explanation (for meaning reference):\n{explanation}\n"
    )

    completion = client.responses.parse(
        model="gpt-5.1",
        temperature=temperature,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        text_format=AnswerParaphraseList,
    )

    parsed: AnswerParaphraseList = completion.output_parsed

    # Safety check
    if len(parsed.answers) != num_paraphrases:
        raise ValueError(
            f"Expected {num_paraphrases} paraphrases, got {len(parsed.answers)}"
        )

    return parsed.answers


def paraphrase_mcq_explanations(
    instruction: dict,
    temperature: float = 0.6,
    num_paraphrases: int = 2,
) -> List[str]:
    """
    Paraphrase the Explanation field for a multiple-choice instruction exactly 2 times.

    Returns:
        List[str]: Two paraphrased explanations.
    """
    question = instruction.get("Question", "")
    selections = instruction.get("Selections", {}) or {}
    output_key = instruction.get("Output", "")
    explanation = instruction.get("Explanation", "")

    # Get correct option text (for context, to avoid changing meaning)
    correct_text = selections.get(output_key, output_key)

    system_prompt = (
        "You are an expert dataset annotation assistant. "
        "Paraphrase ONLY the explanation while preserving the exact meaning. "
        "Keep it consistent with the question and the correct answer choice. "
        "Do not introduce new facts or change numerical meaning. "
        f"Create exactly {num_paraphrases} paraphrases. "
        "Return only JSON matching the provided schema."
    )

    user_prompt = (
        f"Question (context):\n{question}\n\n"
        f"Choices (context):\n{selections}\n\n"
        f"Correct choice key:\n{output_key}\n\n"
        f"Correct choice text:\n{correct_text}\n\n"
        f"Explanation to paraphrase:\n{explanation}\n"
    )

    completion = client.responses.parse(
        model="gpt-5.1",
        temperature=temperature,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        text_format=ExplanationParaphraseList,
    )

    parsed: ExplanationParaphraseList = completion.output_parsed

    # Safety check
    if len(parsed.explanations) != num_paraphrases:
        raise ValueError(
            f"Expected {num_paraphrases} paraphrases, got {len(parsed.explanations)}")

    return parsed.explanations


def swap_mcq_choices(
    instruction: dict,
    num_swaps: int = 5,
) -> List[dict]:
    """
    Generate multiple MCQ variants by swapping answer choices (A/B/C/D).

    Args:
        instruction (dict): Must contain Question, Selections, Output, Explanation
        num_swaps (int): Number of swapped variants to generate
        seed (int | None): Optional random seed for reproducibility

    Returns:
        List[dict]: List of MCQ instructions with shuffled choices
    """

    selections: Dict[str, str] = instruction["Selections"]
    correct_key = instruction["Output"]

    if correct_key not in selections:
        raise ValueError("Output key not found in Selections")

    # Original correct text
    correct_text = selections[correct_key]

    labels = ["A", "B", "C", "D"]
    values = list(selections.values())

    swapped_instructions = []

    for _ in range(num_swaps):
        shuffled_values = values.copy()
        rseed.shuffle(shuffled_values)

        # Rebuild selections with same labels A/B/C/D
        new_selections = dict(zip(labels, shuffled_values))

        # Find new correct label
        new_correct_key = next(
            k for k, v in new_selections.items() if v == correct_text
        )

        swapped_instructions.append({
            "Selections": new_selections,
            "Output": new_correct_key,
        })

    return swapped_instructions


def generate_mcq_negative_choices(
    instruction: dict,
    append_token: str = "I don't know",
) -> List[str]:
    """
    Generate negative choices for a multiple-choice question.

    Returns:
        A list of wrong choice labels (e.g., ["A", "C", "D"])
        with an extra option appended at the end:
        - "none" or
        - "I don't know"
    """
    correct = instruction.get("Output", "")
    all_labels = ["A", "B", "C", "D"]
    wrong_labels = [label for label in all_labels if label != correct]
    wrong_labels.append(append_token)

    return wrong_labels


def multiple_choice_augmentation_one_instruction(
    instruction: dict,
) -> Tuple[List[str], List[str], List[str], List[dict]]:
    """
    Perform data augmentation for a single multiple-choice instruction.

    Returns:
        paraphrased_questions: List of paraphrased question stems (str).
        paraphrased_answers: List of paraphrased correct answer texts (str).
        paraphrased_explanations: List of paraphrased explanations (str).
        swapped_choices_instructions: List of dicts with swapped Selections and Output.
        negative_choices: List of wrong choice labels (str).
    """
    paraphrased_questions = paraphrase_mcq_questions(instruction)
    paraphrased_answers = paraphrase_mcq_correct_answer(instruction)
    paraphrased_explanations = paraphrase_mcq_explanations(instruction)
    swapped_choices_instructions = swap_mcq_choices(instruction)

    return (
        paraphrased_questions,
        paraphrased_answers,
        paraphrased_explanations,
        swapped_choices_instructions,
    )


def multiple_choice_augmentation(instructions: List[dict]) -> dict:
    all_facts: List[str] = [instr["Fact"] for instr in instructions]
    all_questions: List[List[str]] = []
    all_correct_answers: List[List[str]] = []
    all_explanations: List[List[str]] = []
    all_swapped: List[List[dict]] = []

    for instr in instructions:
        q_paras, ans_paras, exp_paras, swapped = (
            multiple_choice_augmentation_one_instruction(instr)
        )

        all_questions.append(q_paras)
        all_correct_answers.append(ans_paras)
        all_explanations.append(exp_paras)
        all_swapped.append(swapped)

    return {
        "Fact": all_facts,
        "questions": all_questions,
        "correct_answers": all_correct_answers,
        "explanations": all_explanations,
        "swapped_choices": all_swapped,
    }


def dataset_augmentation(source_dataset: dict, base_name: str) -> dict:
    instructions = source_dataset["instructions"]
    saa = short_answer_augmentation(instructions[1]['Positive Example'], base_name)
    mca = multiple_choice_augmentation(instructions[0]['Positive Example'])
    return {
        "short_answer_augmentation": saa,
        "multiple_choice_augmentation": mca
    }


if __name__ == "__main__":
    output_dir = "../instruction_dataset_augmented_3"
    final_output_dir = "../instruction_dataset_final_4"
    json_files = glob.glob("../instruction_dataset/*.json")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        for file in json_files:
            print(file)
            if not ("france" in file or "japan" in file):
                continue
            with open(file, 'r') as f:
                base_name = os.path.basename(file)
                data = json.load(f)
                result = dataset_augmentation(data, base_name)
                # Generate output filename based on input filename
                output_file = os.path.join(
                    output_dir, base_name.replace('.json', '_augmented.json'))

                # Save the result
                with open(output_file, 'w') as out_f:
                    json.dump(result, out_f, indent=2, ensure_ascii=False)

                print(f"Augmented data saved to {output_file}")
    if not os.path.exists(final_output_dir):
        os.makedirs(final_output_dir)
    for file in json_files:
        if not ("france" in file or "japan" in file):
                continue
        f = open(file, 'r')
        print(file)
        base_name = os.path.basename(file)
        output_file = os.path.join(
            output_dir, base_name.replace('.json', '_augmented.json'))
        source_dataset = json.load(f)
        seeds = source_dataset["instructions"]
        multiple_choice_seed = seeds[0]
        short_answer_seed = seeds[1]
        with open(output_file, 'r') as out_f:
            augmented_data = json.load(out_f)
        mca = augmented_data["multiple_choice_augmentation"]
        saa = augmented_data["short_answer_augmentation"]
        sa_neg_answer_list = saa["negative_answers"]
        # Combine the generated data together for the final dataset creation
        # Short answer
        # For level 1: the first 10 generated, combine the generated facts and explanations with the questions, has the answer stay the same
        short_answer_level_1_positive = []
        short_answer_level_1_negative = []
        for i in range(10):
            for fact_exp in saa["facts_and_explanations"][i]:
                for question in saa["questions"][i]:
                    fact = fact_exp["Fact"]
                    exp = fact_exp["Explanation"]
                    short_answer_level_1_positive.append({
                        "Fact": fact,
                        "Question": question,
                        "Output": short_answer_seed['Positive Example'][i]['Output'],
                        "Explanation": exp,
                        "Level": 1
                    })
                    # Negative examples should be replacing the correct answer with the generated negative answers
                    short_answer_level_1_negative.append({
                        "Fact": fact,
                        "Question": question,
                        "Output": sa_neg_answer_list[i][rseed.randint(0, 9)],
                        "Explanation": exp,
                        "Level": 1
                    })
        print(f"File {base_name}: generated {len(short_answer_level_1_positive)} short-answer level-1 positive examples (combined facts × questions).")
        print(f"File {base_name}: generated {len(short_answer_level_1_negative)} short-answer level-1 negative examples (combined facts × questions with negative answers).")
        # For Level 2
        short_answer_level_2_positive = []
        short_answer_level_2_negative = []
        for i in range(10, 15):
            for fact_exp in rseed.sample(saa["facts_and_explanations"][i], 4):
                for question in saa["questions"][i]:
                    fact = fact_exp["Fact"]
                    exp = fact_exp["Explanation"]
                    short_answer_level_2_positive.append({
                        "Fact": fact,
                        "Question": question,
                        "Output": short_answer_seed['Positive Example'][i]['Output'],
                        "Explanation": exp,
                        "Level": 2
                    })
                    # Negative examples should be replacing the correct answer with the generated negative answers
                    short_answer_level_2_negative.append({
                        "Fact": fact,
                        "Question": question,
                        "Output": sa_neg_answer_list[i][rseed.randint(0, 9)],
                        "Explanation": exp,
                        "Level": 2
                    })
        print(f"File {base_name}: generated {len(short_answer_level_2_positive)} short-answer level-2 positive examples (combined facts × questions).")
        print(f"File {base_name}: generated {len(short_answer_level_2_negative)} short-answer level-2 negative examples (combined facts × questions with negative answers).")
        # Level 3
        short_answer_level_3_positive = []
        short_answer_level_3_negative = []
        if base_name == "tlc_trip_instructions.json" or base_name == "urban_flow_prediction_survey_instructions.json":
            for i in range(15, 20):
                for fact_exp in saa["facts_and_explanations"][i]:
                    for question in saa["questions"][i]:
                        fact = fact_exp["Fact"]
                        exp = fact_exp["Explanation"]
                        short_answer_level_3_positive.append({
                            "Fact": fact,
                            "Question": question,
                            "Output": short_answer_seed['Positive Example'][i]['Output'],
                            "Explanation": exp,
                            "Level": 3
                        })
                        # Negative examples should be replacing the correct answer with the generated negative answers
                        short_answer_level_3_negative.append({
                            "Fact": fact,
                            "Question": question,
                            "Output": sa_neg_answer_list[i][rseed.randint(0, 9)],
                            "Explanation": exp,
                            "Level": 3
                        })
            print(f"File {base_name}: generated {len(short_answer_level_3_positive)} short-answer level-3 positive examples (combined facts × questions).")
            print(f"File {base_name}: generated {len(short_answer_level_3_negative)} short-answer level-3 negative examples (combined facts × questions with negative answers).")

        # Multiple choice
        # For level 1: the first 10 generated, combine the generated questions, answers and explanations and swapped answers together
        multiple_choice_level_1_positive = []
        multiple_choice_level_1_negative = []
        for i in range(10):
            for question in mca["questions"][i]:
                for explanation in mca["explanations"][i]:
                    for swapped in mca["swapped_choices"][i]:
                        for answer in mca["correct_answers"][i]:
                            swapped["Selections"][swapped["Output"]] = answer
                            multiple_choice_level_1_positive.append({
                                "Fact": mca["Fact"][i],
                                "Question": question,
                                "Selections": swapped["Selections"],
                                "Output": swapped["Output"],
                                "Explanation": explanation,
                                "Level": 1
                            })
                            multiple_choice_level_1_negative.append({
                                "Fact": mca["Fact"][i],
                                "Question": question,
                                "Selections": swapped["Selections"],
                                "Output": generate_mcq_negative_choices(multiple_choice_seed['Positive Example'][i])[rseed.randint(0, 3)],
                                "Explanation": explanation,
                                "Level": 1
                            })
        print(f"File {base_name}: generated {len(multiple_choice_level_1_positive)} multiple-choice level-1 positive examples (combined questions × answers × explanations × swapped).")
        print(f"File {base_name}: generated {len(multiple_choice_level_1_negative)} multiple-choice level-1 negative examples (combined questions × explanations × swapped with negative choices).")
        # For level 2
        multiple_choice_level_2_positive = []
        multiple_choice_level_2_negative = []
        for i in range(10, 14):
            for question in mca["questions"][i]:
                for explanation in mca["explanations"][i]:
                    for swapped in rseed.sample(mca["swapped_choices"][i], 3):
                        for answer in mca["correct_answers"][i]:
                            swapped["Selections"][swapped["Output"]] = answer
                            multiple_choice_level_2_positive.append({
                                "Fact": mca["Fact"][i],
                                "Question": question,
                                "Selections": swapped["Selections"],
                                "Output": swapped["Output"],
                                "Explanation": explanation,
                                "Level": 2
                            })
                            multiple_choice_level_2_negative.append({
                                "Fact": mca["Fact"][i],
                                "Question": question,
                                "Selections": swapped["Selections"],
                                "Output": generate_mcq_negative_choices(multiple_choice_seed['Positive Example'][i])[rseed.randint(0, 3)],
                                "Explanation": explanation,
                                "Level": 2
                            })
        print(f"File {base_name}: generated {len(multiple_choice_level_2_positive)} multiple-choice level-2 positive examples (combined questions × answers × explanations × swapped).")
        print(f"File {base_name}: generated {len(multiple_choice_level_2_negative)} multiple-choice level-2 negative examples (combined questions × explanations × swapped with negative choices).")

        # Save the swapped result to the final dataset
        final_dataset = source_dataset.copy()
        final_dataset["instructions"][0]['Positive Example'] = multiple_choice_seed['Positive Example'] + \
            multiple_choice_level_1_positive + multiple_choice_level_2_positive
        final_dataset["instructions"][0]['Negative Example'] = multiple_choice_level_1_negative + \
            multiple_choice_level_2_negative
        final_dataset["instructions"][1]['Positive Example'] = short_answer_seed['Positive Example'] + \
            short_answer_level_1_positive + \
            short_answer_level_2_positive + short_answer_level_3_positive
        final_dataset["instructions"][1]['Negative Example'] = short_answer_level_1_negative + \
            short_answer_level_2_negative + short_answer_level_3_negative

        final_output_file = os.path.join(
            final_output_dir, base_name.replace('.json', '_final.json'))
        with open(final_output_file, 'w') as final_out_f:
            json.dump(final_dataset, final_out_f, indent=2, ensure_ascii=False)

        print(f"Final dataset saved to {output_file}")
