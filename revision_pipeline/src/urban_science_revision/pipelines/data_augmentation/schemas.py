"""Structured-output schemas used by the augmentation model."""

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FactExplanation(StrictModel):
    Fact: str = Field(description="Meaning-preserving fact that still supports the answer")
    Explanation: str = Field(description="Concise rationale consistent with the fact and answer")


class FactExplanationList(StrictModel):
    items: list[FactExplanation]


class QuestionList(StrictModel):
    questions: list[str]


class AnswerList(StrictModel):
    answers: list[str]


class ExplanationList(StrictModel):
    explanations: list[str]


class NegativeAnswerList(StrictModel):
    answers: list[str]
