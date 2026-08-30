"""OpenAI structured-output adapter with per-request disk caching."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, TypeVar

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

from .schemas import (
    AnswerList,
    ExplanationList,
    FactExplanationList,
    NegativeAnswerList,
    QuestionList,
)

T = TypeVar("T", bound=BaseModel)

MODEL = "gpt-5.1"


class OpenAIAugmenter:
    """Generate validated augmentation components through the Responses API."""

    def __init__(self, cache_dir: str | Path = "data/02_intermediate/api_cache") -> None:
        for candidate in (Path.cwd() / ".env", Path.cwd().parent / ".env"):
            if candidate.is_file():
                load_dotenv(candidate, override=False)
                break
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is required for the data_augmentation pipeline")
        self.client = OpenAI()
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _parse(
        self,
        *,
        system: str,
        user: str,
        schema: type[T],
        temperature: float,
    ) -> T:
        request = {
            "model": MODEL,
            "system": system,
            "user": user,
            "schema": schema.__name__,
            "temperature": temperature,
        }
        cache_key = hashlib.sha256(
            json.dumps(request, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            return schema.model_validate_json(cache_file.read_text(encoding="utf-8"))

        completion = self.client.responses.parse(
            model=MODEL,
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            text_format=schema,
            store=False,
        )
        parsed = completion.output_parsed
        if parsed is None:
            raise RuntimeError(f"{MODEL} returned no parsed {schema.__name__} output")
        cache_file.write_text(parsed.model_dump_json(indent=2), encoding="utf-8")
        return parsed

    @staticmethod
    def _require_count(items: list[Any], expected: int, label: str) -> list[Any]:
        if len(items) != expected:
            raise ValueError(f"Expected {expected} {label}, received {len(items)}")
        return items

    def fact_explanations(self, system: str, user: str, count: int) -> list[dict[str, str]]:
        parsed = self._parse(system=system, user=user, schema=FactExplanationList, temperature=0.2)
        return [item.model_dump() for item in self._require_count(parsed.items, count, "pairs")]

    def questions(self, system: str, user: str, count: int, *, temperature: float) -> list[str]:
        parsed = self._parse(system=system, user=user, schema=QuestionList, temperature=temperature)
        return self._require_count(parsed.questions, count, "questions")

    def answers(self, system: str, user: str, count: int) -> list[str]:
        parsed = self._parse(system=system, user=user, schema=AnswerList, temperature=0.2)
        return self._require_count(parsed.answers, count, "answers")

    def explanations(self, system: str, user: str, count: int) -> list[str]:
        parsed = self._parse(system=system, user=user, schema=ExplanationList, temperature=0.6)
        return self._require_count(parsed.explanations, count, "explanations")

    def negative_answers(self, system: str, user: str, count: int = 10) -> list[str]:
        parsed = self._parse(system=system, user=user, schema=NegativeAnswerList, temperature=0.7)
        return self._require_count(parsed.answers, count, "negative candidates")
