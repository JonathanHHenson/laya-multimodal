"""Typed question and answer schemas shared by every modality."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class QuestionType(str, Enum):
    CHOICE = "choice"
    BINARY = "binary"
    SCORE = "score"


@dataclass(frozen=True, slots=True)
class Option:
    """A stable machine key and its natural-language criterion."""

    key: str
    description: str

    def __post_init__(self) -> None:
        if not self.key.strip():
            raise ValueError("option key cannot be empty")
        if not self.description.strip():
            raise ValueError("option description cannot be empty")


@dataclass(frozen=True, slots=True)
class Question:
    """A runtime-defined decision card.

    Options are deliberately part of the input rather than a fixed model head.
    """

    id: str
    prompt: str
    options: tuple[Option, ...]
    type: QuestionType = QuestionType.CHOICE
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("question id cannot be empty")
        if not self.prompt.strip():
            raise ValueError("question prompt cannot be empty")
        minimum = 2
        if len(self.options) < minimum:
            raise ValueError(f"{self.type.value} questions require at least {minimum} options")
        keys = [option.key for option in self.options]
        if len(keys) != len(set(keys)):
            raise ValueError("option keys must be unique within a question")
        if self.type is QuestionType.BINARY and len(self.options) != 2:
            raise ValueError("binary questions require exactly two options")

    @classmethod
    def choice(
        cls,
        id: str,
        prompt: str,
        options: Mapping[str, str] | Iterable[str],
    ) -> Question:
        return cls(id=id, prompt=prompt, options=_coerce_options(options))

    @classmethod
    def binary(
        cls,
        id: str,
        prompt: str,
        *,
        positive: str = "yes",
        negative: str = "no",
    ) -> Question:
        return cls(
            id=id,
            prompt=prompt,
            type=QuestionType.BINARY,
            options=(Option("yes", positive), Option("no", negative)),
        )

    @classmethod
    def score(cls, id: str, prompt: str, levels: Iterable[str]) -> Question:
        options = tuple(Option(str(i), text) for i, text in enumerate(levels))
        return cls(id=id, prompt=prompt, type=QuestionType.SCORE, options=options)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Question:
        raw_type = str(data.get("type", "choice"))
        question_type = (
            QuestionType.BINARY
            if raw_type in {"noul", "boolean", "bool"}
            else QuestionType(raw_type)
        )
        raw_options = data.get("options", data.get("criteria"))
        if raw_options is None:
            if question_type is QuestionType.BINARY:
                raw_options = {"yes": data.get("positive", "yes"), "no": data.get("negative", "no")}
            else:
                raise ValueError(f"question {data.get('id', '<unknown>')} has no options")
        if question_type is QuestionType.SCORE and not isinstance(raw_options, Mapping):
            options = tuple(Option(str(i), str(text)) for i, text in enumerate(raw_options))
        else:
            options = _coerce_options(raw_options)
        return cls(
            id=str(data["id"]),
            prompt=str(data.get("prompt", data.get("instructions", ""))),
            type=question_type,
            options=options,
            metadata=data.get("metadata", {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value,
            "prompt": self.prompt,
            "options": {option.key: option.description for option in self.options},
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class Answer:
    question_id: str
    type: QuestionType
    choice: str
    probabilities: Mapping[str, float]
    confidence: float
    score: float | None = None
    abstained: bool = False

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "type": self.type.value,
            "choice": self.choice,
            "probabilities": dict(self.probabilities),
            "confidence": self.confidence,
            "abstained": self.abstained,
        }
        if self.score is not None:
            result["score"] = self.score
        return result


def _coerce_options(options: Mapping[str, str] | Iterable[str]) -> tuple[Option, ...]:
    if isinstance(options, Mapping):
        return tuple(Option(str(key), str(value)) for key, value in options.items())
    return tuple(Option(str(value), str(value)) for value in options)
