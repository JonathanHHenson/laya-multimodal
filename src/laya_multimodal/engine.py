"""High-level encode-once, ask-many decision API."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import torch

from .cache import StateCache
from .calibration import TemperatureCalibrator
from .encoders.base import MediaEncoder
from .head import DynamicDecisionHead
from .schemas import Answer, Question, QuestionType
from .state import EncodedState


class DecisionEngine:
    def __init__(
        self,
        encoder: MediaEncoder,
        *,
        decision_head: DynamicDecisionHead | None = None,
        calibrator: TemperatureCalibrator | None = None,
        abstain_below: float | None = None,
        cache: StateCache | None = None,
    ) -> None:
        self.encoder = encoder
        self.decision_head = decision_head
        self.calibrator = calibrator or TemperatureCalibrator()
        self.abstain_below = abstain_below
        self.cache = cache
        if decision_head is not None:
            decision_head.to(encoder.device).eval()

    def encode(self, media: Any, *, use_cache: bool = True) -> EncodedState:
        if use_cache and self.cache is not None:
            state = self.cache.get_or_create(
                media, self.encoder.model_id, lambda: self.encoder.encode(media)
            )
            return state.to(self.encoder.device)
        return self.encoder.encode(media)

    @torch.inference_mode()
    def ask(
        self,
        state: EncodedState,
        questions: Iterable[Question] | Mapping[str, Mapping[str, Any]],
    ) -> dict[str, Answer]:
        if isinstance(questions, Mapping):
            questions = [
                Question.from_dict({"id": key, **value}) for key, value in questions.items()
            ]
        if state.batch_size != 1:
            raise ValueError("ask currently accepts a state containing exactly one media item")
        if state.encoder_id != self.encoder.model_id:
            raise ValueError(
                f"state was produced by {state.encoder_id!r}, not {self.encoder.model_id!r}"
            )
        state = state.to(self.encoder.device)
        answers: dict[str, Answer] = {}
        for question in questions:
            answers[question.id] = self._answer_one(state, question)
        return answers

    def predict(
        self,
        media: Any,
        questions: Iterable[Question] | Mapping[str, Mapping[str, Any]],
        *,
        use_cache: bool = True,
    ) -> dict[str, Answer]:
        return self.ask(self.encode(media, use_cache=use_cache), questions)

    def _answer_one(self, state: EncodedState, question: Question) -> Answer:
        prompts = [
            self.encoder.format_option(question.prompt, option.description)
            for option in question.options
        ]
        option_embeddings = self.encoder.encode_text(prompts)
        similarity = self.encoder.similarity_logits(state.embedding, option_embeddings)
        logits = similarity
        if self.decision_head is not None:
            tokens = state.tokens if state.tokens is not None else state.embedding.unsqueeze(1)
            logits = self.decision_head(
                tokens,
                option_embeddings.unsqueeze(0),
                state_mask=state.token_mask,
                option_mask=torch.ones(
                    1, len(question.options), dtype=torch.bool, device=self.encoder.device
                ),
                similarity_prior=similarity,
            )
        logits = self.calibrator.apply(logits, len(question.options))
        probabilities = logits.softmax(dim=-1)[0]
        best_index = int(probabilities.argmax())
        confidence = float(probabilities[best_index])
        probability_map = {
            option.key: float(probabilities[index]) for index, option in enumerate(question.options)
        }
        score = None
        if question.type is QuestionType.SCORE:
            positions = torch.arange(len(question.options), device=probabilities.device)
            score = float((positions * probabilities).sum())
        abstained = self.abstain_below is not None and confidence < self.abstain_below
        return Answer(
            question_id=question.id,
            type=question.type,
            choice=question.options[best_index].key,
            probabilities=probability_map,
            confidence=confidence,
            score=score,
            abstained=abstained,
        )
