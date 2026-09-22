from collections.abc import Sequence
from typing import Any

import pytest
import torch
from torch.nn import functional as F

from laya_multimodal import DecisionEngine, Question
from laya_multimodal.encoders.base import MediaEncoder
from laya_multimodal.state import EncodedState


class FakeEncoder(MediaEncoder):
    modality = "image"
    model_id = "fake/encoder"
    device = torch.device("cpu")

    @property
    def embedding_dim(self) -> int:
        return 2

    @property
    def token_dim(self) -> int:
        return 2

    def encode(self, media: Any, *, include_tokens: bool = True) -> EncodedState:
        embedding = torch.tensor([[1.0, 0.0]]) if media == "cat" else torch.tensor([[0.0, 1.0]])
        return EncodedState(
            embedding=embedding,
            tokens=embedding.unsqueeze(1) if include_tokens else None,
            token_mask=torch.ones(1, 1, dtype=torch.bool) if include_tokens else None,
            modality=self.modality,
            encoder_id=self.model_id,
        )

    def encode_text(self, texts: Sequence[str]) -> torch.Tensor:
        rows = [
            torch.tensor([1.0, 0.0]) if "cat" in text else torch.tensor([0.0, 1.0])
            for text in texts
        ]
        return F.normalize(torch.stack(rows), dim=-1)

    def format_option(self, question: str, option: str) -> str:
        return option


def test_zero_shot_engine_returns_dynamic_probabilities() -> None:
    engine = DecisionEngine(FakeEncoder())
    result = engine.predict(
        "cat",
        [Question.choice("animal", "What animal?", {"cat": "a cat", "dog": "a dog"})],
    )
    answer = result["animal"]
    assert answer.choice == "cat"
    assert answer.probabilities["cat"] > answer.probabilities["dog"]
    assert sum(answer.probabilities.values()) == pytest.approx(1.0)


def test_score_is_expected_level() -> None:
    engine = DecisionEngine(FakeEncoder())
    answer = engine.predict(
        "cat",
        [Question.score("severity", "Severity?", ["cat-like", "dog-like"])],
    )["severity"]
    assert answer.score is not None
    assert 0 <= answer.score <= 1


def test_abstention_does_not_discard_distribution() -> None:
    engine = DecisionEngine(FakeEncoder(), abstain_below=0.99)
    answer = engine.predict("cat", [Question.choice("animal", "What?", ["cat", "dog"])])["animal"]
    assert answer.abstained is True
    assert len(answer.probabilities) == 2
