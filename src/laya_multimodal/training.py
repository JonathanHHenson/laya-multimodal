"""Training and evaluation for the lightweight decision head."""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from .cache import StateCache
from .calibration import TemperatureCalibrator
from .encoders.base import MediaEncoder
from .head import DecisionHeadConfig, DynamicDecisionHead
from .metrics import report
from .schemas import Question, QuestionType
from .state import EncodedState


@dataclass(frozen=True, slots=True)
class TrainingExample:
    media: str
    question: Question
    target_distribution: tuple[float, ...]

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, base_directory: Path) -> TrainingExample:
        question = Question.from_dict(data["question"])
        media = Path(data["media"])
        if not media.is_absolute():
            media = base_directory / media
        if "target_distribution" in data:
            raw = data["target_distribution"]
            if isinstance(raw, dict):
                distribution = tuple(float(raw.get(option.key, 0.0)) for option in question.options)
            else:
                distribution = tuple(float(value) for value in raw)
        else:
            target = str(data["target"])
            distribution = tuple(float(option.key == target) for option in question.options)
        if len(distribution) != len(question.options):
            raise ValueError("target distribution length must match the number of options")
        total = sum(distribution)
        if total <= 0:
            raise ValueError("target distribution must have positive mass")
        return cls(
            media=str(media),
            question=question,
            target_distribution=tuple(value / total for value in distribution),
        )


def load_jsonl(path: str | Path) -> list[TrainingExample]:
    path = Path(path)
    examples = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                examples.append(
                    TrainingExample.from_dict(json.loads(line), base_directory=path.parent)
                )
            except Exception as exc:
                raise ValueError(f"invalid example at {path}:{line_number}: {exc}") from exc
    if not examples:
        raise ValueError(f"dataset is empty: {path}")
    return examples


@dataclass(slots=True)
class PreparedExample:
    state: EncodedState
    option_embeddings: torch.Tensor
    similarity_prior: torch.Tensor
    target_distribution: torch.Tensor
    question_type: QuestionType


class DecisionHeadTrainer:
    def __init__(
        self,
        encoder: MediaEncoder,
        head: DynamicDecisionHead | None = None,
        *,
        cache_directory: str | Path | None = None,
        seed: int = 42,
    ) -> None:
        self.encoder = encoder
        self.random = random.Random(seed)
        self.cache = StateCache(cache_directory) if cache_directory else None
        self.head = head or DynamicDecisionHead(
            DecisionHeadConfig(
                state_dim=encoder.token_dim,
                option_dim=encoder.embedding_dim,
            )
        )
        self.head.to(encoder.device)
        self._text_cache: dict[tuple[str, ...], torch.Tensor] = {}

    def prepare(self, example: TrainingExample, *, shuffle_options: bool) -> PreparedExample:
        state = self._state(example.media).to(self.encoder.device)
        order = list(range(len(example.question.options)))
        if shuffle_options:
            self.random.shuffle(order)
        prompts = tuple(
            self.encoder.format_option(
                example.question.prompt, example.question.options[index].description
            )
            for index in order
        )
        if prompts not in self._text_cache:
            self._text_cache[prompts] = self.encoder.encode_text(prompts).detach().cpu()
        option_embeddings = self._text_cache[prompts].to(self.encoder.device)
        similarity = self.encoder.similarity_logits(state.embedding, option_embeddings)
        targets = torch.tensor(
            [example.target_distribution[index] for index in order],
            dtype=torch.float32,
            device=self.encoder.device,
        )
        return PreparedExample(
            state=state,
            option_embeddings=option_embeddings,
            similarity_prior=similarity,
            target_distribution=targets,
            question_type=example.question.type,
        )

    def train(
        self,
        examples: Sequence[TrainingExample],
        *,
        output_directory: str | Path,
        epochs: int = 5,
        batch_size: int = 8,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.01,
        loss: str = "log",
        shuffle_options: bool = True,
    ) -> list[dict[str, float]]:
        output_directory = Path(output_directory)
        output_directory.mkdir(parents=True, exist_ok=True)
        optimizer = torch.optim.AdamW(
            self.head.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        history: list[dict[str, float]] = []
        for epoch in range(1, epochs + 1):
            self.head.train()
            indices = list(range(len(examples)))
            self.random.shuffle(indices)
            total_loss = 0.0
            seen = 0
            for start in range(0, len(indices), batch_size):
                prepared = [
                    self.prepare(examples[index], shuffle_options=shuffle_options)
                    for index in indices[start : start + batch_size]
                ]
                batch = collate(prepared, device=self.encoder.device)
                optimizer.zero_grad(set_to_none=True)
                logits = self.head(
                    batch["state_tokens"],
                    batch["option_embeddings"],
                    state_mask=batch["state_mask"],
                    option_mask=batch["option_mask"],
                    similarity_prior=batch["similarity_prior"],
                )
                batch_loss = proper_loss(
                    logits,
                    batch["target_distribution"],
                    batch["option_mask"],
                    kind=loss,
                )
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.head.parameters(), 1.0)
                optimizer.step()
                count = len(prepared)
                total_loss += float(batch_loss.detach()) * count
                seen += count
            record = {"epoch": float(epoch), "loss": total_loss / max(seen, 1)}
            history.append(record)
            self.save(output_directory, history=history)
            print(f"epoch {epoch}/{epochs} loss={record['loss']:.6f}", flush=True)
        return history

    @torch.inference_mode()
    def collect_logits(
        self, examples: Sequence[TrainingExample]
    ) -> tuple[list[torch.Tensor], torch.Tensor, list[int], list[QuestionType]]:
        self.head.eval()
        logits: list[torch.Tensor] = []
        targets: list[int] = []
        cardinalities: list[int] = []
        types: list[QuestionType] = []
        for example in examples:
            item = self.prepare(example, shuffle_options=False)
            state_tokens = (
                item.state.tokens
                if item.state.tokens is not None
                else item.state.embedding.unsqueeze(1)
            )
            value = self.head(
                state_tokens,
                item.option_embeddings.unsqueeze(0),
                state_mask=item.state.token_mask,
                similarity_prior=item.similarity_prior,
            )[0]
            logits.append(value.detach().cpu())
            targets.append(int(item.target_distribution.argmax()))
            cardinalities.append(len(example.question.options))
            types.append(example.question.type)
        return logits, torch.tensor(targets), cardinalities, types

    def calibrate(
        self, examples: Sequence[TrainingExample], *, output_path: str | Path
    ) -> TemperatureCalibrator:
        rows, targets, cardinalities, _ = self.collect_logits(examples)
        unique = set(cardinalities)
        if len(unique) != 1:
            # Fit the shared value using padded -inf logits. Cardinality-specific
            # values are then fitted on the unpadded subsets below.
            max_options = max(unique)
            logits = torch.full((len(rows), max_options), -1e9)
            for index, row in enumerate(rows):
                logits[index, : row.numel()] = row
        else:
            logits = torch.stack(rows)
        calibrator = TemperatureCalibrator().fit(
            logits.to(self.encoder.device),
            targets.to(self.encoder.device),
            cardinalities=cardinalities,
        )
        calibrator.save(output_path)
        return calibrator

    def evaluate(
        self,
        examples: Sequence[TrainingExample],
        *,
        calibrator: TemperatureCalibrator | None = None,
    ) -> dict[str, float]:
        rows, targets, cardinalities, types = self.collect_logits(examples)
        probabilities = []
        for row, cardinality in zip(rows, cardinalities, strict=True):
            if calibrator:
                row = calibrator.apply(row, cardinality)
            probabilities.append(row.softmax(-1))
        max_options = max(cardinalities)
        padded = torch.zeros(len(probabilities), max_options)
        for index, row in enumerate(probabilities):
            padded[index, : row.numel()] = row
        return report(
            padded,
            targets,
            ordinal=all(value is QuestionType.SCORE for value in types),
        )

    def save(self, output_directory: str | Path, *, history: list[dict[str, float]]) -> None:
        output_directory = Path(output_directory)
        self.head.save_pretrained(output_directory)
        (output_directory / "encoder.json").write_text(
            json.dumps(
                {"modality": self.encoder.modality, "model_id": self.encoder.model_id},
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        (output_directory / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )

    def _state(self, media: str) -> EncodedState:
        if self.cache is None:
            return self.encoder.encode(media).detach(cpu=True)
        return self.cache.get_or_create(
            media,
            self.encoder.model_id,
            lambda: self.encoder.encode(media).detach(cpu=True),
        )


def collate(items: Sequence[PreparedExample], *, device: torch.device) -> dict[str, torch.Tensor]:
    batch_size = len(items)
    max_tokens = max(
        (
            item.state.tokens
            if item.state.tokens is not None
            else item.state.embedding.unsqueeze(1)
        ).shape[1]
        for item in items
    )
    max_options = max(item.option_embeddings.shape[0] for item in items)
    state_dim = (
        items[0].state.tokens.shape[-1]
        if items[0].state.tokens is not None
        else items[0].state.embedding.shape[-1]
    )
    option_dim = items[0].option_embeddings.shape[-1]
    state_tokens = torch.zeros(batch_size, max_tokens, state_dim, device=device)
    option_embeddings = torch.zeros(batch_size, max_options, option_dim, device=device)
    similarity_prior = torch.zeros(batch_size, max_options, device=device)
    target_distribution = torch.zeros(batch_size, max_options, device=device)
    state_mask = torch.zeros(batch_size, max_tokens, dtype=torch.bool, device=device)
    option_mask = torch.zeros(batch_size, max_options, dtype=torch.bool, device=device)
    for index, item in enumerate(items):
        tokens = (
            item.state.tokens
            if item.state.tokens is not None
            else item.state.embedding.unsqueeze(1)
        )
        token_count = tokens.shape[1]
        option_count = item.option_embeddings.shape[0]
        state_tokens[index, :token_count] = tokens[0]
        option_embeddings[index, :option_count] = item.option_embeddings
        similarity_prior[index, :option_count] = item.similarity_prior[0]
        target_distribution[index, :option_count] = item.target_distribution
        state_mask[index, :token_count] = True
        option_mask[index, :option_count] = True
    return {
        "state_tokens": state_tokens,
        "option_embeddings": option_embeddings,
        "similarity_prior": similarity_prior,
        "target_distribution": target_distribution,
        "state_mask": state_mask,
        "option_mask": option_mask,
    }


def proper_loss(
    logits: torch.Tensor,
    target_distribution: torch.Tensor,
    option_mask: torch.Tensor,
    *,
    kind: str,
) -> torch.Tensor:
    masked_logits = logits.masked_fill(~option_mask, torch.finfo(logits.dtype).min)
    log_probabilities = F.log_softmax(masked_logits, dim=-1)
    probabilities = log_probabilities.exp()
    if kind == "log":
        return -(target_distribution * log_probabilities).sum(-1).mean()
    if kind == "brier":
        return ((probabilities - target_distribution) ** 2 * option_mask).sum(-1).mean()
    if kind == "spherical":
        numerator = (probabilities * target_distribution).sum(-1)
        denominator = probabilities.square().sum(-1).sqrt().clamp_min(1e-8)
        return -(numerator / denominator).mean()
    if kind == "rps":
        differences = probabilities.cumsum(-1) - target_distribution.cumsum(-1)
        return (differences.square() * option_mask).sum(-1).mean()
    raise ValueError(f"unknown loss {kind!r}; choose log, brier, spherical, or rps")
