"""Encoder contract used by the decision engine."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

import torch
from torch.nn import functional as F

from ..state import EncodedState


class MediaEncoder(ABC):
    modality: str
    model_id: str
    device: torch.device

    @abstractmethod
    def encode(self, media: Any, *, include_tokens: bool = True) -> EncodedState:
        """Encode one media item into a reusable state."""

    @abstractmethod
    def encode_text(self, texts: Sequence[str]) -> torch.Tensor:
        """Encode option descriptions into the media-aligned embedding space."""

    @abstractmethod
    def format_option(self, question: str, option: str) -> str:
        """Format a decision criterion in the encoder's contrastive prompt style."""

    def similarity_logits(
        self, media_embedding: torch.Tensor, text_embeddings: torch.Tensor
    ) -> torch.Tensor:
        media_embedding = F.normalize(media_embedding.float(), dim=-1)
        text_embeddings = F.normalize(text_embeddings.float(), dim=-1)
        return media_embedding @ text_embeddings.transpose(-1, -2)

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Dimension of aligned global embeddings."""

    @property
    @abstractmethod
    def token_dim(self) -> int:
        """Dimension of spatial/temporal state tokens."""


def extract_tensor(value: Any) -> torch.Tensor:
    """Handle both legacy tensor and newer structured feature outputs."""
    if isinstance(value, torch.Tensor):
        return value
    for name in ("image_embeds", "audio_embeds", "text_embeds", "pooler_output"):
        candidate = getattr(value, name, None)
        if isinstance(candidate, torch.Tensor):
            return candidate
    if isinstance(value, (tuple, list)) and value and isinstance(value[0], torch.Tensor):
        return value[0]
    raise TypeError(f"could not extract a tensor from {type(value).__name__}")


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()
    }
