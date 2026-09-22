"""Reusable media state produced by an encoder."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file


@dataclass(slots=True)
class EncodedState:
    """Media representation that can answer many question batches.

    `embedding` is the aligned global representation used by the zero-shot
    baseline. `tokens` retains spatial or temporal structure for the trainable
    decision head.
    """

    embedding: torch.Tensor
    modality: str
    encoder_id: str
    tokens: torch.Tensor | None = None
    token_mask: torch.Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.embedding.ndim == 1:
            self.embedding = self.embedding.unsqueeze(0)
        if self.embedding.ndim != 2:
            raise ValueError("embedding must have shape [batch, dimension]")
        if self.tokens is not None and self.tokens.ndim == 2:
            self.tokens = self.tokens.unsqueeze(0)
        if self.tokens is not None and self.tokens.ndim != 3:
            raise ValueError("tokens must have shape [batch, sequence, dimension]")

    @property
    def batch_size(self) -> int:
        return self.embedding.shape[0]

    def to(self, device: str | torch.device) -> EncodedState:
        return EncodedState(
            embedding=self.embedding.to(device),
            modality=self.modality,
            encoder_id=self.encoder_id,
            tokens=None if self.tokens is None else self.tokens.to(device),
            token_mask=None if self.token_mask is None else self.token_mask.to(device),
            metadata=dict(self.metadata),
        )

    def detach(self, *, cpu: bool = False) -> EncodedState:
        device = "cpu" if cpu else self.embedding.device
        return EncodedState(
            embedding=self.embedding.detach().to(device),
            modality=self.modality,
            encoder_id=self.encoder_id,
            tokens=None if self.tokens is None else self.tokens.detach().to(device),
            token_mask=None if self.token_mask is None else self.token_mask.detach().to(device),
            metadata=dict(self.metadata),
        )

    def save(self, path: str | Path) -> None:
        """Persist tensors safely using safetensors plus adjacent JSON metadata."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tensors = {"embedding": self.embedding.detach().cpu().contiguous()}
        if self.tokens is not None:
            tensors["tokens"] = self.tokens.detach().cpu().contiguous()
        if self.token_mask is not None:
            tensors["token_mask"] = self.token_mask.detach().cpu().contiguous()
        save_file(tensors, str(path))
        payload = {
            "modality": self.modality,
            "encoder_id": self.encoder_id,
            "metadata": self.metadata,
        }
        path.with_suffix(path.suffix + ".json").write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path, device: str | torch.device = "cpu") -> EncodedState:
        path = Path(path)
        tensors = load_file(str(path), device="cpu")
        payload = json.loads(path.with_suffix(path.suffix + ".json").read_text(encoding="utf-8"))
        state = cls(
            embedding=tensors["embedding"],
            tokens=tensors.get("tokens"),
            token_mask=tensors.get("token_mask"),
            modality=payload["modality"],
            encoder_id=payload["encoder_id"],
            metadata=payload.get("metadata", {}),
        )
        return state.to(device)
