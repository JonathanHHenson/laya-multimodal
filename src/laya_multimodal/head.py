"""A small option-query Transformer for trainable multimodal decisions."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import nn


@dataclass(frozen=True, slots=True)
class DecisionHeadConfig:
    state_dim: int
    option_dim: int
    hidden_dim: int = 384
    num_heads: int = 6
    num_layers: int = 2
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.hidden_dim % self.num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")


class DecisionBlock(nn.Module):
    """Permutation-equivariant option competition followed by media attention."""

    def __init__(self, config: DecisionHeadConfig) -> None:
        super().__init__()
        kwargs = {
            "embed_dim": config.hidden_dim,
            "num_heads": config.num_heads,
            "dropout": config.dropout,
            "batch_first": True,
        }
        self.option_attention = nn.MultiheadAttention(**kwargs)
        self.media_attention = nn.MultiheadAttention(**kwargs)
        self.norm1 = nn.LayerNorm(config.hidden_dim)
        self.norm2 = nn.LayerNorm(config.hidden_dim)
        self.norm3 = nn.LayerNorm(config.hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim * 4, config.hidden_dim),
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        options: torch.Tensor,
        media: torch.Tensor,
        *,
        option_mask: torch.Tensor | None,
        media_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        normalized = self.norm1(options)
        attended, _ = self.option_attention(
            normalized,
            normalized,
            normalized,
            key_padding_mask=None if option_mask is None else ~option_mask,
            need_weights=False,
        )
        options = options + self.dropout(attended)
        attended, _ = self.media_attention(
            self.norm2(options),
            media,
            media,
            key_padding_mask=None if media_mask is None else ~media_mask,
            need_weights=False,
        )
        options = options + self.dropout(attended)
        return options + self.dropout(self.ff(self.norm3(options)))


class DynamicDecisionHead(nn.Module):
    """Score any runtime-provided option set against reusable media tokens."""

    def __init__(self, config: DecisionHeadConfig) -> None:
        super().__init__()
        self.config = config
        self.state_projection = nn.Linear(config.state_dim, config.hidden_dim)
        self.option_projection = nn.Linear(config.option_dim, config.hidden_dim)
        self.layers = nn.ModuleList(DecisionBlock(config) for _ in range(config.num_layers))
        self.final_norm = nn.LayerNorm(config.hidden_dim)
        self.scorer = nn.Linear(config.hidden_dim, 1)
        # Start from the useful zero-shot baseline and learn a residual correction.
        self.prior_scale = nn.Parameter(torch.tensor(1.0))
        nn.init.zeros_(self.scorer.weight)
        nn.init.zeros_(self.scorer.bias)

    def forward(
        self,
        state_tokens: torch.Tensor,
        option_embeddings: torch.Tensor,
        *,
        state_mask: torch.Tensor | None = None,
        option_mask: torch.Tensor | None = None,
        similarity_prior: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if state_tokens.ndim != 3 or option_embeddings.ndim != 3:
            raise ValueError("state_tokens and option_embeddings must both be rank 3")
        media = self.state_projection(state_tokens)
        options = self.option_projection(option_embeddings)
        for layer in self.layers:
            options = layer(
                options,
                media,
                option_mask=option_mask,
                media_mask=state_mask,
            )
        logits = self.scorer(self.final_norm(options)).squeeze(-1)
        if similarity_prior is not None:
            logits = logits + self.prior_scale * similarity_prior
        if option_mask is not None:
            logits = logits.masked_fill(~option_mask, torch.finfo(logits.dtype).min)
        return logits

    def save_pretrained(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "config.json").write_text(
            json.dumps(asdict(self.config), indent=2, sort_keys=True), encoding="utf-8"
        )
        save_file(
            {key: value.detach().cpu().contiguous() for key, value in self.state_dict().items()},
            str(directory / "model.safetensors"),
        )

    @classmethod
    def from_pretrained(
        cls, directory: str | Path, *, device: str | torch.device = "cpu"
    ) -> DynamicDecisionHead:
        directory = Path(directory)
        config = DecisionHeadConfig(
            **json.loads((directory / "config.json").read_text(encoding="utf-8"))
        )
        model = cls(config)
        # Some safetensors releases do not recognise MPS as a direct load
        # target. CPU-first loading is portable, and this head is small.
        model.load_state_dict(load_file(str(directory / "model.safetensors"), device="cpu"))
        return model.to(device)
