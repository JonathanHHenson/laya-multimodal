"""Temperature calibration, including separate values by option cardinality."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(slots=True)
class TemperatureCalibrator:
    default: float = 1.0
    by_cardinality: dict[int, float] = field(default_factory=dict)

    def temperature(self, cardinality: int) -> float:
        return max(float(self.by_cardinality.get(cardinality, self.default)), 1e-4)

    def apply(self, logits: torch.Tensor, cardinality: int | None = None) -> torch.Tensor:
        cardinality = cardinality or logits.shape[-1]
        return logits / self.temperature(cardinality)

    def fit(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        *,
        cardinalities: Sequence[int] | None = None,
        max_iter: int = 100,
    ) -> TemperatureCalibrator:
        self.default = fit_temperature(logits, targets, max_iter=max_iter)
        if cardinalities is not None:
            cards = torch.as_tensor(cardinalities, device=logits.device)
            for cardinality in cards.unique().tolist():
                mask = cards == cardinality
                if int(mask.sum()) >= 8:
                    self.by_cardinality[int(cardinality)] = fit_temperature(
                        logits[mask, : int(cardinality)], targets[mask], max_iter=max_iter
                    )
        return self

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(
                {"default": self.default, "by_cardinality": self.by_cardinality},
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> TemperatureCalibrator:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            default=float(data["default"]),
            by_cardinality={
                int(key): float(value) for key, value in data.get("by_cardinality", {}).items()
            },
        )


def fit_temperature(logits: torch.Tensor, targets: torch.Tensor, *, max_iter: int = 100) -> float:
    """Fit one positive temperature by minimising held-out negative log likelihood."""
    logits = logits.detach().float()
    targets = targets.detach().long()
    log_temperature = nn.Parameter(torch.zeros((), device=logits.device))
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=max_iter)

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        temperature = log_temperature.exp().clamp(1e-3, 100)
        loss = F.cross_entropy(logits / temperature, targets)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.detach().exp().clamp(1e-3, 100).cpu())
