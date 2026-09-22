"""Accuracy and probability-quality metrics."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def accuracy(probabilities: torch.Tensor, targets: torch.Tensor) -> float:
    return float((probabilities.argmax(dim=-1) == targets).float().mean())


def negative_log_likelihood(probabilities: torch.Tensor, targets: torch.Tensor) -> float:
    selected = probabilities.clamp_min(1e-12).gather(1, targets[:, None])
    return float(-selected.log().mean())


def brier_score(probabilities: torch.Tensor, targets: torch.Tensor) -> float:
    truth = F.one_hot(targets, num_classes=probabilities.shape[-1]).to(probabilities.dtype)
    return float(((probabilities - truth) ** 2).sum(dim=-1).mean())


def ranked_probability_score(probabilities: torch.Tensor, targets: torch.Tensor) -> float:
    truth = F.one_hot(targets, num_classes=probabilities.shape[-1]).to(probabilities.dtype)
    return float(((probabilities.cumsum(-1) - truth.cumsum(-1)) ** 2).sum(-1).mean())


def expected_calibration_error(
    probabilities: torch.Tensor, targets: torch.Tensor, *, bins: int = 15
) -> float:
    confidence, prediction = probabilities.max(dim=-1)
    correct = prediction.eq(targets).float()
    error = torch.zeros((), device=probabilities.device)
    boundaries = torch.linspace(0, 1, bins + 1, device=probabilities.device)
    for lower, upper in zip(boundaries[:-1], boundaries[1:], strict=True):
        mask = (confidence > lower) & (confidence <= upper)
        if mask.any():
            error += mask.float().mean() * (correct[mask].mean() - confidence[mask].mean()).abs()
    return float(error)


def report(
    probabilities: torch.Tensor, targets: torch.Tensor, *, ordinal: bool = False
) -> dict[str, float]:
    result = {
        "accuracy": accuracy(probabilities, targets),
        "nll": negative_log_likelihood(probabilities, targets),
        "brier": brier_score(probabilities, targets),
        "ece": expected_calibration_error(probabilities, targets),
    }
    if ordinal:
        result["rps"] = ranked_probability_score(probabilities, targets)
    return result
