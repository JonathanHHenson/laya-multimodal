import pytest
import torch

from laya_multimodal.metrics import brier_score, expected_calibration_error, report


def test_perfect_predictions_have_zero_brier_and_ece() -> None:
    probabilities = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    targets = torch.tensor([0, 1])
    assert brier_score(probabilities, targets) == pytest.approx(0.0)
    assert expected_calibration_error(probabilities, targets) == pytest.approx(0.0)


def test_report_contains_probability_metrics() -> None:
    result = report(torch.tensor([[0.8, 0.2]]), torch.tensor([0]))
    assert set(result) == {"accuracy", "nll", "brier", "ece"}
