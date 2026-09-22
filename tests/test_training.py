import torch

from laya_multimodal.training import proper_loss


def test_log_loss_prefers_correct_confident_prediction() -> None:
    target = torch.tensor([[1.0, 0.0, 0.0]])
    mask = torch.ones(1, 3, dtype=torch.bool)
    good = proper_loss(torch.tensor([[4.0, 0.0, -1.0]]), target, mask, kind="log")
    bad = proper_loss(torch.tensor([[-1.0, 4.0, 0.0]]), target, mask, kind="log")
    assert good < bad


def test_all_losses_are_finite_with_padding() -> None:
    logits = torch.tensor([[1.0, 0.0, -2.0], [0.5, -0.5, 99.0]])
    target = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    mask = torch.tensor([[True, True, True], [True, True, False]])
    for kind in ("log", "brier", "spherical", "rps"):
        assert torch.isfinite(proper_loss(logits, target, mask, kind=kind))
