import torch

from laya_multimodal.head import DecisionHeadConfig, DynamicDecisionHead


def make_head() -> DynamicDecisionHead:
    torch.manual_seed(3)
    return DynamicDecisionHead(
        DecisionHeadConfig(
            state_dim=6,
            option_dim=5,
            hidden_dim=12,
            num_heads=3,
            num_layers=2,
            dropout=0.0,
        )
    ).eval()


def test_head_is_equivariant_to_option_order() -> None:
    head = make_head()
    state = torch.randn(1, 4, 6)
    options = torch.randn(1, 3, 5)
    prior = torch.randn(1, 3)
    order = torch.tensor([2, 0, 1])
    original = head(state, options, similarity_prior=prior)
    permuted = head(state, options[:, order], similarity_prior=prior[:, order])
    assert torch.allclose(permuted, original[:, order], atol=1e-6)


def test_initial_head_preserves_similarity_prior() -> None:
    head = make_head()
    prior = torch.tensor([[0.2, 0.4, -0.1]])
    result = head(torch.randn(1, 2, 6), torch.randn(1, 3, 5), similarity_prior=prior)
    assert torch.allclose(result, prior)


def test_head_save_round_trip(tmp_path) -> None:
    head = make_head()
    head.save_pretrained(tmp_path)
    restored = DynamicDecisionHead.from_pretrained(tmp_path)
    state = torch.randn(1, 2, 6)
    options = torch.randn(1, 3, 5)
    assert torch.allclose(head(state, options), restored(state, options))
