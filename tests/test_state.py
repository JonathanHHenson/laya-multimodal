import torch

from laya_multimodal.state import EncodedState


def test_state_safe_tensor_round_trip(tmp_path) -> None:
    original = EncodedState(
        embedding=torch.randn(1, 4),
        tokens=torch.randn(1, 3, 5),
        token_mask=torch.tensor([[True, True, False]]),
        modality="image",
        encoder_id="fake/model",
        metadata={"width": 20},
    )
    path = tmp_path / "state.safetensors"
    original.save(path)
    restored = EncodedState.load(path)
    assert torch.equal(restored.embedding, original.embedding)
    assert torch.equal(restored.tokens, original.tokens)
    assert torch.equal(restored.token_mask, original.token_mask)
    assert restored.metadata == {"width": 20}
