"""SigLIP2 vision encoder and aligned text encoder."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch
from PIL import Image
from torch.nn import functional as F
from transformers import AutoModel, AutoProcessor

from ..devices import resolve_device
from ..state import EncodedState
from .base import MediaEncoder, extract_tensor, move_batch


class SiglipVisionEncoder(MediaEncoder):
    """Encode an image once, then compare it with any number of text criteria."""

    modality = "image"

    def __init__(
        self,
        model_id: str = "google/siglip2-base-patch16-224",
        *,
        device: str = "auto",
        dtype: torch.dtype | None = None,
    ) -> None:
        self.model_id = model_id
        self.device = resolve_device(device)
        model_kwargs = {}
        if dtype is not None:
            model_kwargs["torch_dtype"] = dtype
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id, **model_kwargs).to(self.device).eval()

    @property
    def embedding_dim(self) -> int:
        return int(
            getattr(self.model.config, "projection_dim", self.model.config.text_config.hidden_size)
        )

    @property
    def token_dim(self) -> int:
        return int(self.model.config.vision_config.hidden_size)

    @torch.inference_mode()
    def encode(
        self, media: str | Path | Image.Image, *, include_tokens: bool = True
    ) -> EncodedState:
        image = _load_image(media)
        inputs = move_batch(self.processor(images=image, return_tensors="pt"), self.device)
        image_args = {
            key: value
            for key, value in inputs.items()
            if key in {"pixel_values", "pixel_attention_mask", "spatial_shapes"}
        }
        embedding = extract_tensor(self.model.get_image_features(**image_args))
        embedding = F.normalize(embedding.float(), dim=-1)

        tokens = None
        token_mask = None
        if include_tokens:
            vision_output = self.model.vision_model(**image_args, return_dict=True)
            tokens = vision_output.last_hidden_state.float()
            token_mask = _vision_token_mask(inputs, tokens)

        return EncodedState(
            embedding=embedding,
            tokens=tokens,
            token_mask=token_mask,
            modality=self.modality,
            encoder_id=self.model_id,
            metadata={"width": image.width, "height": image.height},
        )

    @torch.inference_mode()
    def encode_text(self, texts: Sequence[str]) -> torch.Tensor:
        inputs = move_batch(
            self.processor(
                text=list(texts),
                padding="max_length",
                truncation=True,
                max_length=64,
                return_tensors="pt",
            ),
            self.device,
        )
        text_args = {
            key: value for key, value in inputs.items() if key in {"input_ids", "attention_mask"}
        }
        embeddings = extract_tensor(self.model.get_text_features(**text_args))
        return F.normalize(embeddings.float(), dim=-1)

    def format_option(self, question: str, option: str) -> str:
        return f'This is a photo where the answer to "{question}" is "{option}".'

    def similarity_logits(
        self, media_embedding: torch.Tensor, text_embeddings: torch.Tensor
    ) -> torch.Tensor:
        logits = super().similarity_logits(media_embedding, text_embeddings)
        scale = getattr(self.model, "logit_scale", None)
        bias = getattr(self.model, "logit_bias", None)
        if scale is not None:
            logits = logits * scale.detach().float().exp().clamp(max=100)
        if bias is not None:
            logits = logits + bias.detach().float()
        return logits


def _load_image(media: str | Path | Image.Image) -> Image.Image:
    if isinstance(media, Image.Image):
        return media.convert("RGB")
    with Image.open(media) as image:
        return image.convert("RGB")


def _vision_token_mask(inputs: dict[str, torch.Tensor], tokens: torch.Tensor) -> torch.Tensor:
    mask = inputs.get("pixel_attention_mask")
    if mask is None:
        return torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
    if mask.ndim > 2:
        mask = mask.flatten(1)
    if mask.shape[1] != tokens.shape[1]:
        return torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
    return mask.bool()
