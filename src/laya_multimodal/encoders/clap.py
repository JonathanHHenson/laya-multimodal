"""CLAP audio encoder and aligned text encoder."""

from __future__ import annotations

import inspect
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from transformers import AutoModel, AutoProcessor

from ..devices import resolve_device
from ..state import EncodedState
from .base import MediaEncoder, extract_tensor, move_batch


class ClapAudioEncoder(MediaEncoder):
    """Zero-shot decisions over audio using CLAP's joint audio/text space."""

    modality = "audio"

    def __init__(
        self,
        model_id: str = "laion/clap-htsat-fused",
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
        return int(self.model.config.projection_dim)

    @property
    def token_dim(self) -> int:
        # CLAP's HTS-AT backbone does not expose a stable time-token contract
        # across Transformers versions. The global token remains trainable.
        return self.embedding_dim

    @property
    def sampling_rate(self) -> int:
        extractor = getattr(self.processor, "feature_extractor", self.processor)
        return int(getattr(extractor, "sampling_rate", 48_000))

    @torch.inference_mode()
    def encode(
        self,
        media: str | Path | np.ndarray | tuple[np.ndarray, int],
        *,
        include_tokens: bool = True,
    ) -> EncodedState:
        waveform, sampling_rate = _load_audio(media)
        if sampling_rate != self.sampling_rate:
            waveform = _resample(waveform, sampling_rate, self.sampling_rate)
            sampling_rate = self.sampling_rate
        # Transformers 5 standardised `audios=` to `audio=`. Select the
        # advertised name so the library remains compatible with late 4.x.
        audio_parameter = (
            "audio"
            if "audio" in inspect.signature(self.processor.__call__).parameters
            else "audios"
        )
        inputs = move_batch(
            self.processor(
                **{audio_parameter: waveform},
                sampling_rate=sampling_rate,
                return_tensors="pt",
            ),
            self.device,
        )
        audio_args = {
            key: value for key, value in inputs.items() if key in {"input_features", "is_longer"}
        }
        embedding = extract_tensor(self.model.get_audio_features(**audio_args))
        embedding = F.normalize(embedding.float(), dim=-1)

        # A single global state token keeps the decision-head API identical.
        # A future BEATs encoder can supply fine-grained time tokens unchanged.
        tokens = embedding.unsqueeze(1) if include_tokens else None
        token_mask = (
            torch.ones(embedding.shape[0], 1, dtype=torch.bool, device=self.device)
            if include_tokens
            else None
        )
        return EncodedState(
            embedding=embedding,
            tokens=tokens,
            token_mask=token_mask,
            modality=self.modality,
            encoder_id=self.model_id,
            metadata={
                "sampling_rate": sampling_rate,
                "duration_seconds": float(len(waveform) / sampling_rate),
            },
        )

    @torch.inference_mode()
    def encode_text(self, texts: Sequence[str]) -> torch.Tensor:
        inputs = move_batch(
            self.processor(text=list(texts), padding=True, return_tensors="pt"), self.device
        )
        text_args = {
            key: value for key, value in inputs.items() if key in {"input_ids", "attention_mask"}
        }
        embeddings = extract_tensor(self.model.get_text_features(**text_args))
        return F.normalize(embeddings.float(), dim=-1)

    def format_option(self, question: str, option: str) -> str:
        return f'This is an audio recording where the answer to "{question}" is "{option}".'

    def similarity_logits(
        self, media_embedding: torch.Tensor, text_embeddings: torch.Tensor
    ) -> torch.Tensor:
        logits = super().similarity_logits(media_embedding, text_embeddings)
        scale = getattr(self.model, "logit_scale_a", None)
        if scale is None:
            scale = getattr(self.model, "logit_scale", None)
        if scale is not None:
            logits = logits * scale.detach().float().exp().clamp(max=100)
        return logits


def _load_audio(
    media: str | Path | np.ndarray | tuple[np.ndarray, int],
) -> tuple[np.ndarray, int]:
    if isinstance(media, tuple):
        waveform, sampling_rate = media
        return _mono_float(waveform), int(sampling_rate)
    if isinstance(media, np.ndarray):
        raise ValueError("numpy audio input must be supplied as (waveform, sampling_rate)")
    try:
        import soundfile as sf
    except ImportError as exc:
        raise ImportError(
            "file-based audio requires: pip install 'laya-multimodal[audio]'"
        ) from exc
    waveform, sampling_rate = sf.read(str(media), always_2d=False)
    return _mono_float(waveform), int(sampling_rate)


def _mono_float(waveform: np.ndarray) -> np.ndarray:
    waveform = np.asarray(waveform)
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)
    if waveform.ndim != 1:
        raise ValueError("audio waveform must be mono or [samples, channels]")
    return waveform.astype(np.float32, copy=False)


def _resample(waveform: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    try:
        import librosa
    except ImportError as exc:
        raise ImportError(
            "audio resampling requires: pip install 'laya-multimodal[audio]'"
        ) from exc
    return librosa.resample(waveform, orig_sr=source_rate, target_sr=target_rate).astype(np.float32)
