"""Content-addressed cache for expensive media encodings."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from .state import EncodedState


class StateCache:
    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def key(self, media: Any, encoder_id: str) -> str:
        digest = hashlib.sha256()
        digest.update(encoder_id.encode())
        if isinstance(media, (str, Path)):
            path = Path(media)
            digest.update(path.read_bytes())
        elif isinstance(media, tuple) and isinstance(media[0], np.ndarray):
            digest.update(media[0].tobytes())
            digest.update(str(media[1]).encode())
        elif isinstance(media, np.ndarray):
            digest.update(media.tobytes())
        else:
            digest.update(repr(media).encode())
        return digest.hexdigest()

    def path_for(self, key: str) -> Path:
        return self.directory / key[:2] / f"{key}.safetensors"

    def get(self, key: str) -> EncodedState | None:
        path = self.path_for(key)
        if not path.exists():
            return None
        return EncodedState.load(path)

    def put(self, key: str, state: EncodedState) -> Path:
        path = self.path_for(key)
        state.detach(cpu=True).save(path)
        return path

    def get_or_create(
        self, media: Any, encoder_id: str, factory: Callable[[], EncodedState]
    ) -> EncodedState:
        key = self.key(media, encoder_id)
        cached = self.get(key)
        if cached is not None:
            return cached
        state = factory()
        self.put(key, state)
        return state
