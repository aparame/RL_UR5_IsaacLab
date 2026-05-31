#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping
import time

import numpy as np


class RolloutLogger:
    """Append rollout samples into an extensible HDF5 file."""

    def __init__(
        self,
        path: str | None = None,
        run_prefix: str = "rollout",
        flush_interval: int = 100,
        metadata: Mapping[str, Any] | None = None,
    ):
        import h5py

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        resolved = Path(path) if path is not None else Path("./rollout_logs")
        if resolved.suffix.lower() not in {".h5", ".hdf5"}:
            resolved = resolved / f"{run_prefix}_{timestamp}.hdf5"
        resolved = resolved.expanduser().resolve()
        resolved.parent.mkdir(parents=True, exist_ok=True)

        self.path = resolved
        self.flush_interval = max(1, int(flush_interval))
        self._count = 0
        self._max_episode_id = -1
        self._h5 = h5py.File(self.path, "w")
        self._h5.attrs["created_at_unix"] = float(time.time())
        self._h5.attrs["run_prefix"] = run_prefix

        if metadata:
            for key, value in metadata.items():
                if value is None:
                    continue
                self._h5.attrs[key] = value

    def append_batch(self, **arrays: np.ndarray) -> None:
        if not arrays:
            return

        normalized = {name: np.asarray(value) for name, value in arrays.items()}
        batch_size = next(iter(normalized.values())).shape[0]
        if batch_size == 0:
            return

        for name, value in normalized.items():
            if value.shape[0] != batch_size:
                raise ValueError(
                    f"Dataset '{name}' has batch size {value.shape[0]}, expected {batch_size}"
                )

        start = self._count
        end = start + batch_size

        for name, value in normalized.items():
            dataset = self._ensure_dataset(name, value)
            dataset.resize((end,) + value.shape[1:])
            dataset[start:end] = value

        self._count = end

        if "episode_ids" in normalized and normalized["episode_ids"].size > 0:
            self._max_episode_id = max(
                self._max_episode_id, int(np.max(normalized["episode_ids"]))
            )

        if self._count % self.flush_interval == 0:
            self.flush()

    def flush(self) -> None:
        if self._h5:
            self._h5.flush()

    def close(self) -> None:
        if not self._h5:
            return

        self._h5.attrs["total_steps"] = int(self._count)
        self._h5.attrs["total_episodes"] = (
            int(self._max_episode_id + 1) if self._max_episode_id >= 0 else 0
        )
        self._h5.flush()
        self._h5.close()
        self._h5 = None

    def _ensure_dataset(self, name: str, value: np.ndarray):
        if name in self._h5:
            return self._h5[name]

        sample_shape = value.shape[1:]
        chunk_first_dim = min(max(1, value.shape[0]), 64)
        chunks = (chunk_first_dim,) + sample_shape
        compression = "lzf" if value.ndim >= 3 else None

        return self._h5.create_dataset(
            name,
            shape=(0,) + sample_shape,
            maxshape=(None,) + sample_shape,
            chunks=chunks,
            compression=compression,
            dtype=value.dtype,
        )
