"""State buffers used for policy distillation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class StateBuffer:
    """Fixed-capacity reservoir-style state buffer."""

    capacity: int
    obs_dim: int
    seed: int = 0

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        self._states = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self._size = 0
        self._seen = 0

    @property
    def size(self) -> int:
        return self._size

    def add(self, states: np.ndarray) -> None:
        states = np.asarray(states, dtype=np.float32).reshape(-1, self.obs_dim)
        for state in states:
            self._seen += 1
            if self._size < self.capacity:
                self._states[self._size] = state
                self._size += 1
                continue
            idx = int(self._rng.integers(0, self._seen))
            if idx < self.capacity:
                self._states[idx] = state

    def sample(self, batch_size: int) -> np.ndarray:
        if self._size <= 0:
            raise ValueError("Cannot sample from an empty state buffer")
        idx = self._rng.integers(0, self._size, size=batch_size)
        return self._states[idx]

    def as_array(self) -> np.ndarray:
        return self._states[: self._size].copy()
