from __future__ import annotations

import numpy as np


class ObservationNormalizer:
    def __init__(self, shape: tuple[int, ...], epsilon: float = 1e-8):
        if not shape or any(size <= 0 for size in shape):
            raise ValueError("observation shape must contain positive dimensions")
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        self.mean = np.zeros(shape, dtype=np.float64)
        self.variance = np.ones(shape, dtype=np.float64)
        self.count = 0
        self.epsilon = float(epsilon)

    def update(self, observation: np.ndarray) -> None:
        value = np.asarray(observation, dtype=np.float64)
        if value.shape != self.mean.shape:
            raise ValueError(f"expected observation shape {self.mean.shape}, got {value.shape}")
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        second_delta = value - self.mean
        if self.count == 1:
            self.variance.fill(0.0)
        else:
            previous_m2 = self.variance * (self.count - 2)
            self.variance = (previous_m2 + delta * second_delta) / (self.count - 1)

    def normalize(self, observation: np.ndarray) -> np.ndarray:
        value = np.asarray(observation, dtype=np.float32)
        if value.shape != self.mean.shape:
            raise ValueError(f"expected observation shape {self.mean.shape}, got {value.shape}")
        return ((value - self.mean) / np.sqrt(self.variance + self.epsilon)).astype(
            np.float32
        )

    def state_dict(self) -> dict:
        return {
            "mean": self.mean.copy(),
            "variance": self.variance.copy(),
            "count": self.count,
            "epsilon": self.epsilon,
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> "ObservationNormalizer":
        normalizer = cls(np.asarray(state["mean"]).shape, state["epsilon"])
        normalizer.mean = np.asarray(state["mean"], dtype=np.float64)
        normalizer.variance = np.asarray(state["variance"], dtype=np.float64)
        normalizer.count = int(state["count"])
        if normalizer.count < 0:
            raise ValueError("normalizer count must not be negative")
        return normalizer
