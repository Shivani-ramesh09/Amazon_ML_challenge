"""Pair-scorer contract for CPU and future accelerator implementations."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class PairScorer(Protocol):
    def fit(self, train_x: np.ndarray, train_y: np.ndarray,
            validation_x: np.ndarray, validation_y: np.ndarray,
            feature_names: list[str]) -> None:
        """Fit on training entities and early-stop on held-out entities."""

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Return one score in [0, 1] per candidate pair."""
