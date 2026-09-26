"""Minimal retrieval boundary; a future GPU implementation may use the same schema."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class CandidateRetriever(ABC):
    @abstractmethod
    def retrieve(self, query_path: Path, output_path: Path, *, skip_s1_ids: set[str] | None = None) -> dict:
        """Write bounded final channel pairs with scores/ranks; return phase metrics."""
