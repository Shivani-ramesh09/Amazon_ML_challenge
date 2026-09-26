"""Contract for CPU or future GPU pair feature generation."""

from __future__ import annotations

from typing import Protocol

import pyarrow as pa


class PairFeatureBuilder(Protocol):
    def transform(self, candidates: pa.RecordBatch, source_rows: pa.Table,
                  target_rows: pa.Table) -> pa.Table:
        """Return one numeric-feature row per candidate, preserving pair IDs."""
