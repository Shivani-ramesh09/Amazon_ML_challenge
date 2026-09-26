"""Versioned CPU profile configuration (JSON syntax is valid YAML 1.2)."""

from __future__ import annotations

import json
from pathlib import Path


REQUIRED = {
    "data_dir", "artifact_dir", "threads", "memory_budget_gb", "sample_modulus",
    "validation_percent", "split_seed", "candidate_cap", "pair_batch_size",
    "profile_frequency_sample_modulus",
}


def load_config(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    missing = REQUIRED - data.keys()
    if missing:
        raise ValueError(f"{path}: missing config keys {sorted(missing)}")
    if data["threads"] < 1 or data["memory_budget_gb"] <= 0 or data["sample_modulus"] < 1:
        raise ValueError(f"{path}: invalid resource limits")
    if not 0 <= data["validation_percent"] <= 100:
        raise ValueError(f"{path}: invalid validation_percent")
    if not isinstance(data.get("address_channel", False), bool):
        raise ValueError(f"{path}: address_channel must be boolean")
    if any(data.get(key, 1) < 1 for key in
           ("address_max_block_size", "address_per_query_cap", "address_min_chars")):
        raise ValueError(f"{path}: invalid address retrieval limits")
    return data
