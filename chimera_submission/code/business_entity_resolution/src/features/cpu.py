"""Bounded Arrow gathers and CPU lexical/address features for candidate pairs."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import time

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from rapidfuzz import fuzz

ROW_COLUMNS = ["entity_id", "name_clean", "name_core", "address_clean",
               "address_numbers", "postal_candidate", "house_number", "country_key"]
PROVENANCE = ["retrieved_by_exact_name", "retrieved_by_core_name", "retrieved_by_exact_address", "retrieved_by_rare_token",
              "retrieved_by_number_postal", "retrieved_by_tfidf"]
RETRIEVAL_NUMERIC = ["retrieval_channel_count", "exact_block_size", "best_tfidf_score",
                     "tfidf_rank", "best_cheap_score", "preliminary_score"]
FEATURE_NAMES = [
    "name_clean_exact", "name_core_exact", "name_ratio", "name_token_set_ratio",
    "name_word_jaccard", "name_token_intersection", "name_length_ratio",
    "name_prefix_agreement", "name_suffix_agreement", "address_exact", "address_ratio",
    "address_token_set_ratio", "address_word_jaccard", "address_length_ratio",
    "number_intersection", "number_jaccard", "postal_same", "postal_conflict",
    "postal_missing", "house_same", "house_conflict", "house_missing",
    "country_same", "country_different", "country_missing",
    *PROVENANCE, *RETRIEVAL_NUMERIC,
]


def _jaccard(left: str, right: str) -> tuple[float, int]:
    a = set(left.split())
    b = set(right.split())
    intersection = len(a & b)
    union = len(a | b)
    return (intersection / union if union else 0.0, intersection)


def _ratio(left: str, right: str) -> float:
    return fuzz.ratio(left, right) / 100.0 if left and right else 0.0


class CpuPairFeatureBuilder:
    """Gather normalized rows by integer index, then compute bounded pair batches."""

    def __init__(self, source_ids: dict[str, int], target_ids: dict[str, int]):
        self.source_ids = source_ids
        self.target_ids = target_ids

    def transform(self, candidates: pa.RecordBatch, source_rows: pa.Table,
                  target_rows: pa.Table) -> pa.Table:
        n = candidates.num_rows
        s1_ids = candidates.column(candidates.schema.get_field_index("source1_entity_id")).to_pylist()
        target_ids = candidates.column(candidates.schema.get_field_index("candidate_entity_id")).to_pylist()
        try:
            source_indices = pa.array((self.source_ids[value] for value in s1_ids), type=pa.int32())
            target_indices = pa.array((self.target_ids[value] for value in target_ids), type=pa.int32())
        except KeyError as error:
            raise ValueError(f"candidate refers to unknown entity: {error}") from error
        source = source_rows.take(source_indices)
        target = target_rows.take(target_indices)
        fields = {name: source[name].to_pylist() for name in ROW_COLUMNS[1:]}
        target_fields = {name: target[name].to_pylist() for name in ROW_COLUMNS[1:]}
        values = {name: np.zeros(n, dtype=np.float32) for name in FEATURE_NAMES
                  if name not in PROVENANCE and name not in RETRIEVAL_NUMERIC}
        for i in range(n):
            a_name = fields["name_core"][i] or ""
            b_name = target_fields["name_core"][i] or ""
            a_address = fields["address_clean"][i] or ""
            b_address = target_fields["address_clean"][i] or ""
            a_postal = fields["postal_candidate"][i] or ""
            b_postal = target_fields["postal_candidate"][i] or ""
            a_house = fields["house_number"][i] or ""
            b_house = target_fields["house_number"][i] or ""
            a_country = fields["country_key"][i] or ""
            b_country = target_fields["country_key"][i] or ""
            values["name_clean_exact"][i] = bool(fields["name_clean"][i] and
                                                  fields["name_clean"][i] == target_fields["name_clean"][i])
            values["name_core_exact"][i] = bool(a_name and a_name == b_name)
            values["name_ratio"][i] = _ratio(a_name, b_name)
            values["name_token_set_ratio"][i] = fuzz.token_set_ratio(a_name, b_name) / 100.0 if a_name and b_name else 0.0
            values["name_word_jaccard"][i], values["name_token_intersection"][i] = _jaccard(a_name, b_name)
            values["name_length_ratio"][i] = min(len(a_name), len(b_name)) / max(len(a_name), len(b_name)) if a_name and b_name else 0.0
            values["name_prefix_agreement"][i] = bool(a_name and b_name and a_name[:4] == b_name[:4])
            values["name_suffix_agreement"][i] = bool(a_name and b_name and a_name[-4:] == b_name[-4:])
            values["address_exact"][i] = bool(a_address and a_address == b_address)
            values["address_ratio"][i] = _ratio(a_address, b_address)
            values["address_token_set_ratio"][i] = fuzz.token_set_ratio(a_address, b_address) / 100.0 if a_address and b_address else 0.0
            values["address_word_jaccard"][i], _ = _jaccard(a_address, b_address)
            values["address_length_ratio"][i] = min(len(a_address), len(b_address)) / max(len(a_address), len(b_address)) if a_address and b_address else 0.0
            values["number_jaccard"][i], values["number_intersection"][i] = _jaccard(
                fields["address_numbers"][i] or "", target_fields["address_numbers"][i] or "")
            values["postal_missing"][i] = not bool(a_postal and b_postal)
            values["postal_same"][i] = bool(a_postal and a_postal == b_postal)
            values["postal_conflict"][i] = bool(a_postal and b_postal and a_postal != b_postal)
            values["house_missing"][i] = not bool(a_house and b_house)
            values["house_same"][i] = bool(a_house and a_house == b_house)
            values["house_conflict"][i] = bool(a_house and b_house and a_house != b_house)
            values["country_missing"][i] = not bool(a_country and b_country)
            values["country_same"][i] = bool(a_country and a_country == b_country)
            values["country_different"][i] = bool(a_country and b_country and a_country != b_country)
        output = {"source1_entity_id": s1_ids, "candidate_entity_id": target_ids}
        output.update(values)
        for name in PROVENANCE:
            output[name] = np.asarray(candidates.column(candidates.schema.get_field_index(name)).to_numpy(zero_copy_only=False),
                                      dtype=np.float32)
        for name in RETRIEVAL_NUMERIC:
            array = candidates.column(candidates.schema.get_field_index(name))
            output[name] = np.asarray(pc.fill_null(array, 0).to_numpy(zero_copy_only=False), dtype=np.float32)
        return pa.Table.from_pydict(output).select(["source1_entity_id", "candidate_entity_id", *FEATURE_NAMES])


def _load_rows(normalized_dir: Path, split: str) -> tuple[pa.Table, pa.Table, dict[str, int], dict[str, int]]:
    source = pq.read_table(normalized_dir / f"{split}_s1.parquet", columns=ROW_COLUMNS)
    targets = pa.concat_tables([pq.read_table(normalized_dir / f"{split}_s{source_id}.parquet", columns=ROW_COLUMNS)
                                for source_id in (2, 3)])
    source_ids = {value: i for i, value in enumerate(source["entity_id"].to_pylist())}
    target_ids = {value: i for i, value in enumerate(targets["entity_id"].to_pylist())}
    if len(source_ids) != source.num_rows or len(target_ids) != targets.num_rows:
        raise ValueError("duplicate normalized entity IDs")
    return source, targets, source_ids, target_ids


def build_features(normalized_dir: Path, candidates_dir: Path, output_dir: Path,
                   *, split: str, batch_size: int = 250_000) -> dict:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    started_utc = datetime.now(timezone.utc).isoformat()
    start = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    source, targets, source_ids, target_ids = _load_rows(normalized_dir, split)
    load_seconds = time.perf_counter() - start
    builder = CpuPairFeatureBuilder(source_ids, target_ids)
    processed = 0
    batches = 0
    feature_seconds = 0.0
    for shard_path in sorted(candidates_dir.glob("part-*.parquet")):
        parquet = pq.ParquetFile(shard_path)
        for batch in parquet.iter_batches(batch_size=batch_size):
            started = time.perf_counter()
            transformed = builder.transform(batch, source, targets)
            feature_seconds += time.perf_counter() - started
            part_path = output_dir / f"{shard_path.stem}-batch-{batches:05d}.parquet"
            tmp_path = part_path.with_suffix(".parquet.tmp")
            pq.write_table(transformed, tmp_path, compression="zstd")
            os.replace(tmp_path, part_path)
            processed += batch.num_rows
            batches += 1
    return {"input_pairs": processed, "output_pairs": processed, "batches": batches,
            "source_rows": source.num_rows, "target_rows": targets.num_rows,
            "load_seconds": round(load_seconds, 3), "feature_seconds": round(feature_seconds, 3),
            "elapsed_seconds": round(time.perf_counter() - start, 3),
            "started_utc": started_utc, "ended_utc": datetime.now(timezone.utc).isoformat(),
            "peak_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2, 3),
            "artifact_bytes": sum(path.stat().st_size for path in output_dir.glob("part-*.parquet")),
            "feature_names": FEATURE_NAMES, "actual_threads": 1}


def feature_signature(normalized_dir: Path, candidates_dir: Path, split: str, batch_size: int) -> str:
    normalizers = [json.loads((normalized_dir / f"{split}_s{i}.json").read_text())["signature"]
                   for i in (1, 2, 3)]
    candidate = json.loads((candidates_dir / "manifest.json").read_text())["signature"]
    return hashlib.sha256(json.dumps({"normalizers": normalizers, "candidate": candidate,
                                      "feature_code": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                      "batch_size": batch_size}, sort_keys=True).encode()).hexdigest()
