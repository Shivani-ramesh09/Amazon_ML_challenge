"""Compact postings for exact clean/core names with bounded oversized blocks."""

from __future__ import annotations

from array import array
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import pickle
import resource
import time

import pyarrow as pa
import pyarrow.parquet as pq

INDEX_FIELDS = ("entity_id", "name_clean", "name_core", "address_clean", "postal_candidate", "house_number", "country_key")
QUERY_FIELDS = INDEX_FIELDS
ADDRESS_STOPWORDS = frozenset({"road", "street", "avenue", "main", "near", "nagar", "rue", "de", "des", "the", "and"})
CANDIDATE_SCHEMA = pa.schema([
    ("source1_entity_id", pa.string()), ("candidate_entity_id", pa.string()),
    ("channel", pa.string()), ("exact_block_size", pa.int32()),
])


def _sha256(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def _record_batches(path: Path, columns: tuple[str, ...]):
    parquet = pq.ParquetFile(path)
    missing = set(columns) - set(parquet.schema_arrow.names)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    for batch in parquet.iter_batches(batch_size=50_000, columns=list(columns)):
        values = [batch.column(batch.schema.get_field_index(key)).to_pylist() for key in columns]
        yield zip(*values)


class ExactIndex:
    """Per-key uint32 posting positions with optional secondary indexes for large keys.

    Target row metadata is columnar. Python objects scale with *target rows/unique keys*,
    never with the Cartesian product of query and target blocks.
    """

    def __init__(self, max_block_size: int = 64, per_channel_cap: int = 30):
        if max_block_size < 1 or per_channel_cap < 1:
            raise ValueError("block limits must be positive")
        self.max_block_size = max_block_size
        self.per_channel_cap = per_channel_cap
        self.target_ids: list[str] = []
        self.countries: list[str] = []
        self.postals: list[str] = []
        self.houses: list[str] = []
        self.addresses: list[str] = []
        self.names: dict[str, dict[str, array]] = {"exact_clean": {}, "exact_core": {}}
        self.secondary: dict[str, dict[tuple[str, str, str, str], array]] = {"exact_clean": {}, "exact_core": {}}
        self.stats = {}

    def fit(self, target_paths: list[Path]) -> "ExactIndex":
        start = time.perf_counter()
        seen_ids: set[str] = set()
        for path in target_paths:
            for rows in _record_batches(path, INDEX_FIELDS):
                for entity_id, clean, core, address, postal, house, country in rows:
                    if entity_id in seen_ids:
                        raise ValueError(f"duplicate target ID: {entity_id}")
                    seen_ids.add(entity_id)
                    position = len(self.target_ids)
                    if position >= 2**32:
                        raise ValueError("target index exceeds uint32 capacity")
                    self.target_ids.append(entity_id)
                    self.countries.append(country or "")
                    self.postals.append(postal or "")
                    self.houses.append(house or "")
                    self.addresses.append(address or "")
                    for channel, key in (("exact_clean", clean), ("exact_core", core)):
                        if key:
                            self.names[channel].setdefault(key, array("I")).append(position)
        for channel, postings in self.names.items():
            secondary = self.secondary[channel]
            sizes = Counter()
            oversized_keys = 0
            oversized_records = 0
            for key, positions in postings.items():
                size = len(positions)
                sizes[min(size, 1000)] += 1
                if size <= self.max_block_size:
                    continue
                oversized_keys += 1
                oversized_records += size
                for position in positions:
                    country = self.countries[position]
                    for kind, value in (
                        ("postal", self.postals[position]),
                        ("house", self.houses[position]),
                        ("address", self.addresses[position]),
                    ):
                        if value:
                            secondary.setdefault((key, country, kind, value), array("I")).append(position)
                    for token in set(self.addresses[position].split()):
                        if len(token) >= 4 and token not in ADDRESS_STOPWORDS:
                            secondary.setdefault((key, country, "token", token), array("I")).append(position)
            self.stats[channel] = {
                "unique_keys": len(postings), "oversized_keys": oversized_keys,
                "oversized_target_rows": oversized_records,
                "max_block": max((len(x) for x in postings.values()), default=0),
                "block_size_histogram_clipped_1000": dict(sorted(sizes.items())),
            }
        self.stats["target_rows"] = len(self.target_ids)
        self.stats["fit_seconds"] = round(time.perf_counter() - start, 3)
        self.stats["peak_rss_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2, 3)
        return self

    def _positions(self, channel: str, key: str, country: str, postal: str, house: str, address: str):
        positions = self.names[channel].get(key)
        if positions is None:
            return [], 0, False, False
        original_size = len(positions)
        oversized = original_size > self.max_block_size
        if not oversized:
            selected = positions
        else:
            buckets = []
            for kind, value in (("postal", postal), ("house", house), ("address", address)):
                if value:
                    bucket = self.secondary[channel].get((key, country, kind, value))
                    if bucket:
                        buckets.append(bucket)
            for token in set(address.split()):
                if len(token) >= 4 and token not in ADDRESS_STOPWORDS:
                    bucket = self.secondary[channel].get((key, country, "token", token))
                    if bucket and len(bucket) <= 256:
                        buckets.append(bucket)
            if buckets:
                buckets.sort(key=len)
                selected = array("I")
                used = set()
                for bucket in buckets[:8]:
                    for position in bucket:
                        if position not in used:
                            selected.append(position)
                            used.add(position)
                        if len(selected) >= 256:
                            break
                    if len(selected) >= 256:
                        break
            else:
                # No informative secondary key: prefer same-country rows if possible,
                # scanning only a bounded prefix of a very large original block.
                selected = array("I", (p for p in positions[:256] if self.countries[p] == country))
                if not selected:
                    selected = positions[:256]
        if len(selected) > self.per_channel_cap:
            query_tokens = set(address.split())

            def evidence(position: int):
                target_address = self.addresses[position]
                overlap = len(query_tokens.intersection(target_address.split())) if query_tokens else 0
                score = min(overlap, 6)
                score += 12 * bool(address and address == target_address)
                score += 10 * bool(postal and postal == self.postals[position])
                score += 8 * bool(house and house == self.houses[position])
                score += bool(country and country == self.countries[position])
                return (-score, self.target_ids[position])

            selected = sorted(selected, key=evidence)
        trimmed = len(selected) > self.per_channel_cap or (oversized and len(selected) < original_size)
        if len(selected) > self.per_channel_cap:
            selected = selected[: self.per_channel_cap]
        return selected, original_size, oversized, trimmed

    def retrieve(self, query_path: Path, output_path: Path) -> dict:
        start = time.perf_counter()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_suffix(output_path.suffix + ".tmp")
        writer = pq.ParquetWriter(temporary, CANDIDATE_SCHEMA, compression="zstd")
        columns = {field.name: [] for field in CANDIDATE_SCHEMA}
        counts = Counter()

        def flush():
            if not columns["source1_entity_id"]:
                return
            table = pa.Table.from_pydict(columns, schema=CANDIDATE_SCHEMA)
            writer.write_table(table, row_group_size=100_000)
            for column in columns.values():
                column.clear()

        try:
            for rows in _record_batches(query_path, QUERY_FIELDS):
                for s1_id, clean, core, address, postal, house, country in rows:
                    counts["query_rows"] += 1
                    for channel, key in (("exact_clean", clean), ("exact_core", core)):
                        if not key:
                            continue
                        positions, size, oversized, trimmed = self._positions(channel, key, country, postal, house, address)
                        if oversized:
                            counts[f"{channel}_oversized_queries"] += 1
                        if trimmed:
                            counts[f"{channel}_trimmed_queries"] += 1
                            counts[f"{channel}_dropped_raw_hits"] += size - len(positions)
                        for position in positions:
                            columns["source1_entity_id"].append(s1_id)
                            columns["candidate_entity_id"].append(self.target_ids[position])
                            columns["channel"].append(channel)
                            columns["exact_block_size"].append(size)
                            counts[f"{channel}_pairs"] += 1
                        if len(columns["source1_entity_id"]) >= 100_000:
                            flush()
            flush()
        except Exception:
            writer.close()
            temporary.unlink(missing_ok=True)
            raise
        writer.close()
        os.replace(temporary, output_path)
        return {
            **dict(counts), "elapsed_seconds": round(time.perf_counter() - start, 3),
            "peak_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2, 3),
            "artifact_bytes": output_path.stat().st_size, "actual_threads": 1,
        }


def build_and_retrieve(target_paths: list[Path], query_path: Path, index_path: Path, candidates_path: Path,
                       max_block_size: int = 64, per_channel_cap: int = 30) -> dict:
    """Cache index and candidate outputs by exact normalized input/code/settings signature."""
    index_signature = hashlib.sha256(json.dumps({
        "target_hashes": [_sha256(path) for path in target_paths],
        "code_hash": _sha256(Path(__file__)),
        "max_block_size": max_block_size, "per_channel_cap": per_channel_cap,
    }, sort_keys=True).encode()).hexdigest()
    signature = hashlib.sha256((index_signature + _sha256(query_path)).encode()).hexdigest()
    manifest_path = candidates_path.with_suffix(".json")
    index_manifest_path = index_path.with_suffix(".json")
    if candidates_path.is_file() and manifest_path.is_file():
        cached = json.loads(manifest_path.read_text())
        if cached.get("signature") == signature:
            return {"cache_hit": True, **cached}
    start = time.perf_counter()
    reuse_index = False
    if index_path.is_file() and index_manifest_path.is_file():
        saved = json.loads(index_manifest_path.read_text())
        if saved.get("signature") == index_signature:
            with index_path.open("rb") as file:
                index = pickle.load(file)
            reuse_index = True
    if not reuse_index:
        index = ExactIndex(max_block_size, per_channel_cap).fit(target_paths)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_index = index_path.with_suffix(index_path.suffix + ".tmp")
        with temporary_index.open("wb") as file:
            pickle.dump(index, file, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary_index, index_path)
        temporary_index_manifest = index_manifest_path.with_suffix(".json.tmp")
        temporary_index_manifest.write_text(json.dumps({"signature": index_signature, "stats": index.stats}, indent=2, sort_keys=True) + "\n")
        os.replace(temporary_index_manifest, index_manifest_path)
    retrieval = index.retrieve(query_path, candidates_path)
    report = {
        "signature": signature, "index": index.stats, "retrieval": retrieval,
        "index_cache_hit": reuse_index,
        "total_elapsed_seconds": round(time.perf_counter() - start, 3),
        "index_bytes": index_path.stat().st_size,
    }
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary_manifest, manifest_path)
    return report
