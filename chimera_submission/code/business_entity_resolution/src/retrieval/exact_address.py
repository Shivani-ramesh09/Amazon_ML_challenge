"""Frequency-aware exact-address retrieval for measured name-retrieval misses."""

from __future__ import annotations

from array import array
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import pickle
import resource
import time

import pyarrow as pa
import pyarrow.parquet as pq
from rapidfuzz import fuzz

from chimera_submission.code.business_entity_resolution.src.retrieval.interface import CandidateRetriever

ROW_FIELDS = ("entity_id", "address_clean", "name_core", "house_number", "postal_candidate", "country_key")
CANDIDATE_SCHEMA = pa.schema([
    ("source1_entity_id", pa.string()), ("candidate_entity_id", pa.string()),
    ("channel", pa.string()), ("exact_block_size", pa.int32()),
])


def _hash(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def _rows(path: Path):
    for batch in pq.ParquetFile(path).iter_batches(batch_size=50_000, columns=list(ROW_FIELDS)):
        arrays = [batch.column(i).to_pylist() for i in range(len(ROW_FIELDS))]
        yield from zip(*arrays)


class CpuExactAddressRetriever(CandidateRetriever):
    def __init__(self, *, max_block_size: int = 64, per_query_cap: int = 30,
                 min_address_chars: int = 8):
        if min(max_block_size, per_query_cap, min_address_chars) < 1:
            raise ValueError("block limits must be positive")
        self.max_block_size = max_block_size
        self.per_query_cap = per_query_cap
        self.min_address_chars = min_address_chars
        self.postings: dict[tuple[str, str], array] = {}
        self.target_ids: list[str] = []
        self.names: list[str] = []
        self.houses: list[str] = []
        self.postals: list[str] = []
        self.fit_stats: dict = {}

    def fit(self, target_paths: list[Path]) -> "CpuExactAddressRetriever":
        start = time.perf_counter()
        sizes = Counter()
        for path in target_paths:
            for target_id, address, name, house, postal, country in _rows(path):
                position = len(self.target_ids)
                self.target_ids.append(target_id)
                self.names.append(name or "")
                self.houses.append(house or "")
                self.postals.append(postal or "")
                if address and len(address) >= self.min_address_chars:
                    self.postings.setdefault((country or "", address), array("I")).append(position)
        for positions in self.postings.values():
            sizes[min(len(positions), 1000)] += 1
        self.fit_stats = {
            "target_rows": len(self.target_ids), "unique_address_keys": len(self.postings),
            "max_block": max((len(x) for x in self.postings.values()), default=0),
            "oversized_keys": sum(1 for positions in self.postings.values()
                                  if len(positions) > self.max_block_size),
            "block_size_histogram_clipped_1000": dict(sorted(sizes.items())),
            "fit_seconds": round(time.perf_counter() - start, 3),
            "peak_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2, 3),
        }
        return self

    def retrieve(self, query_path: Path, output_path: Path, *, skip_s1_ids: set[str] | None = None) -> dict:
        start_utc = datetime.now(timezone.utc).isoformat()
        start = time.perf_counter()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_suffix(".parquet.tmp")
        writer = pq.ParquetWriter(temporary, CANDIDATE_SCHEMA, compression="zstd")
        columns = {name: [] for name in CANDIDATE_SCHEMA.names}
        counts = Counter()

        def flush():
            if columns["source1_entity_id"]:
                writer.write_table(pa.Table.from_pydict(columns, schema=CANDIDATE_SCHEMA))
                for values in columns.values():
                    values.clear()

        try:
            for s1_id, address, name, house, postal, country in _rows(query_path):
                counts["query_rows"] += 1
                if skip_s1_ids and s1_id in skip_s1_ids:
                    counts["skipped_queries"] += 1
                    continue
                if not address or len(address) < self.min_address_chars:
                    continue
                positions = self.postings.get((country or "", address))
                if not positions:
                    continue
                original_size = len(positions)
                selected = positions
                if original_size > self.max_block_size:
                    counts["oversized_queries"] += 1
                    # Exact address alone is ambiguous on common/generic addresses.
                    selected = array("I", (position for position in positions
                                           if (house and self.houses[position] == house) or
                                              (postal and self.postals[position] == postal)))
                    if not selected or len(selected) > self.max_block_size:
                        counts["unresolved_oversized_queries"] += 1
                        counts["oversized_raw_pairs_dropped"] += original_size
                        continue
                if len(selected) > self.per_query_cap:
                    selected = sorted(selected, key=lambda position: (
                        -fuzz.ratio(name or "", self.names[position]), self.target_ids[position]))[
                            :self.per_query_cap]
                    counts["trimmed_queries"] += 1
                counts["raw_hits_dropped"] += original_size - len(selected)
                for position in selected:
                    columns["source1_entity_id"].append(s1_id)
                    columns["candidate_entity_id"].append(self.target_ids[position])
                    columns["channel"].append("exact_address")
                    columns["exact_block_size"].append(original_size)
                    counts["candidate_pairs"] += 1
                if len(columns["source1_entity_id"]) >= 100_000:
                    flush()
            flush()
        except Exception:
            writer.close()
            temporary.unlink(missing_ok=True)
            raise
        writer.close()
        os.replace(temporary, output_path)
        return {**dict(counts), "elapsed_seconds": round(time.perf_counter() - start, 3),
                "started_utc": start_utc, "ended_utc": datetime.now(timezone.utc).isoformat(),
                "peak_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2, 3),
                "artifact_bytes": output_path.stat().st_size, "actual_threads": 1}


def build_and_retrieve(target_paths: list[Path], query_path: Path, index_path: Path,
                       candidate_path: Path, *, max_block_size: int = 64,
                       per_query_cap: int = 30, min_address_chars: int = 8) -> dict:
    index_signature = hashlib.sha256(json.dumps({
        "target_hashes": [_hash(path) for path in target_paths],
        "code": _hash(Path(__file__)), "max_block_size": max_block_size,
        "per_query_cap": per_query_cap, "min_address_chars": min_address_chars,
    }, sort_keys=True).encode()).hexdigest()
    index_manifest = index_path.with_suffix(".json")
    index_hit = False
    if index_path.is_file() and index_manifest.is_file():
        saved = json.loads(index_manifest.read_text())
        index_hit = saved.get("signature") == index_signature
    if index_hit:
        with index_path.open("rb") as file:
            retriever = pickle.load(file)
    else:
        retriever = CpuExactAddressRetriever(max_block_size=max_block_size,
                                             per_query_cap=per_query_cap,
                                             min_address_chars=min_address_chars).fit(target_paths)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = index_path.with_suffix(".pkl.tmp")
        with temporary.open("wb") as file:
            pickle.dump(retriever, file, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary, index_path)
        index_manifest.write_text(json.dumps({"signature": index_signature,
                                              "stats": retriever.fit_stats}, indent=2, sort_keys=True) + "\n")
    candidate_signature = hashlib.sha256(json.dumps({"index": index_signature,
                                                     "query": _hash(query_path)},
                                                    sort_keys=True).encode()).hexdigest()
    candidate_manifest = candidate_path.with_suffix(".json")
    if candidate_path.is_file() and candidate_manifest.is_file():
        saved = json.loads(candidate_manifest.read_text())
        if saved.get("signature") == candidate_signature:
            return {"cache_hit": True, **saved}
    retrieval = retriever.retrieve(query_path, candidate_path)
    report = {"signature": candidate_signature, "index_cache_hit": index_hit,
              "index": retriever.fit_stats, "retrieval": retrieval,
              "index_bytes": index_path.stat().st_size}
    candidate_manifest.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report
