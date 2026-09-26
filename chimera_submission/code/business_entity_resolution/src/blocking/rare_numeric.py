"""Bounded rare-name-token and name+postal/house retrieval on CPU."""

from __future__ import annotations

from array import array
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import resource
import time

import pyarrow as pa
import pyarrow.parquet as pq

FIELDS = ("entity_id", "name_core", "address_clean", "postal_candidate", "house_number", "country_key")
SCHEMA = pa.schema([
    ("source1_entity_id", pa.string()), ("candidate_entity_id", pa.string()),
    ("channel", pa.string()), ("blocking_key_size", pa.int32()),
    ("retrieval_score", pa.float32()),
])
STOPWORDS = frozenset({"the", "and", "for", "of", "in", "at", "limited", "company", "private", "ltd", "inc"})


def _sha256(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def _rows(path: Path):
    parquet = pq.ParquetFile(path)
    missing = set(FIELDS) - set(parquet.schema_arrow.names)
    if missing:
        raise ValueError(f"{path}: missing {sorted(missing)}")
    for batch in parquet.iter_batches(batch_size=50_000, columns=list(FIELDS)):
        arrays = [batch.column(i).to_pylist() for i in range(len(FIELDS))]
        yield from zip(*arrays)


def tokens(name: str) -> set[str]:
    return {token for token in name.split() if len(token) >= 3 and not token.isdigit() and token not in STOPWORDS}


class RareNumericIndex:
    def __init__(self, rare_max_df: int = 64, composite_max_df: int = 5000,
                 max_postings_per_key: int = 128, rare_cap: int = 15, numeric_cap: int = 10):
        if min(rare_max_df, composite_max_df, max_postings_per_key, rare_cap, numeric_cap) < 1:
            raise ValueError("all limits must be positive")
        self.rare_max_df = rare_max_df
        self.composite_max_df = composite_max_df
        self.max_postings_per_key = max_postings_per_key
        self.rare_cap = rare_cap
        self.numeric_cap = numeric_cap
        self.df = Counter()
        self.target_ids: list[str] = []
        self.names: list[str] = []
        self.addresses: list[str] = []
        self.postals: list[str] = []
        self.houses: list[str] = []
        self.countries: list[str] = []
        self.rare: dict[str, array] = {}
        self.composite: dict[tuple[str, str, str, str], array] = {}
        self.stats = {}

    def fit(self, target_paths: list[Path]) -> "RareNumericIndex":
        start = time.perf_counter()
        for path in target_paths:
            for _, name, *_ in _rows(path):
                self.df.update(tokens(name or ""))
        for path in target_paths:
            for entity_id, name, address, postal, house, country in _rows(path):
                position = len(self.target_ids)
                if position >= 2**32:
                    raise ValueError("target position exceeds uint32")
                self.target_ids.append(entity_id)
                self.names.append(name or "")
                self.addresses.append(address or "")
                self.postals.append(postal or "")
                self.houses.append(house or "")
                self.countries.append(country or "")
                ranked = sorted(tokens(name or ""), key=lambda token: (self.df[token], token))
                for token in ranked[:3]:
                    if self.df[token] <= self.rare_max_df:
                        self.rare.setdefault(token, array("I")).append(position)
                for token in ranked[:2]:
                    if self.df[token] > self.composite_max_df:
                        continue
                    for kind, value in (("postal", postal), ("house", house)):
                        if value:
                            self.composite.setdefault((country or "", kind, value, token), array("I")).append(position)
        self.stats = {
            "target_rows": len(self.target_ids), "token_vocabulary": len(self.df),
            "rare_keys": len(self.rare), "composite_keys": len(self.composite),
            "rare_max_posting": max((len(x) for x in self.rare.values()), default=0),
            "composite_max_posting": max((len(x) for x in self.composite.values()), default=0),
            "rare_posting_histogram_clipped_128": dict(Counter(min(len(x), 128) for x in self.rare.values())),
            "composite_posting_histogram_clipped_128": dict(Counter(min(len(x), 128) for x in self.composite.values())),
            "oversized_composite_keys": sum(len(x) > self.max_postings_per_key for x in self.composite.values()),
            "fit_seconds": round(time.perf_counter() - start, 3),
            "peak_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2, 3),
        }
        return self

    def _retrieve_one(self, name: str, address: str, postal: str, house: str, country: str):
        ranked = sorted(tokens(name or ""), key=lambda token: (self.df.get(token, 0) or 10**9, token))
        channel_scores: dict[str, dict[int, float]] = {"rare_token": {}, "numeric_postal": {}}
        sizes: dict[str, dict[int, int]] = {"rare_token": {}, "numeric_postal": {}}
        scanned = Counter()
        for token in ranked[:3]:
            postings = self.rare.get(token)
            if not postings:
                continue
            weight = math.log1p(len(self.target_ids) / max(1, self.df[token]))
            for position in postings:
                channel_scores["rare_token"][position] = channel_scores["rare_token"].get(position, 0.0) + weight
                sizes["rare_token"][position] = min(sizes["rare_token"].get(position, len(postings)), len(postings))
                scanned["rare_postings"] += 1
        for token in ranked[:2]:
            if self.df.get(token, 10**9) > self.composite_max_df:
                continue
            for kind, value in (("postal", postal), ("house", house)):
                if not value:
                    continue
                postings = self.composite.get((country or "", kind, value, token))
                if not postings:
                    continue
                if len(postings) > self.max_postings_per_key:
                    scanned["oversized_composite_lookups"] += 1
                for position in postings[: self.max_postings_per_key]:
                    weight = math.log1p(len(self.target_ids) / max(1, self.df[token]))
                    weight += 3.0 if kind == "postal" else 2.0
                    channel_scores["numeric_postal"][position] = max(channel_scores["numeric_postal"].get(position, 0.0), weight)
                    sizes["numeric_postal"][position] = min(sizes["numeric_postal"].get(position, len(postings)), len(postings))
                    scanned["composite_postings"] += 1
                    scanned[f"{kind}_postings"] += 1
        query_tokens = tokens(name or "")
        query_address_tokens = set((address or "").split())

        def ranked_hits(channel: str, limit: int):
            scores = channel_scores[channel]
            if not scores:
                return []

            def rank(position: int):
                shared = len(query_tokens.intersection(tokens(self.names[position])))
                address_overlap = len(query_address_tokens.intersection(self.addresses[position].split()))
                evidence = scores[position] + min(shared, 4) + min(address_overlap, 4)
                evidence += 8 * bool(address and address == self.addresses[position])
                evidence += 5 * bool(postal and postal == self.postals[position])
                evidence += 4 * bool(house and house == self.houses[position])
                return (-evidence, self.target_ids[position])

            ordered = sorted(scores, key=rank)
            return [(position, sizes[channel][position], round(-rank(position)[0], 5)) for position in ordered[:limit]]

        return ranked_hits("rare_token", self.rare_cap), ranked_hits("numeric_postal", self.numeric_cap), scanned

    def retrieve(self, query_path: Path, output_path: Path) -> dict:
        start = time.perf_counter()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_suffix(output_path.suffix + ".tmp")
        writer = pq.ParquetWriter(temporary, SCHEMA, compression="zstd")
        columns = {field.name: [] for field in SCHEMA}
        counts = Counter()

        def flush():
            if columns["source1_entity_id"]:
                writer.write_table(pa.Table.from_pydict(columns, schema=SCHEMA), row_group_size=100_000)
                for values in columns.values():
                    values.clear()

        try:
            for s1_id, name, address, postal, house, country in _rows(query_path):
                counts["query_rows"] += 1
                rare, numeric, scanned = self._retrieve_one(name, address, postal, house, country)
                counts.update(scanned)
                for channel, hits in (("rare_token", rare), ("numeric_postal", numeric)):
                    for position, size, score in hits:
                        columns["source1_entity_id"].append(s1_id)
                        columns["candidate_entity_id"].append(self.target_ids[position])
                        columns["channel"].append(channel)
                        columns["blocking_key_size"].append(size)
                        columns["retrieval_score"].append(score)
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
        return {**dict(counts), "elapsed_seconds": round(time.perf_counter() - start, 3),
                "peak_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2, 3),
                "artifact_bytes": output_path.stat().st_size, "actual_threads": 1}


def build_and_retrieve(target_paths: list[Path], query_path: Path, index_path: Path, candidates_path: Path,
                       **limits) -> dict:
    index_signature = hashlib.sha256(json.dumps({
        "target_hashes": [_sha256(path) for path in target_paths], "code_hash": _sha256(Path(__file__)),
        "limits": limits,
    }, sort_keys=True).encode()).hexdigest()
    signature = hashlib.sha256((index_signature + _sha256(query_path)).encode()).hexdigest()
    manifest = candidates_path.with_suffix(".json")
    index_manifest = index_path.with_suffix(".json")
    if candidates_path.is_file() and manifest.is_file():
        report = json.loads(manifest.read_text())
        if report.get("signature") == signature:
            return {"cache_hit": True, **report}
    start = time.perf_counter()
    reuse_index = False
    if index_path.is_file() and index_manifest.is_file():
        saved = json.loads(index_manifest.read_text())
        if saved.get("signature") == index_signature:
            with index_path.open("rb") as file:
                index = pickle.load(file)
            reuse_index = True
    if not reuse_index:
        index = RareNumericIndex(**limits).fit(target_paths)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = index_path.with_suffix(".pkl.tmp")
        with tmp.open("wb") as file:
            pickle.dump(index, file, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, index_path)
        tmp_manifest = index_manifest.with_suffix(".json.tmp")
        tmp_manifest.write_text(json.dumps({"signature": index_signature, "stats": index.stats}, sort_keys=True) + "\n")
        os.replace(tmp_manifest, index_manifest)
    retrieval = index.retrieve(query_path, candidates_path)
    report = {"signature": signature, "index": index.stats, "index_cache_hit": reuse_index,
              "retrieval": retrieval, "total_elapsed_seconds": round(time.perf_counter() - start, 3),
              "index_bytes": index_path.stat().st_size}
    tmp_manifest = manifest.with_suffix(".json.tmp")
    tmp_manifest.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(tmp_manifest, manifest)
    return report
