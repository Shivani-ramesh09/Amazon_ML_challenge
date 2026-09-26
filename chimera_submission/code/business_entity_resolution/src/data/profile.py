"""Bounded TSV contract/profile scan; no third-party dependencies.

Phase 1 deliberately streams source rows. Later phases may convert validated data to
Polars/Arrow partitions; this scan remains a lightweight integrity gate.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import time

from chimera_submission.code.business_entity_resolution.src.config import load_config

SOURCE_HEADER = ("entity_id", "business_name", "business_address", "country")
TRUTH_HEADER = ("source1_entity_id", "matched_entity_ids")
LENGTH_BINS = (0, 10, 20, 40, 80, 160, 320, 640, 1280)
SOURCE_CODES = {"S1": 1, "S2": 2, "S3": 3}


def split_bucket(entity_id: str, seed: int = 42, validation_percent: int = 15) -> str:
    """Stable S1-level split, independent of Python hash seed and input order."""
    if not 0 <= validation_percent <= 100:
        raise ValueError("validation_percent must be in [0, 100]")
    digest = hashlib.blake2b(
        entity_id.encode("utf-8"), key=seed.to_bytes(8, "little", signed=False), digest_size=8
    ).digest()
    return "validation" if int.from_bytes(digest, "little") % 100 < validation_percent else "train"


def encoded_id(raw: str, expected_prefix: str) -> int:
    """Compact unique source-qualified integer ID; reject malformed IDs."""
    prefix, separator, suffix = raw.partition("-")
    if separator != "-" or prefix != expected_prefix or not suffix.isascii() or not suffix.isdecimal():
        raise ValueError(f"invalid {expected_prefix} ID: {raw!r}")
    return (int(suffix) << 2) | SOURCE_CODES[prefix]


def _rows(path: Path, header: tuple[str, ...]):
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.reader(file, delimiter="\t", quoting=csv.QUOTE_NONE)
        actual = tuple(next(reader, ()))
        if actual != header:
            raise ValueError(f"{path}: header {actual!r}, expected {header!r}")
        for line_number, row in enumerate(reader, start=2):
            if len(row) != len(header):
                raise ValueError(f"{path}:{line_number}: expected {len(header)} columns, got {len(row)}")
            yield row


def _length_bucket(length: int) -> str:
    for edge in LENGTH_BINS:
        if length <= edge:
            return f"<={edge}"
    return ">1280"


def _rss_gb() -> float:
    # Linux reports KiB. The target server and development environment are Linux.
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2, 3)


def _file_manifest(path: Path) -> dict:
    stat = path.stat()
    with path.open("rb") as file:
        sha256 = hashlib.file_digest(file, "sha256").hexdigest()
    return {"path": str(path.resolve()), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": sha256}


def _bucket_quantiles(counts: Counter, total: int) -> dict:
    """Report conservative upper bounds; exact lengths are not retained in RAM."""
    order = [f"<={edge}" for edge in LENGTH_BINS] + [">1280"]
    result = {}
    for label, fraction in (("p50_upper", 0.5), ("p90_upper", 0.9), ("p99_upper", 0.99)):
        cumulative = 0
        for bucket in order:
            cumulative += counts[bucket]
            if cumulative >= total * fraction:
                result[label] = bucket
                break
    return result


def profile_source(path: Path, prefix: str, *, sample_modulus: int = 128) -> tuple[dict, set[int]]:
    if sample_modulus < 1:
        raise ValueError("sample_modulus must be positive")
    start = time.perf_counter()
    seen: set[int] = set()
    missing = Counter()
    countries = Counter()
    name_lengths = Counter()
    address_lengths = Counter()
    sampled_names = Counter()
    rows = 0
    for entity_id, name, address, country in _rows(path, SOURCE_HEADER):
        encoded = encoded_id(entity_id, prefix)
        if encoded in seen:
            raise ValueError(f"{path}: duplicate entity_id {entity_id!r}")
        seen.add(encoded)
        rows += 1
        missing.update(k for k, value in (("business_name", name), ("business_address", address), ("country", country)) if value == "")
        countries[country or "<missing>"] += 1
        name_lengths[_length_bucket(len(name))] += 1
        address_lengths[_length_bucket(len(address))] += 1
        if (encoded >> 2) % sample_modulus == 0:
            sampled_names[name.casefold().strip()] += 1
    elapsed = time.perf_counter() - start
    result = {
        "input": _file_manifest(path), "rows": rows, "missing_fields": dict(missing),
        "countries": dict(countries), "name_length_bins": dict(name_lengths),
        "address_length_bins": dict(address_lengths),
        "name_length_quantile_bounds": _bucket_quantiles(name_lengths, rows),
        "address_length_quantile_bounds": _bucket_quantiles(address_lengths, rows),
        "name_frequency_sample_modulus": sample_modulus,
        "sampled_name_top20": sampled_names.most_common(20),
        "elapsed_seconds": round(elapsed, 3), "rows_per_second": round(rows / elapsed, 1),
        "peak_rss_gb": _rss_gb(),
    }
    return result, seen


def profile_truth(path: Path, s1_ids: set[int], target_ids: set[int], *, seed: int = 42, validation_percent: int = 15) -> dict:
    start = time.perf_counter()
    seen: set[int] = set()
    cardinality = Counter()
    split_counts = Counter()
    rows = matches = 0
    for source_id, raw_matches in _rows(path, TRUTH_HEADER):
        encoded = encoded_id(source_id, "S1")
        if encoded not in s1_ids or encoded in seen:
            raise ValueError(f"{path}: unknown or duplicate truth S1 {source_id!r}")
        seen.add(encoded)
        targets = raw_matches.split(",") if raw_matches else []
        if len(targets) != len(set(targets)):
            raise ValueError(f"{path}: repeated target for {source_id!r}")
        for target in targets:
            prefix = target.partition("-")[0]
            if prefix not in ("S2", "S3") or encoded_id(target, prefix) not in target_ids:
                raise ValueError(f"{path}: invalid truth target {target!r} for {source_id!r}")
        rows += 1
        matches += len(targets)
        cardinality[len(targets)] += 1
        split_counts[split_bucket(source_id, seed, validation_percent)] += 1
    if seen != s1_ids:
        raise ValueError(f"{path}: missing {len(s1_ids - seen)} S1 truth rows")
    elapsed = time.perf_counter() - start
    return {
        "input": _file_manifest(path), "rows": rows, "ground_truth_pairs": matches,
        "cardinality": {str(k): v for k, v in sorted(cardinality.items())},
        "split_counts": dict(split_counts), "split_seed": seed, "validation_percent": validation_percent,
        "elapsed_seconds": round(elapsed, 3), "rows_per_second": round(rows / elapsed, 1),
        "peak_rss_gb": _rss_gb(),
    }


def profile_dataset(data_dir: Path, split: str, *, config: dict | None = None) -> dict:
    """Profile a complete train or test directory; release ID sets between splits."""
    start = time.perf_counter()
    reports = {}
    target_ids: set[int] = set()
    s1_ids: set[int] = set()
    for source_number in (1, 2, 3):
        prefix = f"S{source_number}"
        sample_modulus = config["profile_frequency_sample_modulus"] if config else 128
        report, ids = profile_source(data_dir / f"{split}_source{source_number}.tsv", prefix, sample_modulus=sample_modulus)
        reports[prefix] = report
        if source_number == 1:
            s1_ids = ids
        elif split == "train":
            target_ids.update(ids)
    if split == "train":
        seed = config["split_seed"] if config else 42
        validation_percent = config["validation_percent"] if config else 15
        reports["truth"] = profile_truth(data_dir / "train_ground_truth.tsv", s1_ids, target_ids, seed=seed, validation_percent=validation_percent)
    reports["total_elapsed_seconds"] = round(time.perf_counter() - start, 3)
    reports["peak_rss_gb"] = _rss_gb()
    return reports


def main() -> int:
    parser = argparse.ArgumentParser(description="Profile and validate source TSVs")
    parser.add_argument("--data-dir", type=Path, default=Path("dataset/student_resource/dataset"))
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/profile"))
    parser.add_argument("--config", type=Path, default=Path("chimera_submission/code/business_entity_resolution/configs/dev.yaml"))
    args = parser.parse_args()
    config = load_config(args.config)
    data_dir = args.data_dir if args.data_dir != Path("dataset/student_resource/dataset") else Path(config["data_dir"])
    report = profile_dataset(data_dir / args.split, args.split, config=config)
    config_bytes = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    report["profile_config"] = {
        "configured_threads": config["threads"], "actual_scan_threads": 1,
        "frequency_sample_modulus": config["profile_frequency_sample_modulus"],
        "config_path": str(args.config.resolve()), "sha256": hashlib.sha256(config_bytes).hexdigest(),
    }
    report["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    report["git_dirty"] = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"{args.split}.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.split == "train":
        split_manifest = {
            "algorithm": "blake2b-64 over UTF-8 S1 ID keyed by little-endian 8-byte seed; modulo 100",
            "seed": config["split_seed"], "validation_percent": config["validation_percent"],
            "counts": report["truth"]["split_counts"], "input_sha256": report["S1"]["input"]["sha256"],
        }
        (args.output_dir / "split_manifest.json").write_text(json.dumps(split_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(output), "report_bytes": output.stat().st_size, "rows": {k: v["rows"] for k, v in report.items() if isinstance(v, dict) and "rows" in v}, "elapsed_seconds": report["total_elapsed_seconds"], "peak_rss_gb": report["peak_rss_gb"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
