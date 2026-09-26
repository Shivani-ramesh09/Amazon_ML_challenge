"""Bounded, shardwise blocking evaluation against S1 truth sets."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import time

import numpy as np
import polars as pl
import pyarrow.parquet as pq

from chimera_submission.code.business_entity_resolution.src.retrieval.union import _s1_bucket

CHANNELS = ("exact_clean", "exact_core", "rare_token", "numeric_postal", "tfidf_name")


def _metric(counts: np.ndarray, hits: np.ndarray, truth: np.ndarray) -> dict:
    positive = truth > 0
    hit_positive = hits[positive]
    truth_positive = truth[positive]
    oracle = np.ones(len(truth), dtype=np.float64)
    # Perfectly reject unretrieved false pairs: TP=hits, FP=0, FN=truth-hits.
    # F0.5 = 1.25*TP / (1.25*TP + FP + 0.25*FN).
    oracle[positive] = 1.25 * hit_positive / (hit_positive + 0.25 * truth_positive)
    return {
        "candidate_pairs": int(counts.sum()),
        "mean_candidates": float(counts.mean()),
        "median_candidates": float(np.median(counts)),
        "p90_candidates": float(np.percentile(counts, 90)),
        "p95_candidates": float(np.percentile(counts, 95)),
        "p99_candidates": float(np.percentile(counts, 99)),
        "max_candidates": int(counts.max(initial=0)),
        "zero_candidate_entities": int(np.count_nonzero(counts == 0)),
        "gt_pairs_recovered": int(hits.sum()),
        "pair_candidate_recall": float(hits.sum() / truth.sum()) if truth.sum() else None,
        "entity_any_hit_recall": float(np.count_nonzero(hit_positive > 0) / len(hit_positive)) if len(hit_positive) else None,
        "entity_complete_recall": float(np.count_nonzero(hit_positive == truth_positive) / len(hit_positive)) if len(hit_positive) else None,
        "conditional_oracle_macro_f0_5": float(oracle.mean()),
        "conditional_oracle_f0_5_eligible_positive": float(oracle[positive].mean()) if np.any(positive) else None,
    }


def _add_group_counts(frame: pl.DataFrame, arrays: dict[str, np.ndarray], id_to_index: dict[str, int],
                      *, channel_column: bool = False) -> None:
    if frame.is_empty():
        return
    group_keys = ["source1_entity_id", "channel"] if channel_column else ["source1_entity_id"]
    for row in frame.group_by(group_keys).len().iter_rows(named=True):
        key = row["channel"] if channel_column else "union"
        arrays[key][id_to_index[row["source1_entity_id"]]] += row["len"]


def _add_pair_counts(frame: pl.DataFrame, arrays: dict[str, np.ndarray], id_to_index: dict[str, int], key: str) -> None:
    if frame.is_empty():
        return
    for s1_id, count in frame.group_by("source1_entity_id").len().iter_rows():
        arrays[key][id_to_index[s1_id]] += count


def _truth(normalized_dir: Path, truth_path: Path, shards: int, *, sampled_targets: bool) -> tuple[list[str], list[str], np.ndarray, np.ndarray, dict[int, list[tuple[str, str]]]]:
    queries = pl.read_parquet(normalized_dir / "train_s1.parquet", columns=["entity_id", "country"])
    ids = queries["entity_id"].to_list()
    countries = queries["country"].to_list()
    target_ids: set[str] | None = None
    if sampled_targets:
        target_ids = set()
        for source in (2, 3):
            target_ids.update(pl.read_parquet(normalized_dir / f"train_s{source}.parquet", columns=["entity_id"])["entity_id"])
    id_to_index = {s1: i for i, s1 in enumerate(ids)}
    eligible = np.zeros(len(ids), dtype=np.int32)
    original = np.zeros(len(ids), dtype=np.int32)
    by_shard: dict[int, list[tuple[str, str]]] = {part: [] for part in range(shards)}
    with truth_path.open(encoding="utf-8", newline="") as file:
        reader = csv.reader(file, delimiter="\t", quoting=csv.QUOTE_NONE)
        if next(reader) != ["source1_entity_id", "matched_entity_ids"]:
            raise ValueError("unexpected truth schema")
        for s1_id, raw_targets in reader:
            index = id_to_index.get(s1_id)
            if index is None:
                continue
            targets = raw_targets.split(",") if raw_targets else []
            original[index] = len(targets)
            bucket = _s1_bucket(s1_id, shards)
            for target in targets:
                if target_ids is None or target in target_ids:
                    by_shard[bucket].append((s1_id, target))
                    eligible[index] += 1
    return ids, countries, eligible, original, by_shard


def evaluate_blocking(normalized_dir: Path, truth_path: Path, raw_dir: Path,
                      final_dirs: dict[int, Path], *, shards: int = 64,
                      sampled_targets: bool = True) -> dict:
    started_utc = datetime.now(timezone.utc).isoformat()
    start = time.perf_counter()
    ids, countries, gt_counts, original_counts, truth_by_shard = _truth(
        normalized_dir, truth_path, shards, sampled_targets=sampled_targets)
    id_to_index = {s1: i for i, s1 in enumerate(ids)}
    keys = (*CHANNELS, "union", *(f"cap_{cap}" for cap in final_dirs))
    counts = {key: np.zeros(len(ids), dtype=np.int32) for key in keys}
    hits = {key: np.zeros(len(ids), dtype=np.int32) for key in keys}
    exclusive = Counter()
    marginal = Counter()
    raw_pairs = 0
    for bucket in range(shards):
        path = raw_dir / f"part-{bucket:03d}.parquet"
        gt_rows = truth_by_shard[bucket]
        if gt_rows:
            gt = pl.DataFrame(gt_rows, schema=["source1_entity_id", "candidate_entity_id"], orient="row")
        else:
            gt = pl.DataFrame(schema={"source1_entity_id": pl.String, "candidate_entity_id": pl.String})
        if path.is_file():
            raw = pl.read_parquet(path, columns=["source1_entity_id", "candidate_entity_id", "channel"])
            raw_pairs += raw.height
            raw = raw.unique(["source1_entity_id", "candidate_entity_id", "channel"])
            _add_group_counts(raw, counts, id_to_index, channel_column=True)
            pair_keys = ["source1_entity_id", "candidate_entity_id"]
            unique = raw.unique(pair_keys)
            _add_pair_counts(unique, counts, id_to_index, "union")
            recovered = raw.join(gt, on=pair_keys, how="inner")
            _add_group_counts(recovered, hits, id_to_index, channel_column=True)
            unique_hits = recovered.unique(pair_keys)
            _add_pair_counts(unique_hits, hits, id_to_index, "union")
            for row in recovered.group_by(pair_keys).agg(pl.col("channel")).iter_rows(named=True):
                channels = set(row["channel"])
                if len(channels) == 1:
                    exclusive[next(iter(channels))] += 1
                for channel in CHANNELS:
                    if channel in channels:
                        marginal[channel] += 1
                        break
        for cap, final_dir in final_dirs.items():
            final_path = final_dir / f"part-{bucket:03d}.parquet"
            if not final_path.is_file():
                continue
            final = pl.read_parquet(final_path, columns=["source1_entity_id", "candidate_entity_id"])
            _add_pair_counts(final, counts, id_to_index, f"cap_{cap}")
            matched = final.join(gt, on=["source1_entity_id", "candidate_entity_id"], how="inner")
            _add_pair_counts(matched, hits, id_to_index, f"cap_{cap}")
    metrics = {key: _metric(counts[key], hits[key], gt_counts) for key in keys}
    country_slices = {}
    for country in sorted({country or "" for country in countries}):
        mask = np.fromiter(((value or "") == country for value in countries), dtype=bool, count=len(ids))
        if not np.any(mask):
            continue
        country_slices[country or "<missing>"] = {
            key: {"s1_rows": int(mask.sum()), "eligible_gt_pairs": int(gt_counts[mask].sum()),
                  "pair_recall": float(hits[key][mask].sum() / gt_counts[mask].sum()) if gt_counts[mask].sum() else None,
                  "complete_recall": float(np.mean(hits[key][mask & (gt_counts > 0)] == gt_counts[mask & (gt_counts > 0)]))
                  if np.any(mask & (gt_counts > 0)) else None}
            for key in ("union", *(f"cap_{cap}" for cap in final_dirs))
        }
    cardinality_slices = {}
    for label, mask in {"zero": original_counts == 0, "one": original_counts == 1,
                        "many": original_counts > 1}.items():
        cardinality_slices[label] = {
            "s1_rows": int(mask.sum()), "eligible_gt_pairs": int(gt_counts[mask].sum()),
            "cap_recall": {str(cap): float(hits[f"cap_{cap}"][mask].sum() / gt_counts[mask].sum())
                           if gt_counts[mask].sum() else None for cap in final_dirs},
        }
    return {
        "sampled_targets": sampled_targets, "s1_rows": len(ids),
        "eligible_gt_pairs": int(gt_counts.sum()), "original_gt_pairs_for_selected_s1": int(original_counts.sum()),
        "true_singletons": int(np.count_nonzero(original_counts == 0)),
        "conditional_zero_eligible_gt": int(np.count_nonzero(gt_counts == 0)),
        "raw_channel_rows": raw_pairs, "metrics": metrics,
        "exclusive_gt_pairs_by_channel": dict(exclusive),
        "marginal_gt_pairs_by_channel_order": dict(marginal),
        "country_slices": country_slices, "original_cardinality_slices": cardinality_slices,
        "started_utc": started_utc, "ended_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(time.perf_counter() - start, 3),
        "peak_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2, 3),
        "actual_threads": 1,
        "artifact_input_bytes": sum(path.stat().st_size for path in raw_dir.glob("part-*.parquet")) +
                                sum(path.stat().st_size for directory in final_dirs.values() for path in directory.glob("part-*.parquet")),
    }


def _latest_manifest(root: Path, pattern: str) -> Path:
    matches = list(root.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"no manifest at {root / pattern}")
    return max(matches, key=lambda path: path.stat().st_mtime_ns)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-modulus", type=int, default=16)
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--truth", type=Path, default=Path("dataset/student_resource/dataset/train/train_ground_truth.tsv"))
    parser.add_argument("--caps", type=int, nargs="+", default=[5, 10, 15, 20, 30])
    parser.add_argument("--shards", type=int, default=64)
    args = parser.parse_args()
    sample = "full" if args.sample_modulus == 1 else f"sample_{args.sample_modulus}"
    raw_manifest = _latest_manifest(args.artifact_dir / "candidates" / "raw" / sample / "train", "*/manifest.json")
    final_root = args.artifact_dir / "candidates" / "final" / sample / "train"
    final_manifests = {cap: _latest_manifest(final_root, f"k{cap}_*/manifest.json") for cap in args.caps}
    report = evaluate_blocking(args.artifact_dir / "normalized" / sample, args.truth,
                               raw_manifest.parent, {cap: path.parent for cap, path in final_manifests.items()},
                               shards=args.shards, sampled_targets=args.sample_modulus != 1)
    report["manifests"] = {"raw": str(raw_manifest),
                           "caps": {str(cap): str(path) for cap, path in final_manifests.items()}}
    blocking_root = args.artifact_dir / "blocking" / sample
    report["oversized_block_stats"] = {}
    report["channel_artifact_bytes"] = {}
    for name in ("exact", "rare_numeric", "tfidf_df0.001_f60000_all"):
        path = blocking_root / f"train_{name}.json"
        if path.is_file():
            manifest = json.loads(path.read_text())
            report["oversized_block_stats"][name] = manifest.get("index", {}) if name == "exact" else manifest.get("retrieval", {})
        candidate = blocking_root / f"train_{name}.parquet"
        if candidate.is_file():
            report["channel_artifact_bytes"][name] = candidate.stat().st_size
    signature = hashlib.sha256(json.dumps({"manifests": report["manifests"],
                                          "truth_bytes": args.truth.stat().st_size,
                                          "code": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
                                         sort_keys=True).encode()).hexdigest()[:12]
    output = args.artifact_dir / "reports" / "blocking" / f"train_{sample}_{signature}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    print(json.dumps({"report": str(output), "summary": {key: report[key] for key in
                      ("s1_rows", "eligible_gt_pairs", "elapsed_seconds", "peak_rss_gb")},
                      "metrics": report["metrics"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
