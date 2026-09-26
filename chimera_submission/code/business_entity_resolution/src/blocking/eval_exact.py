"""Conditional exact-channel GT audit for a development sample.

This is not full-universe blocking recall: both S1 and targets were independently
sampled. Phase 7 evaluates final candidates against the full target universe.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import pyarrow.parquet as pq


def _names(path: Path) -> dict[str, tuple[str, str]]:
    result = {}
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=100_000, columns=["entity_id", "name_clean", "name_core"]):
        ids = batch.column(0).to_pylist()
        clean = batch.column(1).to_pylist()
        core = batch.column(2).to_pylist()
        result.update(zip(ids, zip(clean, core)))
    return result


def evaluate(query_path: Path, target_paths: list[Path], candidate_path: Path, truth_path: Path) -> dict:
    query = _names(query_path)
    targets = {}
    for path in target_paths:
        targets.update(_names(path))
    candidate_pairs = set()
    parquet = pq.ParquetFile(candidate_path)
    for batch in parquet.iter_batches(batch_size=100_000, columns=["source1_entity_id", "candidate_entity_id"]):
        candidate_pairs.update(zip(batch.column(0).to_pylist(), batch.column(1).to_pylist()))
    counts = Counter()
    with truth_path.open(encoding="utf-8", newline="") as file:
        reader = csv.reader(file, delimiter="\t", quoting=csv.QUOTE_NONE)
        if next(reader) != ["source1_entity_id", "matched_entity_ids"]:
            raise ValueError("unexpected truth schema")
        for s1_id, raw_targets in reader:
            if s1_id not in query:
                continue
            counts["sampled_s1"] += 1
            clean_q, core_q = query[s1_id]
            for target_id in raw_targets.split(",") if raw_targets else []:
                if target_id not in targets:
                    continue
                counts["eligible_gt_pairs"] += 1
                clean_t, core_t = targets[target_id]
                clean_hit = bool(clean_q and clean_q == clean_t)
                core_hit = bool(core_q and core_q == core_t)
                emitted = (s1_id, target_id) in candidate_pairs
                counts["exact_clean_gt_pairs"] += clean_hit
                counts["exact_core_gt_pairs"] += core_hit
                counts["exact_union_before_prune_gt_pairs"] += clean_hit or core_hit
                counts["exact_union_emitted_gt_pairs"] += emitted
                counts["gt_pairs_lost_to_exact_pruning"] += (clean_hit or core_hit) and not emitted
                counts["core_only_gt_pairs"] += core_hit and not clean_hit
    counts["emitted_unique_pairs"] = len(candidate_pairs)
    result = dict(counts)
    eligible = counts["eligible_gt_pairs"]
    result["conditional_recall_before_prune"] = counts["exact_union_before_prune_gt_pairs"] / eligible if eligible else None
    result["conditional_recall_after_prune"] = counts["exact_union_emitted_gt_pairs"] / eligible if eligible else None
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--normalized-dir", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--truth", type=Path, default=Path("dataset/student_resource/dataset/train/train_ground_truth.tsv"))
    args = parser.parse_args()
    result = evaluate(args.normalized_dir / "train_s1.parquet",
                      [args.normalized_dir / "train_s2.parquet", args.normalized_dir / "train_s3.parquet"],
                      args.candidate, args.truth)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
