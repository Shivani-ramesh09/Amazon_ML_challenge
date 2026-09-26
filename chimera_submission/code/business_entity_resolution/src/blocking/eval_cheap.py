"""Conditional GT and incremental-recall audit for sampled cheap channels."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import pyarrow.parquet as pq


def _ids(path: Path) -> set[str]:
    parquet = pq.ParquetFile(path)
    result = set()
    for batch in parquet.iter_batches(batch_size=100_000, columns=["entity_id"]):
        result.update(batch.column(0).to_pylist())
    return result


def _channels(path: Path) -> dict[str, set[tuple[str, str]]]:
    result: dict[str, set[tuple[str, str]]] = {}
    parquet = pq.ParquetFile(path)
    columns = ["source1_entity_id", "candidate_entity_id", "channel"]
    for batch in parquet.iter_batches(batch_size=100_000, columns=columns):
        arrays = [batch.column(i).to_pylist() for i in range(3)]
        for s1, target, channel in zip(*arrays):
            result.setdefault(channel, set()).add((s1, target))
    return result


def evaluate(normalized_dir: Path, exact_path: Path, additional_paths: list[Path], truth_path: Path) -> dict:
    s1_ids = _ids(normalized_dir / "train_s1.parquet")
    target_ids = _ids(normalized_dir / "train_s2.parquet") | _ids(normalized_dir / "train_s3.parquet")
    channels = _channels(exact_path)
    for additional_path in additional_paths:
        for channel, pairs in _channels(additional_path).items():
            channels.setdefault(channel, set()).update(pairs)
    counts = Counter()
    with truth_path.open(encoding="utf-8", newline="") as file:
        reader = csv.reader(file, delimiter="\t", quoting=csv.QUOTE_NONE)
        if next(reader) != ["source1_entity_id", "matched_entity_ids"]:
            raise ValueError("unexpected truth schema")
        for s1_id, raw_targets in reader:
            if s1_id not in s1_ids:
                continue
            counts["sampled_s1"] += 1
            for target_id in raw_targets.split(",") if raw_targets else []:
                if target_id not in target_ids:
                    continue
                counts["eligible_gt_pairs"] += 1
                pair = (s1_id, target_id)
                hits = {channel for channel, pairs in channels.items() if pair in pairs}
                for channel in hits:
                    counts[f"{channel}_gt_pairs"] += 1
                if hits & {"exact_clean", "exact_core"}:
                    counts["exact_union_gt_pairs"] += 1
                if hits & {"exact_clean", "exact_core", "rare_token", "numeric_postal"}:
                    counts["cheap_union_gt_pairs"] += 1
                if hits:
                    counts["all_channel_union_gt_pairs"] += 1
                if "rare_token" in hits and not hits & {"exact_clean", "exact_core"}:
                    counts["rare_incremental_over_exact_gt_pairs"] += 1
                if "numeric_postal" in hits and not hits & {"exact_clean", "exact_core", "rare_token"}:
                    counts["numeric_incremental_over_exact_rare_gt_pairs"] += 1
                if "tfidf_name" in hits and not hits & {"exact_clean", "exact_core", "rare_token", "numeric_postal"}:
                    counts["tfidf_incremental_over_cheap_gt_pairs"] += 1
    result = dict(counts)
    result["channel_pairs"] = {channel: len(pairs) for channel, pairs in channels.items()}
    eligible = counts["eligible_gt_pairs"]
    result["conditional_cheap_union_pair_recall"] = counts["cheap_union_gt_pairs"] / eligible if eligible else None
    result["conditional_all_channel_pair_recall"] = counts["all_channel_union_gt_pairs"] / eligible if eligible else None
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--normalized-dir", type=Path, required=True)
    parser.add_argument("--exact", type=Path, required=True)
    parser.add_argument("--additional", type=Path, action="append", required=True)
    parser.add_argument("--truth", type=Path, default=Path("dataset/student_resource/dataset/train/train_ground_truth.tsv"))
    args = parser.parse_args()
    print(json.dumps(evaluate(args.normalized_dir, args.exact, args.additional, args.truth), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
