"""Measure conditional sample GT loss from final candidate caps."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import polars as pl


def evaluate_caps(normalized_dir: Path, final_root: Path, truth_path: Path) -> dict:
    s1_ids = set(pl.read_parquet(normalized_dir / "train_s1.parquet", columns=["entity_id"])["entity_id"])
    target_ids = set()
    for source in (2, 3):
        target_ids.update(pl.read_parquet(normalized_dir / f"train_s{source}.parquet", columns=["entity_id"])["entity_id"])
    truth = []
    with truth_path.open(encoding="utf-8", newline="") as file:
        reader = csv.reader(file, delimiter="\t", quoting=csv.QUOTE_NONE)
        if next(reader) != ["source1_entity_id", "matched_entity_ids"]:
            raise ValueError("unexpected truth schema")
        for s1_id, raw_targets in reader:
            if s1_id in s1_ids:
                truth.extend((s1_id, target) for target in raw_targets.split(",") if target in target_ids)
    gt = pl.DataFrame(truth, schema=["source1_entity_id", "candidate_entity_id"], orient="row")
    result = {"sampled_s1": len(s1_ids), "eligible_gt_pairs": gt.height, "caps": {}}
    for cap in (5, 10, 15, 20, 30):
        choices = list(final_root.glob(f"k{cap}_*/manifest.json"))
        if not choices:
            raise ValueError(f"missing cached cap {cap}")
        selected = max(choices, key=lambda path: path.stat().st_mtime_ns)
        manifest = json.loads(selected.read_text())
        candidates = pl.scan_parquet(str(selected.parent / "part-*.parquet"))
        hits = gt.lazy().join(candidates.select(["source1_entity_id", "candidate_entity_id"]),
                              on=["source1_entity_id", "candidate_entity_id"], how="semi").collect().height
        result["caps"][str(cap)] = {
            "final_pairs": manifest["final"]["final_pairs"],
            "gt_pairs_recovered": hits,
            "conditional_pair_recall": hits / gt.height if gt.height else None,
            "mean_candidates_per_sampled_s1": manifest["final"]["final_pairs"] / len(s1_ids),
            "manifest": str(selected),
        }
        if cap == 30:
            ranked = candidates.sort(["source1_entity_id", "preliminary_score", "candidate_entity_id"],
                                     descending=[False, True, False]).with_columns(
                pl.col("candidate_entity_id").cum_count().over("source1_entity_id").alias("rank")
            )
            rank_hits = gt.lazy().join(ranked, on=["source1_entity_id", "candidate_entity_id"], how="inner")
            result["gt_rank_bands_at_cap30"] = {
                "1_to_5": rank_hits.filter(pl.col("rank") <= 5).select(pl.len()).collect().item(),
                "6_to_10": rank_hits.filter(pl.col("rank").is_between(6, 10)).select(pl.len()).collect().item(),
                "11_to_20": rank_hits.filter(pl.col("rank").is_between(11, 20)).select(pl.len()).collect().item(),
                "21_to_30": rank_hits.filter(pl.col("rank").is_between(21, 30)).select(pl.len()).collect().item(),
            }
            late = rank_hits.filter(pl.col("rank") > 20)
            result["late_gt_provenance"] = {
                flag: late.filter(pl.col(flag)).select(pl.len()).collect().item()
                for flag in ("retrieved_by_exact_name", "retrieved_by_core_name",
                             "retrieved_by_rare_token", "retrieved_by_number_postal", "retrieved_by_tfidf")
            }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-modulus", type=int, default=16)
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--truth", type=Path, default=Path("dataset/student_resource/dataset/train/train_ground_truth.tsv"))
    args = parser.parse_args()
    sample = "full" if args.sample_modulus == 1 else f"sample_{args.sample_modulus}"
    report = evaluate_caps(args.artifact_dir / "normalized" / sample,
                           args.artifact_dir / "candidates" / "final" / sample / "train", args.truth)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
