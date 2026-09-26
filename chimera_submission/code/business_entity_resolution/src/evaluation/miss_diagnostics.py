"""Profile sampled GT retrieval misses to guide blocking changes."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import polars as pl
from rapidfuzz import fuzz

from chimera_submission.code.business_entity_resolution.src.evaluation.blocking import _truth


def diagnose(normalized_dir: Path, truth_path: Path, raw_dir: Path, *, shards: int = 64) -> dict:
    _, _, _, _, truth_by_shard = _truth(normalized_dir, truth_path, shards, sampled_targets=True)
    truth_rows = [pair for rows in truth_by_shard.values() for pair in rows]
    gt = pl.DataFrame(truth_rows, schema=["source1_entity_id", "candidate_entity_id"], orient="row")
    raw = pl.scan_parquet(str(raw_dir / "part-*.parquet")).select(
        ["source1_entity_id", "candidate_entity_id"]).unique()
    misses = gt.lazy().join(raw, on=["source1_entity_id", "candidate_entity_id"], how="anti").collect()
    columns = ["entity_id", "name_core", "address_clean", "postal_candidate", "country"]
    s1 = pl.read_parquet(normalized_dir / "train_s1.parquet", columns=columns).rename(
        {"entity_id": "source1_entity_id", "name_core": "s1_name", "address_clean": "s1_address",
         "postal_candidate": "s1_postal", "country": "s1_country"})
    targets = pl.concat([pl.read_parquet(normalized_dir / f"train_s{source}.parquet", columns=columns)
                         for source in (2, 3)]).rename(
        {"entity_id": "candidate_entity_id", "name_core": "target_name", "address_clean": "target_address",
         "postal_candidate": "target_postal", "country": "target_country"})
    joined = misses.join(s1, on="source1_entity_id").join(targets, on="candidate_entity_id")
    summary: dict[str, dict] = {}
    for country, frame in joined.partition_by("s1_country", as_dict=True).items():
        label = country[0] or "<missing>"
        counts = Counter()
        examples = []
        for row in frame.iter_rows(named=True):
            name = fuzz.ratio(row["s1_name"] or "", row["target_name"] or "")
            address = fuzz.ratio(row["s1_address"] or "", row["target_address"] or "")
            counts["missed_pairs"] += 1
            counts["name_ratio_below_50"] += name < 50
            counts["name_ratio_50_to_75"] += 50 <= name < 75
            counts["name_ratio_at_least_75"] += name >= 75
            counts["address_ratio_at_least_75"] += address >= 75
            counts["low_name_high_address"] += name < 50 and address >= 75
            counts["postal_equal_present"] += bool(row["s1_postal"] and row["s1_postal"] == row["target_postal"])
            counts["country_mismatch"] += row["s1_country"] != row["target_country"]
            if len(examples) < 8:
                examples.append({"s1": row["source1_entity_id"], "target": row["candidate_entity_id"],
                                 "name_ratio": round(name, 1), "address_ratio": round(address, 1),
                                 "s1_name": row["s1_name"], "target_name": row["target_name"]})
        summary[label] = {"counts": dict(counts), "examples": examples}
    return {"eligible_gt_pairs": gt.height, "union_missed_gt_pairs": misses.height,
            "by_country": summary}


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-modulus", type=int, default=16)
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--truth", type=Path, default=Path("dataset/student_resource/dataset/train/train_ground_truth.tsv"))
    args = parser.parse_args()
    from chimera_submission.code.business_entity_resolution.src.evaluation.blocking import _latest_manifest
    sample = f"sample_{args.sample_modulus}"
    raw_manifest = _latest_manifest(args.artifact_dir / "candidates" / "raw" / sample / "train", "*/manifest.json")
    report = diagnose(args.artifact_dir / "normalized" / sample, args.truth, raw_manifest.parent)
    output = args.artifact_dir / "reports" / "blocking" / f"train_{sample}_misses.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"report": str(output), "eligible_gt_pairs": report["eligible_gt_pairs"],
                      "union_missed_gt_pairs": report["union_missed_gt_pairs"],
                      "by_country": {country: value["counts"] for country, value in report["by_country"].items()}},
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
