"""Tune only on cached train-validation groups and record an ablation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import resource
import time

import pyarrow.parquet as pq

from chimera_submission.code.business_entity_resolution.src.config import load_config
from chimera_submission.code.business_entity_resolution.src.entity_decision.optimize import load_group_arrays, optimize_policy
from chimera_submission.code.business_entity_resolution.src.evaluation.blocking import _latest_manifest
from chimera_submission.code.business_entity_resolution.src.training.pairs import read_truth


ABLATION_FIELDS = ["experiment_id", "candidate_channels", "candidate_cap", "feature_set",
                   "training_negative_policy", "model", "threshold_policy", "candidate_recall",
                   "entity_any_hit", "entity_complete_recall", "precision", "recall",
                   "macro_f0_5", "runtime", "peak_ram", "notes"]


def _append_ablation(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = set()
    if path.is_file():
        with path.open(newline="") as file:
            existing = {row["experiment_id"] for row in csv.DictReader(file)}
    with path.open("a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=ABLATION_FIELDS, lineterminator="\n")
        if not existing and path.stat().st_size == 0:
            writer.writeheader()
        for row in rows:
            if row["experiment_id"] not in existing:
                writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("chimera_submission/code/business_entity_resolution/configs/dev.yaml"))
    parser.add_argument("--sample-modulus", type=int, default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    modulus = config["sample_modulus"] if args.sample_modulus is None else args.sample_modulus
    sample = "full" if modulus == 1 else f"sample_{modulus}"
    root = Path(config["artifact_dir"])
    group_manifest = _latest_manifest(root / "validation" / "grouped" / sample, "*/manifest.json")
    grouped = json.loads(group_manifest.read_text())
    group_dir = group_manifest.parent
    truth_path = Path(config["data_dir"]) / "train" / "train_ground_truth.tsv"
    with truth_path.open("rb") as file:
        truth_hash = hashlib.file_digest(file, "sha256").hexdigest()
    signature = hashlib.sha256(json.dumps({
        "group": grouped["signature"], "truth": truth_hash,
        "optimizer_code": hashlib.sha256(Path(__file__).with_name("optimize.py").read_bytes()).hexdigest(),
    }, sort_keys=True).encode()).hexdigest()
    output = root / "validation" / "policy" / sample / f"{signature[:12]}.json"
    if output.is_file():
        saved = json.loads(output.read_text())
        if saved.get("signature") == signature:
            print(json.dumps({"cache_hit": True, **saved}, indent=2, sort_keys=True))
            return 0
    started = time.perf_counter()
    ids = set()
    for path in sorted(group_dir.glob("part-*.parquet")):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=10_000, columns=["source1_entity_id"]):
            ids.update(batch.column(0).to_pylist())
    truth = read_truth(truth_path, ids)
    arrays = load_group_arrays(group_dir, truth, width=config["candidate_cap"])
    result = optimize_policy(arrays)
    elapsed = time.perf_counter() - started
    result.update({"signature": signature, "group_manifest": str(group_manifest),
                   "elapsed_seconds": round(elapsed, 3),
                   "peak_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2, 3)})
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(".json.tmp")
    temp.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.replace(temp, output)
    blocking_report = _latest_manifest(root / "reports" / "blocking", f"train_{sample}_*.json")
    blocking = json.loads(blocking_report.read_text())
    candidate = blocking["metrics"][f"cap_{config['candidate_cap']}"]
    rows = []
    for name, policy_key, metric_key in (("default", "baseline_policy", "baseline_metrics"),
                                         ("optimized", "selected_policy", "selected_metrics")):
        metric = result[metric_key]
        rows.append({"experiment_id": f"phase12_{name}_{signature[:12]}",
                     "candidate_channels": "exact+core+rare+numeric+char_tfidf",
                     "candidate_cap": config["candidate_cap"], "feature_set": "cpu_36",
                     "training_negative_policy": "per_s1_ratio_5_hard_0.8",
                     "model": "lightgbm_cpu", "threshold_policy": json.dumps(result[policy_key], sort_keys=True),
                     "candidate_recall": candidate["pair_candidate_recall"],
                     "entity_any_hit": candidate["entity_any_hit_recall"],
                     "entity_complete_recall": candidate["entity_complete_recall"],
                     "precision": metric["micro_precision"], "recall": metric["micro_recall"],
                     "macro_f0_5": metric["macro_f0_5"], "runtime": round(elapsed, 3),
                     "peak_ram": result["peak_rss_gb"],
                     "notes": f"{sample}; corrected truth-complete dev targets; full universe pending"})
    _append_ablation(Path("reports/ablation.csv"), rows)
    print(json.dumps({"policy_path": str(output), "elapsed_seconds": result["elapsed_seconds"],
                      "coarse_trials": result["coarse_trials"], "fine_trials": result["fine_trials"],
                      "baseline_macro_f0_5": result["baseline_metrics"]["macro_f0_5"],
                      "selected_policy": result["selected_policy"],
                      "selected_metrics": result["selected_metrics"],
                      "best_raw_macro_f0_5": result["best_raw_macro_f0_5"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
