"""Cache sorted held-out groups and evaluate the initial zero/one/many policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import time

import pyarrow.parquet as pq

from chimera_submission.code.business_entity_resolution.src.config import load_config
from chimera_submission.code.business_entity_resolution.src.data.profile import split_bucket
from chimera_submission.code.business_entity_resolution.src.entity_decision.group import group_scores
from chimera_submission.code.business_entity_resolution.src.entity_decision.policy import DecisionPolicy
from chimera_submission.code.business_entity_resolution.src.evaluation.blocking import _latest_manifest
from chimera_submission.code.business_entity_resolution.src.evaluation.entity import evaluate_groups
from chimera_submission.code.business_entity_resolution.src.training.pairs import read_truth


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("chimera_submission/code/business_entity_resolution/configs/dev.yaml"))
    parser.add_argument("--sample-modulus", type=int, default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    modulus = config["sample_modulus"] if args.sample_modulus is None else args.sample_modulus
    sample = "full" if modulus == 1 else f"sample_{modulus}"
    root = Path(config["artifact_dir"])
    # A later full refit also lives under models/; only held-out models have
    # validation predictions suitable for entity-level threshold selection.
    validation_models = [path for path in (root / "models" / sample).glob("*/manifest.json")
                         if "prediction_dir" in json.loads(path.read_text())]
    if not validation_models:
        raise FileNotFoundError("no held-out model predictions for entity evaluation")
    model_manifest_path = max(validation_models, key=lambda path: path.stat().st_mtime)
    model_manifest = json.loads(model_manifest_path.read_text())
    prediction_dir = Path(model_manifest["prediction_dir"])
    normalized_dir = root / "normalized" / sample
    query = normalized_dir / "train_s1.parquet"
    normalized_signature = json.loads((normalized_dir / "train_s1.json").read_text())["signature"]
    group_signature = hashlib.sha256(json.dumps({
        "model": model_manifest["signature"], "normalized": normalized_signature,
        "group_code": hashlib.sha256(Path(__file__).with_name("group.py").read_bytes()).hexdigest(),
        "validation_percent": config["validation_percent"], "seed": config["split_seed"],
    }, sort_keys=True).encode()).hexdigest()
    group_dir = root / "validation" / "grouped" / sample / group_signature[:12]
    group_manifest = group_dir / "manifest.json"
    if group_manifest.is_file():
        saved = json.loads(group_manifest.read_text())
        group_stats = saved["stats"] if saved.get("signature") == group_signature else None
    else:
        group_stats = None
    if group_stats is None:
        group_stats = group_scores(prediction_dir, query, group_dir,
                                   seed=config["split_seed"], validation_percent=config["validation_percent"])
        report = {"signature": group_signature, "model_manifest": str(model_manifest_path),
                  "group_dir": str(group_dir), "stats": group_stats}
        temp = group_manifest.with_suffix(".json.tmp")
        temp.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        os.replace(temp, group_manifest)
    truth_path = Path(config["data_dir"]) / "train" / "train_ground_truth.tsv"
    policy = DecisionPolicy()
    with truth_path.open("rb") as file:
        truth_hash = hashlib.file_digest(file, "sha256").hexdigest()
    metric_signature = hashlib.sha256(json.dumps({
        "group": group_signature, "truth": truth_hash, "policy": policy.as_dict(),
        "metric_code": hashlib.sha256(Path(__file__).parent.parent.joinpath("evaluation/entity.py").read_bytes()).hexdigest(),
    }, sort_keys=True).encode()).hexdigest()
    output = root / "validation" / "metrics" / sample / f"{metric_signature[:12]}.json"
    if output.is_file():
        saved = json.loads(output.read_text())
        if saved.get("signature") == metric_signature:
            print(json.dumps({"cache_hit": True, **saved}, indent=2, sort_keys=True))
            return 0
    started = time.perf_counter()
    validation_ids = {s1 for batch in pq.ParquetFile(query).iter_batches(
        batch_size=100_000, columns=["entity_id"]) for s1 in batch.column(0).to_pylist()
        if split_bucket(s1, seed=config["split_seed"],
                        validation_percent=config["validation_percent"]) == "validation"}
    truth = read_truth(truth_path, validation_ids)
    metrics = evaluate_groups(group_dir, truth, policy)
    report = {"signature": metric_signature, "group_manifest": str(group_manifest),
              "policy": policy.as_dict(), "metrics": metrics,
              "evaluation_seconds": round(time.perf_counter() - started, 3),
              "peak_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2, 3)}
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(".json.tmp")
    temp.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temp, output)
    print(json.dumps({"group_manifest": str(group_manifest), "group_stats": group_stats,
                      "metrics_path": str(output), "metrics": metrics}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
