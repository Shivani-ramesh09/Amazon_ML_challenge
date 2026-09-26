"""Analyze errors only on cached training validation artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import pyarrow.parquet as pq

from chimera_submission.code.business_entity_resolution.src.config import load_config
from chimera_submission.code.business_entity_resolution.src.entity_decision.policy import DecisionPolicy
from chimera_submission.code.business_entity_resolution.src.evaluation.blocking import _latest_manifest
from chimera_submission.code.business_entity_resolution.src.evaluation.errors import analyze_errors
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
    group_manifest = _latest_manifest(root / "validation" / "grouped" / sample, "*/manifest.json")
    group_dir = group_manifest.parent
    policy_path = _latest_manifest(root / "validation" / "policy" / sample, "*.json")
    policy_manifest = json.loads(policy_path.read_text())
    policy = DecisionPolicy(**policy_manifest["selected_policy"])
    model_manifest = json.loads(Path(json.loads(group_manifest.read_text())["model_manifest"]).read_text())
    training_manifest = json.loads(Path(model_manifest["training_manifest"]).read_text())
    features_dir = Path(training_manifest["output_dir"]) / "validation"
    signature = hashlib.sha256(json.dumps({
        "groups": json.loads(group_manifest.read_text())["signature"],
        "policy": policy_manifest["signature"], "training": training_manifest["signature"],
        "code": hashlib.sha256(Path(__file__).with_name("errors.py").read_bytes()).hexdigest(),
    }, sort_keys=True).encode()).hexdigest()
    output = root / "reports" / "errors" / f"train_{sample}_{signature[:12]}.json"
    if output.is_file():
        saved = json.loads(output.read_text())
        if saved.get("signature") == signature:
            print(json.dumps({"cache_hit": True, "report": str(output),
                              "entity_counts": saved["entity_counts"],
                              "pair_counts": saved["pair_counts"]}, indent=2, sort_keys=True))
            return 0
    ids = set()
    for path in sorted(group_dir.glob("part-*.parquet")):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=10_000, columns=["source1_entity_id"]):
            ids.update(batch.column(0).to_pylist())
    truth = read_truth(Path(config["data_dir"]) / "train" / "train_ground_truth.tsv", ids)
    report = analyze_errors(group_dir, features_dir, root / "normalized" / sample, truth, policy)
    report["group_manifest"] = str(group_manifest)
    report["policy_path"] = str(policy_path)
    report["signature"] = signature
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    print(json.dumps({"report": str(output), "entity_counts": report["entity_counts"],
                      "pair_counts": report["pair_counts"], "by_country": report["by_country"],
                      "elapsed_seconds": report["elapsed_seconds"], "peak_rss_gb": report["peak_rss_gb"]},
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
