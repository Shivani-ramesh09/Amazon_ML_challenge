"""Build or reuse entity-grouped labeled train and complete validation sets."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess

from chimera_submission.code.business_entity_resolution.src.config import load_config
from chimera_submission.code.business_entity_resolution.src.evaluation.blocking import _latest_manifest
from chimera_submission.code.business_entity_resolution.src.training.pairs import build_training_pairs, training_signature


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("chimera_submission/code/business_entity_resolution/configs/dev.yaml"))
    parser.add_argument("--sample-modulus", type=int, default=None)
    parser.add_argument("--negative-ratio", type=int, default=5)
    parser.add_argument("--hard-fraction", type=float, default=0.8)
    args = parser.parse_args()
    config = load_config(args.config)
    modulus = config["sample_modulus"] if args.sample_modulus is None else args.sample_modulus
    sample = "full" if modulus == 1 else f"sample_{modulus}"
    root = Path(config["artifact_dir"])
    feature_manifest = _latest_manifest(root / "features" / sample / "train", "*/manifest.json")
    normalized_s1 = root / "normalized" / sample / "train_s1.parquet"
    truth_path = Path(config["data_dir"]) / "train" / "train_ground_truth.tsv"
    signature = training_signature(feature_manifest, truth_path,
                                   validation_percent=config["validation_percent"],
                                   seed=config["split_seed"], negative_ratio=args.negative_ratio,
                                   hard_fraction=args.hard_fraction)
    output = root / "training" / sample / signature[:12]
    manifest = output / "manifest.json"
    if manifest.is_file():
        saved = json.loads(manifest.read_text())
        if saved.get("signature") == signature:
            print(json.dumps({"cache_hit": True, **saved}, indent=2, sort_keys=True))
            return 0
    stats = build_training_pairs(feature_manifest.parent, normalized_s1, truth_path, output,
                                 validation_percent=config["validation_percent"], seed=config["split_seed"],
                                 negative_ratio=args.negative_ratio, hard_fraction=args.hard_fraction)
    report = {"signature": signature, "feature_manifest": str(feature_manifest),
              "source_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                             text=True, check=False).stdout.strip() or None,
              "output_dir": str(output), "policy": {"negative_ratio": args.negative_ratio,
              "hard_fraction": args.hard_fraction, "validation_percent": config["validation_percent"],
              "seed": config["split_seed"]}, "stats": stats}
    temp = manifest.with_suffix(".json.tmp")
    temp.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temp, manifest)
    print(json.dumps({"manifest": str(manifest), "stats": {key: value for key, value in stats.items()
                      if key != "shards"}}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
