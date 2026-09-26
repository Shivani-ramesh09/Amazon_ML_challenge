"""Train or reuse the CPU pair scorer and score every held-out candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from chimera_submission.code.business_entity_resolution.src.config import load_config
from chimera_submission.code.business_entity_resolution.src.evaluation.blocking import _latest_manifest
from chimera_submission.code.business_entity_resolution.src.scoring.lightgbm_cpu import train_and_score
from chimera_submission.code.business_entity_resolution.src.training.pairs import training_signature


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("chimera_submission/code/business_entity_resolution/configs/dev.yaml"))
    parser.add_argument("--sample-modulus", type=int, default=None)
    parser.add_argument("--negative-ratio", type=int, default=5)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--num-leaves", type=int, default=63)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-rounds", type=int, default=600)
    args = parser.parse_args()
    config = load_config(args.config)
    modulus = config["sample_modulus"] if args.sample_modulus is None else args.sample_modulus
    sample = "full" if modulus == 1 else f"sample_{modulus}"
    root = Path(config["artifact_dir"])
    feature_manifest = _latest_manifest(root / "features" / sample / "train", "*/manifest.json")
    truth_path = Path(config["data_dir"]) / "train" / "train_ground_truth.tsv"
    training_key = training_signature(feature_manifest, truth_path,
                                      validation_percent=config["validation_percent"],
                                      seed=config["split_seed"], negative_ratio=args.negative_ratio,
                                      hard_fraction=0.8)
    training_dir = root / "training" / sample / training_key[:12]
    training_manifest = training_dir / "manifest.json"
    if not training_manifest.is_file():
        raise FileNotFoundError(f"run Phase 9 for ratio {args.negative_ratio}: {training_manifest}")
    threads = config["threads"] if args.threads is None else args.threads
    settings = {"threads": threads, "seed": config["split_seed"], "num_leaves": args.num_leaves,
                "learning_rate": args.learning_rate, "max_rounds": args.max_rounds,
                "early_stopping_rounds": 50}
    signature = hashlib.sha256(json.dumps({"training": training_key, "settings": settings,
                                            "code": hashlib.sha256(Path(__file__).with_name("lightgbm_cpu.py").read_bytes()).hexdigest()},
                                           sort_keys=True).encode()).hexdigest()
    model_dir = root / "models" / sample / signature[:12]
    model_path = model_dir / "lightgbm.txt"
    prediction_dir = root / "predictions" / "validation" / sample / signature[:12]
    manifest = model_dir / "manifest.json"
    if manifest.is_file() and model_path.is_file():
        saved = json.loads(manifest.read_text())
        if saved.get("signature") == signature:
            print(json.dumps({"cache_hit": True, **saved}, indent=2, sort_keys=True))
            return 0
    model_dir.mkdir(parents=True, exist_ok=True)
    stats = train_and_score(training_dir, model_path, prediction_dir, **settings)
    report = {"signature": signature, "training_manifest": str(training_manifest),
              "model_path": str(model_path), "prediction_dir": str(prediction_dir),
              "settings": settings, "stats": stats}
    temp = manifest.with_suffix(".json.tmp")
    temp.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temp, manifest)
    print(json.dumps({"manifest": str(manifest), "model_path": str(model_path),
                      "prediction_dir": str(prediction_dir),
                      "stats": {key: value for key, value in stats.items()
                                if key != "feature_importance_gain"},
                      "top_features": stats["feature_importance_gain"][:12]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
