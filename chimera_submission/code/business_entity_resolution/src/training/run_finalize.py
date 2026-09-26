"""CLI for a frozen all-entity LightGBM refit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from chimera_submission.code.business_entity_resolution.src.config import load_config
from chimera_submission.code.business_entity_resolution.src.evaluation.blocking import _latest_manifest
from chimera_submission.code.business_entity_resolution.src.training.finalize import finalize


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("chimera_submission/code/business_entity_resolution/configs/dev.yaml"))
    parser.add_argument("--sample-modulus", type=int, default=None)
    parser.add_argument("--feature-manifest", type=Path, default=None)
    parser.add_argument("--validated-model-manifest", type=Path, default=None)
    parser.add_argument("--policy-json", type=Path, default=None)
    parser.add_argument("--negative-ratio", type=int, default=5)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.sample_modulus is not None:
        config["sample_modulus"] = args.sample_modulus
    sample = "full" if config["sample_modulus"] == 1 else f"sample_{config['sample_modulus']}"
    root = Path(config["artifact_dir"])
    policy = args.policy_json or _latest_manifest(root / "validation" / "policy" / sample, "*.json")
    grouped = json.loads(Path(json.loads(policy.read_text())["group_manifest"]).read_text())
    model = args.validated_model_manifest or Path(grouped["model_manifest"])
    validated = json.loads(model.read_text())
    training = json.loads(Path(validated["training_manifest"]).read_text())
    features = args.feature_manifest or Path(training["feature_manifest"])
    vectorizer_key = f"df{config['tfidf_max_df']:g}_f{config['tfidf_max_features']}"
    vectorizer = root / "tfidf" / sample / vectorizer_key / "name_vectorizer.pkl"
    result = finalize(features, model, policy,
                      root / "normalized" / sample / "train_s1.parquet",
                      Path(config["data_dir"]) / "train" / "train_ground_truth.tsv",
                      vectorizer, root, config=config, negative_ratio=args.negative_ratio)
    print(json.dumps({key: value for key, value in result.items() if key != "config"},
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
