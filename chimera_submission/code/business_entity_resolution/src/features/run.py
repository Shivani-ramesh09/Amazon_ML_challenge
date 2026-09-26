"""Run cached CPU pair features for the accepted candidate set."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from chimera_submission.code.business_entity_resolution.src.config import load_config
from chimera_submission.code.business_entity_resolution.src.evaluation.blocking import _latest_manifest
from chimera_submission.code.business_entity_resolution.src.features.cpu import build_features, feature_signature


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("chimera_submission/code/business_entity_resolution/configs/dev.yaml"))
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument("--sample-modulus", type=int, default=None)
    parser.add_argument("--cap", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    modulus = config["sample_modulus"] if args.sample_modulus is None else args.sample_modulus
    sample = "full" if modulus == 1 else f"sample_{modulus}"
    cap = config["candidate_cap"] if args.cap is None else args.cap
    batch_size = config["pair_batch_size"] if args.batch_size is None else args.batch_size
    root = Path(config["artifact_dir"])
    normalized = root / "normalized" / sample
    candidate_manifest = _latest_manifest(root / "candidates" / "final" / sample / args.split,
                                          f"k{cap}_*/manifest.json")
    candidates = candidate_manifest.parent
    signature = feature_signature(normalized, candidates, args.split, batch_size)
    output = root / "features" / sample / args.split / signature[:12]
    manifest = output / "manifest.json"
    if manifest.is_file():
        saved = json.loads(manifest.read_text())
        if saved.get("signature") == signature:
            print(json.dumps({"cache_hit": True, **saved}, indent=2, sort_keys=True))
            return 0
    stats = build_features(normalized, candidates, output, split=args.split, batch_size=batch_size)
    report = {"signature": signature, "candidate_manifest": str(candidate_manifest),
              "normalized_dir": str(normalized), "output_dir": str(output), "stats": stats}
    tmp = manifest.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, manifest)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
