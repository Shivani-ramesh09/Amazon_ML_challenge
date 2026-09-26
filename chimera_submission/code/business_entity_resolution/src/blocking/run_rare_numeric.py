"""CLI for bounded rare-token and name-plus-number/postal channels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from chimera_submission.code.business_entity_resolution.src.blocking.rare_numeric import build_and_retrieve
from chimera_submission.code.business_entity_resolution.src.config import load_config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("chimera_submission/code/business_entity_resolution/configs/dev.yaml"))
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument("--sample-modulus", type=int, default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    modulus = config["sample_modulus"] if args.sample_modulus is None else args.sample_modulus
    sample_dir = "full" if modulus == 1 else f"sample_{modulus}"
    root = Path(config["artifact_dir"])
    normalized = root / "normalized" / sample_dir
    target_paths = [normalized / f"{args.split}_s{source}.parquet" for source in (2, 3)]
    query_path = normalized / f"{args.split}_s1.parquet"
    index_path = root / "indexes" / sample_dir / f"{args.split}_rare_numeric.pkl"
    candidate_path = root / "blocking" / sample_dir / f"{args.split}_rare_numeric.parquet"
    print(json.dumps(build_and_retrieve(target_paths, query_path, index_path, candidate_path), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
