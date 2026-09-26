"""Build capped final candidates from the three cached retrieval channels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from chimera_submission.code.business_entity_resolution.src.config import load_config
from chimera_submission.code.business_entity_resolution.src.retrieval.union import build_union


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("chimera_submission/code/business_entity_resolution/configs/dev.yaml"))
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument("--sample-modulus", type=int, default=None)
    parser.add_argument("--cap", type=int, default=None)
    parser.add_argument("--shards", type=int, default=64)
    parser.add_argument("--include-address", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    modulus = config["sample_modulus"] if args.sample_modulus is None else args.sample_modulus
    sample_dir = "full" if modulus == 1 else f"sample_{modulus}"
    root = Path(config["artifact_dir"])
    blocking = root / "blocking" / sample_dir
    channels = [blocking / f"{args.split}_{name}.parquet" for name in
                ("exact", "rare_numeric", "tfidf_df0.001_f60000_all")]
    if args.include_address or config.get("address_channel", False):
        channels.append(blocking / f"{args.split}_exact_address.parquet")
    report = build_union(
        channels, root / "normalized" / sample_dir / f"{args.split}_s1.parquet",
        root / "candidates" / "raw" / sample_dir / args.split,
        root / "candidates" / "final" / sample_dir / args.split,
        cap=config["candidate_cap"] if args.cap is None else args.cap,
        shards=args.shards,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
