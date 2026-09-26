"""Run measured frequency-aware exact-address retrieval channel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from chimera_submission.code.business_entity_resolution.src.config import load_config
from chimera_submission.code.business_entity_resolution.src.retrieval.exact_address import build_and_retrieve


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("chimera_submission/code/business_entity_resolution/configs/dev.yaml"))
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument("--sample-modulus", type=int, default=None)
    parser.add_argument("--max-block-size", type=int, default=None)
    parser.add_argument("--per-query-cap", type=int, default=None)
    parser.add_argument("--min-address-chars", type=int, default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    modulus = config["sample_modulus"] if args.sample_modulus is None else args.sample_modulus
    sample = "full" if modulus == 1 else f"sample_{modulus}"
    root = Path(config["artifact_dir"])
    normalized = root / "normalized" / sample
    result = build_and_retrieve([normalized / f"{args.split}_s{i}.parquet" for i in (2, 3)],
                                normalized / f"{args.split}_s1.parquet",
                                root / "indexes" / sample / f"{args.split}_exact_address.pkl",
                                root / "blocking" / sample / f"{args.split}_exact_address.parquet",
                                max_block_size=config.get("address_max_block_size", 64)
                                if args.max_block_size is None else args.max_block_size,
                                per_query_cap=config.get("address_per_query_cap", 30)
                                if args.per_query_cap is None else args.per_query_cap,
                                min_address_chars=config.get("address_min_chars", 8)
                                if args.min_address_chars is None else args.min_address_chars)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
