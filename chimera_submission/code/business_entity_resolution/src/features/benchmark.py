"""Measure representative per-feature lexical costs on actual candidate pairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import pyarrow as pa
import pyarrow.parquet as pq
from rapidfuzz import fuzz

from chimera_submission.code.business_entity_resolution.src.evaluation.blocking import _latest_manifest
from chimera_submission.code.business_entity_resolution.src.features.cpu import _jaccard, _load_rows, _ratio


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-modulus", type=int, default=16)
    parser.add_argument("--pairs", type=int, default=50_000)
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    sample = "full" if args.sample_modulus == 1 else f"sample_{args.sample_modulus}"
    candidate_manifest = _latest_manifest(args.artifact_dir / "candidates" / "final" / sample / "train", "k30_*/manifest.json")
    path = sorted(candidate_manifest.parent.glob("part-*.parquet"))[0]
    batch = next(pq.ParquetFile(path).iter_batches(batch_size=args.pairs,
                                                 columns=["source1_entity_id", "candidate_entity_id"]))
    source, target, s1_index, target_index = _load_rows(args.artifact_dir / "normalized" / sample, "train")
    s1 = source.take(pa.array([s1_index[value] for value in batch.column(0).to_pylist()], type=pa.int32()))
    s23 = target.take(pa.array([target_index[value] for value in batch.column(1).to_pylist()], type=pa.int32()))
    pairs = {}
    for name, column in (("name", "name_core"), ("address", "address_clean"),
                         ("number", "address_numbers")):
        pairs[name] = list(zip(s1[column].to_pylist(), s23[column].to_pylist()))
    functions = {
        "name_ratio": ("name", lambda a, b: _ratio(a or "", b or "")),
        "name_token_set": ("name", lambda a, b: fuzz.token_set_ratio(a or "", b or "") / 100),
        "name_jaccard": ("name", lambda a, b: _jaccard(a or "", b or "")[0]),
        "address_ratio": ("address", lambda a, b: _ratio(a or "", b or "")),
        "address_token_set": ("address", lambda a, b: fuzz.token_set_ratio(a or "", b or "") / 100),
        "address_jaccard": ("address", lambda a, b: _jaccard(a or "", b or "")[0]),
        "number_jaccard": ("number", lambda a, b: _jaccard(a or "", b or "")[0]),
    }
    times = {}
    for feature, (source_name, fn) in functions.items():
        started = time.perf_counter()
        _ = sum(fn(a, b) for a, b in pairs[source_name])
        times[feature] = round(time.perf_counter() - started, 4)
    print(json.dumps({"candidate_manifest": str(candidate_manifest), "pairs": batch.num_rows,
                      "seconds_by_feature": times}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
