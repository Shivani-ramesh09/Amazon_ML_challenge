"""Build/reuse sparse name index and retrieve top-N without dense products."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import pyarrow.parquet as pq

from chimera_submission.code.business_entity_resolution.src.config import load_config
from chimera_submission.code.business_entity_resolution.src.retrieval.tfidf import (
    build_index, load_retriever, signatures,
)


def _strong_exact_ids(path: Path) -> set[str]:
    """Conservative candidate-based gate to audit before production use."""
    clean, core = {}, {}
    parquet = pq.ParquetFile(path)
    columns = ["source1_entity_id", "candidate_entity_id", "channel", "exact_block_size"]
    for batch in parquet.iter_batches(batch_size=100_000, columns=columns):
        values = [batch.column(i).to_pylist() for i in range(4)]
        for s1_id, target_id, channel, block_size in zip(*values):
            if block_size != 1:
                continue
            if channel == "exact_clean":
                clean[s1_id] = target_id
            elif channel == "exact_core":
                core[s1_id] = target_id
    return {s1_id for s1_id, target in clean.items() if core.get(s1_id) == target}


def _atomic_json(path: Path, value: dict) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("chimera_submission/code/business_entity_resolution/configs/dev.yaml"))
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument("--sample-modulus", type=int, default=None)
    parser.add_argument("--gate", choices=("all", "strong_exact"), default="all")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--max-df", type=float, default=None)
    parser.add_argument("--threads", type=int, default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    modulus = config["sample_modulus"] if args.sample_modulus is None else args.sample_modulus
    top_k = config.get("tfidf_top_k", 20) if args.top_k is None else args.top_k
    threshold = config.get("tfidf_threshold", 0.2) if args.threshold is None else args.threshold
    max_df = config.get("tfidf_max_df", 0.001) if args.max_df is None else args.max_df
    max_features = config.get("tfidf_max_features", 60_000)
    batch_size = config.get("tfidf_batch_size", 1000)
    if not 0 < max_df <= 1:
        raise ValueError("max_df must be in (0, 1]")
    sample_dir = "full" if modulus == 1 else f"sample_{modulus}"
    root = Path(config["artifact_dir"])
    normalized = root / "normalized" / sample_dir
    target_paths = [normalized / f"{args.split}_s{source}.parquet" for source in (2, 3)]
    query_path = normalized / f"{args.split}_s1.parquet"
    vectorizer_key = f"df{max_df:g}_f{max_features}"
    tfidf_dir = root / "tfidf" / sample_dir / vectorizer_key
    tfidf_dir.mkdir(parents=True, exist_ok=True)
    vectorizer_path = tfidf_dir / "name_vectorizer.pkl"
    transpose_path = tfidf_dir / f"{args.split}_target_transpose.npz"
    target_ids_path = tfidf_dir / f"{args.split}_target_ids.parquet"
    index_manifest = tfidf_dir / f"{args.split}_index.json"
    candidate_dir = root / "blocking" / sample_dir
    candidate_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = candidate_dir / f"{args.split}_tfidf_{vectorizer_key}_{args.gate}.parquet"
    candidate_manifest = candidate_path.with_suffix(".json")
    gate_path = candidate_dir / f"{args.split}_exact.parquet" if args.gate == "strong_exact" else None
    index_settings = {"max_df": max_df, "max_features": max_features, "ngram_range": [3, 5]}
    retrieval_settings = {
        "top_k": top_k, "threshold": threshold, "batch_size": batch_size, "gate": args.gate,
        "threads": args.threads or config["threads"],
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    index_signature, candidate_signature = signatures(
        target_paths, query_path, Path(__file__).with_name("tfidf.py"),
        vectorizer_path if args.split == "test" else None,
        index_settings, retrieval_settings, gate_path,
    )
    if candidate_path.is_file() and candidate_manifest.is_file():
        saved = json.loads(candidate_manifest.read_text())
        if saved.get("signature") == candidate_signature:
            print(json.dumps({"cache_hit": True, **saved}, indent=2, sort_keys=True))
            return 0
    index_hit = False
    if all(path.is_file() for path in (transpose_path, target_ids_path, vectorizer_path, index_manifest)):
        saved = json.loads(index_manifest.read_text())
        index_hit = saved.get("signature") == index_signature
    if not index_hit:
        stats = build_index(target_paths, vectorizer_path, transpose_path, target_ids_path,
                            fit_vectorizer=args.split == "train", max_df=max_df, max_features=max_features)
        _atomic_json(index_manifest, {"signature": index_signature, "stats": stats})
    else:
        stats = saved["stats"]
    retriever = load_retriever(vectorizer_path, transpose_path, target_ids_path,
                               top_k=top_k, threshold=threshold, batch_size=batch_size,
                               threads=args.threads or config["threads"])
    skip_ids = _strong_exact_ids(gate_path) if gate_path else None
    retrieval = retriever.retrieve(query_path, candidate_path, skip_s1_ids=skip_ids)
    report = {"signature": candidate_signature, "index_cache_hit": index_hit,
              "index": stats, "retrieval": retrieval, "gate_s1_count": len(skip_ids) if skip_ids else 0,
              "candidate_bytes": candidate_path.stat().st_size}
    _atomic_json(candidate_manifest, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
