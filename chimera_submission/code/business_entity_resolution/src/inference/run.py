"""Run cache-aware frozen test normalization, retrieval, features, and scoring."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

from chimera_submission.code.business_entity_resolution.src.config import load_config
from chimera_submission.code.business_entity_resolution.src.evaluation.blocking import _latest_manifest
from chimera_submission.code.business_entity_resolution.src.inference.score import score_candidates


PREFIX = "chimera_submission.code.business_entity_resolution.src"


def _stage(module: str, config_path: Path, modulus: int, *args: str) -> dict:
    command = [sys.executable, "-m", f"{PREFIX}.{module}", "--config", str(config_path),
               "--split", "test", "--sample-modulus", str(modulus), *args]
    start = time.perf_counter()
    run = subprocess.run(command, text=True, capture_output=True, check=False)
    if run.returncode:
        raise RuntimeError(f"{module} failed ({run.returncode}): {run.stderr[-3000:]}\n{run.stdout[-1000:]}")
    result = json.loads(run.stdout)
    return {"module": module, "elapsed_seconds": round(time.perf_counter() - start, 3),
            "cache_hit": bool(result.get("cache_hit", False)),
            "peak_child_rss_gb_cumulative": round(
                resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024**2, 3)}


def run_inference(frozen_manifest: Path, config_path: Path, *, batch_size: int | None = None,
                  sample_modulus: int | None = None) -> dict:
    frozen = json.loads(frozen_manifest.read_text())
    config = load_config(config_path)
    if sample_modulus is not None:
        config["sample_modulus"] = sample_modulus
    if config != frozen["config"]:
        raise ValueError("inference config differs from frozen training config")
    with Path(frozen["vectorizer_path"]).open("rb") as file:
        if hashlib.file_digest(file, "sha256").hexdigest() != frozen["vectorizer_sha256"]:
            raise ValueError("train-fitted vectorizer differs from frozen manifest")
    modulus = config["sample_modulus"]
    sample = "full" if modulus == 1 else f"sample_{modulus}"
    root = Path(config["artifact_dir"])
    stages = []
    start = time.perf_counter()
    for source in (1, 2, 3):
        stages.append(_stage("normalization.run", config_path, modulus, "--source", str(source)))
    stages.append(_stage("blocking.run_exact", config_path, modulus))
    stages.append(_stage("blocking.run_rare_numeric", config_path, modulus))
    if config.get("address_channel", False):
        stages.append(_stage("retrieval.run_exact_address", config_path, modulus))
    stages.append(_stage("retrieval.run_tfidf", config_path, modulus,
                         "--threads", str(config["threads"])))
    stages.append(_stage("retrieval.run_union", config_path, modulus,
                         "--cap", str(config["candidate_cap"])))
    stages.append(_stage("features.run", config_path, modulus,
                         "--cap", str(config["candidate_cap"])))
    feature_manifest = _latest_manifest(root / "features" / sample / "test", "*/manifest.json")
    scored = score_candidates(frozen_manifest, feature_manifest, root,
                              batch_size=batch_size or config["pair_batch_size"])
    normalized = json.loads((root / "normalized" / sample / "test_s1.json").read_text())
    candidate = json.loads(Path(scored["candidate_manifest"]).read_text())
    if candidate["cap"] != config["candidate_cap"]:
        raise ValueError("final candidate cap differs from frozen config")
    if candidate["final"]["final_pairs"] != scored["scored_pairs"]:
        raise AssertionError("unscored test candidate")
    report = {"frozen_manifest": str(frozen_manifest), "config_path": str(config_path),
              "sample": sample, "test_s1_rows": normalized["written_rows"],
              "test_candidates": candidate["final"]["final_pairs"],
              "test_scored_pairs": scored["scored_pairs"],
              "test_zero_candidate_entities": candidate["final"]["zero_candidate_entities"],
              "score_manifest": str(Path(scored["score_dir"]) / "manifest.json"),
              "stages": stages, "scoring_seconds": scored["elapsed_seconds"],
              "elapsed_seconds": round(time.perf_counter() - start, 3),
              "peak_child_rss_gb_cumulative": round(
                  resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024**2, 3)}
    output = root / "reports" / "inference" / sample / "test.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--frozen-manifest", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--sample-modulus", type=int, default=None)
    args = parser.parse_args()
    print(json.dumps(run_inference(args.frozen_manifest, args.config,
                                   batch_size=args.batch_size,
                                   sample_modulus=args.sample_modulus), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
