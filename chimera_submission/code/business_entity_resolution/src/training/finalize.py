"""Freeze validated settings and refit the CPU scorer on all training entities."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import time

import numpy as np

from chimera_submission.code.business_entity_resolution.src.features.cpu import FEATURE_NAMES
from chimera_submission.code.business_entity_resolution.src.scoring.lightgbm_cpu import (
    LightGBMPairScorer, _matrix,
)
from chimera_submission.code.business_entity_resolution.src.training.pairs import (
    build_training_pairs, training_signature,
)


def _sha(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def finalize(feature_manifest: Path, validated_model_manifest: Path, policy_path: Path,
             normalized_s1: Path, truth_path: Path, vectorizer_path: Path,
             output_root: Path, *, config: dict, negative_ratio: int = 5,
             hard_fraction: float = 0.8) -> dict:
    """Create signed full-train pairs and refit at the validated iteration count.

    The held-out policy and best iteration remain frozen. This refit has no
    independent validation score; the previous validation report is preserved.
    """
    start = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()
    feature = json.loads(feature_manifest.read_text())
    validated = json.loads(validated_model_manifest.read_text())
    policy = json.loads(policy_path.read_text())
    training_manifest = Path(validated["training_manifest"])
    training = json.loads(training_manifest.read_text())
    if Path(training["feature_manifest"]).resolve() != feature_manifest.resolve():
        raise ValueError("validated scorer was trained on a different feature artifact")
    group_manifest = Path(policy["group_manifest"])
    grouped = json.loads(group_manifest.read_text())
    if Path(grouped["model_manifest"]).resolve() != validated_model_manifest.resolve():
        raise ValueError("selected policy was optimized for a different validated scorer")
    if validated["settings"]["seed"] != config["split_seed"]:
        raise ValueError("seed differs from validated scorer")
    if feature["stats"]["feature_names"] != FEATURE_NAMES:
        raise ValueError("feature schema differs from validated scorer")
    if not vectorizer_path.is_file() or not normalized_s1.is_file():
        raise FileNotFoundError("missing frozen train vectorizer or normalized S1")
    rounds = int(validated["stats"]["best_iteration"])
    if rounds < 1:
        raise ValueError("validated scorer has no usable best iteration")

    pair_signature = training_signature(feature_manifest, truth_path, validation_percent=0,
                                        seed=config["split_seed"], negative_ratio=negative_ratio,
                                        hard_fraction=hard_fraction)
    sample = "full" if config["sample_modulus"] == 1 else f"sample_{config['sample_modulus']}"
    full_pairs = output_root / "training" / sample / f"full_{pair_signature[:12]}"
    pairs_manifest = full_pairs / "manifest.json"
    if pairs_manifest.is_file():
        pairs = json.loads(pairs_manifest.read_text())
        if pairs.get("signature") != pair_signature:
            raise ValueError("full-pair cache signature mismatch")
    else:
        stats = build_training_pairs(feature_manifest.parent, normalized_s1, truth_path,
                                     full_pairs, validation_percent=0, seed=config["split_seed"],
                                     negative_ratio=negative_ratio, hard_fraction=hard_fraction)
        if stats["validation_pairs"] != 0 or stats["train_s1_rows"] != stats["selected_s1_rows"]:
            raise AssertionError("final training unexpectedly held out entities")
        pairs = {"signature": pair_signature, "feature_manifest": str(feature_manifest),
                 "stats": stats, "policy": {"negative_ratio": negative_ratio,
                 "hard_fraction": hard_fraction, "seed": config["split_seed"],
                 "validation_percent": 0}}
        _atomic_json(pairs_manifest, pairs)

    frozen_settings = {"rounds": rounds, "threads": config["threads"],
                       "seed": config["split_seed"],
                       "num_leaves": validated["settings"]["num_leaves"],
                       "learning_rate": validated["settings"]["learning_rate"],
                       "negative_ratio": negative_ratio, "hard_fraction": hard_fraction}
    signature = hashlib.sha256(json.dumps({
        "pairs": pair_signature, "validated_model": validated["signature"],
        "policy": policy["signature"], "vectorizer": _sha(vectorizer_path),
        "settings": frozen_settings, "code": _sha(Path(__file__)),
        "scorer_code": _sha(Path(__file__).parent.parent / "scoring" / "lightgbm_cpu.py"),
    }, sort_keys=True).encode()).hexdigest()
    model_dir = output_root / "models" / sample / f"final_{signature[:12]}"
    manifest = model_dir / "manifest.json"
    model_path = model_dir / "lightgbm.txt"
    if manifest.is_file() and model_path.is_file():
        saved = json.loads(manifest.read_text())
        if saved.get("signature") == signature and saved.get("model_sha256") == _sha(model_path):
            return {"cache_hit": True, **saved}

    matrix, labels = _matrix(full_pairs / "train")
    load_seconds = time.perf_counter() - start
    scorer = LightGBMPairScorer(threads=frozen_settings["threads"],
                               seed=frozen_settings["seed"],
                               num_leaves=frozen_settings["num_leaves"],
                               learning_rate=frozen_settings["learning_rate"])
    fit_start = time.perf_counter()
    scorer.fit_full(matrix, labels, FEATURE_NAMES, rounds=rounds)
    fit_seconds = time.perf_counter() - fit_start
    expected = scorer.predict(matrix[:min(128, len(matrix))])
    model_dir.mkdir(parents=True, exist_ok=True)
    temporary_model = model_path.with_suffix(".txt.tmp")
    scorer.save(temporary_model)
    os.replace(temporary_model, model_path)
    loaded = LightGBMPairScorer.load(model_path, threads=config["threads"])
    actual = loaded.predict(matrix[:len(expected)])
    if not np.allclose(expected, actual, rtol=1e-6, atol=1e-7):
        raise AssertionError("saved final model does not reproduce sample scores")
    report = {
        "signature": signature, "model_path": str(model_path),
        "model_sha256": _sha(model_path), "model_bytes": model_path.stat().st_size,
        "full_pairs_manifest": str(pairs_manifest),
        "validated_model_manifest": str(validated_model_manifest),
        "validation_policy_path": str(policy_path),
        "validation_macro_f0_5_pre_refit": policy["selected_metrics"]["macro_f0_5"],
        "validation_note": "pre-refit held-out score; final refit is not independently scored",
        "vectorizer_path": str(vectorizer_path), "vectorizer_sha256": _sha(vectorizer_path),
        "feature_manifest": str(feature_manifest), "feature_names": FEATURE_NAMES,
        "config": config, "settings": frozen_settings,
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                     text=True, check=False).stdout.strip() or None,
        "train_rows": pairs["stats"]["training_pairs"],
        "train_s1_rows": pairs["stats"]["train_s1_rows"],
        "load_seconds": round(load_seconds, 3), "fit_seconds": round(fit_seconds, 3),
        "elapsed_seconds": round(time.perf_counter() - start, 3),
        "started_utc": started_utc, "ended_utc": datetime.now(timezone.utc).isoformat(),
        "peak_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2, 3),
    }
    _atomic_json(manifest, report)
    return {"cache_hit": False, **report}
