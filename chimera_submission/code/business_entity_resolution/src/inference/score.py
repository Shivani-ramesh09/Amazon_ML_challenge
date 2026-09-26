"""Score every final test candidate in bounded Parquet batches."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from chimera_submission.code.business_entity_resolution.src.features.cpu import FEATURE_NAMES
from chimera_submission.code.business_entity_resolution.src.scoring.lightgbm_cpu import LightGBMPairScorer


def _sha(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def score_candidates(frozen_manifest: Path, test_feature_manifest: Path,
                     output_root: Path, *, batch_size: int = 250_000) -> dict:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    frozen = json.loads(frozen_manifest.read_text())
    features = json.loads(test_feature_manifest.read_text())
    if frozen["feature_names"] != FEATURE_NAMES or features["stats"]["feature_names"] != FEATURE_NAMES:
        raise ValueError("test feature schema differs from frozen model")
    candidate_manifest = Path(features["candidate_manifest"])
    candidates = json.loads(candidate_manifest.read_text())
    expected = candidates["final"]["final_pairs"]
    if features["stats"]["output_pairs"] != expected:
        raise ValueError("test features do not cover every final candidate")
    model_path = Path(frozen["model_path"])
    if _sha(model_path) != frozen["model_sha256"]:
        raise ValueError("frozen model hash changed")
    if _sha(Path(frozen["vectorizer_path"])) != frozen["vectorizer_sha256"]:
        raise ValueError("train-fitted vectorizer hash changed")
    signature = hashlib.sha256(json.dumps({
        "frozen": frozen["signature"], "features": features["signature"],
        "candidate": candidates["signature"], "batch_size": batch_size,
        "code": _sha(Path(__file__)),
    }, sort_keys=True).encode()).hexdigest()
    sample = "full" if frozen["config"]["sample_modulus"] == 1 else f"sample_{frozen['config']['sample_modulus']}"
    output = output_root / "predictions" / "test" / sample / signature[:12]
    manifest = output / "manifest.json"
    if manifest.is_file():
        saved = json.loads(manifest.read_text())
        if saved.get("signature") == signature and saved.get("scored_pairs") == expected:
            if all((output / item["file"]).is_file() for item in saved["shards"]):
                return {"cache_hit": True, **saved}
    started_utc = datetime.now(timezone.utc).isoformat()
    start = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    scorer = LightGBMPairScorer.load(model_path, threads=frozen["config"]["threads"])
    rows = 0
    shards = []
    feature_dir = Path(features["output_dir"])
    for path in sorted(feature_dir.glob("part-*-batch-*.parquet")):
        destination = output / path.name
        temporary = destination.with_suffix(".parquet.tmp")
        writer = None
        shard_rows = 0
        try:
            for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size):
                table = pa.Table.from_batches([batch])
                matrix = np.column_stack([
                    table[name].to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
                    for name in FEATURE_NAMES
                ])
                if not np.isfinite(matrix).all():
                    raise ValueError("nonfinite test feature")
                scores = scorer.predict(matrix)
                if not np.isfinite(scores).all():
                    raise ValueError("nonfinite test prediction")
                scored = pa.Table.from_pydict({
                    "source1_entity_id": table["source1_entity_id"],
                    "candidate_entity_id": table["candidate_entity_id"],
                    "score": pa.array(scores),
                })
                if writer is None:
                    writer = pq.ParquetWriter(temporary, scored.schema, compression="zstd")
                writer.write_table(scored)
                shard_rows += scored.num_rows
        except Exception:
            if writer is not None:
                writer.close()
            temporary.unlink(missing_ok=True)
            raise
        if writer is not None:
            writer.close()
            os.replace(temporary, destination)
            shards.append({"file": destination.name, "rows": shard_rows,
                           "bytes": destination.stat().st_size})
            rows += shard_rows
    if rows != expected:
        raise AssertionError(f"scored {rows} of {expected} final test candidates")
    report = {"signature": signature, "frozen_manifest": str(frozen_manifest),
              "feature_manifest": str(test_feature_manifest),
              "candidate_manifest": str(candidate_manifest), "score_dir": str(output),
              "scored_pairs": rows, "shards": shards,
              "artifact_bytes": sum(item["bytes"] for item in shards),
              "elapsed_seconds": round(time.perf_counter() - start, 3),
              "started_utc": started_utc, "ended_utc": datetime.now(timezone.utc).isoformat(),
              "peak_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2, 3),
              "actual_threads": frozen["config"]["threads"]}
    temporary = manifest.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, manifest)
    return {"cache_hit": False, **report}
