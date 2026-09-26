"""Write exactly one matching and scored-candidate TSV row per test S1."""

from __future__ import annotations

from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import time

import pyarrow.parquet as pq

from chimera_submission.code.business_entity_resolution.src.entity_decision.policy import DecisionPolicy
from chimera_submission.code.business_entity_resolution.src.retrieval.union import _s1_bucket


def _sha(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def _pairs(path: Path):
    for batch in pq.ParquetFile(path).iter_batches(batch_size=100_000,
                                                  columns=["source1_entity_id", "candidate_entity_id"]):
        yield from zip(batch.column(0).to_pylist(), batch.column(1).to_pylist())


def _scored_pairs(paths: list[Path]):
    for path in paths:
        for batch in pq.ParquetFile(path).iter_batches(
                batch_size=100_000, columns=["source1_entity_id", "candidate_entity_id", "score"]):
            yield from zip(batch.column(0).to_pylist(), batch.column(1).to_pylist(),
                           batch.column(2).to_pylist())


def _target_ids(normalized_dir: Path) -> set[str]:
    ids = set()
    for source in (2, 3):
        for batch in pq.ParquetFile(normalized_dir / f"test_s{source}.parquet").iter_batches(
                batch_size=100_000, columns=["entity_id"]):
            for target_id in batch.column(0).to_pylist():
                if not target_id.startswith(f"S{source}-") or target_id in ids:
                    raise ValueError(f"invalid or duplicate test target ID {target_id}")
                ids.add(target_id)
    return ids


def write_submission(frozen_manifest: Path, score_manifest: Path, normalized_dir: Path,
                     output_dir: Path, *, shards: int = 64) -> dict:
    started_utc = datetime.now(timezone.utc).isoformat()
    start = time.perf_counter()
    frozen = json.loads(frozen_manifest.read_text())
    scored = json.loads(score_manifest.read_text())
    if Path(scored["frozen_manifest"]).resolve() != frozen_manifest.resolve():
        raise ValueError("scores came from a different frozen model")
    policy_path = Path(frozen["validation_policy_path"])
    if _sha(policy_path) != frozen["validation_policy_sha256"]:
        raise ValueError("frozen validation policy hash changed")
    policy = DecisionPolicy(**json.loads(policy_path.read_text())["selected_policy"])
    candidates = json.loads(Path(scored["candidate_manifest"]).read_text())
    if candidates["final"]["final_pairs"] != scored["scored_pairs"]:
        raise ValueError("score count differs from final candidate count")
    candidate_dir = Path(candidates["final_dir"])
    score_dir = Path(scored["score_dir"])
    valid_targets = _target_ids(normalized_dir)
    query_groups = [[] for _ in range(shards)]
    seen_s1 = set()
    for batch in pq.ParquetFile(normalized_dir / "test_s1.parquet").iter_batches(
            batch_size=100_000, columns=["entity_id"]):
        for s1_id in batch.column(0).to_pylist():
            if not s1_id.startswith("S1-") or s1_id in seen_s1:
                raise ValueError(f"invalid or duplicate S1 ID {s1_id}")
            seen_s1.add(s1_id)
            query_groups[_s1_bucket(s1_id, shards)].append(s1_id)
    if not seen_s1:
        raise ValueError("empty test S1 source")
    score_files = {bucket: [] for bucket in range(shards)}
    for item in scored["shards"]:
        path = score_dir / item["file"]
        bucket = int(path.name.split("-batch-")[0].split("-")[1])
        if bucket >= shards or not path.is_file():
            raise ValueError("missing or out-of-range score shard")
        score_files[bucket].append(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    matching = output_dir / "matching_results.tsv"
    candidate = output_dir / "candidate_pairs.tsv"
    matching_tmp = matching.with_suffix(".tsv.tmp")
    candidate_tmp = candidate.with_suffix(".tsv.tmp")
    counts = Counter()
    try:
        with matching_tmp.open("w", encoding="utf-8", newline="") as match_file, \
                candidate_tmp.open("w", encoding="utf-8", newline="") as candidate_file:
            match_writer = csv.writer(match_file, delimiter="\t", lineterminator="\n")
            candidate_writer = csv.writer(candidate_file, delimiter="\t", lineterminator="\n")
            match_writer.writerow(["source1_entity_id", "matched_entity_ids"])
            candidate_writer.writerow(["source1_entity_id", "candidate_entity_ids"])
            for bucket, query_ids in enumerate(query_groups):
                candidate_path = candidate_dir / f"part-{bucket:03d}.parquet"
                score_paths = sorted(score_files[bucket])
                if candidate_path.is_file() != bool(score_paths):
                    raise ValueError(f"candidate/score shard mismatch at bucket {bucket}")
                grouped = {}
                if score_paths:
                    expected_pairs = _pairs(candidate_path)
                    for source_id, target_id, score in _scored_pairs(score_paths):
                        if next(expected_pairs, None) != (source_id, target_id):
                            raise ValueError("scored pair differs from final candidate set")
                        if source_id not in seen_s1 or target_id not in valid_targets:
                            raise ValueError(f"invalid scored pair {source_id}, {target_id}")
                        grouped.setdefault(source_id, []).append((target_id, float(score)))
                        counts["scored_pairs_verified"] += 1
                    if next(expected_pairs, None) is not None:
                        raise ValueError("final candidate shard has unscored pairs")
                if not set(grouped).issubset(query_ids):
                    raise ValueError("score shard has S1 IDs assigned to another bucket")
                for s1_id in query_ids:
                    pairs = grouped.get(s1_id, [])
                    if len(pairs) > frozen["config"]["candidate_cap"]:
                        raise ValueError("candidate cap exceeded")
                    pairs.sort(key=lambda item: (-item[1], item[0]))
                    ids = [item[0] for item in pairs]
                    if len(ids) != len(set(ids)):
                        raise ValueError(f"duplicate scored candidate for {s1_id}")
                    matches = policy.decide(ids, [item[1] for item in pairs])
                    if not set(matches).issubset(ids):
                        raise AssertionError("matched IDs must be scored candidates")
                    match_writer.writerow([s1_id, ",".join(matches)])
                    candidate_writer.writerow([s1_id, ",".join(ids)])
                    counts["s1_rows"] += 1
                    counts["matched_pairs"] += len(matches)
                    counts["predicted_zero" if not matches else "predicted_one" if len(matches) == 1
                           else "predicted_many"] += 1
        if counts["s1_rows"] != len(seen_s1) or counts["scored_pairs_verified"] != scored["scored_pairs"]:
            raise AssertionError("submission coverage mismatch")
        os.replace(matching_tmp, matching)
        os.replace(candidate_tmp, candidate)
    except Exception:
        matching_tmp.unlink(missing_ok=True)
        candidate_tmp.unlink(missing_ok=True)
        raise
    report = {"frozen_manifest": str(frozen_manifest), "score_manifest": str(score_manifest),
              "matching_path": str(matching), "candidate_path": str(candidate),
              **dict(counts), "matching_bytes": matching.stat().st_size,
              "candidate_bytes": candidate.stat().st_size,
              "elapsed_seconds": round(time.perf_counter() - start, 3),
              "started_utc": started_utc, "ended_utc": datetime.now(timezone.utc).isoformat(),
              "peak_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2, 3)}
    report_path = output_dir / "submission_manifest.json"
    temporary = report_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, report_path)
    return report
