"""Cache one sorted score/list row per held-out S1, including no-candidate rows."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import resource
import time

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from chimera_submission.code.business_entity_resolution.src.data.profile import split_bucket
from chimera_submission.code.business_entity_resolution.src.retrieval.union import _s1_bucket


GROUP_SCHEMA = pa.schema([
    ("source1_entity_id", pa.string()), ("country", pa.string()),
    ("candidate_ids", pa.list_(pa.string())), ("scores", pa.list_(pa.float32())),
    ("channel_counts", pa.list_(pa.float32())), ("tfidf_ranks", pa.list_(pa.float32())),
    ("candidate_count", pa.int16()), ("top_score", pa.float32()),
    ("second_score", pa.float32()), ("third_score", pa.float32()),
    ("top_second_gap", pa.float32()), ("mean_top_3", pa.float32()),
    ("top_channel_count", pa.float32()), ("top_tfidf_rank", pa.float32()),
])


def group_scores(prediction_dir: Path, normalized_s1: Path, output_dir: Path,
                 *, seed: int = 42, validation_percent: int = 15, shards: int = 64) -> dict:
    started_utc = datetime.now(timezone.utc).isoformat()
    start = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = {bucket: [] for bucket in range(shards)}
    query = pq.read_table(normalized_s1, columns=["entity_id", "country"])
    for s1_id, country in zip(query["entity_id"].to_pylist(), query["country"].to_pylist()):
        if split_bucket(s1_id, seed=seed, validation_percent=validation_percent) == "validation":
            selected[_s1_bucket(s1_id, shards)].append((s1_id, country or ""))
    counts = Counter()
    for bucket in range(shards):
        query_rows = selected[bucket]
        if not query_rows:
            continue
        predictions = prediction_dir / f"part-{bucket:03d}.parquet"
        groups = {}
        if predictions.is_file():
            frame = pl.read_parquet(predictions)
            counts["input_pairs"] += frame.height
            frame = frame.sort(["source1_entity_id", "score", "candidate_entity_id"],
                               descending=[False, True, False])
            grouped = frame.group_by("source1_entity_id", maintain_order=True).agg(
                pl.col("candidate_entity_id").alias("candidate_ids"),
                pl.col("score").alias("scores"),
                pl.col("retrieval_channel_count").alias("channel_counts"),
                pl.col("tfidf_rank").alias("tfidf_ranks"),
            )
            groups = {row["source1_entity_id"]: row for row in grouped.to_dicts()}
        output = {name: [] for name in GROUP_SCHEMA.names}
        for s1_id, country in query_rows:
            group = groups.get(s1_id)
            ids = group["candidate_ids"] if group else []
            scores = group["scores"] if group else []
            channels = group["channel_counts"] if group else []
            ranks = group["tfidf_ranks"] if group else []
            count = len(ids)
            output["source1_entity_id"].append(s1_id)
            output["country"].append(country)
            output["candidate_ids"].append(ids)
            output["scores"].append(scores)
            output["channel_counts"].append(channels)
            output["tfidf_ranks"].append(ranks)
            output["candidate_count"].append(count)
            output["top_score"].append(scores[0] if count else 0.0)
            output["second_score"].append(scores[1] if count > 1 else 0.0)
            output["third_score"].append(scores[2] if count > 2 else 0.0)
            output["top_second_gap"].append(scores[0] - scores[1] if count > 1 else scores[0] if count else 0.0)
            output["mean_top_3"].append(sum(scores[:3]) / min(count, 3) if count else 0.0)
            output["top_channel_count"].append(channels[0] if count else 0.0)
            output["top_tfidf_rank"].append(ranks[0] if count else 0.0)
            counts["entities"] += 1
            counts["zero_candidate_entities"] += count == 0
            counts["output_pairs"] += count
        pq.write_table(pa.Table.from_pydict(output, schema=GROUP_SCHEMA),
                       output_dir / f"part-{bucket:03d}.parquet", compression="zstd")
        counts["shard_files"] += 1
    if counts["input_pairs"] != counts["output_pairs"]:
        raise ValueError("grouping lost validation candidate scores")
    return {**dict(counts), "elapsed_seconds": round(time.perf_counter() - start, 3),
            "started_utc": started_utc, "ended_utc": datetime.now(timezone.utc).isoformat(),
            "peak_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2, 3),
            "artifact_bytes": sum(path.stat().st_size for path in output_dir.glob("part-*.parquet")),
            "actual_threads": 1}
