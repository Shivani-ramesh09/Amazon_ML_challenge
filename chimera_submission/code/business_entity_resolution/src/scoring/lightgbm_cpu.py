"""Bounded CPU LightGBM training and sharded validation prediction."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import resource
import time

import lightgbm as lgb
import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score, precision_score, recall_score

from chimera_submission.code.business_entity_resolution.src.features.cpu import FEATURE_NAMES


class LightGBMPairScorer:
    def __init__(self, *, threads: int = 8, seed: int = 42, num_leaves: int = 63,
                 learning_rate: float = 0.05, max_rounds: int = 600, early_stopping_rounds: int = 50):
        self.threads = threads
        self.seed = seed
        self.num_leaves = num_leaves
        self.learning_rate = learning_rate
        self.max_rounds = max_rounds
        self.early_stopping_rounds = early_stopping_rounds
        self.model: lgb.Booster | None = None

    def fit(self, train_x: np.ndarray, train_y: np.ndarray,
            validation_x: np.ndarray, validation_y: np.ndarray,
            feature_names: list[str]) -> None:
        if train_x.shape[1] != len(feature_names) or validation_x.shape[1] != len(feature_names):
            raise ValueError("feature matrix does not match feature names")
        if len(np.unique(train_y)) < 2 or len(np.unique(validation_y)) < 2:
            raise ValueError("training and validation each need both classes")
        train = lgb.Dataset(train_x, label=train_y, feature_name=feature_names, free_raw_data=True)
        validation = lgb.Dataset(validation_x, label=validation_y, reference=train,
                                 feature_name=feature_names, free_raw_data=True)
        params = {
            "objective": "binary", "metric": "binary_logloss", "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves, "max_depth": 10, "min_data_in_leaf": 50,
            "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
            "num_threads": self.threads, "seed": self.seed,
            "feature_fraction_seed": self.seed, "bagging_seed": self.seed,
            "data_random_seed": self.seed, "verbosity": -1,
        }
        self.model = lgb.train(params, train, num_boost_round=self.max_rounds,
                               valid_sets=[validation], valid_names=["held_out_entities"],
                               callbacks=[lgb.early_stopping(self.early_stopping_rounds, verbose=False)])

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("model is not fitted")
        return np.asarray(self.model.predict(features, num_threads=self.threads,
                                             num_iteration=self.model.best_iteration), dtype=np.float32)

    def save(self, path: Path) -> None:
        if self.model is None:
            raise ValueError("model is not fitted")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(path), num_iteration=self.model.best_iteration)

    @classmethod
    def load(cls, path: Path, *, threads: int = 8) -> "LightGBMPairScorer":
        scorer = cls(threads=threads)
        scorer.model = lgb.Booster(model_file=str(path))
        return scorer


def _matrix(path: Path) -> tuple[np.ndarray, np.ndarray]:
    frame = pl.read_parquet(str(path / "*.parquet"), columns=[*FEATURE_NAMES, "label"])
    labels = frame["label"].to_numpy().astype(np.int8, copy=False)
    matrix = frame.select(FEATURE_NAMES).to_numpy().astype(np.float32, copy=False)
    if not np.isfinite(matrix).all():
        raise ValueError("nonfinite pair feature")
    return matrix, labels


def train_and_score(training_dir: Path, model_path: Path, prediction_dir: Path,
                    *, threads: int = 8, seed: int = 42, num_leaves: int = 63,
                    learning_rate: float = 0.05, max_rounds: int = 600,
                    early_stopping_rounds: int = 50, batch_size: int = 250_000) -> dict:
    started_utc = datetime.now(timezone.utc).isoformat()
    start = time.perf_counter()
    train_x, train_y = _matrix(training_dir / "train")
    val_x, val_y = _matrix(training_dir / "validation")
    load_seconds = time.perf_counter() - start
    scorer = LightGBMPairScorer(threads=threads, seed=seed, num_leaves=num_leaves,
                               learning_rate=learning_rate, max_rounds=max_rounds,
                               early_stopping_rounds=early_stopping_rounds)
    fit_start = time.perf_counter()
    scorer.fit(train_x, train_y, val_x, val_y, FEATURE_NAMES)
    fit_seconds = time.perf_counter() - fit_start
    scorer.save(model_path)
    del train_x, train_y, val_x, val_y
    prediction_dir.mkdir(parents=True, exist_ok=True)
    prediction_start = time.perf_counter()
    scores_all = []
    labels_all = []
    output_rows = 0
    for path in sorted((training_dir / "validation").glob("*.parquet")):
        writer = None
        output_path = prediction_dir / path.name
        try:
            for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size):
                frame = pa.Table.from_batches([batch])
                matrix = pl.from_arrow(frame.select(FEATURE_NAMES)).to_numpy().astype(np.float32, copy=False)
                score = scorer.predict(matrix)
                labels = frame["label"].to_numpy(zero_copy_only=False).astype(np.int8, copy=False)
                output = pa.Table.from_pydict({
                    "source1_entity_id": frame["source1_entity_id"],
                    "candidate_entity_id": frame["candidate_entity_id"],
                    "score": pa.array(score), "label": pa.array(labels),
                    "retrieval_channel_count": frame["retrieval_channel_count"],
                    "tfidf_rank": frame["tfidf_rank"],
                })
                if writer is None:
                    writer = pq.ParquetWriter(output_path, output.schema, compression="zstd")
                writer.write_table(output)
                scores_all.append(score)
                labels_all.append(labels)
                output_rows += len(score)
        finally:
            if writer is not None:
                writer.close()
    scores = np.concatenate(scores_all) if scores_all else np.empty(0, dtype=np.float32)
    labels = np.concatenate(labels_all) if labels_all else np.empty(0, dtype=np.int8)
    if not len(scores):
        raise ValueError("empty validation prediction set")
    expected_validation = json.loads((training_dir / "manifest.json").read_text())["stats"]["validation_pairs"]
    if output_rows != expected_validation or not np.isfinite(scores).all():
        raise ValueError("validation predictions are incomplete or nonfinite")
    predicted = scores >= 0.5
    booster = scorer.model
    assert booster is not None
    importance = sorted(zip(FEATURE_NAMES, booster.feature_importance(importance_type="gain")),
                        key=lambda value: value[1], reverse=True)
    return {
        "train_rows": int(json.loads((training_dir / "manifest.json").read_text())["stats"]["training_pairs"]),
        "validation_rows": output_rows, "validation_positive_pairs": int(labels.sum()),
        "validation_pair_average_precision": float(average_precision_score(labels, scores)),
        "validation_pair_precision_at_0_5": float(precision_score(labels, predicted, zero_division=0)),
        "validation_pair_recall_at_0_5": float(recall_score(labels, predicted, zero_division=0)),
        "best_iteration": booster.best_iteration,
        "best_validation_logloss": float(booster.best_score["held_out_entities"]["binary_logloss"]),
        "feature_importance_gain": [{"feature": name, "gain": float(gain)} for name, gain in importance],
        "load_seconds": round(load_seconds, 3), "fit_seconds": round(fit_seconds, 3),
        "prediction_seconds": round(time.perf_counter() - prediction_start, 3),
        "elapsed_seconds": round(time.perf_counter() - start, 3),
        "started_utc": started_utc, "ended_utc": datetime.now(timezone.utc).isoformat(),
        "peak_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2, 3),
        "model_bytes": model_path.stat().st_size,
        "prediction_bytes": sum(path.stat().st_size for path in prediction_dir.glob("*.parquet")),
        "actual_threads": threads,
    }
