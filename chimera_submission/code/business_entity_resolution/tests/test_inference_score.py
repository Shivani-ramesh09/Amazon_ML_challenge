"""Frozen inference scores the actual capped candidate set exactly once."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from chimera_submission.code.business_entity_resolution.src.features.cpu import FEATURE_NAMES
from chimera_submission.code.business_entity_resolution.src.inference.score import score_candidates
from chimera_submission.code.business_entity_resolution.src.scoring.lightgbm_cpu import LightGBMPairScorer


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class InferenceScoreTests(unittest.TestCase):
    def test_scores_all_final_pairs_and_reuses_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_x = np.zeros((120, len(FEATURE_NAMES)), dtype=np.float32)
            train_x[:, 0] = np.arange(120) / 120
            labels = (train_x[:, 0] > 0.5).astype(np.int8)
            scorer = LightGBMPairScorer(threads=2)
            scorer.fit_full(train_x, labels, FEATURE_NAMES, rounds=5)
            model = root / "model.txt"
            scorer.save(model)
            vectorizer = root / "vectorizer.pkl"
            vectorizer.write_bytes(b"frozen-train-vectorizer")
            frozen = root / "frozen.json"
            frozen.write_text(json.dumps({"model_path": str(model), "model_sha256": _sha(model),
                                          "vectorizer_path": str(vectorizer),
                                          "vectorizer_sha256": _sha(vectorizer),
                                          "feature_names": FEATURE_NAMES,
                                          "config": {"sample_modulus": 16, "threads": 2},
                                          "signature": "frozen-test"}))
            candidates = root / "candidate.json"
            candidates.write_text(json.dumps({"signature": "candidate-test", "final": {"final_pairs": 2}}))
            feature_dir = root / "features"
            feature_dir.mkdir()
            values = {name: [0.1, 0.9] if name == FEATURE_NAMES[0] else [0.0, 0.0]
                      for name in FEATURE_NAMES}
            pq.write_table(pa.Table.from_pydict({"source1_entity_id": ["S1-1", "S1-1"],
                                                  "candidate_entity_id": ["S2-1", "S2-2"],
                                                  **values}), feature_dir / "part-000-batch-00000.parquet")
            features = root / "features.json"
            features.write_text(json.dumps({"candidate_manifest": str(candidates),
                                            "signature": "feature-test", "output_dir": str(feature_dir),
                                            "stats": {"output_pairs": 2, "feature_names": FEATURE_NAMES}}))
            result = score_candidates(frozen, features, root, batch_size=1)
            self.assertEqual(result["scored_pairs"], 2)
            self.assertTrue(score_candidates(frozen, features, root, batch_size=1)["cache_hit"])
            scored = pq.read_table(Path(result["score_dir"]) / "part-000-batch-00000.parquet")
            self.assertEqual(scored.num_rows, 2)
            vectorizer.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "vectorizer hash"):
                score_candidates(frozen, features, root, batch_size=1)

    def test_rejects_incomplete_feature_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frozen = root / "frozen.json"
            frozen.write_text(json.dumps({"feature_names": FEATURE_NAMES,
                                          "model_path": str(root / "unused"),
                                          "model_sha256": "", "vectorizer_path": str(root / "unused"),
                                          "vectorizer_sha256": ""}))
            features = root / "features.json"
            candidates = root / "candidates.json"
            candidates.write_text(json.dumps({"final": {"final_pairs": 3}}))
            features.write_text(json.dumps({"stats": {"feature_names": FEATURE_NAMES,
                                                       "output_pairs": 2},
                                            "candidate_manifest": str(candidates)}))
            with self.assertRaisesRegex(ValueError, "do not cover every final candidate"):
                score_candidates(frozen, features, root)
