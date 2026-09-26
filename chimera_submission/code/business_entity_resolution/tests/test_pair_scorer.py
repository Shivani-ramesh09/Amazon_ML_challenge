"""CPU scorer accepts fixed schema, early-stops and round-trips model scores."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from chimera_submission.code.business_entity_resolution.src.scoring.lightgbm_cpu import LightGBMPairScorer


class PairScorerTests(unittest.TestCase):
    def test_fit_predict_save_load(self):
        rng = np.random.default_rng(42)
        train_x = rng.normal(size=(300, 3)).astype(np.float32)
        val_x = rng.normal(size=(100, 3)).astype(np.float32)
        train_y = (train_x[:, 0] + train_x[:, 1] > 0).astype(np.int8)
        val_y = (val_x[:, 0] + val_x[:, 1] > 0).astype(np.int8)
        scorer = LightGBMPairScorer(threads=2, max_rounds=40, early_stopping_rounds=5)
        scorer.fit(train_x, train_y, val_x, val_y, ["a", "b", "c"])
        scores = scorer.predict(val_x)
        self.assertEqual(scores.shape, (100,))
        self.assertTrue(np.all((scores >= 0) & (scores <= 1)))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.txt"
            scorer.save(path)
            loaded = LightGBMPairScorer.load(path, threads=2)
            np.testing.assert_allclose(scores, loaded.predict(val_x), rtol=1e-5)

    def test_shape_and_class_checks(self):
        scorer = LightGBMPairScorer(threads=1)
        with self.assertRaises(ValueError):
            scorer.predict(np.ones((2, 2), dtype=np.float32))
        with self.assertRaises(ValueError):
            scorer.fit(np.ones((4, 2), dtype=np.float32), np.ones(4),
                       np.ones((4, 2), dtype=np.float32), np.ones(4), ["only_one"])


if __name__ == "__main__":
    unittest.main()
