"""Make targets select only manifests linked to the intended config/model."""

import json
import tempfile
import unittest
from pathlib import Path

from chimera_submission.code.business_entity_resolution.src.output.select_manifest import select_manifest


class ManifestSelectionTests(unittest.TestCase):
    def test_frozen_and_scores_follow_same_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {"data_dir": "unused", "artifact_dir": str(root), "threads": 2,
                      "memory_budget_gb": 4, "sample_modulus": 16,
                      "validation_percent": 15, "split_seed": 42,
                      "candidate_cap": 30, "pair_batch_size": 100,
                      "profile_frequency_sample_modulus": 128}
            config_path = root / "config.yaml"
            config_path.write_text(json.dumps(config))
            model_dir = root / "models" / "sample_16" / "final_good"
            model_dir.mkdir(parents=True)
            (model_dir / "lightgbm.txt").write_text("model")
            frozen = model_dir / "manifest.json"
            frozen.write_text(json.dumps({"config": config,
                                          "model_path": str(model_dir / "lightgbm.txt")}))
            other_dir = root / "models" / "sample_16" / "final_other"
            other_dir.mkdir()
            other = other_dir / "manifest.json"
            other.write_text(json.dumps({"config": {**config, "candidate_cap": 20},
                                         "model_path": str(model_dir / "lightgbm.txt")}))
            self.assertEqual(select_manifest(config_path, 16, "frozen"), frozen)
            score_dir = root / "predictions" / "test" / "sample_16" / "scores"
            score_dir.mkdir(parents=True)
            (score_dir / "part-000.parquet").write_text("fixture")
            scores = score_dir / "manifest.json"
            scores.write_text(json.dumps({"frozen_manifest": str(frozen),
                                          "score_dir": str(score_dir),
                                          "scored_pairs": 2,
                                          "shards": [{"file": "part-000.parquet", "rows": 2}]}))
            self.assertEqual(select_manifest(config_path, 16, "scores",
                                             frozen_manifest=frozen), scores)
            scores.write_text(json.dumps({"frozen_manifest": str(other),
                                          "score_dir": str(score_dir),
                                          "scored_pairs": 2,
                                          "shards": [{"file": "part-000.parquet", "rows": 2}]}))
            with self.assertRaises(FileNotFoundError):
                select_manifest(config_path, 16, "scores", frozen_manifest=frozen)


if __name__ == "__main__":
    unittest.main()
