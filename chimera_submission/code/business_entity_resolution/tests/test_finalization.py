"""Final refit must preserve validation lineage and avoid hidden score claims."""

import json
import tempfile
import unittest
from pathlib import Path

from chimera_submission.code.business_entity_resolution.src.features.cpu import FEATURE_NAMES
from chimera_submission.code.business_entity_resolution.src.training.finalize import finalize


class FinalizationTests(unittest.TestCase):
    def test_rejects_mismatched_validation_lineage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            features = root / "features.json"
            features.write_text(json.dumps({"stats": {"feature_names": FEATURE_NAMES}}))
            prior_training = root / "prior_training.json"
            prior_training.write_text(json.dumps({"feature_manifest": str(features)}))
            model = root / "model.json"
            model.write_text(json.dumps({"training_manifest": str(prior_training),
                                         "settings": {"seed": 42}}))
            grouped = root / "grouped.json"
            grouped.write_text(json.dumps({"model_manifest": str(root / "wrong_model.json")}))
            policy = root / "policy.json"
            policy.write_text(json.dumps({"group_manifest": str(grouped)}))
            with self.assertRaisesRegex(ValueError, "different validated scorer"):
                finalize(features, model, policy, root / "s1.parquet", root / "truth.tsv",
                         root / "vectorizer.pkl", root,
                         config={"split_seed": 42, "sample_modulus": 16, "threads": 2})
