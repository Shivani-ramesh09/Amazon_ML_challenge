"""Training construction retains positives and isolates S1 validation entities."""

import tempfile
import unittest
from pathlib import Path

import polars as pl

from chimera_submission.code.business_entity_resolution.src.data.profile import split_bucket
from chimera_submission.code.business_entity_resolution.src.training.pairs import (
    label_and_split, read_truth, select_train_pairs,
)


class TrainingPairTests(unittest.TestCase):
    def test_labels_grouped_split_and_positive_retention(self):
        with tempfile.TemporaryDirectory() as directory:
            truth_path = Path(directory) / "truth.tsv"
            truth_path.write_text("source1_entity_id\tmatched_entity_ids\n"
                                  "S1-1\tS2-1,S3-1\nS1-2\t\n")
            truth = read_truth(truth_path)
            frame = pl.DataFrame({
                "source1_entity_id": ["S1-1"] * 8 + ["S1-2"] * 3,
                "candidate_entity_id": ["S2-1", "S3-1"] + [f"S2-{i}" for i in range(3, 9)] +
                                       ["S2-9", "S2-10", "S2-11"],
                "name_ratio": [1.0, 0.9] + [0.8, 0.7, 0.6, 0.5, 0.4, 0.3] + [0.9, 0.2, 0.1],
                "name_token_set_ratio": [0.8] * 11, "address_ratio": [0.7] * 11,
                "name_core_exact": [0.0] * 11, "name_clean_exact": [0.0] * 11,
                "best_tfidf_score": [0.0] * 11, "retrieval_channel_count": [1.0] * 11,
                "postal_same": [0.0] * 11, "house_same": [0.0] * 11,
            })
            validation_ids = {"S1-2"}
            labeled = label_and_split(frame, truth, validation_ids)
            self.assertEqual(labeled.filter(pl.col("label")).height, 2)
            self.assertEqual(labeled.filter(pl.col("is_validation")).height, 3)
            selected, stats = select_train_pairs(labeled.filter(~pl.col("is_validation")),
                                                 negative_ratio=2, hard_fraction=0.75)
            self.assertEqual(stats["positive_pairs"], 2)
            self.assertEqual(stats["hard_negative_pairs"], 3)
            self.assertEqual(stats["easy_negative_pairs"], 1)
            self.assertEqual(selected.filter(pl.col("label")).height, 2)
            self.assertEqual(selected.height, 6)
            selected_again, _ = select_train_pairs(labeled.filter(~pl.col("is_validation")),
                                                   negative_ratio=2, hard_fraction=0.75)
            self.assertEqual(set(selected["candidate_entity_id"]), set(selected_again["candidate_entity_id"]))
            self.assertNotEqual(split_bucket("S1-1", validation_percent=100), "train")

    def test_validation_not_subsampled_and_policy_checked(self):
        frame = pl.DataFrame({"source1_entity_id": ["S1-1"], "candidate_entity_id": ["S2-1"],
                              "label": [False], "is_validation": [True]})
        with self.assertRaises(ValueError):
            select_train_pairs(frame)
        with self.assertRaises(ValueError):
            select_train_pairs(frame.filter(~pl.col("is_validation")), negative_ratio=0)


if __name__ == "__main__":
    unittest.main()
