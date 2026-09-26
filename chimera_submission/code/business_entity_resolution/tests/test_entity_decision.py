"""Official F0.5 and zero/one/many policy with complete S1 accounting."""

import tempfile
import unittest
from pathlib import Path

import polars as pl

from chimera_submission.code.business_entity_resolution.src.entity_decision.group import group_scores
from chimera_submission.code.business_entity_resolution.src.entity_decision.policy import DecisionPolicy
from chimera_submission.code.business_entity_resolution.src.evaluation.entity import entity_f0_5, evaluate_groups


class EntityDecisionTests(unittest.TestCase):
    def test_official_f05_examples_and_policy(self):
        self.assertEqual(entity_f0_5(set(), set()), 1.0)
        self.assertEqual(entity_f0_5(set(), {"S2-1"}), 0.0)
        self.assertEqual(entity_f0_5({"S2-1"}, set()), 0.0)
        self.assertAlmostEqual(entity_f0_5({"S2-1", "S3-1"}, {"S2-1", "S3-1", "S2-2"}), 5 / 7)
        self.assertAlmostEqual(entity_f0_5({"S2-1", "S3-1"}, {"S2-1"}), 1.25 / 1.5)
        policy = DecisionPolicy(singleton_threshold=0.6, strong_match_threshold=0.95,
                                pair_threshold=0.7, gap_threshold=0.2, max_match_count=3)
        self.assertEqual(policy.decide([], []), [])
        self.assertEqual(policy.decide(["a"], [0.59]), [])
        self.assertEqual(policy.decide(["a", "b"], [0.99, 0.5]), ["a"])
        self.assertEqual(policy.decide(["a", "b", "c"], [0.99, 0.95, 0.75]), ["a", "b", "c"])
        with self.assertRaises(ValueError):
            policy.decide(["a"], [0.9, 0.8])

    def test_groups_include_no_candidate_s1_and_keep_rank_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prediction_dir = root / "predictions"
            prediction_dir.mkdir()
            pl.DataFrame({"entity_id": ["S1-1", "S1-2", "S1-3"],
                          "country": ["India", "France", "US"]}).write_parquet(root / "s1.parquet")
            pl.DataFrame({"source1_entity_id": ["S1-1", "S1-1", "S1-3"],
                          "candidate_entity_id": ["S2-2", "S2-1", "S3-1"],
                          "score": [0.7, 0.9, 0.1], "label": [False, True, False],
                          "retrieval_channel_count": [1.0, 3.0, 1.0],
                          "tfidf_rank": [2.0, 1.0, 1.0]}).write_parquet(prediction_dir / "part-001.parquet")
            stats = group_scores(prediction_dir, root / "s1.parquet", root / "groups",
                                 validation_percent=100, shards=2)
            self.assertEqual(stats["entities"], 3)
            self.assertEqual(stats["zero_candidate_entities"], 1)
            self.assertEqual(stats["input_pairs"], stats["output_pairs"])
            grouped = pl.read_parquet(str(root / "groups" / "*.parquet"))
            s1 = grouped.filter(pl.col("source1_entity_id") == "S1-1").row(0, named=True)
            self.assertEqual(s1["candidate_ids"], ["S2-1", "S2-2"])
            self.assertAlmostEqual(s1["top_second_gap"], 0.2)
            empty = grouped.filter(pl.col("source1_entity_id") == "S1-2").row(0, named=True)
            self.assertEqual(empty["candidate_ids"], [])
            metrics = evaluate_groups(root / "groups", {"S1-1": {"S2-1"}, "S1-2": set(),
                                                       "S1-3": {"S3-2"}}, DecisionPolicy())
            self.assertEqual(metrics["counts"]["entities"], 3)
            self.assertEqual(metrics["singleton_accuracy"], 1.0)
            self.assertEqual(metrics["cardinality_confusion"]["true_zero_pred_zero"], 1)


if __name__ == "__main__":
    unittest.main()
