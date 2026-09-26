"""Threshold search uses the exact entity set objective and improves a hand case."""

import unittest

import numpy as np

from chimera_submission.code.business_entity_resolution.src.entity_decision.optimize import evaluate_policy, optimize_policy
from chimera_submission.code.business_entity_resolution.src.entity_decision.policy import DecisionPolicy
from chimera_submission.code.business_entity_resolution.src.evaluation.entity import entity_f0_5


class ThresholdOptimizationTests(unittest.TestCase):
    def test_vectorized_matches_exact_zero_one_many(self):
        data = {"scores": np.array([[0.99, 0.2, -1], [0.9, 0.85, 0.7],
                                     [0.4, -1, -1], [-1, -1, -1]], dtype=np.float32),
                "truth_mask": np.array([[True, False, False], [True, True, False],
                                          [False, False, False], [False, False, False]]),
                "truth_counts": np.array([1, 2, 0, 0], dtype=np.int16)}
        policy = DecisionPolicy(singleton_threshold=0.5, strong_match_threshold=0.95,
                                pair_threshold=0.5, gap_threshold=0.2, max_match_count=3)
        actual = evaluate_policy(data, policy)
        expected = np.mean([entity_f0_5({"a"}, {"a"}),
                            entity_f0_5({"a", "b"}, {"a", "b", "c"}),
                            entity_f0_5(set(), set()), entity_f0_5(set(), set())])
        self.assertAlmostEqual(actual["macro_f0_5"], expected)
        self.assertEqual(actual["counts"]["correct_singletons"], 2)

    def test_search_reduces_false_singleton_links(self):
        scores = np.vstack([np.tile(np.array([0.55, -1, -1], dtype=np.float32), (30, 1)),
                            np.tile(np.array([0.92, -1, -1], dtype=np.float32), (20, 1))])
        truth = np.zeros_like(scores, dtype=bool)
        truth[30:, 0] = True
        data = {"scores": scores, "truth_mask": truth,
                "truth_counts": np.array([0] * 30 + [1] * 20, dtype=np.int16)}
        result = optimize_policy(data)
        self.assertGreater(result["selected_metrics"]["macro_f0_5"],
                           result["baseline_metrics"]["macro_f0_5"])
        self.assertEqual(result["selected_metrics"]["singleton_false_positive_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
