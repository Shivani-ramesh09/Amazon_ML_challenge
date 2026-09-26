"""Error tags cover retrieval, decision, singleton and ambiguity cases."""

import unittest

from chimera_submission.code.business_entity_resolution.src.evaluation.errors import classify_entity


class ErrorAnalysisTests(unittest.TestCase):
    def test_retrieval_and_missing_extra_match(self):
        tags = classify_entity({"a", "b"}, ["a", "c"], [0.9, 0.8], ["a"])
        self.assertIn("retrieval_miss", tags)
        self.assertIn("missing_extra_multi_match", tags)
        self.assertNotIn("singleton_false_positive", tags)

    def test_singleton_false_positive_and_wrong_top(self):
        self.assertIn("singleton_false_positive", classify_entity(set(), ["a"], [0.9], ["a"]))
        tags = classify_entity({"a"}, ["b", "a"], [0.99, 0.4], ["b"],
                               {"b": {"name_core_exact": 1, "address_ratio": 0.2}})
        self.assertIn("wrong_top_candidate", tags)
        self.assertIn("true_candidate_ranked_below_false", tags)
        self.assertIn("common_name_ambiguity", tags)
        self.assertIn("retrieved_false_negative", tags)

    def test_numeric_and_address_conflicts(self):
        tags = classify_entity({"a"}, ["a"], [0.1], [],
                               {"a": {"address_ratio": 0.2, "postal_conflict": 1,
                                      "house_conflict": 0}})
        self.assertIn("address_conflict", tags)
        self.assertIn("numeric_postal_conflict", tags)
        self.assertIn("false_negative_entity", tags)


if __name__ == "__main__":
    unittest.main()
