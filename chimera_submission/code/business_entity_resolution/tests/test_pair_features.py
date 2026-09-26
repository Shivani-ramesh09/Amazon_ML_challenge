"""Exact, missing, contradiction and open-country pair features."""

import unittest

import pyarrow as pa

from chimera_submission.code.business_entity_resolution.src.features.cpu import CpuPairFeatureBuilder, ROW_COLUMNS


class PairFeatureTests(unittest.TestCase):
    def test_feature_values_and_missingness(self):
        source = pa.Table.from_pydict({
            "entity_id": ["S1-1", "S1-2"], "name_clean": ["alpha and co", ""],
            "name_core": ["alpha", ""], "address_clean": ["12 main street", ""],
            "address_numbers": ["12", ""], "postal_candidate": ["75001", ""],
            "house_number": ["12", ""], "country_key": ["france", ""],
        }).select(ROW_COLUMNS)
        target = pa.Table.from_pydict({
            "entity_id": ["S2-1", "S2-2"], "name_clean": ["alpha and co", "beta"],
            "name_core": ["alpha", "beta"], "address_clean": ["12 main street", "99 rue"],
            "address_numbers": ["12", "99"], "postal_candidate": ["75001", "75002"],
            "house_number": ["12", "99"], "country_key": ["france", "us"],
        }).select(ROW_COLUMNS)
        candidates = pa.RecordBatch.from_pydict({
            "source1_entity_id": ["S1-1", "S1-1", "S1-2"],
            "candidate_entity_id": ["S2-1", "S2-2", "S2-2"],
            "retrieved_by_exact_name": [True, False, False],
            "retrieved_by_core_name": [True, False, False],
            "retrieved_by_rare_token": [False, True, False],
            "retrieved_by_number_postal": [True, False, False],
            "retrieved_by_tfidf": [True, True, False],
            "retrieval_channel_count": [4, 2, 0], "exact_block_size": [1, None, None],
            "best_tfidf_score": [0.99, 0.7, None], "tfidf_rank": [1, 2, None],
            "best_cheap_score": [4.0, 2.0, None], "preliminary_score": [10.0, 5.0, 0.0],
        })
        builder = CpuPairFeatureBuilder({"S1-1": 0, "S1-2": 1}, {"S2-1": 0, "S2-2": 1})
        result = builder.transform(candidates, source, target).to_pydict()
        self.assertEqual(result["name_core_exact"], [1.0, 0.0, 0.0])
        self.assertEqual(result["address_exact"], [1.0, 0.0, 0.0])
        self.assertEqual(result["number_intersection"], [1.0, 0.0, 0.0])
        self.assertEqual(result["postal_same"], [1.0, 0.0, 0.0])
        self.assertEqual(result["postal_conflict"], [0.0, 1.0, 0.0])
        self.assertEqual(result["house_conflict"], [0.0, 1.0, 0.0])
        self.assertEqual(result["country_same"], [1.0, 0.0, 0.0])
        self.assertEqual(result["country_different"], [0.0, 1.0, 0.0])
        self.assertEqual(result["country_missing"], [0.0, 0.0, 1.0])
        self.assertEqual(result["best_tfidf_score"], [0.9900000095367432, 0.699999988079071, 0.0])
        self.assertEqual(result["retrieved_by_exact_name"], [1.0, 0.0, 0.0])

    def test_unknown_candidate_rejected(self):
        row = pa.Table.from_pydict({name: ["a"] for name in ROW_COLUMNS})
        candidate = pa.RecordBatch.from_pydict({"source1_entity_id": ["missing"],
                                               "candidate_entity_id": ["a"]})
        builder = CpuPairFeatureBuilder({"a": 0}, {"a": 0})
        with self.assertRaises(ValueError):
            builder.transform(candidate, row, row)


if __name__ == "__main__":
    unittest.main()
