"""Exact-address retrieval handles cross-script names and common blocks safely."""

import tempfile
import unittest
from pathlib import Path

import polars as pl

from chimera_submission.code.business_entity_resolution.src.retrieval.exact_address import (
    CpuExactAddressRetriever, build_and_retrieve,
)


class ExactAddressTests(unittest.TestCase):
    def test_cross_script_same_address_and_open_country(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pl.DataFrame({"entity_id": ["S2-1", "S3-1", "S2-2"],
                          "address_clean": ["12 mg road pune", "12 mg road pune", "99 main street"],
                          "name_core": ["देवनागरी", "different", "other"],
                          "house_number": ["12", "12", "99"],
                          "postal_candidate": ["", "", "75001"],
                          "country_key": ["india", "france", "france"]}).write_parquet(root / "targets.parquet")
            pl.DataFrame({"entity_id": ["S1-1"], "address_clean": ["12 mg road pune"],
                          "name_core": ["latin name"], "house_number": ["12"],
                          "postal_candidate": [""], "country_key": ["india"]}).write_parquet(root / "query.parquet")
            report = build_and_retrieve([root / "targets.parquet"], root / "query.parquet",
                                        root / "index.pkl", root / "pairs.parquet")
            self.assertEqual(report["retrieval"]["candidate_pairs"], 1)
            pairs = pl.read_parquet(root / "pairs.parquet")
            self.assertEqual(pairs["candidate_entity_id"].to_list(), ["S2-1"])
            self.assertEqual(pairs["channel"].to_list(), ["exact_address"])
            cached = build_and_retrieve([root / "targets.parquet"], root / "query.parquet",
                                        root / "index.pkl", root / "pairs.parquet")
            self.assertTrue(cached["cache_hit"])

    def test_oversized_requires_secondary_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = pl.DataFrame({"entity_id": [f"S2-{i}" for i in range(5)],
                                   "address_clean": ["same long address"] * 5,
                                   "name_core": [f"name {i}" for i in range(5)],
                                   "house_number": ["", "", "5", "", ""],
                                   "postal_candidate": [""] * 5,
                                   "country_key": ["india"] * 5})
            target.write_parquet(root / "targets.parquet")
            query = pl.DataFrame({"entity_id": ["S1-1", "S1-2"],
                                  "address_clean": ["same long address"] * 2,
                                  "name_core": ["unrelated"] * 2,
                                  "house_number": ["", "5"],
                                  "postal_candidate": [""] * 2,
                                  "country_key": ["india"] * 2})
            query.write_parquet(root / "query.parquet")
            retriever = CpuExactAddressRetriever(max_block_size=2, per_query_cap=2).fit([root / "targets.parquet"])
            stats = retriever.retrieve(root / "query.parquet", root / "pairs.parquet")
            self.assertEqual(stats["oversized_queries"], 2)
            self.assertEqual(stats["unresolved_oversized_queries"], 1)
            self.assertEqual(stats["candidate_pairs"], 1)
            self.assertEqual(pl.read_parquet(root / "pairs.parquet")["candidate_entity_id"].to_list(), ["S2-2"])


if __name__ == "__main__":
    unittest.main()
