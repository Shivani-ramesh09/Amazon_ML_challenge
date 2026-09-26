import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from chimera_submission.code.business_entity_resolution.src.blocking.rare_numeric import RareNumericIndex, build_and_retrieve


def write_rows(path, rows):
    keys = ["entity_id", "name_core", "address_clean", "postal_candidate", "house_number", "country_key"]
    pq.write_table(pa.table({key: pa.array([row[i] for row in rows], type=pa.string()) for i, key in enumerate(keys)}), path)


class RareNumericTests(unittest.TestCase):
    def test_cache_reuses_target_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target, query = root / "target.parquet", root / "query.parquet"
            index_path, candidate_path = root / "index.pkl", root / "candidate.parquet"
            write_rows(target, [("S2-1", "alpha bakery", "7 lane", "", "7", "us")])
            write_rows(query, [("S1-1", "alpha bakery", "7 lane", "", "7", "us")])
            first = build_and_retrieve([target], query, index_path, candidate_path)
            self.assertGreater(first["retrieval"]["rare_token_pairs"], 0)
            self.assertTrue(build_and_retrieve([target], query, index_path, candidate_path)["cache_hit"])
            write_rows(query, [("S1-2", "alpha bakery", "7 lane", "", "7", "us")])
            changed = build_and_retrieve([target], query, index_path, candidate_path)
            self.assertTrue(changed["index_cache_hit"])

    def test_rare_and_name_plus_house(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target, query, output = root / "target.parquet", root / "query.parquet", root / "output.parquet"
            write_rows(target, [
                ("S2-1", "acme coffee", "12 elm road", "", "12", "us"),
                ("S2-2", "acme laundry", "99 elm road", "", "99", "us"),
                ("S3-1", "globex coffee", "12 oak road", "", "12", "france"),
            ])
            write_rows(query, [
                ("S1-1", "acme", "12 elm road", "", "12", "us"),
                ("S1-2", "globex", "12 oak road", "", "12", "france"),
            ])
            index = RareNumericIndex(rare_max_df=1, composite_max_df=5).fit([target])
            index.retrieve(query, output)
            pairs = pq.read_table(output).to_pylist()
            self.assertIn(("S1-1", "S2-1", "numeric_postal"),
                          {(p["source1_entity_id"], p["candidate_entity_id"], p["channel"]) for p in pairs})
            self.assertIn(("S1-2", "S3-1", "rare_token"),
                          {(p["source1_entity_id"], p["candidate_entity_id"], p["channel"]) for p in pairs})
            self.assertFalse(any(p["channel"] == "rare_token" and p["source1_entity_id"] == "S1-1" for p in pairs))

    def test_candidate_cap_and_no_number_only_join(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target, query, output = root / "target.parquet", root / "query.parquet", root / "output.parquet"
            write_rows(target, [(f"S2-{i}", "alpha business", f"7 lane {i}", "", "7", "us") for i in range(1, 8)])
            write_rows(query, [("S1-1", "alpha business", "7 lane", "", "7", "us"),
                               ("S1-2", "unrelated", "7 lane", "", "7", "us")])
            index = RareNumericIndex(rare_max_df=7, composite_max_df=7, rare_cap=3, numeric_cap=2).fit([target])
            index.retrieve(query, output)
            pairs = pq.read_table(output).to_pylist()
            self.assertLessEqual(sum(p["channel"] == "rare_token" for p in pairs), 3)
            self.assertLessEqual(sum(p["channel"] == "numeric_postal" for p in pairs), 2)
            self.assertFalse(any(p["source1_entity_id"] == "S1-2" for p in pairs))


if __name__ == "__main__":
    unittest.main()
