import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from chimera_submission.code.business_entity_resolution.src.blocking.exact import ExactIndex, build_and_retrieve


def write_normalized(path, rows):
    fields = ["entity_id", "name_clean", "name_core", "address_clean", "postal_candidate", "house_number", "country_key"]
    pq.write_table(pa.table({field: pa.array([row[i] for row in rows], type=pa.string()) for i, field in enumerate(fields)}), path)


class ExactBlockingTests(unittest.TestCase):
    def test_open_country_exact_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            targets = root / "targets.parquet"
            queries = root / "queries.parquet"
            write_normalized(targets, [("S3-7", "bonjour cafe", "bonjour cafe", "7 rue paris", "", "7", "france")])
            write_normalized(queries, [("S1-7", "bonjour cafe", "bonjour cafe", "7 rue paris", "", "7", "france")])
            index = ExactIndex().fit([targets])
            index.retrieve(queries, root / "pairs.parquet")
            self.assertEqual({row["candidate_entity_id"] for row in pq.read_table(root / "pairs.parquet").to_pylist()}, {"S3-7"})

    def test_normal_block_ranks_address_evidence_before_cap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            targets = root / "targets.parquet"
            queries = root / "queries.parquet"
            write_normalized(targets, [
                (f"S2-{i}", "star enterprises", "star enterprises", f"{i} elm road", "", str(i), "india")
                for i in range(1, 5)
            ])
            write_normalized(queries, [("S1-1", "star enterprises", "star enterprises", "4 elm road", "", "4", "india")])
            index = ExactIndex(max_block_size=10, per_channel_cap=2).fit([targets])
            index.retrieve(queries, root / "pairs.parquet")
            pairs = pq.read_table(root / "pairs.parquet").to_pylist()
            self.assertIn("S2-4", {pair["candidate_entity_id"] for pair in pairs})

    def test_oversized_refines_and_reports_trim(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            targets = root / "targets.parquet"
            queries = root / "queries.parquet"
            write_normalized(targets, [
                (f"S2-{i}", "acme", "acme", f"{i} main road", "", str(i), "us") for i in range(1, 5)
            ])
            write_normalized(queries, [
                ("S1-1", "acme", "acme", "3 main road", "", "3", "us"),
                ("S1-2", "acme", "acme", "", "", "", "us"),
                ("S1-3", "", "", "", "", "", "france"),
            ])
            index = ExactIndex(max_block_size=2, per_channel_cap=2).fit([targets])
            self.assertEqual(index.stats["exact_clean"]["oversized_keys"], 1)
            result = index.retrieve(queries, root / "candidates.parquet")
            pairs = pq.read_table(root / "candidates.parquet").to_pylist()
            self.assertEqual({p["candidate_entity_id"] for p in pairs if p["source1_entity_id"] == "S1-1"}, {"S2-3"})
            self.assertEqual({p["candidate_entity_id"] for p in pairs if p["source1_entity_id"] == "S1-2"}, {"S2-1", "S2-2"})
            self.assertFalse(any(p["source1_entity_id"] == "S1-3" for p in pairs))
            self.assertEqual(result["exact_clean_trimmed_queries"], 2)

    def test_index_and_candidate_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            targets = root / "targets.parquet"
            queries = root / "queries.parquet"
            write_normalized(targets, [("S2-1", "a", "a", "1 road", "", "1", "us")])
            write_normalized(queries, [("S1-1", "a", "a", "1 road", "", "1", "us")])
            index_path, candidate_path = root / "index.pkl", root / "candidates.parquet"
            initial = build_and_retrieve([targets], queries, index_path, candidate_path)
            self.assertEqual(initial["retrieval"]["exact_clean_pairs"], 1)
            cached = build_and_retrieve([targets], queries, index_path, candidate_path)
            self.assertTrue(cached["cache_hit"])
            write_normalized(queries, [("S1-2", "a", "a", "1 road", "", "1", "us")])
            changed = build_and_retrieve([targets], queries, index_path, candidate_path)
            self.assertTrue(changed["index_cache_hit"])
            self.assertEqual(pq.read_table(candidate_path)["source1_entity_id"][0].as_py(), "S1-2")


if __name__ == "__main__":
    unittest.main()
