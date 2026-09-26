"""Candidate union preserves evidence while bounding per-entity pair counts."""

import tempfile
import unittest
from pathlib import Path

import polars as pl

from chimera_submission.code.business_entity_resolution.src.retrieval.union import build_union


class CandidateUnionTests(unittest.TestCase):
    def test_dedup_provenance_cap_and_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pl.DataFrame({"source1_entity_id": ["S1-1", "S1-1", "S1-1"],
                          "candidate_entity_id": ["S2-1", "S2-2", "S2-3"],
                          "channel": ["exact_clean", "exact_core", "exact_core"],
                          "exact_block_size": [1, 2, 2]}).write_parquet(root / "exact.parquet")
            pl.DataFrame({"source1_entity_id": ["S1-1", "S1-1"],
                          "candidate_entity_id": ["S2-1", "S2-4"],
                          "channel": ["rare_token", "numeric_postal"],
                          "blocking_key_size": [2, 1],
                          "retrieval_score": [3.0, 2.0]}).write_parquet(root / "rare.parquet")
            pl.DataFrame({"source1_entity_id": ["S1-1", "S1-1"],
                          "candidate_entity_id": ["S2-1", "S2-2"],
                          "channel": ["tfidf_name", "tfidf_name"],
                          "tfidf_score": [0.95, 0.8],
                          "tfidf_rank": [1, 2]}).write_parquet(root / "tfidf.parquet")
            pl.DataFrame({"entity_id": ["S1-1", "S1-2"]}).write_parquet(root / "query.parquet")
            channels = [root / f"{name}.parquet" for name in ("exact", "rare", "tfidf")]
            result = build_union(channels, root / "query.parquet", root / "raw", root / "final", cap=2, shards=2)
            self.assertEqual(result["final"]["raw_pairs"], 7)
            self.assertEqual(result["final"]["unique_pairs_before_cap"], 4)
            self.assertEqual(result["final"]["final_pairs"], 2)
            self.assertEqual(result["final"]["zero_candidate_entities"], 1)
            pairs = pl.read_parquet(str(Path(result["final_dir"]) / "*.parquet"))
            self.assertEqual(pairs["candidate_entity_id"].to_list(), ["S2-1", "S2-2"])
            top = pairs.row(0, named=True)
            self.assertTrue(top["retrieved_by_exact_name"])
            self.assertTrue(top["retrieved_by_rare_token"])
            self.assertTrue(top["retrieved_by_tfidf"])
            self.assertEqual(top["retrieval_channel_count"], 3)
            self.assertEqual(top["tfidf_rank"], 1)
            cached = build_union(channels, root / "query.parquet", root / "raw", root / "final", cap=2, shards=2)
            self.assertTrue(cached["cache_hit"])
            wider = build_union(channels, root / "query.parquet", root / "raw", root / "final", cap=4, shards=2)
            self.assertTrue(wider["raw_cache_hit"])
            self.assertEqual(wider["final"]["final_pairs"], 4)

    def test_stable_tie_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pl.DataFrame({"source1_entity_id": ["S1-1", "S1-1"],
                          "candidate_entity_id": ["S2-2", "S2-1"],
                          "channel": ["exact_core", "exact_core"],
                          "exact_block_size": [2, 2]}).write_parquet(root / "exact.parquet")
            pl.DataFrame({"source1_entity_id": [], "candidate_entity_id": [],
                          "channel": [], "blocking_key_size": [], "retrieval_score": []},
                         schema={"source1_entity_id": pl.String, "candidate_entity_id": pl.String,
                                 "channel": pl.String, "blocking_key_size": pl.Int32,
                                 "retrieval_score": pl.Float32}).write_parquet(root / "rare.parquet")
            pl.DataFrame({"source1_entity_id": [], "candidate_entity_id": [],
                          "channel": [], "tfidf_score": [], "tfidf_rank": []},
                         schema={"source1_entity_id": pl.String, "candidate_entity_id": pl.String,
                                 "channel": pl.String, "tfidf_score": pl.Float32,
                                 "tfidf_rank": pl.Int16}).write_parquet(root / "tfidf.parquet")
            pl.DataFrame({"entity_id": ["S1-1"]}).write_parquet(root / "query.parquet")
            result = build_union([root / f"{name}.parquet" for name in ("exact", "rare", "tfidf")],
                                 root / "query.parquet", root / "raw", root / "final", cap=1, shards=2)
            pairs = pl.read_parquet(str(Path(result["final_dir"]) / "*.parquet"))
            self.assertEqual(pairs["candidate_entity_id"].to_list(), ["S2-1"])


if __name__ == "__main__":
    unittest.main()
