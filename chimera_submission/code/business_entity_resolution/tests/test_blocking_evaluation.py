"""Check pair and entity blocking metrics, including singleton and multi-match."""

import tempfile
import unittest
from pathlib import Path

import polars as pl

from chimera_submission.code.business_entity_resolution.src.evaluation.blocking import evaluate_blocking
from chimera_submission.code.business_entity_resolution.src.retrieval.union import build_union


class BlockingEvaluationTests(unittest.TestCase):
    def test_multi_match_singleton_and_empty_entity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            normalized = root / "normalized"
            normalized.mkdir()
            pl.DataFrame({"entity_id": ["S1-1", "S1-2", "S1-3"],
                          "country": ["US", "FRANCE", "INDIA"]}).write_parquet(normalized / "train_s1.parquet")
            pl.DataFrame({"source1_entity_id": ["S1-1", "S1-1", "S1-2"],
                          "candidate_entity_id": ["S2-1", "S2-2", "S2-9"],
                          "channel": ["exact_clean", "exact_core", "tfidf_name"],
                          "exact_block_size": [1, 2, None],
                          "tfidf_score": [None, None, 0.9],
                          "tfidf_rank": [None, None, 1]}).write_parquet(root / "channels.parquet")
            (root / "truth.tsv").write_text("source1_entity_id\tmatched_entity_ids\n"
                                            "S1-1\tS2-1,S3-1\nS1-2\t\nS1-3\tS2-3\n")
            result = build_union([root / "channels.parquet"], normalized / "train_s1.parquet",
                                 root / "raw", root / "final", cap=2, shards=2)
            report = evaluate_blocking(normalized, root / "truth.tsv", Path(result["raw_dir"]),
                                       {2: Path(result["final_dir"])}, shards=2, sampled_targets=False)
            metric = report["metrics"]["cap_2"]
            self.assertEqual(metric["gt_pairs_recovered"], 1)
            self.assertEqual(metric["pair_candidate_recall"], 1 / 3)
            self.assertEqual(metric["entity_any_hit_recall"], 1 / 2)
            self.assertEqual(metric["entity_complete_recall"], 0)
            self.assertEqual(metric["zero_candidate_entities"], 1)
            self.assertEqual(metric["max_candidates"], 2)
            self.assertAlmostEqual(metric["conditional_oracle_macro_f0_5"], (1.25 / 2.25 + 1) / 3)
            self.assertEqual(report["true_singletons"], 1)
            self.assertEqual(report["exclusive_gt_pairs_by_channel"]["exact_clean"], 1)
            self.assertEqual(report["original_cardinality_slices"]["many"]["s1_rows"], 1)


if __name__ == "__main__":
    unittest.main()
