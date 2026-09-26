import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from chimera_submission.code.business_entity_resolution.src.blocking.eval_cheap import evaluate


def ids(path, values):
    pq.write_table(pa.table({"entity_id": pa.array(values, type=pa.string())}), path)


def pairs(path, values):
    pq.write_table(pa.table({
        "source1_entity_id": pa.array([v[0] for v in values], type=pa.string()),
        "candidate_entity_id": pa.array([v[1] for v in values], type=pa.string()),
        "channel": pa.array([v[2] for v in values], type=pa.string()),
    }), path)


class RetrievalAuditTests(unittest.TestCase):
    def test_cheap_and_fuzzy_unions_stay_distinct(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ids(root / "train_s1.parquet", ["S1-1", "S1-2", "S1-3"])
            ids(root / "train_s2.parquet", ["S2-1", "S2-2"])
            ids(root / "train_s3.parquet", [])
            exact, cheap, fuzzy = root / "exact.parquet", root / "cheap.parquet", root / "fuzzy.parquet"
            pairs(exact, [("S1-1", "S2-1", "exact_core")])
            pairs(cheap, [])
            pairs(fuzzy, [("S1-1", "S2-2", "tfidf_name")])
            truth = root / "truth.tsv"
            truth.write_text("source1_entity_id\tmatched_entity_ids\nS1-1\tS2-1,S2-2\nS1-2\t\nS1-3\t\n")
            result = evaluate(root, exact, [cheap, fuzzy], truth)
            self.assertEqual(result["eligible_gt_pairs"], 2)
            self.assertEqual(result["cheap_union_gt_pairs"], 1)
            self.assertEqual(result["all_channel_union_gt_pairs"], 2)
            self.assertEqual(result["tfidf_incremental_over_cheap_gt_pairs"], 1)


if __name__ == "__main__":
    unittest.main()
