import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.feature_extraction.text import TfidfVectorizer

from chimera_submission.code.business_entity_resolution.src.retrieval.tfidf import (
    CpuTfidfRetriever, build_index, load_retriever,
)


def write_names(path, rows):
    pq.write_table(pa.table({
        "entity_id": pa.array([row[0] for row in rows], type=pa.string()),
        "name_clean": pa.array([row[1] for row in rows], type=pa.string()),
    }), path)


class TfidfTests(unittest.TestCase):
    def test_sparse_top_n_and_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            names = ["acme bakery", "acme laundry", "globex clinic"]
            vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), dtype=np.float32)
            matrix = vectorizer.fit_transform(names)
            retriever = CpuTfidfRetriever(vectorizer, matrix.T.tocsr(), ["S2-1", "S2-2", "S3-1"],
                                           top_k=2, threshold=0.1, threads=1)
            query = root / "query.parquet"
            output = root / "pairs.parquet"
            write_names(query, [("S1-1", "acme bakery"), ("S1-2", "globex clinc"), ("S1-3", "")])
            report = retriever.retrieve(query, output, skip_s1_ids={"S1-2"})
            pairs = pq.read_table(output).to_pylist()
            self.assertEqual(report["searched_rows"], 2)
            self.assertEqual(report["skipped_rows"], 1)
            self.assertEqual(report["zero_feature_rows"], 1)
            self.assertEqual(pairs[0]["candidate_entity_id"], "S2-1")
            self.assertEqual(pairs[0]["tfidf_rank"], 1)
            self.assertLessEqual(len(pairs), 2)

    def test_frozen_vectorizer_for_test_targets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train = root / "train.parquet"
            test = root / "test.parquet"
            write_names(train, [("S2-1", "acme bakery"), ("S3-1", "acme bakeries"), ("S2-2", "globex clinic")])
            write_names(test, [("S2-3", "acme bakery paris")])
            vectorizer = root / "vectorizer.pkl"
            matrix = root / "target.npz"
            ids = root / "target_ids.parquet"
            stats = build_index([train], vectorizer, matrix, ids, fit_vectorizer=True, max_df=1.0)
            self.assertGreater(stats["nnz"], 0)
            test_stats = build_index([test], vectorizer, root / "test_target.npz", root / "test_ids.parquet",
                                     fit_vectorizer=False, max_df=1.0)
            self.assertEqual(test_stats["features"], stats["features"])
            loaded = load_retriever(vectorizer, root / "test_target.npz", root / "test_ids.parquet", threads=1)
            self.assertEqual(loaded.target_ids, ["S2-3"])


if __name__ == "__main__":
    unittest.main()
