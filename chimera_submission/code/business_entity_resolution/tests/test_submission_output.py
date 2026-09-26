"""Submission writer enforces zero/one/many and scored-candidate lineage."""

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from chimera_submission.code.business_entity_resolution.src.output.write import write_submission
from chimera_submission.code.business_entity_resolution.src.output.validate import official_validate


def _table(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pydict(values), path)


class SubmissionOutputTests(unittest.TestCase):
    def test_zero_one_many_and_invalid_scored_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            normalized = root / "normalized"
            _table(normalized / "test_s1.parquet", {"entity_id": ["S1-1", "S1-2", "S1-3"]})
            _table(normalized / "test_s2.parquet", {"entity_id": ["S2-1", "S2-2"]})
            _table(normalized / "test_s3.parquet", {"entity_id": ["S3-1"]})
            policy = root / "policy.json"
            policy.write_text(json.dumps({"selected_policy": {
                "singleton_threshold": 0.65, "pair_threshold": 0.65,
                "strong_match_threshold": 0.9, "gap_threshold": 0.4,
                "max_match_count": 30}}))
            frozen = root / "frozen.json"
            frozen.write_text(json.dumps({"validation_policy_path": str(policy),
                                          "validation_policy_sha256": hashlib.sha256(policy.read_bytes()).hexdigest(),
                                          "config": {"candidate_cap": 30}}))
            final_dir = root / "final"
            score_dir = root / "scores"
            for bucket, s1_ids, targets, scores in ((2, ["S1-2"], ["S2-1"], [0.93]),
                                                     (3, ["S1-3", "S1-3"], ["S2-2", "S3-1"],
                                                      [0.95, 0.92])):
                values = {"source1_entity_id": s1_ids, "candidate_entity_id": targets}
                _table(final_dir / f"part-{bucket:03d}.parquet", values)
                _table(score_dir / f"part-{bucket:03d}-batch-00000.parquet",
                       {**values, "score": scores})
            candidate_manifest = root / "candidates.json"
            candidate_manifest.write_text(json.dumps({"final_dir": str(final_dir),
                                                      "final": {"final_pairs": 3}}))
            score_manifest = root / "scored.json"
            score_manifest.write_text(json.dumps({"frozen_manifest": str(frozen),
                "candidate_manifest": str(candidate_manifest), "score_dir": str(score_dir),
                "scored_pairs": 3, "shards": [
                    {"file": "part-002-batch-00000.parquet"},
                    {"file": "part-003-batch-00000.parquet"}]}))
            result = write_submission(frozen, score_manifest, normalized, root / "output")
            self.assertEqual(result["s1_rows"], 3)
            self.assertEqual(result["predicted_zero"], 1)
            self.assertEqual(result["predicted_one"], 1)
            self.assertEqual(result["predicted_many"], 1)
            with Path(result["matching_path"]).open(newline="") as file:
                matching = dict(csv.reader(file, delimiter="\t"))
            self.assertEqual(matching["S1-1"], "")
            self.assertEqual(matching["S1-2"], "S2-1")
            self.assertEqual(matching["S1-3"], "S2-2,S3-1")
            _table(score_dir / "part-002-batch-00000.parquet",
                   {"source1_entity_id": ["S1-2"], "candidate_entity_id": ["S2-2"], "score": [0.93]})
            with self.assertRaisesRegex(ValueError, "scored pair differs"):
                write_submission(frozen, score_manifest, normalized, root / "bad_output")

    def test_official_validator_rejects_duplicates_invalid_ids_and_subset_violation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            test_dir = root / "test"
            test_dir.mkdir()
            for source in (1, 2, 3):
                (test_dir / f"test_source{source}.tsv").write_text(
                    f"entity_id\nS{source}-1\n")
            matching = root / "matching_results.tsv"
            candidate = root / "candidate_pairs.tsv"
            matching.write_text("source1_entity_id\tmatched_entity_ids\nS1-1\tS2-1\n")
            candidate.write_text("source1_entity_id\tcandidate_entity_ids\nS1-1\tS2-1,S3-1\n")
            self.assertTrue(official_validate(matching, candidate, test_dir=test_dir)[
                "official_validator_passed"])
            matching.write_text("source1_entity_id\tmatched_entity_ids\nS1-1\tS2-1,S2-1\n")
            with self.assertRaises(ValueError):
                official_validate(matching, candidate, test_dir=test_dir)
            matching.write_text("source1_entity_id\tmatched_entity_ids\nS1-1\tS2-9\n")
            with self.assertRaises(ValueError):
                official_validate(matching, candidate, test_dir=test_dir)
            matching.write_text("source1_entity_id\tmatched_entity_ids\nS1-1\tS3-1\n")
            candidate.write_text("source1_entity_id\tcandidate_entity_ids\nS1-1\tS2-1\n")
            with self.assertRaisesRegex(ValueError, "not present in candidate_pairs"):
                official_validate(matching, candidate, test_dir=test_dir)


if __name__ == "__main__":
    unittest.main()
