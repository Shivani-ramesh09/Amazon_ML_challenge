import tempfile
import unittest
from pathlib import Path

from chimera_submission.code.business_entity_resolution.src.data.profile import (
    encoded_id, profile_dataset, split_bucket,
)
from chimera_submission.code.business_entity_resolution.src.config import load_config


class ProfileTests(unittest.TestCase):
    def test_profile_configs(self):
        base = Path("chimera_submission/code/business_entity_resolution/configs")
        dev = load_config(base / "dev.yaml")
        aws = load_config(base / "aws_cpu.yaml")
        self.assertEqual(aws["threads"], 8)
        self.assertLess(dev["memory_budget_gb"], aws["memory_budget_gb"])

    def test_split_is_stable(self):
        self.assertEqual(split_bucket("S1-123"), split_bucket("S1-123"))
        self.assertIn(split_bucket("S1-123"), ("train", "validation"))

    def test_id_validation(self):
        self.assertNotEqual(encoded_id("S2-3", "S2"), encoded_id("S3-3", "S3"))
        for value in ("S2-x", "S3-3", "S2-3-4", "S2-"):
            with self.assertRaises(ValueError):
                encoded_id(value, "S2")

    def test_train_contract_and_empty_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / "train_source1.tsv").write_text("entity_id\tbusiness_name\tbusiness_address\tcountry\nS1-1\tCafe\t\tFrance\nS1-2\t\t1 Main St\t\n")
            for source in (2, 3):
                (path / f"train_source{source}.tsv").write_text(f"entity_id\tbusiness_name\tbusiness_address\tcountry\nS{source}-1\tCafe\t1 Main St\tFrance\n")
            truth = path / "train_ground_truth.tsv"
            truth.write_text("source1_entity_id\tmatched_entity_ids\nS1-1\tS2-1,S3-1\nS1-2\t\n")
            report = profile_dataset(path, "train")
            self.assertEqual(report["S1"]["missing_fields"]["business_name"], 1)
            self.assertEqual(report["S1"]["countries"]["France"], 1)
            from chimera_submission.code.business_entity_resolution.src.data.profile import profile_source
            sampled, _ = profile_source(path / "train_source1.tsv", "S1", sample_modulus=1)
            self.assertIn(("cafe", 1), sampled["sampled_name_top20"])
            self.assertEqual(report["truth"]["cardinality"], {"0": 1, "2": 1})
            truth.write_text("source1_entity_id\tmatched_entity_ids\nS1-1\tS2-999\nS1-2\t\n")
            with self.assertRaisesRegex(ValueError, "invalid truth target"):
                profile_dataset(path, "train")

    def test_duplicate_and_bad_tsv(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test_source1.tsv"
            header = "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
            path.write_text(header + "S1-1\tA\tB\tUS\nS1-1\tA\tB\tUS\n")
            from chimera_submission.code.business_entity_resolution.src.data.profile import profile_source
            with self.assertRaisesRegex(ValueError, "duplicate"):
                profile_source(path, "S1")
            path.write_text(header + "S1-1\tA\tB\n")
            with self.assertRaisesRegex(ValueError, "expected 4 columns"):
                profile_source(path, "S1")


if __name__ == "__main__":
    unittest.main()
