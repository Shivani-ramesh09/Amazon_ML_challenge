import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import io
import contextlib
import json

import pyarrow.parquet as pq

from chimera_submission.code.business_entity_resolution.src.normalization.address import address_views, country_key
from chimera_submission.code.business_entity_resolution.src.normalization.run import normalize_file, main
from chimera_submission.code.business_entity_resolution.src.normalization.text import clean_text, name_views


class NormalizationTests(unittest.TestCase):
    def test_unicode_and_suffix(self):
        self.assertEqual(name_views("  ＡＣＭＥ & Sons Pvt. Ltd.  "), ("acme and sons pvt ltd", "acme and sons"))
        self.assertEqual(name_views("Company"), ("company", "company"))
        self.assertEqual(name_views("O’Reilly, Inc."), ("oreilly inc", "oreilly"))
        self.assertEqual(clean_text(clean_text("A & B")), clean_text("A & B"))

    def test_country_specific_and_generic(self):
        self.assertEqual(address_views("Plot 12, MG Rd, Pune 411001", "India")[2:], ("411001", "12"))
        self.assertEqual(address_views("12 Main Ave, Tyler TX 75701-1234", "US")[2:], ("75701", "12"))
        self.assertEqual(address_views("8 Rue Victor Hugo, 75001 Paris", "France")[2:], ("75001", "8"))
        self.assertEqual(address_views("9 Queen Rd, AB12 3CD", "UK")[2], "ab12")
        self.assertEqual(country_key("  FRANCE "), "france")

    def test_streamed_tsv_preserves_raw_and_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.tsv"
            output = root / "normalized.parquet"
            source.write_text(
                "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
                "S1-1\tACME & Sons Pvt. Ltd.\tPlot 12, MG Rd, Pune 411001\tIndia\n"
                "S1-2\t\t\tFrance\n",
                encoding="utf-8",
            )
            report = normalize_file(source, output)
            self.assertEqual((report["scanned_rows"], report["written_rows"]), (2, 2))
            rows = pq.read_table(output).to_pylist()
            self.assertEqual(rows[0]["name_core"], "acme and sons")
            self.assertEqual(rows[0]["name_raw"], "ACME & Sons Pvt. Ltd.")
            self.assertEqual(rows[1]["name_clean"], "")
            self.assertEqual(rows[1]["country_key"], "france")
            sampled = normalize_file(source, root / "sample.parquet", sample_modulus=2)
            self.assertEqual(sampled["written_rows"], 1)

    def test_cli_reuses_normalized_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = root / "data" / "train"
            source_dir.mkdir(parents=True)
            (source_dir / "train_source1.tsv").write_text(
                "entity_id\tbusiness_name\tbusiness_address\tcountry\nS1-2\tACME Ltd\t8 Main Rd\tUS\n"
            )
            config = {
                "data_dir": str(root / "data"), "artifact_dir": str(root / "artifacts"),
                "threads": 1, "memory_budget_gb": 1, "sample_modulus": 1,
                "validation_percent": 15, "split_seed": 42, "candidate_cap": 10,
                "pair_batch_size": 100, "profile_frequency_sample_modulus": 128,
            }
            config_path = root / "config.yaml"
            config_path.write_text(json.dumps(config))
            args = ["normalize", "--config", str(config_path), "--split", "train", "--source", "1"]
            with patch("sys.argv", args), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(), 0)
            output = root / "artifacts" / "normalized" / "full" / "train_s1.parquet"
            original_mtime = output.stat().st_mtime_ns
            stdout = io.StringIO()
            with patch("sys.argv", args), contextlib.redirect_stdout(stdout):
                self.assertEqual(main(), 0)
            self.assertIn('"cache_hit": true', stdout.getvalue())
            self.assertEqual(output.stat().st_mtime_ns, original_mtime)
            (source_dir / "train_source1.tsv").write_text(
                "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
                "S1-2\tACME Ltd\t8 Main Rd\tUS\nS1-4\tBeta LLC\t\tFrance\n"
            )
            with patch("sys.argv", args), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(), 0)
            self.assertEqual(pq.read_table(output).num_rows, 2)


if __name__ == "__main__":
    unittest.main()
