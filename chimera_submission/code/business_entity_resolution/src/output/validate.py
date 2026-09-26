"""Run the official validator with fatal subset/ID warnings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import pyarrow.parquet as pq

from utils.validate_submission import validate


def _sample_ids(normalized_dir: Path, test_dir: Path) -> None:
    """Project sampled normalized IDs into the validator's expected TSV layout."""
    for source in (1, 2, 3):
        with (test_dir / f"test_source{source}.tsv").open("w", encoding="utf-8") as file:
            file.write("entity_id\n")
            parquet = pq.ParquetFile(normalized_dir / f"test_s{source}.parquet")
            for batch in parquet.iter_batches(batch_size=100_000, columns=["entity_id"]):
                for entity_id in batch.column(0).to_pylist():
                    file.write(entity_id + "\n")


def official_validate(matching: Path, candidate: Path, *, test_dir: Path | None = None,
                      normalized_dir: Path | None = None) -> dict:
    if (test_dir is None) == (normalized_dir is None):
        raise ValueError("specify exactly one of raw test_dir or sampled normalized_dir")
    if normalized_dir is not None:
        with tempfile.TemporaryDirectory(prefix="entity_resolution_validator_") as directory:
            projected = Path(directory)
            _sample_ids(normalized_dir, projected)
            errors, warnings = validate(str(matching), str(candidate), str(projected), check_ids=True)
    else:
        errors, warnings = validate(str(matching), str(candidate), str(test_dir), check_ids=True)
    fatal_warnings = [warning for warning in warnings if
                      "not present in candidate_pairs.tsv" in warning or
                      "skipping" in warning or "not checking" in warning]
    report = {"official_validator_passed": not errors and not fatal_warnings,
              "errors": errors, "warnings": warnings, "fatal_warnings": fatal_warnings,
              "matching_path": str(matching), "candidate_path": str(candidate),
              "test_scope": "sampled_normalized_ids" if normalized_dir else "raw_full_test"}
    if errors or fatal_warnings:
        raise ValueError(json.dumps(report, indent=2, sort_keys=True))
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matching", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--test-dir", type=Path)
    choice.add_argument("--normalized-dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(official_validate(args.matching, args.candidate,
                                        test_dir=args.test_dir,
                                        normalized_dir=args.normalized_dir),
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
