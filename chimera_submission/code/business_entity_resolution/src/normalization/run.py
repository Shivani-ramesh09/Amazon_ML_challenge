"""Streaming Arrow TSV -> normalized Parquet partitions, same path for dev and AWS."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import time

import pyarrow as pa
import pyarrow.csv as pc
import pyarrow.parquet as pq

from chimera_submission.code.business_entity_resolution.src.config import load_config
from chimera_submission.code.business_entity_resolution.src.normalization.address import address_views, country_key
from chimera_submission.code.business_entity_resolution.src.normalization.text import name_views

INPUT_COLUMNS = ("entity_id", "business_name", "business_address", "country")
OUTPUT_COLUMNS = (
    "entity_id", "name_raw", "name_clean", "name_core", "address_raw", "address_clean",
    "address_numbers", "postal_candidate", "house_number", "country", "country_key",
)


def _reader(path: Path):
    return pc.open_csv(
        path, read_options=pc.ReadOptions(block_size=8 * 1024 * 1024, use_threads=False),
        parse_options=pc.ParseOptions(delimiter="\t", quote_char=False),
        convert_options=pc.ConvertOptions(
            column_types={column: pa.string() for column in INPUT_COLUMNS},
            strings_can_be_null=False, null_values=[],
        ),
    )


def _sha256(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def _normalizer_signature(input_path: Path, modulus: int) -> str:
    """Invalidate only on input, normalizer implementation, or sampling changes."""
    digest = hashlib.sha256()
    digest.update(_sha256(input_path).encode())
    digest.update(str(modulus).encode())
    for name in ("text.py", "address.py", "run.py"):
        digest.update(_sha256(Path(__file__).with_name(name)).encode())
    return digest.hexdigest()


def normalize_file(path: Path, output: Path, *, sample_modulus: int = 1, compression: str = "zstd") -> dict:
    if sample_modulus < 1:
        raise ValueError("sample_modulus must be positive")
    start = time.perf_counter()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    writer = None
    scanned = written = 0
    try:
        reader = _reader(path)
        if tuple(reader.schema.names) != INPUT_COLUMNS:
            raise ValueError(f"{path}: unexpected TSV columns {reader.schema.names!r}")
        for batch in reader:
            values = {key: batch.column(batch.schema.get_field_index(key)).to_pylist() for key in INPUT_COLUMNS}
            out = {key: [] for key in OUTPUT_COLUMNS}
            for entity_id, name, address, country in zip(*(values[key] for key in INPUT_COLUMNS)):
                scanned += 1
                if sample_modulus > 1 and int(entity_id.partition("-")[2]) % sample_modulus:
                    continue
                clean_name, core_name = name_views(name or "")
                clean_address, numbers, postal, house = address_views(address or "", country or "")
                row = (
                    entity_id, name or "", clean_name, core_name, address or "", clean_address,
                    numbers, postal, house, country or "", country_key(country or ""),
                )
                for key, value in zip(OUTPUT_COLUMNS, row):
                    out[key].append(value)
                written += 1
            if not out["entity_id"]:
                continue
            table = pa.table({key: pa.array(out[key], type=pa.string()) for key in OUTPUT_COLUMNS})
            if writer is None:
                writer = pq.ParquetWriter(temporary, table.schema, compression=compression)
            writer.write_table(table, row_group_size=min(len(table), 100_000))
        if writer is None:
            table = pa.table({key: pa.array([], type=pa.string()) for key in OUTPUT_COLUMNS})
            writer = pq.ParquetWriter(temporary, table.schema, compression=compression)
        writer.close()
        writer = None
        os.replace(temporary, output)
    finally:
        if writer is not None:
            writer.close()
        if temporary.exists():
            temporary.unlink()
    elapsed = time.perf_counter() - start
    return {
        "input": str(path.resolve()), "output": str(output.resolve()), "input_bytes": path.stat().st_size,
        "output_bytes": output.stat().st_size, "scanned_rows": scanned, "written_rows": written,
        "sample_modulus": sample_modulus, "elapsed_seconds": round(elapsed, 3),
        "rows_per_second": round(scanned / elapsed, 1),
        "peak_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2, 3),
        "actual_threads": 1,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Normalize TSV files in bounded Arrow batches")
    parser.add_argument("--config", type=Path, default=Path("chimera_submission/code/business_entity_resolution/configs/dev.yaml"))
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument("--source", choices=(1, 2, 3), type=int, required=True)
    parser.add_argument("--sample-modulus", type=int, default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    modulus = args.sample_modulus if args.sample_modulus is not None else config["sample_modulus"]
    input_path = Path(config["data_dir"]) / args.split / f"{args.split}_source{args.source}.tsv"
    output_dir = Path(config["artifact_dir"]) / "normalized" / ("full" if modulus == 1 else f"sample_{modulus}")
    output_path = output_dir / f"{args.split}_s{args.source}.parquet"
    manifest_path = output_dir / f"{args.split}_s{args.source}.json"
    signature = _normalizer_signature(input_path, modulus)
    if output_path.is_file() and manifest_path.is_file():
        cached = json.loads(manifest_path.read_text())
        if cached.get("signature") == signature:
            print(json.dumps({"cache_hit": True, **cached}, indent=2))
            return 0
    report = normalize_file(input_path, output_path, sample_modulus=modulus)
    report["signature"] = signature
    report["config_sha256"] = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    manifest_tmp = manifest_path.with_suffix(".json.tmp")
    manifest_tmp.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(manifest_tmp, manifest_path)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
