"""Resolve the cached manifests that belong to one frozen run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from chimera_submission.code.business_entity_resolution.src.config import load_config


def select_manifest(config_path: Path, sample_modulus: int, kind: str,
                    *, frozen_manifest: Path | None = None) -> Path:
    config = load_config(config_path)
    if sample_modulus < 1:
        raise ValueError("sample modulus must be positive")
    config["sample_modulus"] = sample_modulus
    sample = "full" if sample_modulus == 1 else f"sample_{sample_modulus}"
    root = Path(config["artifact_dir"])
    if kind == "frozen":
        candidates = (root / "models" / sample).glob("final_*/manifest.json")
        valid = []
        for path in candidates:
            record = json.loads(path.read_text())
            if record.get("config") == config and Path(record["model_path"]).is_file():
                valid.append(path)
    elif kind == "scores":
        if frozen_manifest is None:
            frozen_manifest = select_manifest(config_path, sample_modulus, "frozen")
        frozen_manifest = frozen_manifest.resolve()
        candidates = (root / "predictions" / "test" / sample).glob("*/manifest.json")
        valid = []
        for path in candidates:
            record = json.loads(path.read_text())
            if (record.get("score_dir") and
                    Path(record.get("frozen_manifest", "")).resolve() == frozen_manifest and
                    record.get("scored_pairs") == sum(
                        item["rows"] for item in record.get("shards", [])) and
                    all((Path(record["score_dir"]) / item["file"]).is_file()
                        for item in record.get("shards", []))):
                valid.append(path)
    else:
        raise ValueError(f"unknown manifest kind: {kind}")
    if not valid:
        raise FileNotFoundError(f"no {kind} manifest for {sample} and {config_path}")
    return max(valid, key=lambda path: path.stat().st_mtime_ns)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("frozen", "scores"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--sample-modulus", type=int, required=True)
    parser.add_argument("--frozen-manifest", type=Path)
    args = parser.parse_args()
    print(select_manifest(args.config, args.sample_modulus, args.kind,
                          frozen_manifest=args.frozen_manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
