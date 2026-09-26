"""Generate submission TSVs from a frozen model and its scored test candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from chimera_submission.code.business_entity_resolution.src.output.write import write_submission


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen-manifest", type=Path, required=True)
    parser.add_argument("--score-manifest", type=Path, required=True)
    parser.add_argument("--normalized-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("chimera_submission/output"))
    args = parser.parse_args()
    print(json.dumps(write_submission(args.frozen_manifest, args.score_manifest,
                                      args.normalized_dir, args.output_dir),
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
