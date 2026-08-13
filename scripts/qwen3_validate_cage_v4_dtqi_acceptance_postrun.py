#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.qwen3_cage_v4_dtqi_postrun import build_postrun_audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate two frozen CAGE-v4-DTQI GPU acceptance repeats")
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--failed-attempts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite DTQI GPU acceptance postrun audit: {output}")
    audit = build_postrun_audit(
        args.artifacts,
        failed_attempts_path=args.failed_attempts,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(audit, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
