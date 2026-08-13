#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.qwen3_cage_v4_dtqi_closeout import build_closeout


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze negative closeout for CAGE-v4-DTQI")
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite CAGE-v4 closeout: {output}")
    closeout = build_closeout(args.artifacts)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(closeout, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(closeout, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
