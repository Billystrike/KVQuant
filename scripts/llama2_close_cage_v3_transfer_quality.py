#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.llama2_cage_v3_transfer_quality_closeout import build_closeout


def main() -> None:
    parser = argparse.ArgumentParser(description="Close the frozen Llama-2 CAGE-v3 transfer promotion study")
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite transfer closeout: {output}")
    closeout = build_closeout(args.artifacts.resolve(), repo_root=REPO_ROOT)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(closeout, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps(closeout, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
