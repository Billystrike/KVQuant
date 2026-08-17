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

from utils.qwen3_cage_v3_promotion_closeout import build_closeout


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze the negative CAGE-v3 promotion decision")
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite CAGE-v3 promotion closeout: {output}")
    report = build_closeout(args.artifacts)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    rendered = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary.write_text(rendered, encoding="utf-8")
    os.replace(temporary, output)
    print(rendered, end="")


if __name__ == "__main__":
    main()
