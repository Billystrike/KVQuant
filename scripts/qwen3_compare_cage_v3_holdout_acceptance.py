#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.qwen3_cage_v3_execution import canonical_sha256, load_json
from utils.qwen3_cage_v3_holdout import PARTITIONS
from utils.qwen3_cage_v3_protocol import file_sha256
from utils.qwen3_cage_v3_screen import SCIENTIFIC_FIELDS


def _payload(root: Path, *, partition: str) -> tuple[dict, list[dict]]:
    summary = load_json(root / "summary.json")
    expected = 6 if partition == "cage_qwen3" else 3
    if (
        summary.get("status") != "pass"
        or summary.get("stage") != "holdout_acceptance"
        or summary.get("partition") != partition
        or summary.get("completed_cases") != expected
        or summary.get("new_cases") != expected
        or summary.get("resumed_cases") != 0
        or summary.get("failure_records") != 0
    ):
        raise RuntimeError(f"incomplete CAGE-v3 holdout acceptance: {root}")
    records = []
    for path in sorted((root / "cases").glob("*.json")):
        value = load_json(path)
        if value.get("status") != "completed" or path.stem != value.get("case_id"):
            raise RuntimeError(f"invalid holdout acceptance case: {path}")
        records.append({field: value[field] for field in SCIENTIFIC_FIELDS})
    if len(records) != expected:
        raise RuntimeError(f"holdout acceptance case count mismatch: {root}")
    records.sort(key=lambda record: record["case_id"])
    return summary, records


def compare_repeats(*, partition: str, first_root: Path, second_root: Path) -> dict:
    if first_root.resolve() == second_root.resolve():
        raise RuntimeError("holdout acceptance repeat directories must differ")
    first_summary, first = _payload(first_root.resolve(), partition=partition)
    second_summary, second = _payload(second_root.resolve(), partition=partition)
    if first_summary["identity"] != second_summary["identity"]:
        raise RuntimeError("holdout acceptance identities differ")
    first_by_id = {record["case_id"]: record for record in first}
    second_by_id = {record["case_id"]: record for record in second}
    mismatches = sorted(
        case_id for case_id in set(first_by_id) | set(second_by_id)
        if first_by_id.get(case_id) != second_by_id.get(case_id)
    )
    return {
        "schema_version": 1,
        "status": "pass" if not mismatches else "fail",
        "claim_eligible": False,
        "partition": partition,
        "case_count": len(first),
        "fields": list(SCIENTIFIC_FIELDS),
        "required_consistency": "bitwise_equal_json_numeric_payload",
        "mismatch_case_ids": mismatches,
        "scientific_payload_sha256": canonical_sha256(first),
        "comparator_sha256": file_sha256(Path(__file__).resolve()),
        "repeat_a": str(first_root.resolve()),
        "repeat_b": str(second_root.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare fresh CAGE-v3 holdout acceptance repeats")
    parser.add_argument("--partition", choices=PARTITIONS, required=True)
    parser.add_argument("--repeat-a", type=Path, required=True)
    parser.add_argument("--repeat-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare_repeats(
        partition=args.partition,
        first_root=args.repeat_a,
        second_root=args.repeat_b,
    )
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite holdout comparison: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    if result["mismatch_case_ids"]:
        raise RuntimeError(f"CAGE-v3 holdout acceptance mismatches: {result['mismatch_case_ids']}")


if __name__ == "__main__":
    main()
