#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _track_key(track: dict[str, Any]) -> tuple[str, str]:
    return str(track.get("family_id")), str(track.get("target"))


def validate_decision(decision: dict[str, Any], analysis: dict[str, Any]) -> dict[str, Any]:
    if decision.get("schema_version") != 1:
        raise ValueError("decision schema mismatch")
    if decision.get("status") != "closed_negative_development_result":
        raise ValueError("CAGE-v2 decision must remain closed")
    if decision.get("claim_eligible") is not False:
        raise ValueError("CAGE-v2 decision must remain claim-ineligible")
    if analysis.get("status") != "pass" or analysis.get("claim_eligible") is not False:
        raise ValueError("analysis boundary mismatch")
    frozen = decision["frozen_outcome"]
    tracks = analysis.get("track_reports")
    if not isinstance(tracks, list) or len(tracks) != frozen["track_count"]:
        raise ValueError("analysis track count mismatch")
    if analysis.get("advanced_family_count") != frozen["advanced_family_count"]:
        raise ValueError("advanced family count mismatch")
    if analysis.get("advanced_families") != []:
        raise ValueError("negative decision requires no advanced families")
    if not all(track["gates"]["memory_all_three"] is True for track in tracks):
        raise ValueError("not every frozen track passed memory")
    if not all(track["gates"]["no_worse_than_kitty_at_least_two"] is False for track in tracks):
        raise ValueError("at least one track unexpectedly passed the Kitty gate")

    best = frozen["best_track"]
    matching = [track for track in tracks if _track_key(track) == (best["family_id"], best["target"])]
    if len(matching) != 1:
        raise ValueError("best track identity mismatch")
    track = matching[0]
    actual_length_deltas = {
        str(row["prompt_length"]): row["versus_kitty"]["mean_delta"]
        for row in track["lengths"]
    }
    checks = {
        "overall_kitty_delta": track["overall_15_case_versus_kitty"]["mean_delta"]
        == best["overall_15_case_mean_delta_vs_kitty"],
        "kitty_favor_count": track["overall_15_case_versus_kitty"]["favor_count"]
        == best["kitty_favor_count"],
        "paired_count": track["overall_15_case_versus_kitty"]["paired_count"]
        == best["paired_count"],
        "length_deltas": actual_length_deltas == best["length_mean_deltas_vs_kitty"],
        "overall_v1_delta": track["overall_15_case_versus_cage_v1"]["mean_delta"]
        == best["overall_15_case_mean_delta_vs_cage_v1"],
        "v1_favor_count": track["overall_15_case_versus_cage_v1"]["favor_count"]
        == best["cage_v1_favor_count"],
    }
    failures = sorted(name for name, passed in checks.items() if not passed)
    if failures:
        raise ValueError(f"frozen best-track facts differ: {failures}")
    return {
        "status": "pass",
        "decision_id": decision["decision_id"],
        "advanced_family_count": analysis["advanced_family_count"],
        "track_count": len(tracks),
        "best_track": {"family_id": best["family_id"], "target": best["target"]},
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the frozen negative CAGE-v2 round1 decision")
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--analysis-manifest", type=Path)
    parser.add_argument("--analysis-markdown", type=Path)
    parser.add_argument("--analysis-log", type=Path)
    args = parser.parse_args()
    decision_path = args.decision.resolve()
    analysis_path = args.analysis.resolve()
    decision = load_json(decision_path)
    artifact = decision["analysis_artifact"]
    if analysis_path.name != artifact["filename"]:
        raise ValueError("analysis filename differs from decision")
    if file_sha256(analysis_path) != artifact["sha256"]:
        raise ValueError("analysis hash differs from decision")
    if analysis_path.stat().st_size != artifact["size_bytes"]:
        raise ValueError("analysis size differs from decision")
    optional_artifacts = (
        (args.analysis_manifest, "analysis_manifest_sha256", None, "analysis manifest"),
        (args.analysis_markdown, "markdown_sha256", "markdown_size_bytes", "analysis markdown"),
        (args.analysis_log, "analysis_log_sha256", "analysis_log_size_bytes", "analysis log"),
    )
    verified_optional = {}
    for supplied, hash_key, size_key, label in optional_artifacts:
        if supplied is None:
            continue
        path = supplied.resolve()
        if file_sha256(path) != artifact[hash_key]:
            raise ValueError(f"{label} hash differs from decision")
        if size_key is not None and path.stat().st_size != artifact[size_key]:
            raise ValueError(f"{label} size differs from decision")
        verified_optional[label] = file_sha256(path)
    receipt_path = REPO_ROOT / decision["results_receipt"]["path"]
    if file_sha256(receipt_path) != decision["results_receipt"]["sha256"]:
        raise ValueError("results receipt hash differs from decision")
    report = validate_decision(decision, load_json(analysis_path))
    report.update(
        {
            "decision_sha256": file_sha256(decision_path),
            "analysis_sha256": file_sha256(analysis_path),
            "results_receipt_sha256": file_sha256(receipt_path),
            "verified_optional_artifacts": verified_optional,
        }
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
