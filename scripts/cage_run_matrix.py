"""Run every resolved CAGE experiment job and rebuild its summaries."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.cage_experiment_config import expand_jobs, load_and_resolve_manifest
from utils.cage_experiment_io import (
    aggregate_completed_runs, atomic_write_json, load_prompt_records, point_id,
    source_state_identity,
)


WORKER_EXIT_CODES = {0, 2, 3, 4, 5, 6}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--retry-transient-once", action="store_true")
    return parser.parse_args(argv)


def _run_worker(command: list[str], retry_transient_once: bool) -> int:
    completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
    returncode = completed.returncode
    if returncode not in WORKER_EXIT_CODES:
        raise RuntimeError(f"unexpected worker exit code {returncode}: {command}")
    if returncode == 5 and retry_transient_once:
        completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
        returncode = completed.returncode
        if returncode not in WORKER_EXIT_CODES:
            raise RuntimeError(f"unexpected worker exit code {returncode}: {command}")
    return returncode


def run_matrix(manifest_path: str | Path, retry_transient_once: bool = False) -> int:
    manifest = load_and_resolve_manifest(manifest_path)
    manifest_dir = Path(manifest_path).resolve().parent
    for field in ("prompts_file", "output_dir"):
        path = Path(manifest[field])
        if not path.is_absolute():
            manifest[field] = str((manifest_dir / path).resolve())
    jobs = expand_jobs(manifest)
    output_dir = Path(manifest["output_dir"])
    prompts = load_prompt_records(manifest["prompts_file"])
    missing = sorted(set(manifest["sample_ids"]) - set(prompts))
    if missing:
        raise ValueError(f"selected sample_ids missing from prompts: {missing}")
    source = source_state_identity(REPO_ROOT)
    expected_run_ids = {
        point_id(job, prompts[sample_id], prompt_length, source)
        for job in jobs
        for sample_id in job["sample_ids"]
        for prompt_length in job["prompt_lengths"]
    }
    resolved_path = output_dir / "manifest.resolved.json"
    atomic_write_json(resolved_path, {**manifest, "jobs": jobs})

    result = 0
    for job_index in range(len(jobs)):
        command = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "cage_experiment_worker.py"),
            "--manifest", str(resolved_path),
            "--job-index", str(job_index),
        ]
        returncode = _run_worker(command, retry_transient_once)
        if returncode and result == 0:
            result = returncode
        if returncode == 2:
            break

    aggregate_completed_runs(output_dir, expected_run_ids=expected_run_ids)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return run_matrix(args.manifest, args.retry_transient_once)
    except ValueError as error:
        print(f"manifest/input error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
