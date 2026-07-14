"""Reproducible input, identity, provenance, and output helpers."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import posixpath
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import transformers


_EXPERIMENT_CODE_SUFFIXES = {
    ".py", ".pyi", ".cu", ".cuh", ".c", ".cc", ".cpp", ".h", ".hpp",
    ".sh", ".ps1", ".toml", ".yaml", ".yml",
}
_PATH_COMMAND_FLAGS = {"--manifest", "--output-dir", "--prompts-file"}

@dataclass(frozen=True)
class PromptRecord:
    sample_id: str
    text: str


def load_prompt_records(path: str | Path) -> dict[str, PromptRecord]:
    """Load the exact ``sample_id``/``text`` JSONL prompt schema."""
    source = Path(path)
    records: dict[str, PromptRecord] = {}
    try:
        handle = source.open("r", encoding="utf-8")
    except OSError as error:
        raise ValueError(f"cannot load prompts {source}: {error}") from error
    with handle:
        for line_number, line in enumerate(handle, 1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON on line {line_number}: {error.msg}") from error
            if not isinstance(value, dict):
                raise ValueError(f"line {line_number} must be a JSON object")
            unknown = set(value) - {"sample_id", "text"}
            missing = {"sample_id", "text"} - set(value)
            if missing:
                raise ValueError(f"line {line_number} missing fields: {sorted(missing)}")
            if unknown:
                raise ValueError(f"line {line_number} has unknown fields: {sorted(unknown)}")
            sample_id, text = value["sample_id"], value["text"]
            if not isinstance(sample_id, str) or not sample_id.strip():
                raise ValueError(f"line {line_number} sample_id must be a non-empty string")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"line {line_number} text must be a non-empty string")
            if sample_id in records:
                raise ValueError(f"duplicate sample_id {sample_id!r} on line {line_number}")
            records[sample_id] = PromptRecord(sample_id=sample_id, text=text)
    if not records:
        raise ValueError("prompts file must contain at least one record")
    return records


def prepare_prompt(tokenizer: Any, text: str, prompt_length: int) -> dict[str, Any]:
    if isinstance(prompt_length, bool) or not isinstance(prompt_length, int) or prompt_length <= 0:
        raise ValueError("prompt_length must be a positive integer")
    encoded = tokenizer(text, add_special_tokens=True, return_tensors="pt")["input_ids"]
    if encoded.shape[-1] < prompt_length + 1:
        raise ValueError(f"text has {encoded.shape[-1]} tokens; need {prompt_length + 1}")
    return {
        "prompt_ids": encoded[:, :prompt_length].contiguous(),
        "continuation_ids": encoded[:, prompt_length:prompt_length + 1].contiguous(),
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "effective_prompt_length": prompt_length,
    }


def stable_run_id(payload: dict) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:20]


def point_identity(
    job: dict[str, Any], sample: PromptRecord, prompt_length: int,
    source: dict[str, Any],
) -> dict[str, Any]:
    """Return only the scientific identity local to one experiment point."""

    model = dict(job["model"])
    reference = model.get("reference")
    if isinstance(reference, str):
        model["reference"] = posixpath.normpath(reference.replace("\\", "/"))
    return {
        "model": model,
        "method": {
            "name": job["method"],
            "resolved_config": job["method_config"],
        },
        "measurement": job["measurement"],
        "input": {
            "sample_id": sample.sample_id,
            "text_sha256": hashlib.sha256(sample.text.encode("utf-8")).hexdigest(),
            "prompt_length": prompt_length,
        },
        "source_state": source,
    }


def point_id(
    job: dict[str, Any], sample: PromptRecord, prompt_length: int,
    source: dict[str, Any],
) -> str:
    """Return the stable identifier for one scientific experiment point."""

    return stable_run_id(point_identity(job, sample, prompt_length, source))


def _git(repo_root: Path, *arguments: str) -> bytes:
    try:
        return subprocess.run(
            ["git", *arguments], cwd=repo_root, check=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError(f"cannot inspect Git source state at {repo_root}: {error}") from error


def source_state_identity(repo_root: str | Path) -> dict[str, Any]:
    """Return commit identity plus a deterministic digest of dirty source state."""
    root = Path(repo_root).resolve()
    commit = _git(root, "rev-parse", "HEAD").decode("ascii").strip()
    tracked_diff = _git(root, "diff", "--binary", "HEAD")
    untracked_output = _git(root, "ls-files", "--others", "--exclude-standard", "-z")
    untracked_paths = sorted(
        (item.decode("utf-8", errors="surrogateescape") for item in untracked_output.split(b"\0") if item),
        key=lambda item: item.encode("utf-8", errors="surrogateescape"),
    )
    digest = hashlib.sha256()
    digest.update(tracked_diff)
    included_paths: list[str] = []
    for relative in untracked_paths:
        candidate = root / relative
        if not candidate.is_file() or candidate.suffix.lower() not in _EXPERIMENT_CODE_SUFFIXES:
            continue
        encoded_path = relative.replace("\\", "/").encode("utf-8", errors="surrogateescape")
        digest.update(b"\0untracked\0" + encoded_path + b"\0")
        digest.update(candidate.read_bytes())
        included_paths.append(relative.replace("\\", "/"))
    dirty = bool(tracked_diff or included_paths)
    return {
        "git_commit": commit,
        "dirty": dirty,
        "dirty_sha256": digest.hexdigest() if dirty else None,
        "untracked_paths": included_paths,
    }


def _cuda_driver_version() -> Any:
    getter = getattr(getattr(torch, "_C", None), "_cuda_getDriverVersion", None)
    if getter is None:
        return None
    try:
        return getter()
    except Exception:
        return None


def normalize_command_arguments(
    command_args: Iterable[Any], repo_root: str | Path,
) -> list[str]:
    """Return portable command arguments with experiment paths made absolute.

    Values belonging to ``--manifest``, ``--output-dir``, and
    ``--prompts-file`` are resolved against ``repo_root`` when relative,
    normalized to collapse ``.`` and ``..``, and emitted with forward slashes.
    Both ``--flag value`` and ``--flag=value`` forms retain their original
    shape. All other arguments are preserved after conversion to ``str``.
    """
    root = Path(repo_root).resolve()
    arguments = [str(argument) for argument in command_args]

    def normalize_path(value: str) -> str:
        path = Path(value)
        if not path.is_absolute():
            path = root / path
        return path.resolve().as_posix()

    normalized: list[str] = []
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if argument in _PATH_COMMAND_FLAGS and index + 1 < len(arguments):
            normalized.extend((argument, normalize_path(arguments[index + 1])))
            index += 2
            continue
        flag, separator, value = argument.partition("=")
        if separator and flag in _PATH_COMMAND_FLAGS:
            normalized.append(f"{flag}={normalize_path(value)}")
        else:
            normalized.append(argument)
        index += 1
    return normalized


def collect_provenance(
    repo_root: str | Path, *, deterministic_seed: int,
) -> dict[str, Any]:
    source = source_state_identity(repo_root)
    has_cuda = torch.cuda.is_available()
    deterministic_getter = getattr(
        torch, "are_deterministic_algorithms_enabled", None
    )
    warn_only_getter = getattr(
        torch, "is_deterministic_algorithms_warn_only_enabled", None
    )
    cudnn = getattr(getattr(torch, "backends", None), "cudnn", None)
    return {
        "source_state": source,
        "dirty": source["dirty"],
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_runtime": torch.version.cuda if has_cuda else None,
        "cuda_driver": _cuda_driver_version() if has_cuda else None,
        "gpu_name": torch.cuda.get_device_name(0) if has_cuda else None,
        "deterministic_seed": int(deterministic_seed),
        "deterministic_algorithms_enabled": (
            bool(deterministic_getter()) if callable(deterministic_getter) else None
        ),
        "deterministic_algorithms_warn_only": (
            bool(warn_only_getter()) if callable(warn_only_getter) else None
        ),
        "cudnn_deterministic": (
            bool(getattr(cudnn, "deterministic")) if cudnn is not None else None
        ),
        "cudnn_benchmark": (
            bool(getattr(cudnn, "benchmark")) if cudnn is not None else None
        ),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "command": normalize_command_arguments(sys.argv, repo_root),
    }


def _atomic_write(path: str | Path, writer: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", dir=destination.parent,
            prefix=f".{destination.name}.", suffix=".tmp", delete=False,
        ) as handle:
            temporary_name = handle.name
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
        temporary_name = None
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass


def atomic_write_json(path: str | Path, value: Any) -> None:
    _atomic_write(path, lambda handle: json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2))


def atomic_write_jsonl(path: str | Path, records: Iterable[Any]) -> None:
    def write(handle: Any) -> None:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
            handle.write("\n")
    _atomic_write(path, write)


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    if isinstance(value, dict):
        for key in sorted(value):
            name = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(_flatten(value[key], name))
    else:
        flattened[prefix] = json.dumps(value, ensure_ascii=False, separators=(",", ":")) if isinstance(value, (list, dict)) else value
    return flattened


def aggregate_completed_runs(
    output_dir: str | Path, *, expected_run_ids: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    """Rebuild canonical JSONL and flattened CSV summaries from completed runs."""
    root = Path(output_dir)
    records: list[dict[str, Any]] = []
    runs_dir = root / "runs"
    if runs_dir.exists():
        if expected_run_ids is None:
            paths = sorted(runs_dir.glob("*.json"), key=lambda item: item.name)
        else:
            paths = [
                runs_dir / f"{run_id}.json"
                for run_id in sorted(set(expected_run_ids))
                if (runs_dir / f"{run_id}.json").exists()
            ]
        for path in paths:
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise ValueError(f"cannot load run record {path}: {error}") from error
            if not isinstance(record, dict) or record.get("status") != "completed":
                continue
            from utils.cage_experiment_schema import validate_completed_point

            records.append(validate_completed_point(root, path.stem))
    records.sort(key=lambda record: str(record.get("run_id", "")))
    summary = root / "summary"
    atomic_write_jsonl(summary / "runs.jsonl", records)
    flat_rows = [_flatten(record) for record in records]
    fields = sorted({field for row in flat_rows for field in row})
    if "run_id" in fields:
        fields.remove("run_id"); fields.insert(0, "run_id")

    def write_csv(handle: Any) -> None:
        if not fields:
            return
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise", lineterminator="\n")
        writer.writeheader()
        writer.writerows(flat_rows)
    _atomic_write(summary / "runs.csv", write_csv)
    return records


__all__ = [
    "PromptRecord", "aggregate_completed_runs", "atomic_write_json", "atomic_write_jsonl",
    "collect_provenance", "load_prompt_records", "normalize_command_arguments", "point_id",
    "point_identity", "prepare_prompt", "source_state_identity", "stable_run_id",
]
