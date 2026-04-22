"""JSON contract between the validator host and the inference machine.

The validator process (chain, wallet, saved scores) usually runs on a
CPU-only box. Model inference runs elsewhere—typically a GPU host with
no wallet. They communicate by writing files the other side reads:

    job.json       validator → inference   (who to score, model settings)
    results.json   inference → validator   (scores per challenger)

This module defines those payloads as plain dataclasses with JSON
helpers. It deliberately avoids importing torch, bittensor, or
HuggingFace so the validator can serialize jobs and parse results with
only the standard library + these types.

`SCHEMA_VERSION` here is only for the job/results JSON shape. It is
separate from `validator.state.SCHEMA_VERSION` (the on-disk validator
state file);
bump this when fields on the wire change, not when king/eval history
storage changes.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SCHEMA_VERSION: int = 2
"""
Bump history:
  1 → initial shape with `model`/`revision`.
  2 → rename `model` → `repo`; add `source_hash` to `ChallengerJob` and
      `ChallengerResult` (sha256 of policy.py bytes, verified on the pod).
"""

JOB_FILE_NAME: str = "job.json"
RESULTS_FILE_NAME: str = "results.json"


# --------------------------------------------------------------------------- #
# Job (CPU → pod)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ChallengerJob:
    """One challenger to evaluate on the pod.

    `policy_path` is a filesystem path **on the pod** where `policy.py`
    has already been staged (fetched from HF by the CPU side and uploaded
    / mounted). The pod never talks to HF directly.

    `source_hash` is sha256 of the policy.py bytes as fetched on the CPU
    side. The pod re-hashes the file at `policy_path` and DQs the
    challenger if it doesn't match, so a corrupted / tampered upload
    can't silently score something other than what the CPU reviewed.
    Empty string means "caller didn't compute it" — pod will treat as
    verified to support legacy callers.
    """
    uid: int
    hotkey: str
    commit_block: int
    repo: str                   # HF repo pointer, informational on the pod
    revision: str               # git sha, informational on the pod
    policy_path: str
    source_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChallengerJob:
        known = {f: data[f] for f in cls.__dataclass_fields__ if f in data}
        return cls(**known)


@dataclass(frozen=True)
class EvaluationJob:
    """One tick's worth of work for the pod.

    All challengers share the same baseline (identical prompts +
    model + decoder settings), so baseline runs at most once per job.
    The pod caches baseline artifacts at `baseline_cache_dir` keyed by
    `baseline_cache_key` — next tick reuses them if the key matches.
    """
    schema_version: int
    job_id: str
    current_block: int
    block_hash: str | None
    model_name: str
    max_new_tokens: int
    n_prompts: int
    baseline_cache_dir: str
    baseline_cache_key: str
    challengers: list[ChallengerJob] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "job_id": self.job_id,
            "current_block": self.current_block,
            "block_hash": self.block_hash,
            "model_name": self.model_name,
            "max_new_tokens": self.max_new_tokens,
            "n_prompts": self.n_prompts,
            "baseline_cache_dir": self.baseline_cache_dir,
            "baseline_cache_key": self.baseline_cache_key,
            "challengers": [c.to_dict() for c in self.challengers],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvaluationJob:
        version = int(data.get("schema_version", 1))
        if version > SCHEMA_VERSION:
            raise ValueError(
                f"EvaluationJob schema_version={version} is newer than "
                f"this binary (supports up to {SCHEMA_VERSION}); "
                f"upgrade before parsing."
            )
        return cls(
            schema_version=version,
            job_id=str(data["job_id"]),
            current_block=int(data["current_block"]),
            block_hash=data.get("block_hash"),
            model_name=str(data["model_name"]),
            max_new_tokens=int(data["max_new_tokens"]),
            n_prompts=int(data["n_prompts"]),
            baseline_cache_dir=str(data["baseline_cache_dir"]),
            baseline_cache_key=str(data["baseline_cache_key"]),
            challengers=[
                ChallengerJob.from_dict(c)
                for c in (data.get("challengers") or [])
            ],
        )


# --------------------------------------------------------------------------- #
# Result (pod → CPU)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ChallengerResult:
    """One challenger's post-scoring output.

    Maps 1:1 onto `validator.state.EvaluationRecord` on the CPU side —
    the CPU adds `evaluated_at` + `evaluation_block` from its own clock /
    chain view when recording. `source_hash` echoes back the CPU-provided
    hash after the pod verified it matches the on-disk `policy.py`
    (DQ'd with ``source_hash_mismatch`` if not).
    """
    uid: int
    hotkey: str
    commit_block: int
    repo: str
    revision: str
    score: float
    kl_divergence: float
    memory_reduction: float
    latency_improvement: float
    disqualified: bool
    disqualify_reason: str | None
    source_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChallengerResult:
        known = {f: data[f] for f in cls.__dataclass_fields__ if f in data}
        return cls(**known)


@dataclass(frozen=True)
class BaselineMetrics:
    """Informational — logged on CPU to catch regressions in the baseline
    run itself (e.g. pod thermal throttling)."""
    latency_s: float
    peak_memory_bytes: int
    cached: bool
    """True if the pod loaded baseline from `baseline_cache_dir` rather
    than recomputing."""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BaselineMetrics:
        known = {f: data[f] for f in cls.__dataclass_fields__ if f in data}
        return cls(**known)


@dataclass(frozen=True)
class EvaluationResult:
    """Full pod output for one tick. One `ChallengerResult` per
    challenger in the job — including DQ'd ones with `score=0.0`."""
    schema_version: int
    job_id: str
    current_block: int
    block_hash: str | None
    baseline: BaselineMetrics
    challenger_results: list[ChallengerResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "job_id": self.job_id,
            "current_block": self.current_block,
            "block_hash": self.block_hash,
            "baseline": self.baseline.to_dict(),
            "challenger_results": [
                c.to_dict() for c in self.challenger_results
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvaluationResult:
        version = int(data.get("schema_version", 1))
        if version > SCHEMA_VERSION:
            raise ValueError(
                f"EvaluationResult schema_version={version} is newer than "
                f"this binary (supports up to {SCHEMA_VERSION}); "
                f"upgrade before parsing."
            )
        return cls(
            schema_version=version,
            job_id=str(data["job_id"]),
            current_block=int(data["current_block"]),
            block_hash=data.get("block_hash"),
            baseline=BaselineMetrics.from_dict(data["baseline"]),
            challenger_results=[
                ChallengerResult.from_dict(c)
                for c in (data.get("challenger_results") or [])
            ],
        )


# --------------------------------------------------------------------------- #
# Disk helpers — both sides use these so the on-wire format stays identical
# --------------------------------------------------------------------------- #


def write_job(job: EvaluationJob, path: str | os.PathLike) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(job.to_dict(), f, indent=2, sort_keys=True)


def read_job(path: str | os.PathLike) -> EvaluationJob:
    with open(path) as f:
        return EvaluationJob.from_dict(json.load(f))


def write_results(result: EvaluationResult, path: str | os.PathLike) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(result.to_dict(), f, indent=2, sort_keys=True)


def read_results(path: str | os.PathLike) -> EvaluationResult:
    with open(path) as f:
        return EvaluationResult.from_dict(json.load(f))
