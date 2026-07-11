"""Validator-owned scoring arenas and immutable policy fingerprints.

An end-to-end speed ratio is meaningful only inside the exact runtime and workload
that produced it.  ``target=attention...`` is not enough: a score against a short,
graphs-off, TP=1 debug launch must never enter the same championship bracket as a
long-prefill, graphs-on, TP=4 production run.

This module is deliberately data-only.  An arena pins the validator image, sglang
revision, hardware shape, engine configuration, workload, fidelity lane, and scoring
policy.  Its SHA-256 fingerprint is stamped into qualification reports and ledger
rows; changing any score-affecting field creates a new bracket automatically.

Ad-hoc ``optima evaluate`` remains useful for development, but only a registered,
competable arena may emit a settlement report or write a crownable ledger score.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.metadata
import json
import os
import re
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from optima.compat import PINNED_SGLANG
from optima.referee_release import (
    APPROVED_REFEREE_SOURCE_DIGEST,
    APPROVED_REFEREE_TREE_DIGEST,
)
from optima.runtime_overlay import RuntimeFileOverlay, normalize_runtime_overlays


class ArenaPolicyError(ValueError):
    """A requested run does not match a registered scoring arena."""


def referee_source_digest(package_root: str | Path | None = None) -> str:
    """Content identity of the trusted Optima referee implementation.

    The serving image currently pins the dependency/runtime stack, while Optima is
    mounted into that image read-only.  Hashing every Python source file closes the
    otherwise serious consensus gap where two validators could claim the same arena
    fingerprint while running different scoring or qualification code.

    The digest intentionally excludes bytecode, tests, docs, and git metadata.  A
    dependency change rotates ``validator_image``; a referee-code change rotates this
    digest automatically.
    """
    root = Path(package_root) if package_root is not None else Path(__file__).parent
    root = root.resolve()
    if not root.is_dir():
        raise ArenaPolicyError(f"referee package root does not exist: {root}")
    files = sorted(
        path for path in root.rglob("*.py")
        if ("__pycache__" not in path.parts and path.is_file()
            and path.name != "referee_release.py")
    )
    if not files:
        raise ArenaPolicyError(f"referee package root contains no Python source: {root}")
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return f"sha256:{digest.hexdigest()}"


def huggingface_model_manifest(
    model_path: str | Path,
) -> tuple[str, str]:
    """Return ``(revision, manifest_digest)`` from a local HF download receipt.

    Hugging Face's per-file metadata records the immutable repository revision and
    object identity for every downloaded artifact.  Hashing the sorted receipt is a
    cheap startup preflight (including 239 GB of weights takes milliseconds rather
    than re-hashing them on every launch) and makes a bare mutable filesystem path
    insufficient as an arena identity.
    """
    root = Path(model_path)
    metadata_root = root / ".cache" / "huggingface" / "download"
    if not metadata_root.is_dir():
        raise ArenaPolicyError(
            f"model {root} has no Hugging Face download metadata receipt"
        )
    rows: list[tuple[str, str, str]] = []
    for path in sorted(metadata_root.rglob("*.metadata")):
        lines = path.read_text().splitlines()
        if len(lines) < 2 or not lines[0] or not lines[1]:
            raise ArenaPolicyError(f"malformed model metadata receipt: {path}")
        relative = path.relative_to(metadata_root).as_posix()
        relative = relative.removesuffix(".metadata")
        rows.append((relative, lines[0], lines[1]))
    if not rows:
        raise ArenaPolicyError(f"model {root} has an empty metadata receipt")
    revisions = {revision for _, revision, _ in rows}
    if len(revisions) != 1:
        raise ArenaPolicyError(
            f"model receipt mixes revisions: {sorted(revisions)!r}"
        )
    payload = "".join(
        f"{relative}\0{revision}\0{object_id}\n"
        for relative, revision, object_id in rows
    ).encode("utf-8")
    return next(iter(revisions)), f"sha256:{hashlib.sha256(payload).hexdigest()}"


def verify_model_content_seal(
    model_path: str | Path,
    *,
    expected_digest: str,
    verify_bytes: bool = False,
) -> None:
    """Verify a validator-generated manifest of the actual model bytes.

    The seal is produced once while provisioning the model volume from the 37
    observed files, then the volume is mounted read-only into every candidate.  The
    cheap default validates the complete path/size/hash receipt and filesystem shape;
    ``verify_bytes`` re-hashes every file (use at provisioning or daemon startup).
    """
    root = Path(model_path)
    seal_path = root / ".optima-content-sha256.json"
    try:
        raw = json.loads(seal_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ArenaPolicyError(f"missing/malformed model content seal: {exc}") from exc
    if not isinstance(raw, dict) or raw.get("version") != 1:
        raise ArenaPolicyError("unsupported model content seal")
    files = raw.get("files")
    if not isinstance(files, list) or not files:
        raise ArenaPolicyError("model content seal has no files")
    rows: list[tuple[str, str]] = []
    expected_paths: set[str] = set()
    for item in files:
        if not isinstance(item, dict):
            raise ArenaPolicyError("model content seal row must be an object")
        relative = item.get("path")
        sha = item.get("sha256")
        size = item.get("size")
        if (not isinstance(relative, str) or not relative or relative.startswith("/")
                or ".." in Path(relative).parts
                or not re.fullmatch(r"[0-9a-f]{64}", str(sha or ""))
                or type(size) is not int or size < 0):
            raise ArenaPolicyError("invalid model content seal row")
        path = root / relative
        try:
            stat_result = path.stat()
        except OSError as exc:
            raise ArenaPolicyError(f"sealed model file is missing: {relative}") from exc
        if not path.is_file() or path.is_symlink() or stat_result.st_size != size:
            raise ArenaPolicyError(f"sealed model file shape changed: {relative}")
        if relative in expected_paths:
            raise ArenaPolicyError(f"duplicate model content seal path: {relative}")
        expected_paths.add(relative)
        rows.append((relative, sha))

    observed_paths = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if (path.is_file() and ".cache" not in path.relative_to(root).parts
            and path.name != seal_path.name)
    }
    if observed_paths != expected_paths:
        raise ArenaPolicyError(
            "model content seal file set mismatch: "
            f"missing={sorted(expected_paths - observed_paths)!r} "
            f"extra={sorted(observed_paths - expected_paths)!r}"
        )
    payload = "".join(
        f"{relative}\0{sha}\n" for relative, sha in sorted(rows)
    ).encode("utf-8")
    actual_digest = f"sha256:{hashlib.sha256(payload).hexdigest()}"
    if raw.get("content_digest") != actual_digest or actual_digest != expected_digest:
        raise ArenaPolicyError(
            f"model content digest mismatch: {actual_digest!r} != {expected_digest!r}"
        )
    if verify_bytes:
        for relative, sha in rows:
            digest = hashlib.sha256()
            with (root / relative).open("rb") as handle:
                for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != sha:
                raise ArenaPolicyError(f"model byte hash mismatch: {relative}")


def _canonical_json_value(value: Any, *, path: str = "value") -> Any:
    """Validate/canonicalize the JSON subset allowed in consensus policy."""
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        if not __import__("math").isfinite(value):
            raise ArenaPolicyError(f"{path} must be finite")
        return value
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _canonical_json_value(
                getattr(value, field.name), path=f"{path}.{field.name}"
            )
            for field in dataclasses.fields(value)
        }
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for key in sorted(value):
            if not isinstance(key, str) or not key:
                raise ArenaPolicyError(f"{path} keys must be non-empty strings")
            out[key] = _canonical_json_value(value[key], path=f"{path}.{key}")
        return out
    if isinstance(value, (tuple, list)):
        return [
            _canonical_json_value(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise ArenaPolicyError(
        f"{path} contains unsupported consensus value {type(value).__name__}"
    )


@dataclass(frozen=True)
class WorkloadProfile:
    regime: str
    prompt_generator: str
    input_len: int | None
    num_prompts: int
    max_new_tokens: int
    top_logprobs: int = 20
    ignore_eos: bool = True
    prompt_engine_version: str = "one-shot-v2"
    prompt_seed_scheme: str = "post-commit-blockhash-v1"

    def __post_init__(self) -> None:
        if not (self.regime and self.prompt_generator and self.prompt_engine_version
                and self.prompt_seed_scheme):
            raise ArenaPolicyError(
                "workload regime/prompt generator/engine/seed scheme must be non-empty"
            )
        if self.input_len is not None and self.input_len <= 0:
            raise ArenaPolicyError("workload input_len must be positive")
        if self.num_prompts <= 0 or self.max_new_tokens <= 0:
            raise ArenaPolicyError("workload prompt/token counts must be positive")
        if self.top_logprobs <= 0:
            raise ArenaPolicyError("a crownable workload requires top-logprob evidence")
        if not self.ignore_eos:
            raise ArenaPolicyError("crownable throughput must use a fixed token budget")


@dataclass(frozen=True)
class TeacherForcedQualityPolicy:
    """Registered post-hoc non-inferiority policy.

    ``uncalibrated`` is an explicit fail-closed state: the evaluator may collect
    stock/candidate evidence on new hardware, but it can only return NO_DECISION.
    Moving to ``provisional-rtx`` or ``frozen-b300`` requires a new arena fingerprint.
    ``familywise_z`` is already corrected for the registered metric family; runtime
    code never chooses a confidence level or a tolerance.
    """

    protocol: str
    calibration_state: str
    clusters_per_batch: int
    nll_clip: float
    tail_nll_threshold: float
    familywise_z: float
    stock_mean_nll_envelope: float
    stock_worst_nll_envelope: float
    stock_tail_rate_envelope: float
    stock_topk_kl_envelope: float
    stock_argmax_rate_envelope: float
    stock_coverage_envelope: float
    mean_nll_delta: float
    worst_nll_delta: float
    tail_rate_delta: float
    topk_kl_delta: float
    argmax_rate_delta: float
    coverage_delta: float
    require_hidden_tasks: bool
    stock_hidden_score_envelope: float
    hidden_score_delta: float
    hidden_score_floor: float

    def __post_init__(self) -> None:
        from optima.eval.external_quality import TEACHER_FORCED_QUALITY_PROTOCOL_V2

        if self.protocol != TEACHER_FORCED_QUALITY_PROTOCOL_V2:
            raise ArenaPolicyError("teacher-forced protocol must be the reviewed v2 protocol")
        if self.calibration_state not in {
            "uncalibrated", "provisional-rtx", "frozen-b300",
        }:
            raise ArenaPolicyError("teacher-forced calibration state is invalid")
        if type(self.clusters_per_batch) is not int or not 2 <= self.clusters_per_batch <= 64:
            raise ArenaPolicyError("teacher-forced clusters_per_batch must be in [2, 64]")
        if type(self.require_hidden_tasks) is not bool:
            raise ArenaPolicyError("teacher-forced hidden-task policy must be boolean")
        for name in (
            "nll_clip", "tail_nll_threshold", "familywise_z",
            "stock_mean_nll_envelope", "stock_worst_nll_envelope",
            "stock_tail_rate_envelope", "stock_topk_kl_envelope",
            "stock_argmax_rate_envelope", "stock_coverage_envelope",
            "mean_nll_delta", "worst_nll_delta", "tail_rate_delta",
            "topk_kl_delta", "argmax_rate_delta", "coverage_delta",
            "stock_hidden_score_envelope", "hidden_score_delta",
            "hidden_score_floor",
        ):
            value = getattr(self, name)
            if (
                type(value) not in (int, float)
                or not __import__("math").isfinite(float(value))
                or float(value) < 0.0
            ):
                raise ArenaPolicyError(f"teacher-forced {name} must be finite and non-negative")
        if not 0.0 < self.nll_clip <= 1_000.0:
            raise ArenaPolicyError("teacher-forced nll_clip must be in (0, 1000]")
        if not 0.0 < self.tail_nll_threshold <= self.nll_clip:
            raise ArenaPolicyError("teacher-forced tail threshold must be in (0, nll_clip]")
        if not 0.0 < self.familywise_z <= 10.0:
            raise ArenaPolicyError("teacher-forced familywise_z must be in (0, 10]")
        for name in (
            "stock_tail_rate_envelope", "stock_argmax_rate_envelope",
            "stock_coverage_envelope", "tail_rate_delta", "argmax_rate_delta",
            "coverage_delta", "stock_hidden_score_envelope", "hidden_score_delta",
            "hidden_score_floor",
        ):
            if getattr(self, name) > 1.0:
                raise ArenaPolicyError(f"teacher-forced {name} must be in [0, 1]")


@dataclass(frozen=True)
class FidelityProfile:
    mode: str
    audit_rate: float
    audit_min_calls: int
    external_quality_gate: str
    teacher_forced_policy: TeacherForcedQualityPolicy
    kl_threshold: float | None = None
    argmax_disagree_rate_threshold: float | None = None
    p99_kl_threshold: float | None = None
    coverage_dev_threshold: float | None = None

    def __post_init__(self) -> None:
        if self.mode not in {"kl", "audit"}:
            raise ArenaPolicyError("fidelity mode must be 'kl' or 'audit'")
        if not (0.0 < self.audit_rate <= 1.0):
            raise ArenaPolicyError("audit_rate must be in (0, 1]")
        if self.audit_min_calls <= 0:
            raise ArenaPolicyError("audit_min_calls must be positive")
        if not self.external_quality_gate:
            raise ArenaPolicyError(
                "in-engine audit/receipts are diagnostic; an external quality gate is required"
            )
        if type(self.teacher_forced_policy) is not TeacherForcedQualityPolicy:
            raise ArenaPolicyError("fidelity must pin a TeacherForcedQualityPolicy")
        if self.external_quality_gate != self.teacher_forced_policy.protocol:
            raise ArenaPolicyError("external quality gate/policy protocol mismatch")
        for name in (
            "kl_threshold", "argmax_disagree_rate_threshold",
            "p99_kl_threshold", "coverage_dev_threshold",
        ):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ArenaPolicyError(f"{name} must be non-negative or null")
        if self.mode == "kl" and self.kl_threshold is None:
            raise ArenaPolicyError("KL fidelity arenas must pin kl_threshold")


@dataclass(frozen=True)
class ScoringProfile:
    timed_iters: int
    warmup_iters: int
    conditioning_iters: int = 2
    speedup_margin: float = 0.005
    score_k: float = 2.0
    max_noise: float = 0.10
    bookend_baseline: bool = True
    cuda_graphs: bool = True

    def __post_init__(self) -> None:
        if self.timed_iters < 2 or self.warmup_iters < 2:
            raise ArenaPolicyError("crownable scoring needs >=2 timed and >=2 warmup rounds")
        if not 2 <= self.conditioning_iters <= self.warmup_iters:
            raise ArenaPolicyError(
                "crownable scoring needs 2..warmup_iters charged conditioning rounds"
            )
        if self.speedup_margin <= 0 or self.score_k <= 0 or self.max_noise <= 0:
            raise ArenaPolicyError("scoring margins/noise policy must be positive")
        if not self.bookend_baseline:
            raise ArenaPolicyError("crownable scoring requires B,C,B' bookends")
        if not self.cuda_graphs:
            raise ArenaPolicyError("graphs-on is the only crownable serving regime")


@dataclass(frozen=True)
class SettlementProfile:
    """Consensus policy applied after a qualified score is produced.

    Attempt budgets count the initial failed evaluation as attempt one. Three
    infrastructure failures, four noisy no-decisions, or six cumulative attempts
    therefore enter nonterminal operator hold at that exact boundary.
    """

    dethrone_margin: float = 0.02
    round_blocks: int = 100
    weights_refresh_blocks: int = 360
    retry_backoff_blocks: int = 20
    retry_max_backoff_blocks: int = 360
    retry_max_automatic_infrastructure_attempts: int = 3
    retry_max_automatic_no_decision_attempts: int = 4
    retry_max_total_attempts: int = 6
    emission_policy: str = "equal-per-target-koth-v1"
    chain_scope_scheme: str = "genesis-netuid-v1"

    def __post_init__(self) -> None:
        if not (0.0 < self.dethrone_margin < 1.0):
            raise ArenaPolicyError("settlement dethrone_margin must be in (0, 1)")
        if self.round_blocks <= 0 or self.weights_refresh_blocks <= 0:
            raise ArenaPolicyError("settlement block cadences must be positive")
        if (self.retry_backoff_blocks <= 0
                or self.retry_max_backoff_blocks < self.retry_backoff_blocks):
            raise ArenaPolicyError("retry backoff policy is invalid")
        if (
            type(self.retry_max_automatic_infrastructure_attempts) is not int
            or self.retry_max_automatic_infrastructure_attempts <= 0
        ):
            raise ArenaPolicyError(
                "automatic infrastructure retry attempts must be a positive integer"
            )
        if (
            type(self.retry_max_automatic_no_decision_attempts) is not int
            or self.retry_max_automatic_no_decision_attempts <= 0
            or type(self.retry_max_total_attempts) is not int
            or self.retry_max_total_attempts
            < max(
                self.retry_max_automatic_infrastructure_attempts,
                self.retry_max_automatic_no_decision_attempts,
            )
        ):
            raise ArenaPolicyError(
                "no-decision/total retry attempt budgets are invalid"
            )
        if not self.emission_policy or not self.chain_scope_scheme:
            raise ArenaPolicyError("settlement policy/scope scheme must be named")


@dataclass(frozen=True)
class OCIResourceProfile:
    """Consensus-bound host/container resources for crownable engine launches."""

    gpu_count: int
    cpu_logical_count: int
    cpu_model: str
    affinity_policy: str
    cpu_limit: float
    memory_limit_bytes: int
    shm_size: str
    scratch_tmpfs_size: str
    artifact_tmpfs_size: str
    scratch_tmpfs_inodes: int
    artifact_tmpfs_inodes: int
    artifact_max_bytes: int
    artifact_max_files: int
    require_host_tmpfs: bool
    prebuild_timeout_s: float
    bracket_timeout_s: float
    init_timeout_s: float
    batch_timeout_s: float
    pids_limit: int
    nofile_limit: int
    worker_uid: int
    worker_gid: int

    def __post_init__(self) -> None:
        if (
            type(self.gpu_count) is not int
            or self.gpu_count <= 0
            or type(self.cpu_logical_count) is not int
            or self.cpu_logical_count <= 0
        ):
            raise ArenaPolicyError("OCI GPU/CPU counts must be explicit and positive")
        if (
            not self.cpu_model
            or self.affinity_policy != "single-numa-local-v1"
        ):
            raise ArenaPolicyError("OCI CPU class/affinity policy is invalid")
        if (
            isinstance(self.cpu_limit, bool)
            or not isinstance(self.cpu_limit, (int, float))
            or not 1 <= float(self.cpu_limit) <= 1024
            or type(self.memory_limit_bytes) is not int
            or not (1 << 30) <= self.memory_limit_bytes <= (1 << 42)
        ):
            raise ArenaPolicyError("OCI CPU/memory limits are invalid")
        for name in ("shm_size", "scratch_tmpfs_size", "artifact_tmpfs_size"):
            if re.fullmatch(r"[1-9][0-9]*(?:[kKmMgG])?", getattr(self, name)) is None:
                raise ArenaPolicyError(f"OCI {name} must be an explicit positive size")
        if (
            type(self.artifact_max_bytes) is not int
            or not (1 << 20) <= self.artifact_max_bytes <= (1 << 40)
            or type(self.artifact_max_files) is not int
            or not 1 <= self.artifact_max_files <= 1_000_000
            or type(self.pids_limit) is not int
            or not 256 <= self.pids_limit <= 1_048_576
            or type(self.nofile_limit) is not int
            or not 1024 <= self.nofile_limit <= 1_048_576
        ):
            raise ArenaPolicyError("OCI artifact/process/file resource limits are invalid")
        if (
            type(self.scratch_tmpfs_inodes) is not int
            or type(self.artifact_tmpfs_inodes) is not int
            or not 1024 <= self.scratch_tmpfs_inodes <= 10_000_000
            or not 1024 <= self.artifact_tmpfs_inodes <= 10_000_000
        ):
            raise ArenaPolicyError("OCI tmpfs inode limits are invalid")
        if type(self.require_host_tmpfs) is not bool or not self.require_host_tmpfs:
            raise ArenaPolicyError("crownable OCI staging/JIT must require host tmpfs")
        for name in (
            "prebuild_timeout_s", "bracket_timeout_s", "init_timeout_s",
            "batch_timeout_s",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not 1 <= float(value) <= 86_400
            ):
                raise ArenaPolicyError(f"OCI {name} must be in [1, 86400]")
        if self.prebuild_timeout_s >= self.bracket_timeout_s:
            raise ArenaPolicyError("OCI prebuild timeout must be below whole-bracket timeout")
        if (
            type(self.worker_uid) is not int
            or type(self.worker_gid) is not int
            or not 1 <= self.worker_uid <= 2_147_483_647
            or not 1 <= self.worker_gid <= 2_147_483_647
        ):
            raise ArenaPolicyError("crownable OCI worker UID/GID must be nonzero")


@dataclass(frozen=True)
class DeviceStateClassProfile:
    """Consensus GPU configuration and active/idle conditioning envelope."""

    power_limit_mw: int
    compute_mode: str
    persistence_mode: str
    application_graphics_clock_mhz: int | None
    application_memory_clock_mhz: int | None
    max_graphics_clock_mhz: int
    max_memory_clock_mhz: int
    require_process_on_every_gpu: bool
    maximum_temperature_c: int
    maximum_gpu_utilization_percent: int
    maximum_memory_utilization_percent: int
    required_consecutive_idle_samples: int
    poll_interval_s: float
    ready_poll_interval_s: float
    drain_timeout_s: float
    maximum_samples: int

    def __post_init__(self) -> None:
        if (
            type(self.power_limit_mw) is not int
            or self.power_limit_mw <= 0
            or self.compute_mode not in {"Default", "Exclusive_Process"}
            or self.persistence_mode not in {"Enabled", "Disabled"}
            or type(self.max_graphics_clock_mhz) is not int
            or type(self.max_memory_clock_mhz) is not int
            or self.max_graphics_clock_mhz <= 0
            or self.max_memory_clock_mhz <= 0
        ):
            raise ArenaPolicyError("GPU immutable management configuration is invalid")
        for name in (
            "application_graphics_clock_mhz", "application_memory_clock_mhz",
        ):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value <= 0):
                raise ArenaPolicyError(f"GPU {name} must be positive or null")
        if type(self.require_process_on_every_gpu) is not bool:
            raise ArenaPolicyError("GPU active process-presence policy is invalid")
        for name, low, high in (
            ("maximum_temperature_c", 0, 120),
            ("maximum_gpu_utilization_percent", 0, 25),
            ("maximum_memory_utilization_percent", 0, 25),
            ("required_consecutive_idle_samples", 2, 32),
            ("maximum_samples", 2, 4096),
        ):
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ArenaPolicyError(f"GPU {name} is invalid")
        if (
            isinstance(self.poll_interval_s, bool)
            or not 0.05 <= float(self.poll_interval_s) <= 60
            or isinstance(self.drain_timeout_s, bool)
            or not 1 <= float(self.drain_timeout_s) <= 3600
        ):
            raise ArenaPolicyError("GPU conditioning cadence/deadline is invalid")
        if (
            isinstance(self.ready_poll_interval_s, bool)
            or not 0.05 <= float(self.ready_poll_interval_s) <= 5.0
        ):
            raise ArenaPolicyError("GPU ready-sampling cadence is invalid")


@dataclass(frozen=True)
class ArenaProfile:
    name: str
    model_path: str
    model_id: str
    model_revision: str
    model_manifest_digest: str
    model_content_digest: str
    dtype: str
    sglang_version: str
    validator_image: str
    referee_source_digest: str
    referee_tree_digest: str
    gpu_architecture: str
    gpu_topology_sha256: str
    gpu_name: str
    gpu_memory_mib: int
    driver_version: str
    runtime_overlays: tuple[RuntimeFileOverlay, ...]
    oci_resources: OCIResourceProfile
    device_state: DeviceStateClassProfile
    tp_size: int
    attention_backend: str
    moe_runner_backend: str
    mem_fraction_static: float
    max_running_requests: int | None
    engine_kwargs: Mapping[str, Any]
    environment: Mapping[str, str]
    workload: WorkloadProfile
    fidelity: FidelityProfile
    scoring: ScoringProfile
    settlement: SettlementProfile = field(default_factory=SettlementProfile)
    model_seed: int = 0
    temperature: float = 0.0
    log_level: str = "warning"
    deterministic: bool = False
    disable_custom_all_reduce: bool = False
    require_isolation: bool = True
    allow_unsafe_no_isolation: bool = False
    allow_framework: bool = False
    allow_candidate_engine_overrides: bool = False

    def __post_init__(self) -> None:
        if not (self.name and self.model_path and self.model_id
                and self.model_revision and self.sglang_version):
            raise ArenaPolicyError(
                "arena name/model identity/revision/sglang pin must be non-empty"
            )
        if not re.fullmatch(r"[0-9a-f]{40,64}", self.model_revision):
            raise ArenaPolicyError("model_revision must be an immutable hex revision")
        for name in (
            "model_manifest_digest", "model_content_digest", "referee_source_digest",
            "referee_tree_digest",
        ):
            if not re.fullmatch(r"sha256:[0-9a-f]{64}", getattr(self, name)):
                raise ArenaPolicyError(f"{name} must be a sha256 content identity")
        if not re.fullmatch(r"[^@]+@sha256:[0-9a-f]{64}", self.validator_image):
            raise ArenaPolicyError("validator_image must be pinned by immutable sha256 digest")
        if not self.gpu_architecture or self.tp_size <= 0:
            raise ArenaPolicyError("arena architecture/TP size must be explicit")
        if re.fullmatch(r"[0-9a-f]{64}", self.gpu_topology_sha256) is None:
            raise ArenaPolicyError("arena GPU topology must be a SHA-256 identity")
        if (not self.gpu_name or self.gpu_memory_mib <= 0
                or re.fullmatch(r"[0-9]+(?:\.[0-9]+){1,3}", self.driver_version) is None):
            raise ArenaPolicyError(
                "arena GPU name/memory/host-driver identity must be explicit"
            )
        try:
            runtime_overlays = normalize_runtime_overlays(self.runtime_overlays)
        except Exception as exc:
            raise ArenaPolicyError(f"invalid arena runtime overlays: {exc}") from None
        object.__setattr__(self, "runtime_overlays", runtime_overlays)
        if type(self.oci_resources) is not OCIResourceProfile:
            raise ArenaPolicyError("arena must pin an exact OCIResourceProfile")
        if self.oci_resources.gpu_count != self.tp_size:
            raise ArenaPolicyError("arena TP size must equal its pinned OCI GPU count")
        if type(self.device_state) is not DeviceStateClassProfile:
            raise ArenaPolicyError("arena must pin an exact DeviceStateClassProfile")
        if self.dtype not in {"bfloat16", "float16", "float32"}:
            raise ArenaPolicyError(f"unsupported arena dtype {self.dtype!r}")
        if not (0.0 < self.mem_fraction_static < 1.0):
            raise ArenaPolicyError("mem_fraction_static must be in (0, 1)")
        if self.max_running_requests is not None and self.max_running_requests <= 0:
            raise ArenaPolicyError("max_running_requests must be positive")
        if self.model_seed < 0 or self.temperature < 0 or not self.log_level:
            raise ArenaPolicyError("model seed/temperature/log level must be valid")
        kwargs = _canonical_json_value(dict(self.engine_kwargs), path="engine_kwargs")
        env = _canonical_json_value(dict(self.environment), path="environment")
        if any(not isinstance(value, str) for value in env.values()):
            raise ArenaPolicyError("arena environment values must be strings")
        object.__setattr__(self, "engine_kwargs", MappingProxyType(kwargs))
        object.__setattr__(self, "environment", MappingProxyType(env))

    def canonical_payload(self) -> dict[str, Any]:
        # Walk fields ourselves: dataclasses.asdict deep-copies values and therefore
        # cannot consume immutable MappingProxyType policy maps on Python 3.11.
        return _canonical_json_value(self, path="arena")

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(
            self.canonical_payload(), sort_keys=True, separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @property
    def bracket(self) -> str:
        """Immutable arena+regime championship namespace."""
        return f"{self.name}:{self.workload.regime}@{self.fingerprint}"

    @property
    def competable(self) -> bool:
        return bool(
            self.validator_image
            and self.sglang_version
            and self.fidelity.external_quality_gate
            and self.scoring.bookend_baseline
            and self.scoring.cuda_graphs
            and self.require_isolation
            and not self.allow_unsafe_no_isolation
            and not self.allow_framework
            and not self.allow_candidate_engine_overrides
        )

    def eval_config_kwargs(self) -> dict[str, Any]:
        """Return the complete crownable ``EvalConfig`` policy for this arena.

        The CLI deliberately constructs a scored run from this mapping instead of
        merging miner/operator flags into it.  Adding a score-affecting evaluator
        knob therefore requires adding it here (and thus to the fingerprint), not
        remembering to reject one more command-line override.
        """
        if not self.competable:
            raise ArenaPolicyError(f"arena {self.name!r} is not crownable")
        if self.workload.prompt_generator not in {
            "optima.eval.prompts.long-v2-one-shot",
            "optima.eval.prompts.short-v2-one-shot",
        }:
            raise ArenaPolicyError(
                f"unsupported prompt generator {self.workload.prompt_generator!r}"
            )
        return {
            "model_path": self.model_path,
            "dtype": self.dtype,
            "max_new_tokens": self.workload.max_new_tokens,
            "num_prompts": self.workload.num_prompts,
            "timed_iters": self.scoring.timed_iters,
            "top_logprobs_num": self.workload.top_logprobs,
            "ignore_eos": self.workload.ignore_eos,
            "warmup_iters": self.scoring.warmup_iters,
            "conditioning_iters": self.scoring.conditioning_iters,
            "deterministic": self.deterministic,
            "temperature": self.temperature,
            "kl_threshold": self.fidelity.kl_threshold,
            "argmax_disagree_rate_threshold": self.fidelity.argmax_disagree_rate_threshold,
            "p99_kl_threshold": self.fidelity.p99_kl_threshold,
            "coverage_dev_threshold": self.fidelity.coverage_dev_threshold,
            "framework_mode": False,
            "isolate": self.require_isolation,
            "allow_unsafe_no_isolation": self.allow_unsafe_no_isolation,
            "seed": self.model_seed,
            "prompt_seed": 0,
            "input_len": self.workload.input_len,
            "speedup_margin": self.scoring.speedup_margin,
            "bookend_baseline": self.scoring.bookend_baseline,
            "score_k": self.scoring.score_k,
            "max_noise": self.scoring.max_noise,
            "attention_backend": self.attention_backend,
            "disable_cuda_graph": not self.scoring.cuda_graphs,
            "mem_fraction_static": self.mem_fraction_static,
            "log_level": self.log_level,
            "max_running_requests": self.max_running_requests,
            "tp_size": self.tp_size,
            "moe_runner_backend": self.moe_runner_backend,
            "disable_custom_all_reduce": self.disable_custom_all_reduce,
            "candidate_attention_backend": None,
            "candidate_moe_runner_backend": None,
            "candidate_disable_custom_all_reduce": None,
            "extra_engine_kwargs": dict(self.engine_kwargs),
            "candidate_extra_engine_kwargs": {},
            "fidelity_mode": self.fidelity.mode,
            "audit_rate": self.fidelity.audit_rate,
            "audit_min_calls": self.fidelity.audit_min_calls,
        }

    def verify_model_receipt(
        self,
        model_path: str | Path | None = None,
        *,
        verify_bytes: bool | None = None,
    ) -> None:
        """Fail closed if the runtime model is not the arena-pinned artifact set."""
        runtime_path = self.model_path if model_path is None else model_path
        revision, manifest_digest = huggingface_model_manifest(runtime_path)
        if revision != self.model_revision:
            raise ArenaPolicyError(
                f"model revision mismatch for {self.name!r}: "
                f"{revision!r} != {self.model_revision!r}"
            )
        if manifest_digest != self.model_manifest_digest:
            raise ArenaPolicyError(
                f"model manifest mismatch for {self.name!r}: "
                f"{manifest_digest!r} != {self.model_manifest_digest!r}"
            )
        if verify_bytes is None:
            verify_bytes = os.environ.get("OPTIMA_VERIFY_MODEL_BYTES", "").lower() in {
                "1", "true", "yes", "on",
            }
        if type(verify_bytes) is not bool:
            raise ArenaPolicyError("verify_bytes must be boolean")
        verify_model_content_seal(
            runtime_path,
            expected_digest=self.model_content_digest,
            verify_bytes=verify_bytes,
        )

    def verify_referee_source(self) -> None:
        """Fail closed if the mounted referee differs from the registered arena."""
        actual = referee_source_digest()
        if actual != self.referee_source_digest:
            raise ArenaPolicyError(
                f"referee source mismatch for {self.name!r}: "
                f"{actual!r} != {self.referee_source_digest!r}"
            )

    def verify_runtime_packages(self) -> None:
        """Attest the package actually installed in the immutable runtime image."""
        try:
            installed = importlib.metadata.version("sglang")
        except importlib.metadata.PackageNotFoundError as exc:
            raise ArenaPolicyError("sglang distribution is not installed") from exc
        if installed != self.sglang_version:
            raise ArenaPolicyError(
                f"installed sglang mismatch for {self.name!r}: "
                f"{installed!r} != {self.sglang_version!r}"
            )


@contextmanager
def arena_environment(arena: ArenaProfile):
    """Apply the immutable arena environment for all spawned engine launches."""
    saved = {key: os.environ.get(key) for key in arena.environment}
    os.environ.update(arena.environment)
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def derive_prompt_seed(
    arena: ArenaProfile,
    *,
    bundle_hash: str,
    round_id: int,
    block_hash: str,
) -> int:
    """Derive the hidden workload seed from post-commit chain entropy.

    The reveal/bundle hash is already fixed before ``block_hash`` exists. Including
    the arena and bundle prevents one observed seed from being replayed into another
    championship. The 63-bit result survives argparse/JSON and Python RNG paths.
    """
    if arena.workload.prompt_seed_scheme != "post-commit-blockhash-v1":
        raise ArenaPolicyError(
            f"unsupported prompt seed scheme {arena.workload.prompt_seed_scheme!r}"
        )
    if not block_hash:
        raise ArenaPolicyError("post-commit prompt seed requires a block hash")
    material = (
        f"optima:{arena.workload.prompt_seed_scheme}:{arena.fingerprint}:"
        f"{bundle_hash}:{int(round_id)}:{block_hash}"
    ).encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "big") & ((1 << 63) - 1)
    # Zero is reserved as an unmistakable ad-hoc/development seed.
    return seed or 1


_M3_IMAGE = (
    "lmsysorg/sglang@sha256:"
    "de63ac56df5d7b064451e21147eaab89634a02332d830ca8c01cb8c033b3a78f"
)
# A registered arena with a second, hand-copied revision string would be unusable
# at best and a split-brain championship at worst.
_M3_SGLANG = PINNED_SGLANG
_M3_MODEL_ID = "MiniMax-M3-NVFP4"
_M3_MODEL_REVISION = "668435825700a0047399441720f430bdd8eca0ab"
_M3_MODEL_MANIFEST = (
    "sha256:455066f95b805a646e9ca2fca67ffabe6069cb01265effc9dbe619574f0882fd"
)
_M3_MODEL_CONTENT = (
    "sha256:bb1f4be5e15631b6c8997ed0d8e55afe2f314923f41a42c3c3ef139eae623621"
)
_M3_GPU_TOPOLOGY = "0599c1e63ab0bcbe6226702838731f0027e9b381a7c30d3693c0faca5574dbbb"
_M3_GPU_NAME = "NVIDIA B300 SXM6 AC"
_M3_GPU_MEMORY_MIB = 275040
_M3_DRIVER_VERSION = "595.71.05"
_M3_RUNTIME_OVERLAYS = (
    RuntimeFileOverlay(
        source="sglang_patch/flashinfer_trtllm.py",
        target=(
            "/sgl-workspace/sglang/python/sglang/srt/layers/moe/moe_runner/"
            "flashinfer_trtllm.py"
        ),
        sha256="4eada0776f23f7aa4633d57dea69e55fadbd335fa6d01cc2983cbba11981f289",
        size=48_224,
    ),
    RuntimeFileOverlay(
        source="sglang_patch/modelopt_quant.py",
        target=(
            "/sgl-workspace/sglang/python/sglang/srt/layers/quantization/"
            "modelopt_quant.py"
        ),
        sha256="53f92fac550b46ed4b297a229b6f10fe9235424730d69143ce970d828f7b36b5",
        size=103_027,
    ),
)
_M3_OCI_RESOURCES = OCIResourceProfile(
    gpu_count=4,
    cpu_logical_count=96,
    cpu_model="Intel(R) Xeon(R) 6740P",
    affinity_policy="single-numa-local-v1",
    cpu_limit=96.0,
    memory_limit_bytes=1 << 40,
    shm_size="256g",
    scratch_tmpfs_size="16g",
    artifact_tmpfs_size="32g",
    scratch_tmpfs_inodes=1_000_000,
    artifact_tmpfs_inodes=65_536,
    artifact_max_bytes=16 * 1024 * 1024 * 1024,
    artifact_max_files=16_384,
    require_host_tmpfs=True,
    prebuild_timeout_s=1_800.0,
    bracket_timeout_s=7_200.0,
    init_timeout_s=1_800.0,
    batch_timeout_s=1_800.0,
    pids_limit=65_536,
    nofile_limit=65_536,
    worker_uid=65_532,
    worker_gid=65_532,
)
_M3_DEVICE_STATE = DeviceStateClassProfile(
    power_limit_mw=1_100_000,
    compute_mode="Default",
    persistence_mode="Disabled",
    application_graphics_clock_mhz=None,
    application_memory_clock_mhz=None,
    max_graphics_clock_mhz=2_032,
    max_memory_clock_mhz=3_996,
    require_process_on_every_gpu=True,
    maximum_temperature_c=65,
    maximum_gpu_utilization_percent=5,
    maximum_memory_utilization_percent=5,
    required_consecutive_idle_samples=3,
    poll_interval_s=2.0,
    ready_poll_interval_s=0.1,
    drain_timeout_s=300.0,
    maximum_samples=256,
)
_REFEREE_SOURCE_DIGEST = APPROVED_REFEREE_SOURCE_DIGEST
_REFEREE_TREE_DIGEST = APPROVED_REFEREE_TREE_DIGEST
_M3_COMMON_ENGINE = {
    "quantization": "modelopt_fp4",
    "page_size": 128,
    "trust_remote_code": True,
    "cuda_graph_backend_prefill": "disabled",
}

# Evidence collection is enabled, but B300 acceptance constants have not yet been
# frozen from independent stock brackets.  Zero acceptance envelopes are deliberate
# inert sentinels: ``calibration_state='uncalibrated'`` returns NO_DECISION before
# consulting them, so this branch cannot accidentally crown from made-up constants.
_M3_TEACHER_POLICY_UNCALIBRATED = TeacherForcedQualityPolicy(
    protocol="controller-posthoc-teacher-forced-v2",
    calibration_state="uncalibrated",
    clusters_per_batch=8,
    nll_clip=1_000.0,
    tail_nll_threshold=1_000.0,
    familywise_z=3.0,
    stock_mean_nll_envelope=0.0,
    stock_worst_nll_envelope=0.0,
    stock_tail_rate_envelope=0.0,
    stock_topk_kl_envelope=0.0,
    stock_argmax_rate_envelope=0.0,
    stock_coverage_envelope=0.0,
    mean_nll_delta=0.0,
    worst_nll_delta=0.0,
    tail_rate_delta=0.0,
    topk_kl_delta=0.0,
    argmax_rate_delta=0.0,
    coverage_delta=0.0,
    require_hidden_tasks=True,
    stock_hidden_score_envelope=0.0,
    hidden_score_delta=0.0,
    hidden_score_floor=0.0,
)


MINIMAX_M3_B300_TP4_LONGPREFILL_V1 = ArenaProfile(
    name="minimax-m3-b300-tp4-longprefill-v1",
    model_path="/models/MiniMax-M3-NVFP4",
    model_id=_M3_MODEL_ID,
    model_revision=_M3_MODEL_REVISION,
    model_manifest_digest=_M3_MODEL_MANIFEST,
    model_content_digest=_M3_MODEL_CONTENT,
    dtype="bfloat16",
    sglang_version=_M3_SGLANG,
    validator_image=_M3_IMAGE,
    referee_source_digest=_REFEREE_SOURCE_DIGEST,
    referee_tree_digest=_REFEREE_TREE_DIGEST,
    gpu_architecture="sm103",
    gpu_topology_sha256=_M3_GPU_TOPOLOGY,
    gpu_name=_M3_GPU_NAME,
    gpu_memory_mib=_M3_GPU_MEMORY_MIB,
    driver_version=_M3_DRIVER_VERSION,
    runtime_overlays=_M3_RUNTIME_OVERLAYS,
    oci_resources=_M3_OCI_RESOURCES,
    device_state=_M3_DEVICE_STATE,
    tp_size=4,
    attention_backend="fa4",
    moe_runner_backend="flashinfer_cutlass",
    mem_fraction_static=0.88,
    max_running_requests=None,
    engine_kwargs={
        **_M3_COMMON_ENGINE,
        "kv_cache_dtype": "auto",
        "context_length": 262144,
        "chunked_prefill_size": 32768,
        "max_prefill_tokens": 32768,
        "disable_radix_cache": True,
    },
    environment={
        "OPTIMA_MSA_PREFILL_SEAM": "1",
        "SGLANG_FLASHINFER_AUTOTUNE_CACHE": "0",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    },
    workload=WorkloadProfile(
        regime="longprefill-255k-c8-out64",
        prompt_generator="optima.eval.prompts.long-v2-one-shot",
        input_len=225000,
        num_prompts=8,
        max_new_tokens=64,
    ),
    fidelity=FidelityProfile(
        mode="audit",
        audit_rate=0.20,
        audit_min_calls=32,
        # This names the required trusted-controller gate.  It is deliberately not
        # satisfied by scheduler-written audit/receipt files alone.
        external_quality_gate="controller-posthoc-teacher-forced-v2",
        teacher_forced_policy=_M3_TEACHER_POLICY_UNCALIBRATED,
    ),
    scoring=ScoringProfile(timed_iters=3, warmup_iters=3),
)


MINIMAX_M3_B300_TP4_DECODE_V1 = ArenaProfile(
    name="minimax-m3-b300-tp4-decode-v1",
    model_path="/models/MiniMax-M3-NVFP4",
    model_id=_M3_MODEL_ID,
    model_revision=_M3_MODEL_REVISION,
    model_manifest_digest=_M3_MODEL_MANIFEST,
    model_content_digest=_M3_MODEL_CONTENT,
    dtype="bfloat16",
    sglang_version=_M3_SGLANG,
    validator_image=_M3_IMAGE,
    referee_source_digest=_REFEREE_SOURCE_DIGEST,
    referee_tree_digest=_REFEREE_TREE_DIGEST,
    gpu_architecture="sm103",
    gpu_topology_sha256=_M3_GPU_TOPOLOGY,
    gpu_name=_M3_GPU_NAME,
    gpu_memory_mib=_M3_GPU_MEMORY_MIB,
    driver_version=_M3_DRIVER_VERSION,
    runtime_overlays=_M3_RUNTIME_OVERLAYS,
    oci_resources=_M3_OCI_RESOURCES,
    device_state=_M3_DEVICE_STATE,
    tp_size=4,
    attention_backend="fa4",
    moe_runner_backend="flashinfer_cutlass",
    mem_fraction_static=0.90,
    max_running_requests=256,
    engine_kwargs={
        **_M3_COMMON_ENGINE,
        "context_length": 16384,
        "chunked_prefill_size": 4096,
        "enable_flashinfer_allreduce_fusion": True,
    },
    environment={
        "OPTIMA_ARFUSION_SEAM": "1",
        "SGLANG_FLASHINFER_AUTOTUNE_CACHE": "0",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    },
    workload=WorkloadProfile(
        regime="decode-c256-out256",
        prompt_generator="optima.eval.prompts.short-v2-one-shot",
        input_len=None,
        num_prompts=256,
        max_new_tokens=256,
    ),
    fidelity=FidelityProfile(
        mode="audit",
        audit_rate=0.05,
        audit_min_calls=32,
        external_quality_gate="controller-posthoc-teacher-forced-v2",
        teacher_forced_policy=_M3_TEACHER_POLICY_UNCALIBRATED,
    ),
    scoring=ScoringProfile(timed_iters=3, warmup_iters=4),
)


ARENAS: Mapping[str, ArenaProfile] = MappingProxyType({
    arena.name: arena
    for arena in (
        MINIMAX_M3_B300_TP4_LONGPREFILL_V1,
        MINIMAX_M3_B300_TP4_DECODE_V1,
    )
})


def get_arena(name: str) -> ArenaProfile:
    try:
        return ARENAS[name]
    except KeyError:
        known = ", ".join(sorted(ARENAS)) or "(none)"
        raise ArenaPolicyError(f"unknown scoring arena {name!r}; known: {known}") from None


def list_arenas() -> list[str]:
    return sorted(ARENAS)
