"""Controller-owned publication of authoritative host attestation evidence.

The candidate never mounts this sidecar.  It binds the stock-runtime preflight,
chain/arena provenance, and the trusted host's exact pre/active/post GPU receipts
into one immutable content-addressed JSON object.  The active receipt spans the
final warmup conditioning boundary for each arm. Publication is no-replace and
directory-relative so a hostile candidate artifact tree cannot redirect it.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import secrets
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


HOST_ATTESTATION_SCHEMA = "optima-host-attestation-v4"
RUNTIME_PREFLIGHT_SCHEMA = "optima-stock-runtime-preflight-v1"
DEVICE_RECEIPT_SCHEMA = "optima.device-state-receipt.v1"
ACTIVE_DEVICE_RECEIPT_SCHEMA = "optima.device-state-active-receipt.v2"
HOST_ATTESTATION_DIRECTORY = "host_attestations"

MAX_DEVICE_RECEIPTS = 96
MAX_RUNTIME_PREFLIGHT_BYTES = 64 * 1024
MAX_HOST_ATTESTATION_BYTES = 16 * 1024 * 1024
MAX_DEVICE_SAMPLES = 4096
MAX_GPU_COUNT = 64
MAX_PROCESSES_PER_SAMPLE = 4096

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SHA256_ID = re.compile(r"sha256:[0-9a-f]{64}\Z")
_HEX_REVISION = re.compile(r"[0-9a-f]{40,64}\Z")
_IMAGE = re.compile(r"[a-z0-9][a-z0-9._/:+-]{0,255}@sha256:[0-9a-f]{64}\Z")
_ID = re.compile(r"[A-Za-z0-9_.-]{1,128}\Z")
_ARM = re.compile(r"(baseline|candidate|bookend)-([1-9][0-9]{0,8})\Z")
_UUID = re.compile(r"GPU-[0-9A-Fa-f-]{16,64}\Z")
_PSTATE = re.compile(r"P[0-9]{1,2}\Z")
_PROCESS_KIND = re.compile(r"[A-Z][A-Z+/]{0,7}\Z")

_CONTEXT_KEYS = frozenset({
    "arena_name",
    "arena_fingerprint",
    "arena_bracket",
    "regime",
    "bundle_hash",
    "sglang_version",
    "validator_image",
    "referee_source_digest",
    "referee_tree_digest",
    "model_revision",
    "model_manifest_digest",
    "model_content_digest",
    "prompt_seed",
    "prompt_engine_version",
    "prompt_seed_scheme",
    "seed_round_id",
    "seed_block",
    "seed_block_hash",
    "chain_scope",
    "validator_hotkey",
    "evaluation_id",
    "miner_hotkey",
    "settlement_round_id",
    "evaluation_block",
    "target",
    "mode",
    "member_slots",
    "score",
    "passed_quality",
    "passed_timed_quality",
    "passed_warmup_quality",
    "passed_speedup",
    "confident",
    "crownable",
    "quality_evidence",
    "qualification_evidence_sha256",
})
_RUNTIME_KEYS = frozenset({
    "schema",
    "requested_image",
    "requested_manifest_digest",
    "local_image_id",
    "repo_digests",
    "docker_binary",
    "uid",
    "gid",
    "sglang_version",
    "python",
    "packages",
    "cuda",
    "security_argv_sha256",
})
_DEVICE_KEYS = frozenset({
    "schema",
    "sequence",
    "arm",
    "phase",
    "selected_physical_gpu_ids",
    "configuration_sha256",
    "policy_sha256",
    "started_monotonic_s",
    "completed_monotonic_s",
    "consecutive_idle_samples",
    "samples",
})
_ACTIVE_DEVICE_KEYS = frozenset({
    "schema",
    "sequence",
    "arm",
    "event",
    "selected_physical_gpu_ids",
    "configuration_sha256",
    "policy_sha256",
    "started_monotonic_s",
    "completed_monotonic_s",
    "consecutive_active_samples",
    "release_sample_index",
    "post_release_ready_samples",
    "samples",
})
_SAMPLE_KEYS = frozenset({
    "monotonic_s",
    "telemetry",
    "processes",
    "idle",
    "idle_reason",
    "active_envelope_passed",
    "active_envelope_reason",
})
_TELEMETRY_KEYS = frozenset({
    "physical_id",
    "uuid",
    "pstate",
    "temperature_c",
    "gpu_utilization_percent",
    "memory_utilization_percent",
    "current_graphics_clock_mhz",
    "current_memory_clock_mhz",
    "power_draw_mw",
})
_PROCESS_KEYS = frozenset({
    "physical_id", "pid", "kind", "process_name",
})
_PACKAGE_KEYS = frozenset({
    "cuda-python",
    "flashinfer-python",
    "nvidia-cuda-runtime-cu12",
    "torch",
    "triton",
})


class HostAttestationError(RuntimeError):
    """Trusted controller evidence or publication state is invalid."""

    validator_fault = True
    retryable = False


@dataclass(frozen=True, slots=True)
class HostAttestationReference:
    sha256: str
    path: str
    receipt_count: int
    runtime_preflight_sha256: str
    qualification_evidence_sha256: str
    device_configuration_sha256: str
    device_policy_sha256: str
    arms: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "sha256": self.sha256,
            "path": self.path,
            "receipt_count": self.receipt_count,
            "runtime_preflight_sha256": self.runtime_preflight_sha256,
            "qualification_evidence_sha256": self.qualification_evidence_sha256,
            "device_configuration_sha256": self.device_configuration_sha256,
            "device_policy_sha256": self.device_policy_sha256,
            "arms": list(self.arms),
        }


def host_attestation_context(
    arena,
    *,
    bundle_hash: str,
    prompt_seed: int,
    seed_round_id: int,
    seed_block: int,
    seed_block_hash: str,
    chain_scope: str,
    validator_hotkey: str,
    evaluation_id: str,
    miner_hotkey: str,
    settlement_round_id: int,
    evaluation_block: int,
    target: str,
    mode: str,
    member_slots: Sequence[str],
    score: float,
    passed_quality: bool,
    passed_timed_quality: bool,
    passed_warmup_quality: bool,
    passed_speedup: bool,
    confident: bool,
    crownable: bool,
    quality_evidence: str,
    qualification_evidence_sha256: str,
) -> dict[str, object]:
    """Build the one canonical arena/bundle/seed context used everywhere.

    Chain publication, direct CLI publication, qualification inspection, and
    settlement verification must not hand-maintain parallel provenance mappings.
    """

    try:
        context = {
            "arena_name": arena.name,
            "arena_fingerprint": arena.fingerprint,
            "arena_bracket": arena.bracket,
            "regime": arena.workload.regime,
            "bundle_hash": bundle_hash,
            "sglang_version": arena.sglang_version,
            "validator_image": arena.validator_image,
            "referee_source_digest": arena.referee_source_digest,
            "referee_tree_digest": arena.referee_tree_digest,
            "model_revision": arena.model_revision,
            "model_manifest_digest": arena.model_manifest_digest,
            "model_content_digest": arena.model_content_digest,
            "prompt_seed": prompt_seed,
            "prompt_engine_version": arena.workload.prompt_engine_version,
            "prompt_seed_scheme": arena.workload.prompt_seed_scheme,
            "seed_round_id": seed_round_id,
            "seed_block": seed_block,
            "seed_block_hash": seed_block_hash,
            "chain_scope": chain_scope,
            "validator_hotkey": validator_hotkey,
            "evaluation_id": evaluation_id,
            "miner_hotkey": miner_hotkey,
            "settlement_round_id": settlement_round_id,
            "evaluation_block": evaluation_block,
            "target": target,
            "mode": mode,
            "member_slots": list(member_slots),
            "score": score,
            "passed_quality": passed_quality,
            "passed_timed_quality": passed_timed_quality,
            "passed_warmup_quality": passed_warmup_quality,
            "passed_speedup": passed_speedup,
            "confident": confident,
            "crownable": crownable,
            "quality_evidence": quality_evidence,
            "qualification_evidence_sha256": qualification_evidence_sha256,
        }
    except (AttributeError, TypeError) as exc:
        raise HostAttestationError(
            f"cannot build host attestation context from arena: {exc}"
        ) from None
    normalized = _normalize_context(context)
    try:
        round_blocks = arena.settlement.round_blocks
    except (AttributeError, TypeError):
        raise HostAttestationError(
            "arena lacks an immutable settlement round policy"
        ) from None
    if normalized["settlement_round_id"] != normalized["evaluation_block"] // round_blocks:
        raise HostAttestationError(
            "settlement round differs from the evaluation block"
        )
    return normalized


def _exact_mapping(value: object, keys: frozenset[str], *, label: str) -> Mapping:
    if not isinstance(value, Mapping) or set(value) != keys:
        actual = sorted(value) if isinstance(value, Mapping) else type(value).__name__
        raise HostAttestationError(
            f"{label} fields/type mismatch: expected={sorted(keys)!r} actual={actual!r}"
        )
    if any(type(key) is not str for key in value):
        raise HostAttestationError(f"{label} keys must be exact strings")
    return value


def _text(value: object, *, label: str, maximum: int = 4096,
          allow_empty: bool = False) -> str:
    if (
        not isinstance(value, str)
        or len(value) > maximum
        or any(char in value for char in "\x00\r\n")
        or (not allow_empty and not value)
    ):
        raise HostAttestationError(f"{label} must be a bounded single-line string")
    return value


def _integer(value: object, *, label: str, low: int, high: int) -> int:
    if type(value) is not int or not low <= value <= high:
        raise HostAttestationError(f"{label} must be an integer in [{low}, {high}]")
    return value


def _number(value: object, *, label: str, low: float = 0.0,
            high: float = 1e15) -> float:
    if (
        type(value) not in (int, float)
        or not math.isfinite(float(value))
        or not low <= float(value) <= high
    ):
        raise HostAttestationError(f"{label} must be a bounded finite number")
    return float(value)


def _optional_integer(value: object, *, label: str, low: int,
                      high: int) -> int | None:
    if value is None:
        return None
    return _integer(value, label=label, low=low, high=high)


def _canonical_json(value: object, *, label: str, maximum: int,
                    trailing_newline: bool = False) -> bytes:
    try:
        raw = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError, RecursionError) as exc:
        raise HostAttestationError(f"{label} is not canonical JSON: {exc}") from None
    if not raw or len(raw) + int(trailing_newline) > maximum:
        raise HostAttestationError(f"{label} exceeds its {maximum}-byte bound")
    return raw + (b"\n" if trailing_newline else b"")


def _normalize_context(context: Mapping[str, Any]) -> dict[str, object]:
    data = _exact_mapping(context, _CONTEXT_KEYS, label="attestation context")
    normalized: dict[str, object] = {}
    for name in (
        "arena_name", "regime", "sglang_version", "prompt_engine_version",
        "prompt_seed_scheme",
    ):
        normalized[name] = _text(data[name], label=f"context.{name}", maximum=256)
    normalized["arena_bracket"] = _text(
        data["arena_bracket"], label="context.arena_bracket"
    )
    for name in ("arena_fingerprint", "bundle_hash"):
        value = _text(data[name], label=f"context.{name}", maximum=64)
        if _SHA256.fullmatch(value) is None:
            raise HostAttestationError(f"context.{name} must be lowercase SHA-256")
        normalized[name] = value
    image = _text(data["validator_image"], label="context.validator_image")
    if _IMAGE.fullmatch(image) is None:
        raise HostAttestationError("context.validator_image must be an immutable image")
    normalized["validator_image"] = image
    for name in (
        "referee_source_digest", "referee_tree_digest", "model_manifest_digest",
        "model_content_digest",
    ):
        value = _text(data[name], label=f"context.{name}", maximum=71)
        if _SHA256_ID.fullmatch(value) is None:
            raise HostAttestationError(f"context.{name} must be a SHA-256 identity")
        normalized[name] = value
    revision = _text(data["model_revision"], label="context.model_revision", maximum=64)
    if _HEX_REVISION.fullmatch(revision) is None:
        raise HostAttestationError("context.model_revision must be immutable hex")
    normalized["model_revision"] = revision
    normalized["prompt_seed"] = _integer(
        data["prompt_seed"], label="context.prompt_seed", low=1, high=(1 << 63) - 1
    )
    normalized["seed_round_id"] = _integer(
        data["seed_round_id"], label="context.seed_round_id", low=0, high=(1 << 63) - 1
    )
    normalized["seed_block"] = _integer(
        data["seed_block"], label="context.seed_block", low=0, high=(1 << 63) - 1
    )
    block_hash = _text(
        data["seed_block_hash"], label="context.seed_block_hash", maximum=66
    )
    if re.fullmatch(r"0x[0-9a-f]{64}", block_hash) is None:
        raise HostAttestationError("context.seed_block_hash must be canonical lowercase hex")
    normalized["seed_block_hash"] = block_hash
    chain_scope = _text(
        data["chain_scope"], label="context.chain_scope", maximum=256
    )
    if re.fullmatch(
        r"[A-Za-z0-9_.-]{1,128}:sha256:[0-9a-f]{64}", chain_scope
    ) is None:
        raise HostAttestationError("context.chain_scope must be canonical")
    normalized["chain_scope"] = chain_scope
    validator_hotkey = _text(
        data["validator_hotkey"],
        label="context.validator_hotkey",
        maximum=256,
    )
    if validator_hotkey.strip() != validator_hotkey:
        raise HostAttestationError("context.validator_hotkey is not canonical")
    normalized["validator_hotkey"] = validator_hotkey
    evaluation_id = _text(
        data["evaluation_id"], label="context.evaluation_id", maximum=64
    )
    if _SHA256.fullmatch(evaluation_id) is None:
        raise HostAttestationError("context.evaluation_id must be exact lowercase hex")
    normalized["evaluation_id"] = evaluation_id
    miner_hotkey = _text(
        data["miner_hotkey"], label="context.miner_hotkey", maximum=256
    )
    if miner_hotkey.strip() != miner_hotkey:
        raise HostAttestationError("context.miner_hotkey is not canonical")
    normalized["miner_hotkey"] = miner_hotkey
    settlement_round_id = _integer(
        data["settlement_round_id"],
        label="context.settlement_round_id",
        low=0,
        high=(1 << 63) - 1,
    )
    evaluation_block = _integer(
        data["evaluation_block"],
        label="context.evaluation_block",
        low=0,
        high=(1 << 63) - 1,
    )
    if evaluation_block < normalized["seed_block"]:
        raise HostAttestationError("context evaluation predates its reveal seed")
    normalized["settlement_round_id"] = settlement_round_id
    normalized["evaluation_block"] = evaluation_block
    target = _text(data["target"], label="context.target", maximum=256)
    mode = _text(data["mode"], label="context.mode", maximum=16)
    raw_members = data["member_slots"]
    if (
        not isinstance(raw_members, list)
        or len(raw_members) > 64
        or any(
            not isinstance(member, str)
            or not member.strip()
            or len(member) > 256
            for member in raw_members
        )
        or len(set(raw_members)) != len(raw_members)
    ):
        raise HostAttestationError("context.member_slots is not canonical")
    members = list(raw_members)
    if (
        (mode == "slot" and members != [target])
        or (mode == "atomic" and len(members) < 2)
        or (mode == "system" and members)
        or mode not in {"slot", "atomic", "system"}
    ):
        raise HostAttestationError("context competition projection is inconsistent")
    normalized["target"] = target
    normalized["mode"] = mode
    normalized["member_slots"] = members
    score = _number(data["score"], label="context.score", low=0.0, high=1e12)
    normalized["score"] = score
    decisions: dict[str, bool] = {}
    for name in (
        "passed_quality", "passed_timed_quality", "passed_warmup_quality",
        "passed_speedup", "confident", "crownable",
    ):
        value = data[name]
        if type(value) is not bool:
            raise HostAttestationError(f"context.{name} must be an exact boolean")
        decisions[name] = value
        normalized[name] = value
    if decisions["passed_quality"] != (
        decisions["passed_timed_quality"]
        and decisions["passed_warmup_quality"]
    ):
        raise HostAttestationError("context phase-quality projection is inconsistent")
    expected_crownable = (
        decisions["passed_quality"]
        and decisions["passed_speedup"]
        and decisions["confident"]
    )
    if decisions["crownable"] != expected_crownable:
        raise HostAttestationError("context crown decision is inconsistent")
    if (decisions["crownable"] and score <= 1.0) or (
        not decisions["crownable"] and score != 0.0
    ):
        raise HostAttestationError("context score disagrees with its crown decision")
    normalized["quality_evidence"] = _text(
        data["quality_evidence"], label="context.quality_evidence", maximum=4096
    )
    evidence_hash = _text(
        data["qualification_evidence_sha256"],
        label="context.qualification_evidence_sha256",
        maximum=71,
    )
    if _SHA256_ID.fullmatch(evidence_hash) is None:
        raise HostAttestationError(
            "context.qualification_evidence_sha256 must be a SHA-256 identity"
        )
    normalized["qualification_evidence_sha256"] = evidence_hash
    return normalized


def _normalize_runtime(runtime: Mapping[str, Any]) -> tuple[dict[str, object], str]:
    data = _exact_mapping(runtime, _RUNTIME_KEYS, label="runtime preflight receipt")
    if data["schema"] != RUNTIME_PREFLIGHT_SCHEMA:
        raise HostAttestationError("runtime preflight schema mismatch")
    requested_image = _text(data["requested_image"], label="runtime.requested_image")
    if _IMAGE.fullmatch(requested_image) is None:
        raise HostAttestationError("runtime requested image is not immutable")
    manifest = _text(
        data["requested_manifest_digest"],
        label="runtime.requested_manifest_digest",
        maximum=71,
    )
    image_id = _text(data["local_image_id"], label="runtime.local_image_id", maximum=71)
    if _SHA256_ID.fullmatch(manifest) is None or _SHA256_ID.fullmatch(image_id) is None:
        raise HostAttestationError("runtime image identities must be SHA-256 values")
    if requested_image.rsplit("@", 1)[1] != manifest:
        raise HostAttestationError("runtime manifest digest differs from requested image")
    repo_digests = data["repo_digests"]
    if (
        not isinstance(repo_digests, list)
        or not 1 <= len(repo_digests) <= 64
        or any(not isinstance(item, str) or _IMAGE.fullmatch(item) is None
               for item in repo_digests)
        or repo_digests != sorted(set(repo_digests))
        or requested_image not in repo_digests
    ):
        raise HostAttestationError("runtime repo_digests are not canonical")
    python = _exact_mapping(
        data["python"],
        frozenset({"implementation", "version", "abi", "platform", "machine"}),
        label="runtime.python",
    )
    packages = _exact_mapping(
        data["packages"], _PACKAGE_KEYS, label="runtime.packages"
    )
    cuda = _exact_mapping(
        data["cuda"],
        frozenset({
            "cudart_library", "cuda_visible_devices", "nvidia_visible_devices",
        }),
        label="runtime.cuda",
    )
    normalized_packages: dict[str, str | None] = {}
    for name in sorted(_PACKAGE_KEYS):
        value = packages[name]
        normalized_packages[name] = (
            None if value is None
            else _text(value, label=f"runtime.packages.{name}", maximum=256)
        )
    cudart = cuda["cudart_library"]
    if cudart is not None:
        cudart = _text(cudart, label="runtime.cuda.cudart_library", maximum=256)
    cuda_visible = _text(
        cuda["cuda_visible_devices"],
        label="runtime.cuda.cuda_visible_devices",
        maximum=256,
        allow_empty=True,
    )
    nvidia_visible = _text(
        cuda["nvidia_visible_devices"],
        label="runtime.cuda.nvidia_visible_devices",
        maximum=256,
    )
    if cuda_visible != "" or nvidia_visible != "void":
        raise HostAttestationError("runtime preflight receipt is not from a no-GPU run")
    security_hash = _text(
        data["security_argv_sha256"],
        label="runtime.security_argv_sha256",
        maximum=64,
    )
    if _SHA256.fullmatch(security_hash) is None:
        raise HostAttestationError("runtime security argv hash is invalid")
    docker_binary = _text(
        data["docker_binary"], label="runtime.docker_binary", maximum=4096
    )
    docker_path = Path(docker_binary)
    if (
        not docker_path.is_absolute()
        or docker_path.name != "docker"
        or ".." in docker_path.parts
        or str(docker_path) != docker_binary
    ):
        raise HostAttestationError("runtime docker binary is not a normalized absolute path")
    normalized = {
        "schema": RUNTIME_PREFLIGHT_SCHEMA,
        "requested_image": requested_image,
        "requested_manifest_digest": manifest,
        "local_image_id": image_id,
        "repo_digests": list(repo_digests),
        "docker_binary": docker_binary,
        "uid": _integer(data["uid"], label="runtime.uid", low=1,
                        high=2_147_483_647),
        "gid": _integer(data["gid"], label="runtime.gid", low=1,
                        high=2_147_483_647),
        "sglang_version": _text(
            data["sglang_version"], label="runtime.sglang_version", maximum=128
        ),
        "python": {
            name: _text(
                python[name], label=f"runtime.python.{name}", maximum=256,
                allow_empty=(name == "abi"),
            )
            for name in ("implementation", "version", "abi", "platform", "machine")
        },
        "packages": normalized_packages,
        "cuda": {
            "cudart_library": cudart,
            "cuda_visible_devices": cuda_visible,
            "nvidia_visible_devices": nvidia_visible,
        },
        "security_argv_sha256": security_hash,
    }
    raw = _canonical_json(
        normalized,
        label="runtime preflight receipt",
        maximum=MAX_RUNTIME_PREFLIGHT_BYTES,
    )
    return normalized, "sha256:" + hashlib.sha256(raw).hexdigest()


def _cross_check_context_runtime(
    context: Mapping[str, object], runtime: Mapping[str, object]
) -> None:
    if runtime["requested_image"] != context["validator_image"]:
        raise HostAttestationError(
            "runtime requested image differs from attestation arena context"
        )
    if runtime["sglang_version"] != context["sglang_version"]:
        raise HostAttestationError(
            "runtime sglang version differs from attestation arena context"
        )


def _normalize_qualification_evidence(
    evidence: Mapping[str, Any], context: Mapping[str, object]
) -> dict[str, object]:
    """Reconstitute and independently grade the exact retained result evidence."""

    from optima.eval.qualification import (
        QualificationReport,
        QualificationReportError,
    )

    try:
        report = QualificationReport.from_evidence_dict(
            evidence,
            qualification_evidence_sha256=str(
                context["qualification_evidence_sha256"]
            ),
        )
        normalized = report.evidence_dict()
        report_context = report.attestation_context()
    except QualificationReportError as exc:
        raise HostAttestationError(
            f"qualification evidence is invalid: {exc}"
        ) from None
    if report_context != dict(context):
        raise HostAttestationError(
            "qualification evidence provenance differs from host attestation context"
        )
    return normalized


def _normalize_telemetry(
    raw: object, *, selected: tuple[int, ...], label: str,
) -> tuple[list[dict[str, object]], dict[int, str]]:
    if not isinstance(raw, list) or len(raw) != len(selected):
        raise HostAttestationError(f"{label} must contain one row per selected GPU")
    rows: list[dict[str, object]] = []
    uuid_by_gpu: dict[int, str] = {}
    for index, value in enumerate(raw):
        data = _exact_mapping(value, _TELEMETRY_KEYS, label=f"{label}[{index}]")
        physical_id = _integer(
            data["physical_id"], label=f"{label}[{index}].physical_id",
            low=0, high=65_535,
        )
        uuid = _text(data["uuid"], label=f"{label}[{index}].uuid", maximum=80)
        pstate = _text(data["pstate"], label=f"{label}[{index}].pstate", maximum=4)
        if _UUID.fullmatch(uuid) is None or _PSTATE.fullmatch(pstate) is None:
            raise HostAttestationError(f"{label}[{index}] GPU identity/state is invalid")
        rows.append({
            "physical_id": physical_id,
            "uuid": uuid,
            "pstate": pstate,
            "temperature_c": _integer(
                data["temperature_c"], label=f"{label}[{index}].temperature_c",
                low=-20, high=150,
            ),
            "gpu_utilization_percent": _integer(
                data["gpu_utilization_percent"],
                label=f"{label}[{index}].gpu_utilization_percent", low=0, high=100,
            ),
            "memory_utilization_percent": _integer(
                data["memory_utilization_percent"],
                label=f"{label}[{index}].memory_utilization_percent", low=0, high=100,
            ),
            "current_graphics_clock_mhz": _optional_integer(
                data["current_graphics_clock_mhz"],
                label=f"{label}[{index}].current_graphics_clock_mhz",
                low=1, high=100_000,
            ),
            "current_memory_clock_mhz": _optional_integer(
                data["current_memory_clock_mhz"],
                label=f"{label}[{index}].current_memory_clock_mhz",
                low=1, high=100_000,
            ),
            "power_draw_mw": _optional_integer(
                data["power_draw_mw"], label=f"{label}[{index}].power_draw_mw",
                low=0, high=10_000_000,
            ),
        })
        uuid_by_gpu[physical_id] = uuid
    if tuple(row["physical_id"] for row in rows) != selected:
        raise HostAttestationError(f"{label} physical GPU order/set mismatch")
    if len(set(uuid_by_gpu.values())) != len(selected):
        raise HostAttestationError(f"{label} contains duplicate physical GPU UUIDs")
    return rows, uuid_by_gpu


def _normalize_processes(raw: object, *, selected: tuple[int, ...],
                         label: str) -> list[dict[str, object]]:
    if not isinstance(raw, list) or len(raw) > MAX_PROCESSES_PER_SAMPLE:
        raise HostAttestationError(f"{label} must be a bounded process list")
    result: list[dict[str, object]] = []
    seen: set[tuple[int, int, str]] = set()
    selected_set = set(selected)
    for index, value in enumerate(raw):
        data = _exact_mapping(value, _PROCESS_KEYS, label=f"{label}[{index}]")
        physical_id = _integer(
            data["physical_id"], label=f"{label}[{index}].physical_id",
            low=0, high=65_535,
        )
        pid = _integer(
            data["pid"], label=f"{label}[{index}].pid", low=1,
            high=2_147_483_647,
        )
        kind = _text(data["kind"], label=f"{label}[{index}].kind", maximum=8)
        if physical_id not in selected_set or _PROCESS_KIND.fullmatch(kind) is None:
            raise HostAttestationError(f"{label}[{index}] process identity is invalid")
        key = (physical_id, pid, kind)
        if key in seen:
            raise HostAttestationError(f"{label} contains a duplicate GPU process")
        seen.add(key)
        result.append({
            "physical_id": physical_id,
            "pid": pid,
            "kind": kind,
            "process_name": _text(
                data["process_name"], label=f"{label}[{index}].process_name",
                maximum=256,
            ),
        })
    if result != sorted(
        result, key=lambda item: (item["physical_id"], item["pid"], item["kind"])
    ):
        raise HostAttestationError(f"{label} process list is not canonical")
    return result


def _normalize_device_receipts(
    receipts: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, object]], str, str, tuple[str, ...]]:
    if (
        not isinstance(receipts, (list, tuple))
        or not 9 <= len(receipts) <= MAX_DEVICE_RECEIPTS
        or len(receipts) % 3
    ):
        raise HostAttestationError(
            "device receipts must contain 9..96 adjacent pre/active/post entries"
        )
    normalized: list[dict[str, object]] = []
    previous_sequence = 0
    previous_completed = -math.inf
    selected_identity: tuple[int, ...] | None = None
    configuration_hash: str | None = None
    policy_hash: str | None = None
    uuid_identity: dict[int, str] | None = None

    for receipt_index, value in enumerate(receipts):
        label = f"device_receipts[{receipt_index}]"
        if not isinstance(value, Mapping):
            raise HostAttestationError(f"{label} must be a receipt mapping")
        schema = value.get("schema")
        active_receipt = schema == ACTIVE_DEVICE_RECEIPT_SCHEMA
        data = _exact_mapping(
            value,
            _ACTIVE_DEVICE_KEYS if active_receipt else _DEVICE_KEYS,
            label=label,
        )
        if schema not in {DEVICE_RECEIPT_SCHEMA, ACTIVE_DEVICE_RECEIPT_SCHEMA}:
            raise HostAttestationError(f"{label} schema mismatch")
        sequence = _integer(
            data["sequence"], label=f"{label}.sequence", low=1, high=1_000_000_000
        )
        if sequence <= previous_sequence:
            raise HostAttestationError(
                "device receipt sequences must be strictly increasing and unique"
            )
        arm = _text(data["arm"], label=f"{label}.arm", maximum=128)
        if _ARM.fullmatch(arm) is None:
            raise HostAttestationError(f"{label}.arm is not baseline/candidate/bookend-N")
        if active_receipt:
            event = _text(data["event"], label=f"{label}.event", maximum=128)
            if event != "final-warmup-conditioning":
                raise HostAttestationError(
                    f"{label}.event must be the final-warmup conditioning boundary"
                )
            phase = "active"
        else:
            phase = data["phase"]
            if phase not in {"pre", "post"}:
                raise HostAttestationError(f"{label}.phase must be pre or post")
        selected_raw = data["selected_physical_gpu_ids"]
        if (
            not isinstance(selected_raw, list)
            or not 1 <= len(selected_raw) <= MAX_GPU_COUNT
            or any(type(item) is not int or not 0 <= item <= 65_535
                   for item in selected_raw)
            or selected_raw != sorted(set(selected_raw))
        ):
            raise HostAttestationError(f"{label} selected GPU set is invalid")
        selected = tuple(selected_raw)
        config_hash = _text(
            data["configuration_sha256"], label=f"{label}.configuration_sha256",
            maximum=64,
        )
        this_policy_hash = _text(
            data["policy_sha256"], label=f"{label}.policy_sha256", maximum=64
        )
        if _SHA256.fullmatch(config_hash) is None or _SHA256.fullmatch(
            this_policy_hash
        ) is None:
            raise HostAttestationError(f"{label} device hashes are invalid")
        started = _number(
            data["started_monotonic_s"], label=f"{label}.started_monotonic_s"
        )
        completed = _number(
            data["completed_monotonic_s"], label=f"{label}.completed_monotonic_s"
        )
        if not previous_completed < started < completed:
            raise HostAttestationError(
                "device receipt timestamps must be globally strictly increasing"
            )
        consecutive_field = (
            "consecutive_active_samples"
            if active_receipt else "consecutive_idle_samples"
        )
        consecutive = _integer(
            data[consecutive_field],
            label=f"{label}.{consecutive_field}", low=2, high=32,
        )
        release_sample_index = None
        post_release_ready_samples = None
        if active_receipt:
            release_sample_index = _integer(
                data["release_sample_index"],
                label=f"{label}.release_sample_index", low=2,
                high=MAX_DEVICE_SAMPLES - 1,
            )
            post_release_ready_samples = _integer(
                data["post_release_ready_samples"],
                label=f"{label}.post_release_ready_samples", low=1, high=1,
            )
        samples_raw = data["samples"]
        if (
            not isinstance(samples_raw, list)
            or not consecutive <= len(samples_raw) <= MAX_DEVICE_SAMPLES
        ):
            raise HostAttestationError(f"{label}.samples has an invalid bound")
        samples: list[dict[str, object]] = []
        prior_sample_time = started
        for sample_index, sample_value in enumerate(samples_raw):
            sample_label = f"{label}.samples[{sample_index}]"
            sample_data = _exact_mapping(
                sample_value, _SAMPLE_KEYS, label=sample_label
            )
            sampled = _number(
                sample_data["monotonic_s"], label=f"{sample_label}.monotonic_s"
            )
            if not prior_sample_time < sampled < completed:
                raise HostAttestationError(
                    f"{sample_label} timestamps must increase within the receipt"
                )
            prior_sample_time = sampled
            telemetry, sample_uuids = _normalize_telemetry(
                sample_data["telemetry"], selected=selected,
                label=f"{sample_label}.telemetry",
            )
            processes = _normalize_processes(
                sample_data["processes"], selected=selected,
                label=f"{sample_label}.processes",
            )
            idle = sample_data["idle"]
            active_passed = sample_data["active_envelope_passed"]
            if type(idle) is not bool or type(active_passed) is not bool:
                raise HostAttestationError(f"{sample_label} verdicts must be booleans")
            if idle and processes:
                raise HostAttestationError(f"{sample_label} cannot be idle with processes")
            if uuid_identity is None:
                uuid_identity = sample_uuids
            elif sample_uuids != uuid_identity:
                raise HostAttestationError("physical GPU UUID identity changed across receipts")
            samples.append({
                "monotonic_s": sampled,
                "telemetry": telemetry,
                "processes": processes,
                "idle": idle,
                "idle_reason": _text(
                    sample_data["idle_reason"], label=f"{sample_label}.idle_reason",
                    maximum=4096,
                ),
                "active_envelope_passed": active_passed,
                "active_envelope_reason": _text(
                    sample_data["active_envelope_reason"],
                    label=f"{sample_label}.active_envelope_reason", maximum=4096,
                ),
            })
        if active_receipt:
            assert release_sample_index is not None
            assert post_release_ready_samples is not None
            if (
                release_sample_index < consecutive
                or release_sample_index >= len(samples)
                or len(samples) - release_sample_index
                < post_release_ready_samples
            ):
                raise HostAttestationError(
                    f"{label} has an invalid host-release sample boundary"
                )
            pre_release_run = samples[
                release_sample_index - consecutive:release_sample_index
            ]
            post_release_run = samples[-post_release_ready_samples:]
            if not all(
                sample["active_envelope_passed"]
                for sample in pre_release_run
            ):
                raise HostAttestationError(
                    f"{label} lacks its claimed pre-release active-envelope run"
                )
            if not all(
                sample["active_envelope_passed"] for sample in post_release_run
            ):
                raise HostAttestationError(
                    f"{label} lacks its claimed post-release ready sample"
                )
            selected_set = set(selected)
            for sample in (*pre_release_run, *post_release_run):
                process_gpus = {
                    process["physical_id"] for process in sample["processes"]
                }
                if sample["idle"] or process_gpus != selected_set:
                    raise HostAttestationError(
                        f"{label} active/ready evidence lacks a process on every selected GPU"
                    )
        elif not all(sample["idle"] for sample in samples[-consecutive:]):
            raise HostAttestationError(f"{label} does not end in its claimed idle run")
        if selected_identity is None:
            selected_identity = selected
            configuration_hash = config_hash
            policy_hash = this_policy_hash
        elif (
            selected != selected_identity
            or config_hash != configuration_hash
            or this_policy_hash != policy_hash
        ):
            raise HostAttestationError(
                "device GPU set/configuration/policy changed across the bracket"
            )
        normalized_receipt = {
            "schema": schema,
            "sequence": sequence,
            "arm": arm,
            "selected_physical_gpu_ids": list(selected),
            "configuration_sha256": config_hash,
            "policy_sha256": this_policy_hash,
            "started_monotonic_s": started,
            "completed_monotonic_s": completed,
            "samples": samples,
        }
        if active_receipt:
            normalized_receipt.update(
                event="final-warmup-conditioning",
                consecutive_active_samples=consecutive,
                release_sample_index=release_sample_index,
                post_release_ready_samples=post_release_ready_samples,
            )
        else:
            normalized_receipt.update(
                phase=phase,
                consecutive_idle_samples=consecutive,
            )
        normalized.append(normalized_receipt)
        previous_sequence = sequence
        previous_completed = completed

    arms: list[str] = []
    stages: list[int] = []
    ordinals: dict[str, list[int]] = {
        "baseline": [], "candidate": [], "bookend": [],
    }
    seen_arms: set[str] = set()
    stage_index = {"baseline": 0, "candidate": 1, "bookend": 2}
    for index in range(0, len(normalized), 3):
        before, active, after = normalized[index:index + 3]
        if (
            before.get("phase") != "pre"
            or active.get("schema") != ACTIVE_DEVICE_RECEIPT_SCHEMA
            or after.get("phase") != "post"
            or active.get("arm") != before.get("arm")
            or before.get("arm") != after.get("arm")
        ):
            raise HostAttestationError(
                "device receipts must be adjacent same-arm pre/active/post triplets"
            )
        arm = str(before["arm"])
        if arm in seen_arms:
            raise HostAttestationError("device receipt arm labels must be unique")
        seen_arms.add(arm)
        arms.append(arm)
        match = _ARM.fullmatch(arm)
        assert match is not None
        stage = match.group(1)
        stages.append(stage_index[stage])
        ordinals[stage].append(int(match.group(2)))
    if stages != sorted(stages) or set(stages) != {0, 1, 2}:
        raise HostAttestationError(
            "device bracket requires ordered complete baseline, candidate, and bookend triplets"
        )
    for stage, observed in ordinals.items():
        if observed != sorted(observed) or len(set(observed)) != len(observed):
            raise HostAttestationError(
                f"device {stage} arm ordinals must be strictly increasing and unique"
            )
    assert configuration_hash is not None and policy_hash is not None
    return normalized, configuration_hash, policy_hash, tuple(arms)


def _validate_directory(fd: int, *, label: str, private: bool) -> os.stat_result:
    try:
        info = os.fstat(fd)
    except OSError as exc:
        raise HostAttestationError(f"cannot inspect {label}: {exc}") from None
    mode = stat.S_IMODE(info.st_mode)
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid():
        raise HostAttestationError(f"{label} must be a controller-owned directory")
    if mode & 0o022 or (private and mode != 0o700):
        raise HostAttestationError(f"{label} permissions are not private/read-only-safe")
    return info


def _open_publication_root(value: str | os.PathLike[str]) -> tuple[Path, int, os.stat_result]:
    raw = Path(value)
    if (
        not raw.is_absolute()
        or ".." in raw.parts
        or any(char in str(raw) for char in "\x00\r\n")
    ):
        raise HostAttestationError("publication root must be an absolute normalized path")
    try:
        path_info = raw.lstat()
    except OSError as exc:
        raise HostAttestationError(f"publication root is unavailable: {exc}") from None
    if stat.S_ISLNK(path_info.st_mode) or not stat.S_ISDIR(path_info.st_mode):
        raise HostAttestationError("publication root may not be a symlink/non-directory")
    resolved = Path(os.path.realpath(raw))
    if "published" in resolved.parts:
        raise HostAttestationError(
            "host attestation sidecar may not live inside candidate-mounted published/"
        )
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(raw, flags)
    except OSError as exc:
        raise HostAttestationError(f"cannot open publication root safely: {exc}") from None
    try:
        info = _validate_directory(fd, label="publication root", private=False)
        stable = ("st_dev", "st_ino", "st_mode", "st_uid", "st_gid")
        if any(getattr(path_info, field) != getattr(info, field) for field in stable):
            raise HostAttestationError("publication root changed while opening")
        return resolved, fd, info
    except BaseException:
        os.close(fd)
        raise


def _open_sidecar_directory(root_fd: int, *, create: bool = True) -> tuple[int, bool]:
    created = False
    if create:
        try:
            os.mkdir(HOST_ATTESTATION_DIRECTORY, 0o700, dir_fd=root_fd)
            created = True
        except FileExistsError:
            pass
        except OSError as exc:
            raise HostAttestationError(
                f"cannot create host attestation directory: {exc}"
            ) from None
    if created:
        try:
            os.fsync(root_fd)
        except OSError as exc:
            raise HostAttestationError(
                f"cannot fsync host attestation parent directory: {exc}"
            ) from None
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(HOST_ATTESTATION_DIRECTORY, flags, dir_fd=root_fd)
    except OSError as exc:
        raise HostAttestationError(f"cannot open host attestation directory: {exc}") from None
    try:
        _validate_directory(fd, label="host attestation directory", private=True)
    except BaseException:
        os.close(fd)
        raise
    return fd, created


def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        try:
            written = os.write(fd, view)
        except OSError as exc:
            raise HostAttestationError(f"host attestation write failed: {exc}") from None
        if written <= 0:
            raise HostAttestationError("host attestation write made no progress")
        view = view[written:]


def _read_published(sidecar_fd: int, filename: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(filename, flags, dir_fd=sidecar_fd)
    except OSError as exc:
        raise HostAttestationError(
            f"existing host attestation is missing/unsafe: {exc}"
        ) from None
    try:
        before = os.fstat(fd)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_uid != os.geteuid()
            or stat.S_IMODE(before.st_mode) != 0o444
            or before.st_size <= 0
            or before.st_size > MAX_HOST_ATTESTATION_BYTES
        ):
            raise HostAttestationError("existing host attestation file shape is unsafe")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            try:
                chunk = os.read(fd, min(1024 * 1024, remaining))
            except OSError as exc:
                raise HostAttestationError(
                    f"existing host attestation read failed: {exc}"
                ) from None
            if not chunk:
                raise HostAttestationError("existing host attestation was truncated")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(fd, 1):
            raise HostAttestationError("existing host attestation grew while reading")
        after = os.fstat(fd)
        stable = (
            "st_dev", "st_ino", "st_mode", "st_nlink", "st_uid", "st_gid",
            "st_size", "st_mtime_ns", "st_ctime_ns",
        )
        if any(getattr(before, field) != getattr(after, field) for field in stable):
            raise HostAttestationError("existing host attestation changed while reading")
        try:
            named = os.stat(filename, dir_fd=sidecar_fd, follow_symlinks=False)
        except OSError as exc:
            raise HostAttestationError(
                f"existing host attestation name changed while reading: {exc}"
            ) from None
        if (
            not stat.S_ISREG(named.st_mode)
            or (named.st_dev, named.st_ino) != (after.st_dev, after.st_ino)
            or named.st_nlink != 1
            or named.st_uid != os.geteuid()
            or stat.S_IMODE(named.st_mode) != 0o444
            or named.st_size != after.st_size
        ):
            raise HostAttestationError(
                "existing host attestation name was replaced while reading"
            )
        return b"".join(chunks)
    finally:
        os.close(fd)


def _read_existing(sidecar_fd: int, filename: str, expected: bytes) -> None:
    if _read_published(sidecar_fd, filename) != expected:
        raise HostAttestationError(
            "existing content-addressed host attestation bytes are corrupt"
        )


def _publish_bytes(sidecar_fd: int, filename: str, data: bytes) -> None:
    temporary = ".tmp-" + secrets.token_hex(16)
    flags = (
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    fd = -1
    temporary_exists = False
    try:
        try:
            fd = os.open(temporary, flags, 0o600, dir_fd=sidecar_fd)
            temporary_exists = True
        except OSError as exc:
            raise HostAttestationError(
                f"cannot create exclusive host attestation stage: {exc}"
            ) from None
        _write_all(fd, data)
        try:
            os.fchmod(fd, 0o444)
            os.fsync(fd)
            info = os.fstat(fd)
        except OSError as exc:
            raise HostAttestationError(f"cannot freeze/fsync host attestation: {exc}") from None
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
            or info.st_uid != os.geteuid()
            or stat.S_IMODE(info.st_mode) != 0o444
            or info.st_size != len(data)
        ):
            raise HostAttestationError("staged host attestation file shape is unsafe")
        try:
            staged_name = os.stat(
                temporary, dir_fd=sidecar_fd, follow_symlinks=False
            )
        except OSError as exc:
            raise HostAttestationError(
                f"staged host attestation name changed before publication: {exc}"
            ) from None
        if (
            not stat.S_ISREG(staged_name.st_mode)
            or (staged_name.st_dev, staged_name.st_ino) != (info.st_dev, info.st_ino)
        ):
            raise HostAttestationError(
                "staged host attestation name was replaced before publication"
            )
        linked = False
        try:
            # linkat is an atomic no-replace publication: unlike rename(), it can
            # never overwrite a same-digest path created by another controller.
            os.link(
                temporary,
                filename,
                src_dir_fd=sidecar_fd,
                dst_dir_fd=sidecar_fd,
                follow_symlinks=False,
            )
            linked = True
        except FileExistsError:
            pass
        except OSError as exc:
            raise HostAttestationError(
                f"atomic host attestation publication failed: {exc}"
            ) from None
        if linked:
            try:
                linked_fd = os.fstat(fd)
                linked_name = os.stat(
                    filename, dir_fd=sidecar_fd, follow_symlinks=False
                )
            except OSError as exc:
                raise HostAttestationError(
                    f"published host attestation binding cannot be verified: {exc}"
                ) from None
            if (
                linked_fd.st_nlink != 2
                or not stat.S_ISREG(linked_name.st_mode)
                or (linked_name.st_dev, linked_name.st_ino)
                    != (linked_fd.st_dev, linked_fd.st_ino)
            ):
                raise HostAttestationError(
                    "published host attestation link does not bind the staged inode"
                )
        try:
            os.unlink(temporary, dir_fd=sidecar_fd)
            temporary_exists = False
            os.fsync(sidecar_fd)
        except OSError as exc:
            raise HostAttestationError(
                f"cannot finalize/fsync host attestation publication: {exc}"
            ) from None
        if linked and os.fstat(fd).st_nlink != 1:
            raise HostAttestationError(
                "published host attestation retained an unexpected hardlink"
            )
        os.close(fd)
        fd = -1
        _read_existing(sidecar_fd, filename, data)
    finally:
        if fd >= 0:
            os.close(fd)
        if temporary_exists:
            try:
                os.unlink(temporary, dir_fd=sidecar_fd)
                os.fsync(sidecar_fd)
            except OSError:
                pass


def _strict_json_bytes(raw: bytes, *, label: str) -> object:
    if not isinstance(raw, bytes) or not raw or len(raw) > MAX_HOST_ATTESTATION_BYTES:
        raise HostAttestationError(f"{label} bytes are empty or oversized")
    try:
        text = raw.decode("utf-8", errors="strict")

        def object_pairs(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"duplicate JSON key {key!r}")
                result[key] = value
            return result

        return json.loads(
            text,
            object_pairs_hook=object_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"invalid JSON constant {value}")
            ),
        )
    except (
        UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError,
    ) as exc:
        raise HostAttestationError(f"{label} is malformed JSON: {exc}") from None


def _verify_sidecar_binding(root_fd: int, sidecar_fd: int) -> None:
    opened = _validate_directory(
        sidecar_fd, label="host attestation directory", private=True
    )
    try:
        named = os.stat(
            HOST_ATTESTATION_DIRECTORY,
            dir_fd=root_fd,
            follow_symlinks=False,
        )
    except OSError as exc:
        raise HostAttestationError(
            f"host attestation directory changed during publication: {exc}"
        ) from None
    if (
        not stat.S_ISDIR(named.st_mode)
        or (named.st_dev, named.st_ino) != (opened.st_dev, opened.st_ino)
    ):
        raise HostAttestationError(
            "host attestation directory was replaced during publication"
        )


def _verify_root_binding(root: Path, expected: os.stat_result) -> None:
    try:
        observed = os.stat(root, follow_symlinks=False)
    except OSError as exc:
        raise HostAttestationError(
            f"publication root changed during host attestation access: {exc}"
        ) from None
    if (
        not stat.S_ISDIR(observed.st_mode)
        or (observed.st_dev, observed.st_ino) != (expected.st_dev, expected.st_ino)
    ):
        raise HostAttestationError("publication root was replaced during access")


def _make_reference(
    *,
    root: Path,
    digest: str,
    receipt_count: int,
    runtime_hash: str,
    qualification_hash: str,
    config_hash: str,
    policy_hash: str,
    arms: tuple[str, ...],
) -> HostAttestationReference:
    filename = f"sha256-{digest}.json"
    return HostAttestationReference(
        sha256="sha256:" + digest,
        path=str(root / HOST_ATTESTATION_DIRECTORY / filename),
        receipt_count=receipt_count,
        runtime_preflight_sha256=runtime_hash,
        qualification_evidence_sha256=qualification_hash,
        device_configuration_sha256=config_hash,
        device_policy_sha256=policy_hash,
        arms=arms,
    )


def publish_host_attestation(
    publication_root: str | os.PathLike[str],
    *,
    context: Mapping[str, Any],
    runtime_preflight: Mapping[str, Any],
    device_receipts: Sequence[Mapping[str, Any]],
    qualification_evidence: Mapping[str, Any],
) -> HostAttestationReference:
    """Validate and immutably publish one authoritative host-side evidence object."""

    normalized_context = _normalize_context(context)
    normalized_runtime, runtime_hash = _normalize_runtime(runtime_preflight)
    _cross_check_context_runtime(normalized_context, normalized_runtime)
    normalized_qualification = _normalize_qualification_evidence(
        qualification_evidence, normalized_context
    )
    normalized_devices, config_hash, policy_hash, arms = _normalize_device_receipts(
        device_receipts
    )
    payload = {
        "schema": HOST_ATTESTATION_SCHEMA,
        "context": normalized_context,
        "runtime_preflight": normalized_runtime,
        "device_receipts": normalized_devices,
        "qualification_evidence": normalized_qualification,
    }
    raw = _canonical_json(
        payload,
        label="host attestation",
        maximum=MAX_HOST_ATTESTATION_BYTES,
        trailing_newline=True,
    )
    digest = hashlib.sha256(raw).hexdigest()
    filename = f"sha256-{digest}.json"
    root, root_fd, root_info = _open_publication_root(publication_root)
    sidecar_fd = -1
    try:
        sidecar_fd, _ = _open_sidecar_directory(root_fd)
        _publish_bytes(sidecar_fd, filename, raw)
        _verify_sidecar_binding(root_fd, sidecar_fd)
        _verify_root_binding(root, root_info)
    finally:
        if sidecar_fd >= 0:
            os.close(sidecar_fd)
        os.close(root_fd)
    path = root / HOST_ATTESTATION_DIRECTORY / filename
    published_subtree = root / "published"
    if path == published_subtree or published_subtree in path.parents:
        raise HostAttestationError("host attestation path entered candidate publication")
    return _make_reference(
        root=root,
        digest=digest,
        receipt_count=len(normalized_devices),
        runtime_hash=runtime_hash,
        qualification_hash=str(
            normalized_context["qualification_evidence_sha256"]
        ),
        config_hash=config_hash,
        policy_hash=policy_hash,
        arms=arms,
    )


def verify_host_attestation(
    publication_root: str | os.PathLike[str],
    reference: str | HostAttestationReference,
    *,
    expected_context: Mapping[str, Any],
) -> HostAttestationReference:
    """Re-open and fully verify retained evidence before settlement/emission."""

    expected = _normalize_context(expected_context)
    if type(reference) is HostAttestationReference:
        expected_digest = reference.sha256
    elif isinstance(reference, str):
        expected_digest = reference
    else:
        raise HostAttestationError(
            "host attestation reference must be a digest or frozen reference"
        )
    if _SHA256_ID.fullmatch(expected_digest) is None:
        raise HostAttestationError("host attestation reference digest is invalid")
    digest = expected_digest.removeprefix("sha256:")
    filename = f"sha256-{digest}.json"
    root, root_fd, root_info = _open_publication_root(publication_root)
    sidecar_fd = -1
    try:
        sidecar_fd, _ = _open_sidecar_directory(root_fd, create=False)
        raw = _read_published(sidecar_fd, filename)
        if hashlib.sha256(raw).hexdigest() != digest:
            raise HostAttestationError("retained host attestation content hash mismatch")
        payload = _exact_mapping(
            _strict_json_bytes(raw, label="retained host attestation"),
            frozenset({
                "schema", "context", "runtime_preflight", "device_receipts",
                "qualification_evidence",
            }),
            label="retained host attestation",
        )
        if payload["schema"] != HOST_ATTESTATION_SCHEMA:
            raise HostAttestationError("retained host attestation schema mismatch")
        context = _normalize_context(payload["context"])
        runtime, runtime_hash = _normalize_runtime(payload["runtime_preflight"])
        _cross_check_context_runtime(context, runtime)
        qualification = _normalize_qualification_evidence(
            payload["qualification_evidence"], context
        )
        devices, config_hash, policy_hash, arms = _normalize_device_receipts(
            payload["device_receipts"]
        )
        if context != expected:
            raise HostAttestationError(
                "retained host attestation context differs from settlement context"
            )
        canonical = _canonical_json(
            {
                "schema": HOST_ATTESTATION_SCHEMA,
                "context": context,
                "runtime_preflight": runtime,
                "device_receipts": devices,
                "qualification_evidence": qualification,
            },
            label="retained host attestation",
            maximum=MAX_HOST_ATTESTATION_BYTES,
            trailing_newline=True,
        )
        if canonical != raw:
            raise HostAttestationError(
                "retained host attestation is not byte-canonical"
            )
        _verify_sidecar_binding(root_fd, sidecar_fd)
        _verify_root_binding(root, root_info)
    finally:
        if sidecar_fd >= 0:
            os.close(sidecar_fd)
        os.close(root_fd)
    observed = _make_reference(
        root=root,
        digest=digest,
        receipt_count=len(devices),
        runtime_hash=runtime_hash,
        qualification_hash=str(context["qualification_evidence_sha256"]),
        config_hash=config_hash,
        policy_hash=policy_hash,
        arms=arms,
    )
    if type(reference) is HostAttestationReference and reference != observed:
        raise HostAttestationError(
            "frozen host attestation reference metadata/path differs from retained evidence"
        )
    return observed


__all__ = [
    "ACTIVE_DEVICE_RECEIPT_SCHEMA",
    "DEVICE_RECEIPT_SCHEMA",
    "HOST_ATTESTATION_SCHEMA",
    "HostAttestationError",
    "HostAttestationReference",
    "host_attestation_context",
    "publish_host_attestation",
    "verify_host_attestation",
]
