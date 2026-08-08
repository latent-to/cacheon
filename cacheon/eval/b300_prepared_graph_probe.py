"""Prepared-runtime execution of commissioned B300 focused-graph probes.

The controller derives a closed, path-free request from an already prepared
qualification arm.  An isolated worker reopens that exact materialized tree,
uses only validator-owned slot routing, and returns the existing canonical graph
artifact.  Candidate diagnostic strings are deliberately not an evidence API.
"""

from __future__ import annotations

import json
import math
import re
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from cacheon.arena_service import ArenaCandidateBinding
from cacheon.engine_tree import EngineTreeError, reopen_materialized_engine_tree
from cacheon.eval.b300_qualification_graph_provider import (
    B300QualificationGraphArtifact,
    B300QualificationGraphBinding,
    B300QualificationGraphProviderError,
    _tree_identity_digest,
)
from cacheon.eval.b300_qualification_capabilities import (
    B300QualificationCapabilityError,
    StructuredGraphShapeRecord,
    StructuredGraphVariantRecord,
)
from cacheon.eval.engine_launch import EngineLaunchError, EngineLaunchSpec
from cacheon.eval.marginal_runtime import PreparedCandidateRuntime
from cacheon.manifest import Manifest, ManifestError, OpEntry, load_manifest
from cacheon.registry import Eligibility, eligibility_from_metadata
from cacheon.slots import SlotSpec, slot_for_model
from cacheon.stack_identity import canonical_digest, canonical_json_bytes
from cacheon.verification_outcomes import (
    GraphPhaseOutcome,
    PhaseDisposition,
    ShapeResult,
    VerificationCaseDescriptor,
    VerificationCaseKind,
    VerifyResult,
)
from cacheon.verify import verify_entry_from_source
from cacheon.verify_collective import verify_collective


REQUEST_SCHEMA = "cacheon.eval.b300-prepared-graph-probe-request.v1"
POLICY_SCHEMA = "cacheon.eval.b300-prepared-graph-probe-policy.v1"
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_ARCHITECTURE = re.compile(r"sm[0-9]{2,3}[a-z]?\Z")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,255}\Z")


class PreparedGraphProbeError(RuntimeError):
    """The prepared request or materialized worker authority is inconsistent."""


class PreparedGraphProbeIncompleteError(PreparedGraphProbeError):
    """A probe cannot publish candidate evidence and must produce HOLD."""

    decision = "HOLD"


def _require_digest(value: object, field: str) -> str:
    if type(value) is not str or _DIGEST.fullmatch(value) is None:
        raise PreparedGraphProbeError(f"{field} must be one lowercase SHA-256 digest")
    return value


def _require_identifier(value: object, field: str) -> str:
    if type(value) is not str or _IDENTIFIER.fullmatch(value) is None:
        raise PreparedGraphProbeError(f"{field} must be one canonical identifier")
    return value


def _require_positive(value: object, field: str, *, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise PreparedGraphProbeError(f"{field} must be an integer >= {minimum}")
    return value


def _strict_object(value: object, fields: frozenset[str], field: str) -> dict[str, object]:
    if type(value) is not dict or set(value) != fields:
        raise PreparedGraphProbeError(f"{field} fields do not match the closed schema")
    return value


def _canonical_object(payload: bytes) -> dict[str, object]:
    def reject_number(value: str) -> object:
        raise PreparedGraphProbeError(f"probe request contains unsupported number {value!r}")

    def unique(rows: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in rows:
            if key in result:
                raise PreparedGraphProbeError(f"probe request repeats key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            payload.decode("utf-8"),
            parse_constant=reject_number,
            object_pairs_hook=unique,
        )
    except PreparedGraphProbeError:
        raise
    except (UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise PreparedGraphProbeError(f"probe request is malformed: {exc}") from None
    if type(value) is not dict or canonical_json_bytes(value) != payload:
        raise PreparedGraphProbeError("probe request is not exact canonical JSON")
    return value


@dataclass(frozen=True)
class PreparedGraphProbePolicy:
    """Closed validator policy for one B300 graph-only worker execution."""

    verification_policy_digest: str
    expected_graph_replays: int
    dtype_name: str
    architecture: str
    tp_size: int
    world_size: int
    graph_mode: str
    model_profile_key: str
    seed: int
    jitter_seed: int
    collective_timeout_s: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "verification_policy_digest",
            _require_digest(self.verification_policy_digest, "verification policy digest"),
        )
        object.__setattr__(
            self,
            "expected_graph_replays",
            _require_positive(self.expected_graph_replays, "expected graph replays", minimum=2),
        )
        object.__setattr__(self, "dtype_name", _require_identifier(self.dtype_name, "dtype"))
        if type(self.architecture) is not str or _ARCHITECTURE.fullmatch(self.architecture) is None:
            raise PreparedGraphProbeError("architecture must be a canonical CUDA architecture")
        object.__setattr__(self, "tp_size", _require_positive(self.tp_size, "tp_size"))
        object.__setattr__(self, "world_size", _require_positive(self.world_size, "world_size"))
        if self.world_size != self.tp_size:
            raise PreparedGraphProbeError("collective world_size must equal tp_size")
        if self.graph_mode != "cuda_graph":
            raise PreparedGraphProbeError("prepared graph probe mode must be cuda_graph")
        object.__setattr__(
            self,
            "model_profile_key",
            _require_identifier(self.model_profile_key, "model profile key"),
        )
        for field in ("seed", "jitter_seed"):
            value = getattr(self, field)
            if type(value) is not int or value < 0:
                raise PreparedGraphProbeError(f"{field} must be a nonnegative integer")
        object.__setattr__(
            self,
            "collective_timeout_s",
            _require_positive(self.collective_timeout_s, "collective timeout"),
        )
        try:
            finite_timeout = math.isfinite(float(self.collective_timeout_s))
        except OverflowError:
            finite_timeout = False
        if not finite_timeout:
            raise PreparedGraphProbeError("collective timeout must be finite")

    def to_dict(self) -> dict[str, object]:
        return {
            "architecture": self.architecture,
            "collective_timeout_s": self.collective_timeout_s,
            "dtype_name": self.dtype_name,
            "expected_graph_replays": self.expected_graph_replays,
            "graph_mode": self.graph_mode,
            "jitter_seed": self.jitter_seed,
            "model_profile_key": self.model_profile_key,
            "schema": POLICY_SCHEMA,
            "seed": self.seed,
            "tp_size": self.tp_size,
            "verification_policy_digest": self.verification_policy_digest,
            "world_size": self.world_size,
        }

    @classmethod
    def from_dict(cls, value: object) -> "PreparedGraphProbePolicy":
        row = _strict_object(
            value,
            frozenset(
                {
                    "architecture",
                    "collective_timeout_s",
                    "dtype_name",
                    "expected_graph_replays",
                    "graph_mode",
                    "jitter_seed",
                    "model_profile_key",
                    "schema",
                    "seed",
                    "tp_size",
                    "verification_policy_digest",
                    "world_size",
                }
            ),
            "probe policy",
        )
        if row["schema"] != POLICY_SCHEMA:
            raise PreparedGraphProbeError("probe policy schema is unsupported")
        return cls(**{key: value for key, value in row.items() if key != "schema"})  # type: ignore[arg-type]

    @property
    def digest(self) -> str:
        return canonical_digest(POLICY_SCHEMA, self.to_dict())


@dataclass(frozen=True)
class PreparedGraphVariantAuthority:
    """Canonical member/variant authority selected from the prepared manifest."""

    slot_id: str
    variant_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "slot_id", _require_identifier(self.slot_id, "slot ID"))
        object.__setattr__(self, "variant_id", _require_identifier(self.variant_id, "variant ID"))

    def to_dict(self) -> dict[str, str]:
        return {"slot_id": self.slot_id, "variant_id": self.variant_id}

    @classmethod
    def from_dict(cls, value: object) -> "PreparedGraphVariantAuthority":
        row = _strict_object(value, frozenset({"slot_id", "variant_id"}), "variant authority")
        return cls(row["slot_id"], row["variant_id"])  # type: ignore[arg-type]


@dataclass(frozen=True)
class PreparedGraphProbeRequest:
    """Path-free commissioned input for an isolated prepared-runtime worker."""

    binding: B300QualificationGraphBinding
    launch: EngineLaunchSpec
    policy: PreparedGraphProbePolicy
    target_variants: tuple[PreparedGraphVariantAuthority, ...]

    def __post_init__(self) -> None:
        if type(self.binding) is not B300QualificationGraphBinding:
            raise PreparedGraphProbeError("request binding is not exactly typed")
        if type(self.launch) is not EngineLaunchSpec:
            raise PreparedGraphProbeError("request launch is not exactly typed")
        if type(self.policy) is not PreparedGraphProbePolicy:
            raise PreparedGraphProbeError("request policy is not exactly typed")
        variants = self.target_variants
        keys = tuple((row.slot_id, row.variant_id) for row in variants) if type(variants) is tuple else ()
        if (
            not variants
            or any(type(row) is not PreparedGraphVariantAuthority for row in variants)
            or keys != tuple(sorted(set(keys)))
        ):
            raise PreparedGraphProbeError("target member/variant authority is not canonical")
        if tuple(sorted({row.slot_id for row in variants})) != self.binding.target_members:
            raise PreparedGraphProbeError("target variant authority differs from exact target members")
        if (
            self.launch.digest != self.binding.prepared_launch_digest
            or self.launch.stack_digest != self.binding.materialized_stack_digest
            or self.launch.tree_digest != self.binding.materialized_tree_digest
            or self.launch.native_build_spec_digest != self.binding.native_build_spec_digest
        ):
            raise PreparedGraphProbeError("request launch differs from the prepared graph binding")
        if (
            self.launch.hardware.architecture != self.policy.architecture
            or self.launch.hardware.tp_size != self.policy.tp_size
            or self.policy.world_size != self.policy.tp_size
        ):
            raise PreparedGraphProbeError("probe execution topology differs from prepared launch hardware")

    @classmethod
    def derive(
        cls,
        binding: B300QualificationGraphBinding,
        candidate: ArenaCandidateBinding,
        prepared: PreparedCandidateRuntime,
        policy: PreparedGraphProbePolicy,
    ) -> "PreparedGraphProbeRequest":
        if type(binding) is not B300QualificationGraphBinding:
            raise PreparedGraphProbeError("commissioned binding is not exactly typed")
        if type(candidate) is not ArenaCandidateBinding or type(prepared) is not PreparedCandidateRuntime:
            raise PreparedGraphProbeError("candidate/prepared request inputs are not exactly typed")
        try:
            derived = B300QualificationGraphBinding.derive(candidate, prepared)
        except B300QualificationGraphProviderError as exc:
            raise PreparedGraphProbeError(f"cannot derive prepared graph binding: {exc}") from None
        if binding != derived:
            raise PreparedGraphProbeError("commissioned binding differs from candidate/prepared derivation")
        tree = prepared.binding.tree
        try:
            reopened = reopen_materialized_engine_tree(
                tree.root, expected_tree_digest=tree.tree_digest
            )
            if reopened != tree or _tree_identity_digest(reopened) != binding.trusted_tree_identity_digest:
                raise PreparedGraphProbeError("prepared materialized tree identity drifted")
            manifest = load_manifest(reopened.root)
        except PreparedGraphProbeError:
            raise
        except (EngineTreeError, ManifestError, OSError, ValueError) as exc:
            raise PreparedGraphProbeError(f"cannot reopen prepared manifest: {exc}") from None
        variants = _target_variant_authority(binding, manifest)
        return cls(binding, prepared.launch, policy, variants)

    def to_dict(self) -> dict[str, object]:
        return {
            "binding": self.binding.to_dict(),
            "binding_digest": self.binding.digest,
            "launch": self.launch.to_dict(),
            "launch_digest": self.launch.digest,
            "policy": self.policy.to_dict(),
            "policy_digest": self.policy.digest,
            "schema": REQUEST_SCHEMA,
            "target_variants": [row.to_dict() for row in self.target_variants],
        }

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    @property
    def digest(self) -> str:
        return canonical_digest(REQUEST_SCHEMA, self.to_dict())

    @classmethod
    def from_canonical_bytes(cls, payload: bytes) -> "PreparedGraphProbeRequest":
        if type(payload) is not bytes or not payload:
            raise PreparedGraphProbeError("probe request bytes must be exact and nonempty")
        row = _strict_object(
            _canonical_object(payload),
            frozenset(
                {
                    "binding",
                    "binding_digest",
                    "launch",
                    "launch_digest",
                    "policy",
                    "policy_digest",
                    "schema",
                    "target_variants",
                }
            ),
            "probe request",
        )
        if row["schema"] != REQUEST_SCHEMA or type(row["target_variants"]) is not list:
            raise PreparedGraphProbeError("probe request schema or target variants are invalid")
        try:
            binding = B300QualificationGraphBinding.from_dict(row["binding"])
            launch = EngineLaunchSpec.from_dict(row["launch"])
            policy = PreparedGraphProbePolicy.from_dict(row["policy"])
        except (B300QualificationGraphProviderError, EngineLaunchError, TypeError, ValueError) as exc:
            raise PreparedGraphProbeError(f"probe request authority is invalid: {exc}") from None
        if row["binding_digest"] != binding.digest or row["launch_digest"] != launch.digest:
            raise PreparedGraphProbeError("probe request binding or launch digest is mismatched")
        if row["policy_digest"] != policy.digest:
            raise PreparedGraphProbeError("probe request policy digest is mismatched")
        request = cls(
            binding,
            launch,
            policy,
            tuple(PreparedGraphVariantAuthority.from_dict(value) for value in row["target_variants"]),
        )
        if request.canonical_bytes != payload:
            raise PreparedGraphProbeError("probe request bytes changed during typed parsing")
        return request


def _target_variant_authority(
    binding: B300QualificationGraphBinding,
    manifest: Manifest,
) -> tuple[PreparedGraphVariantAuthority, ...]:
    rows: list[PreparedGraphVariantAuthority] = []
    observed_order: list[tuple[str, str]] = []
    for member in binding.target_members:
        member_ops = manifest.ops_for(member)
        if not member_ops:
            raise PreparedGraphProbeError(f"prepared manifest is missing target member {member!r}")
        for op in member_ops:
            if type(op) is not OpEntry:
                raise PreparedGraphProbeError("prepared manifest op is not exactly typed")
            observed_order.append((op.slot, op.variant))
            rows.append(PreparedGraphVariantAuthority(op.slot, op.variant))
    keys = tuple(observed_order)
    if len(keys) != len(set(keys)):
        raise PreparedGraphProbeError("prepared manifest repeats member/variant authority")
    if keys != tuple(sorted(keys)):
        raise PreparedGraphProbeError("prepared manifest reorders member/variant authority")
    rows.sort(key=lambda row: (row.slot_id, row.variant_id))
    return tuple(rows)


def _canonical_root(value: str | Path) -> Path:
    try:
        path = Path(value)
        before = path.lstat()
        resolved = path.resolve(strict=True)
    except (TypeError, OSError) as exc:
        raise PreparedGraphProbeError(f"materialized root cannot be reopened: {exc}") from None
    if not path.is_absolute() or resolved != path or stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
        raise PreparedGraphProbeError("materialized root must be one canonical nonsymlink directory")
    return path


def _contained_regular(root: Path, relative: str, field: str) -> Path:
    logical = PurePosixPath(relative) if type(relative) is str else None
    if (
        logical is None
        or logical.is_absolute()
        or logical.as_posix() != relative
        or any(part in {"", ".", ".."} for part in logical.parts)
    ):
        raise PreparedGraphProbeError(f"{field} path is not canonical and relative")
    candidate = root.joinpath(*logical.parts)
    try:
        before = candidate.lstat()
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise PreparedGraphProbeError(f"cannot reopen {field}: {exc}") from None
    try:
        resolved.relative_to(root)
    except ValueError:
        raise PreparedGraphProbeError(f"{field} escapes the materialized tree") from None
    if resolved != candidate or stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise PreparedGraphProbeError(f"{field} must be a contained regular nonsymlink file")
    return candidate


def _read_metadata(root: Path, op: OpEntry) -> dict | None:
    if op.metadata is None:
        return None
    path = _contained_regular(root, op.metadata, "variant metadata")
    try:
        before = path.stat()
        first = path.read_bytes()
        middle = path.stat()
        second = path.read_bytes()
        after = path.stat()
        if (
            (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            != (middle.st_dev, middle.st_ino, middle.st_size, middle.st_mtime_ns)
            or (middle.st_dev, middle.st_ino, middle.st_size, middle.st_mtime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            or first != second
        ):
            raise PreparedGraphProbeError("variant metadata changed while reopening")
        value = json.loads(first.decode("utf-8"))
    except PreparedGraphProbeError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PreparedGraphProbeError(f"variant metadata is invalid: {exc}") from None
    if type(value) is not dict:
        raise PreparedGraphProbeError("variant metadata must be an exact JSON object")
    return value


def _shape_record(
    row: ShapeResult,
    *,
    slot_id: str,
    variant_id: str,
    policy: PreparedGraphProbePolicy,
    collective: bool,
) -> StructuredGraphShapeRecord | None:
    if type(row) is not ShapeResult or type(row.phase_outcome) is not GraphPhaseOutcome:
        raise PreparedGraphProbeIncompleteError("verifier returned an untyped shape outcome")
    if (
        type(row.applicable) is not bool
        or type(row.passed) is not bool
        or type(row.graph_replays) is not int
        or row.graph_replays < 0
    ):
        raise PreparedGraphProbeIncompleteError("verifier returned malformed shape flags")
    descriptor = row.case_descriptor
    if type(descriptor) is not VerificationCaseDescriptor:
        raise PreparedGraphProbeIncompleteError("verifier omitted the typed case descriptor")
    if descriptor.slot_id != slot_id or descriptor.variant_id != variant_id:
        raise PreparedGraphProbeError("case descriptor names another slot or variant")
    if row.dtype != policy.dtype_name:
        raise PreparedGraphProbeError("shape dtype differs from commissioned policy")
    phase = row.phase_outcome
    if row.graph_replays != phase.replay_count:
        raise PreparedGraphProbeError("legacy graph_replays disagrees with typed replay_count")
    calls = tuple(dict(call) for call in descriptor.calls)
    for call in calls:
        if (
            call.get("dtype") != policy.dtype_name
            or call.get("architecture") != policy.architecture
            or call.get("tp_size") != policy.tp_size
            or call.get("world_size") != policy.world_size
        ):
            raise PreparedGraphProbeError("case descriptor execution context differs from policy")

    temporal = descriptor.case_kind is VerificationCaseKind.COLLECTIVE_TEMPORAL_EAGER
    expected_kinds = (
        {
            VerificationCaseKind.COLLECTIVE_SINGLE,
            VerificationCaseKind.COLLECTIVE_TEMPORAL_EAGER,
            VerificationCaseKind.COLLECTIVE_GRAPH_SEQUENCE,
        }
        if collective
        else {VerificationCaseKind.ORDINARY_SINGLE}
    )
    if descriptor.case_kind not in expected_kinds:
        raise PreparedGraphProbeError("case descriptor kind differs from validator slot routing")
    expected_graph_mode = "eager" if temporal else policy.graph_mode
    if any(call.get("graph_mode") != expected_graph_mode for call in calls):
        raise PreparedGraphProbeError("case descriptor graph mode differs from commissioned phase")

    if temporal:
        if (
            not row.applicable
            or not row.passed
            or phase != GraphPhaseOutcome.eager_only_passed()
            or row.graph_replays != 0
        ):
            raise PreparedGraphProbeIncompleteError(
                f"collective temporal-eager precondition failed for {(slot_id, variant_id)!r}"
            )
        return None

    if not row.applicable:
        if not row.passed or phase != GraphPhaseOutcome.not_applicable():
            raise PreparedGraphProbeError("nonapplicable row has inconsistent typed outcome")
        return StructuredGraphShapeRecord(
            descriptor.digest, False, False, False, 0, False, True, False
        )
    if not phase.observation_complete:
        raise PreparedGraphProbeIncompleteError(
            f"graph observation is incomplete for {(slot_id, variant_id)!r}"
        )
    candidate_failed = phase.failure_is_candidate_attributable
    passed = (
        phase.eager is PhaseDisposition.PASSED
        and phase.capture is PhaseDisposition.PASSED
        and phase.replay is PhaseDisposition.PASSED
    )
    if row.passed != passed:
        raise PreparedGraphProbeError("shape pass flag disagrees with typed graph phases")
    if passed and phase.replay_count != policy.expected_graph_replays:
        raise PreparedGraphProbeIncompleteError("passing graph row has incomplete replay coverage")
    if phase.replay_count > policy.expected_graph_replays:
        raise PreparedGraphProbeError("graph row exceeds commissioned replay authority")
    if not passed and not candidate_failed:
        raise PreparedGraphProbeIncompleteError("nonpassing graph row is not candidate-attributable")
    return StructuredGraphShapeRecord(
        descriptor.digest,
        True,
        phase.eager_passed,
        phase.capture_succeeded,
        phase.replay_count,
        phase.replay_passed,
        phase.observation_complete,
        candidate_failed,
    )


def _variant_record(
    result: VerifyResult,
    *,
    slot_id: str,
    variant_id: str,
    policy: PreparedGraphProbePolicy,
    collective: bool,
) -> StructuredGraphVariantRecord:
    if (
        type(result) is not VerifyResult
        or result.slot != slot_id
        or result.dtype != policy.dtype_name
    ):
        raise PreparedGraphProbeIncompleteError("verifier returned an unbound result")
    for field in (
        "passed",
        "graph_required",
        "graph_verified",
        "context_inapplicable",
        "domain_coverage_complete",
    ):
        if type(getattr(result, field)) is not bool:
            raise PreparedGraphProbeIncompleteError("verifier returned malformed result flags")
    if type(result.coverage_required) is not int or result.coverage_required < 0:
        raise PreparedGraphProbeIncompleteError("verifier returned malformed coverage authority")
    if not result.graph_required:
        raise PreparedGraphProbeIncompleteError("verifier did not execute the required graph policy")
    if not result.domain_coverage_complete:
        raise PreparedGraphProbeIncompleteError("verifier reported incomplete graph-domain coverage")
    if type(result.shape_results) is not list or not result.shape_results:
        raise PreparedGraphProbeIncompleteError("verifier returned no typed shape rows")
    records: list[StructuredGraphShapeRecord] = []
    temporal_count = 0
    graph_sequence_count = 0
    for row in result.shape_results:
        if (
            type(row) is ShapeResult
            and type(row.case_descriptor) is VerificationCaseDescriptor
            and row.case_descriptor.case_kind is VerificationCaseKind.COLLECTIVE_TEMPORAL_EAGER
        ):
            temporal_count += 1
        if (
            type(row) is ShapeResult
            and type(row.case_descriptor) is VerificationCaseDescriptor
            and row.case_descriptor.case_kind is VerificationCaseKind.COLLECTIVE_GRAPH_SEQUENCE
        ):
            graph_sequence_count += 1
        record = _shape_record(
            row,
            slot_id=slot_id,
            variant_id=variant_id,
            policy=policy,
            collective=collective,
        )
        if record is not None:
            records.append(record)
    records.sort(key=lambda row: row.descriptor_digest)
    if not records or len({row.descriptor_digest for row in records}) != len(records):
        raise PreparedGraphProbeError("graph descriptor authority is empty or duplicated")
    context_applicable = any(row.applicable for row in records)
    if collective and temporal_count != int(context_applicable):
        raise PreparedGraphProbeIncompleteError(
            "collective verifier omitted or duplicated its temporal-eager precondition"
        )
    if collective and graph_sequence_count != int(context_applicable):
        raise PreparedGraphProbeIncompleteError(
            "collective verifier omitted or duplicated its graph-sequence evidence"
        )
    if result.context_inapplicable and context_applicable:
        raise PreparedGraphProbeError("verifier context applicability is inconsistent")
    shapes_passed = all(not row.failed for row in records)
    expected_pass = result.coverage_sufficient and shapes_passed
    if result.passed != expected_pass:
        raise PreparedGraphProbeError("aggregate verifier pass disagrees with typed shape evidence")
    if result.graph_verified != (context_applicable and shapes_passed):
        raise PreparedGraphProbeError("aggregate graph verification flag is inconsistent")
    try:
        return StructuredGraphVariantRecord(
            slot_id,
            variant_id,
            context_applicable,
            True,
            tuple(records),
        )
    except B300QualificationCapabilityError as exc:
        raise PreparedGraphProbeError(f"typed graph variant is inconsistent: {exc}") from None


# Production callers cannot inject request-selected behavior.  Tests may replace
# these module-private validator seams inside their trusted process.
_VERIFY_ENTRY_FROM_SOURCE = verify_entry_from_source
_VERIFY_COLLECTIVE = verify_collective


def _execute_variant(root: Path, op: OpEntry, policy: PreparedGraphProbePolicy) -> VerifyResult:
    source = _contained_regular(root, op.source, "variant source")
    metadata = _read_metadata(root, op)
    try:
        eligibility = eligibility_from_metadata(metadata, op.dtypes, op.architectures)
        slot = slot_for_model(op.slot, policy.model_profile_key)
    except (KeyError, TypeError, ValueError) as exc:
        raise PreparedGraphProbeError(f"validator slot or eligibility is invalid: {exc}") from None
    if type(slot) is not SlotSpec:
        raise PreparedGraphProbeError("validator slot resolver returned an inexact slot")
    common = {
        "dtype_name": policy.dtype_name,
        "seed": policy.seed,
        "jitter_seed": policy.jitter_seed,
        "model_key": policy.model_profile_key,
        "bundle_path": str(root),
        "variant_name": op.variant,
        "graph_safe": True,
        "graph_replays": policy.expected_graph_replays,
        "tp_size": policy.tp_size,
    }
    if slot.kind in {"op", "block"}:
        try:
            return _VERIFY_ENTRY_FROM_SOURCE(
                op.slot,
                str(source),
                op.entry,
                prepare_name=op.prepare,
                device="cuda",
                override_point=op.override_point,
                eligibility_metadata=metadata,
                manifest_dtypes=op.dtypes,
                manifest_architectures=op.architectures,
                world_size=policy.world_size,
                **common,
            )
        except Exception as exc:  # noqa: BLE001 - absent typed outcome is HOLD
            raise PreparedGraphProbeIncompleteError(
                f"ordinary verifier did not return typed evidence: {type(exc).__name__}"
            ) from None
    if slot.kind == "collective":
        if op.override_point is not None:
            raise PreparedGraphProbeError("collective validator path does not support overrides")
        if type(eligibility) is not Eligibility:
            raise PreparedGraphProbeError("collective eligibility is not exactly typed")
        try:
            return _VERIFY_COLLECTIVE(
                slot,
                str(source),
                op.entry,
                prepare_name=op.prepare,
                world_size=policy.world_size,
                backend="nccl",
                device="cuda",
                timeout_s=float(policy.collective_timeout_s),
                eligibility=eligibility,
                **common,
            )
        except Exception as exc:  # noqa: BLE001 - absent typed outcome is HOLD
            raise PreparedGraphProbeIncompleteError(
                f"collective verifier did not return typed evidence: {type(exc).__name__}"
            ) from None
    raise PreparedGraphProbeError(f"validator slot kind {slot.kind!r} is unsupported")


def execute_prepared_graph_probe(
    request: PreparedGraphProbeRequest,
    materialized_root: str | Path,
) -> B300QualificationGraphArtifact:
    """Execute one commissioned request inside its isolated prepared worker.

    This function performs no filesystem writes and returns only the final domain
    artifact.  Publication/storage remains a separate private controller authority.
    """

    if type(request) is not PreparedGraphProbeRequest:
        raise PreparedGraphProbeError("prepared graph request is not exactly typed")
    root = _canonical_root(materialized_root)
    try:
        tree = reopen_materialized_engine_tree(
            root, expected_tree_digest=request.binding.materialized_tree_digest
        )
        trusted_identity = _tree_identity_digest(tree)
        manifest = load_manifest(root)
    except (EngineTreeError, ManifestError, B300QualificationGraphProviderError, OSError, ValueError) as exc:
        raise PreparedGraphProbeError(f"cannot reopen exact materialized graph tree: {exc}") from None
    if (
        tree.stack_digest != request.binding.materialized_stack_digest
        or tree.tree_digest != request.launch.tree_digest
        or trusted_identity != request.binding.trusted_tree_identity_digest
        or request.launch.stack_digest != tree.stack_digest
    ):
        raise PreparedGraphProbeError("materialized tree differs from commissioned binding/launch")
    observed_authority = _target_variant_authority(request.binding, manifest)
    if observed_authority != request.target_variants:
        raise PreparedGraphProbeError("materialized member/variant authority differs from request")

    by_key = {(op.slot, op.variant): op for op in manifest.ops}
    records: list[StructuredGraphVariantRecord] = []
    for authority in request.target_variants:
        key = (authority.slot_id, authority.variant_id)
        op = by_key.get(key)
        if op is None:
            raise PreparedGraphProbeError(f"materialized manifest is missing authority {key!r}")
        result = _execute_variant(root, op, request.policy)
        slot = slot_for_model(op.slot, request.policy.model_profile_key)
        records.append(
            _variant_record(
                result,
                slot_id=op.slot,
                variant_id=op.variant,
                policy=request.policy,
                collective=slot.kind == "collective",
            )
        )
    records.sort(key=lambda row: (row.slot_id, row.variant_id))
    try:
        return B300QualificationGraphArtifact(
            request.binding,
            request.policy.verification_policy_digest,
            request.policy.expected_graph_replays,
            tuple(records),
        )
    except (B300QualificationGraphProviderError, B300QualificationCapabilityError) as exc:
        raise PreparedGraphProbeError(f"final graph artifact rejected: {exc}") from None


__all__ = [
    "PreparedGraphProbeError",
    "PreparedGraphProbeIncompleteError",
    "PreparedGraphProbePolicy",
    "PreparedGraphProbeRequest",
    "PreparedGraphVariantAuthority",
    "execute_prepared_graph_probe",
]

from cacheon.eval.b300_prepared_graph_oci import B300PreparedGraphOCIExecutor  # noqa: E402

__all__.append("B300PreparedGraphOCIExecutor")
