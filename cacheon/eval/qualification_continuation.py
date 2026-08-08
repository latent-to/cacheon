"""Authenticated, immutable continuation records for one sealed qualification.

Every record is request/authority/source-bound, canonical, fsynced, and
create-once. Reopened typed evidence reruns its constructor validation; partial,
foreign, or ambiguous evaluator state is a HOLD rather than permission to rerun.
"""

from __future__ import annotations

import math
import os
import uuid
from dataclasses import dataclass
from pathlib import Path

from cacheon.eval.continuation_codec import (
    ContinuationCodec,
    ContinuationCodecError,
)
from cacheon.eval.crossover_runtime import ResidentCrossoverEvidence
from cacheon.eval.device_state import DeviceStateReceipt
from cacheon.eval.evidence_store import EvidenceArtifactRef
from cacheon.eval.marginal_runtime import (
    CandidateLifecycleEvidence,
    MarginalLifecycleEvidence,
    MarginalRuntimeError,
    PreparedMarginalRuntime,
)
from cacheon.eval.oci_backend import (
    CandidateFreeRuntimeIdentity,
    EngineExecutionEvidence,
    PristineReferenceExecutionEvidence,
)
from cacheon.eval.oci_process import OCIQuiescenceReceipt
from cacheon.eval.qualification import SelectionEntropyReceipt
from cacheon.eval.reference_protocol import ReferenceRequest
from cacheon.eval.resident_pair_crossover import (
    ResidentPairCrossoverError,
    ResidentPairCrossoverEvidence,
    ResidentPairCrossoverPlan,
)
from cacheon.eval.resident_pair_binding import (
    ResidentPairRuntimeBinding,
)
from cacheon.stack_identity import (
    canonical_digest,
    canonical_json_bytes,
    require_sha256_hex,
)


RECORD_SCHEMA = "cacheon.eval.qualification-continuation-record.v2"
RESIDENT_COUNT_QUALITY_CHECKPOINT_SCHEMA = (
    "cacheon.eval.resident-count-quality-checkpoint.v2"
)
RESIDENT_COUNT_QUALITY_PAYLOAD_SCHEMA = (
    "cacheon.eval.resident-count-quality-continuation-payload.v2"
)
RESIDENT_PAIR_RETIREMENT_CHECKPOINT_SCHEMA = (
    "cacheon.eval.resident-pair-retirement-checkpoint.v1"
)
RESIDENT_PAIR_RETIREMENT_PAYLOAD_SCHEMA = (
    "cacheon.eval.resident-pair-retirement-continuation-payload.v1"
)
_STAGES = ("speed", "resident_count", "retirement", "audit_armed",
           "audit_completed", "t_armed", "quality", "final")


def _decimal(value: object, where: str) -> str:
    """Encode one finite timing fact as the canonical .17g decimal string."""

    if type(value) is bool or type(value) not in (int, float):
        raise QualificationContinuationError(f"{where} is not a real number")
    number = float(value)
    if not math.isfinite(number):
        raise QualificationContinuationError(f"{where} is not finite")
    return format(number, ".17g")


def _decimal_value(value: object, where: str) -> float:
    """Reopen one canonical .17g decimal string; reject any other spelling."""

    if type(value) is not str:
        raise QualificationContinuationError(
            f"{where} is not a canonical decimal string"
        )
    try:
        number = float(value)
    except ValueError:
        raise QualificationContinuationError(
            f"{where} is not a canonical decimal string"
        ) from None
    if not math.isfinite(number) or format(number, ".17g") != value:
        raise QualificationContinuationError(f"{where} is not canonical")
    return number


class QualificationContinuationError(RuntimeError):
    """A continuation record is absent-where-required, foreign, or mutated.

    Callers must treat this as HOLD: neither rerun the expensive stage nor
    fabricate a result.
    """


@dataclass(frozen=True)
class ResidentCountQualityCheckpoint:
    """Artifact and authority bindings for one completed resident count run."""

    raw_execution_evidence: EvidenceArtifactRef
    raw_execution_evidence_semantic_digest: str
    candidate_observation: EvidenceArtifactRef
    candidate_observation_semantic_digest: str
    execution_plan_digest: str
    fixed_stock_authority_digest: str
    pair_binding_digest: str
    schema: str = RESIDENT_COUNT_QUALITY_CHECKPOINT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != RESIDENT_COUNT_QUALITY_CHECKPOINT_SCHEMA:
            raise QualificationContinuationError(
                "resident count quality checkpoint schema is unsupported"
            )
        for field, value in (
            ("raw execution evidence", self.raw_execution_evidence),
            ("candidate observation", self.candidate_observation),
        ):
            if type(value) is not EvidenceArtifactRef:
                raise QualificationContinuationError(
                    f"resident count {field} is not an exact evidence reference"
                )
        for field, value in (
            (
                "raw execution evidence semantic digest",
                self.raw_execution_evidence_semantic_digest,
            ),
            (
                "candidate observation semantic digest",
                self.candidate_observation_semantic_digest,
            ),
            ("resident count execution-plan digest", self.execution_plan_digest),
            ("fixed-stock authority digest", self.fixed_stock_authority_digest),
            ("resident pair binding digest", self.pair_binding_digest),
        ):
            if type(value) is not str:
                raise QualificationContinuationError(f"{field} is not exactly str")
            try:
                require_sha256_hex(value, field=field)
            except ValueError as exc:
                raise QualificationContinuationError(str(exc)) from None


@dataclass(frozen=True)
class ResidentPairLaneRetirement:
    """Path-free projection of one stock lifetime and its empty namespace."""

    lane_id: str
    commissioning_digest: str
    runtime_identity: CandidateFreeRuntimeIdentity
    runtime_preflight_receipt_sha256: str
    arena_model_receipt_digest: str
    resource_policy_digest: str
    native_publication_digest: str
    runtime_argv_sha256: str
    recovered_lease_ids: tuple[str, ...]
    device_receipts: tuple[DeviceStateReceipt, DeviceStateReceipt]
    session_ready_completed_at: float
    session_completed_at: float
    session_digest: str
    quiescence: OCIQuiescenceReceipt

    def __post_init__(self) -> None:
        if (
            self.lane_id not in ("A", "B")
            or type(self.runtime_identity) is not CandidateFreeRuntimeIdentity
            or type(self.recovered_lease_ids) is not tuple
            or any(not isinstance(row, str) or not row for row in self.recovered_lease_ids)
            or len(set(self.recovered_lease_ids)) != len(self.recovered_lease_ids)
            or type(self.device_receipts) is not tuple
            or len(self.device_receipts) != 2
            or any(type(row) is not DeviceStateReceipt for row in self.device_receipts)
            or type(self.quiescence) is not OCIQuiescenceReceipt
        ):
            raise QualificationContinuationError("resident pair lane retirement is malformed")
        for field in (
            "commissioning_digest", "runtime_preflight_receipt_sha256",
            "arena_model_receipt_digest", "resource_policy_digest",
            "native_publication_digest", "runtime_argv_sha256", "session_digest",
        ):
            try:
                object.__setattr__(self, field, require_sha256_hex(getattr(self, field), field=field))
            except (TypeError, ValueError) as exc:
                raise QualificationContinuationError(str(exc)) from None
        for field in ("session_ready_completed_at", "session_completed_at"):
            if type(getattr(self, field)) is not float or not math.isfinite(getattr(self, field)):
                raise QualificationContinuationError("resident pair session time is malformed")
        if self.session_ready_completed_at > self.session_completed_at:
            raise QualificationContinuationError("resident pair session time is reordered")


@dataclass(frozen=True)
class ResidentPairRetirementCheckpoint:
    """One immutable request-bound A/B retirement product."""

    authenticated_request_digest: str
    qualification_authority_digest: str
    target_profile_digest: str
    request_epoch_digest: str
    pair_binding: ResidentPairRuntimeBinding
    speed_plan_digest: str
    speed_evidence_digest: str
    count_plan_digest: str | None
    count_evidence_digest: str | None
    request_history_slice_digests: tuple[str, ...]
    lanes: tuple[ResidentPairLaneRetirement, ResidentPairLaneRetirement]
    schema: str = RESIDENT_PAIR_RETIREMENT_CHECKPOINT_SCHEMA

    def __post_init__(self) -> None:
        if (
            self.schema != RESIDENT_PAIR_RETIREMENT_CHECKPOINT_SCHEMA
            or type(self.pair_binding) is not ResidentPairRuntimeBinding
            or type(self.request_history_slice_digests) is not tuple
            or not self.request_history_slice_digests
            or type(self.lanes) is not tuple
            or len(self.lanes) != 2
            or any(type(row) is not ResidentPairLaneRetirement for row in self.lanes)
            or tuple(row.lane_id for row in self.lanes) != ("A", "B")
        ):
            raise QualificationContinuationError("resident pair retirement is malformed")
        for field in (
            "authenticated_request_digest", "qualification_authority_digest",
            "target_profile_digest", "request_epoch_digest", "speed_plan_digest",
            "speed_evidence_digest",
        ):
            try:
                object.__setattr__(self, field, require_sha256_hex(getattr(self, field), field=field))
            except (TypeError, ValueError) as exc:
                raise QualificationContinuationError(str(exc)) from None
        if (self.count_plan_digest is None) != (self.count_evidence_digest is None):
            raise QualificationContinuationError("resident pair count retirement is partial")
        for field in ("count_plan_digest", "count_evidence_digest"):
            if getattr(self, field) is not None:
                try:
                    object.__setattr__(self, field, require_sha256_hex(getattr(self, field), field=field))
                except (TypeError, ValueError) as exc:
                    raise QualificationContinuationError(str(exc)) from None
        for row in self.request_history_slice_digests:
            try:
                require_sha256_hex(row, field="request history slice digest")
            except (TypeError, ValueError) as exc:
                raise QualificationContinuationError(str(exc)) from None

    @property
    def digest(self) -> str:
        try:
            payload = _codec().encode(self)
        except ContinuationCodecError as exc:
            raise QualificationContinuationError(str(exc)) from None
        return canonical_digest(RESIDENT_PAIR_RETIREMENT_CHECKPOINT_SCHEMA, payload)


def _codec() -> ContinuationCodec:
    # Deferred import breaks the cycle with qualification_runner (AuditWitness).
    from cacheon.eval.qualification_runner import AuditWitness

    return ContinuationCodec(
        (
            ResidentCrossoverEvidence,
            EngineExecutionEvidence,
            PristineReferenceExecutionEvidence,
            OCIQuiescenceReceipt,
            SelectionEntropyReceipt,
            ReferenceRequest,
            AuditWitness,
            EvidenceArtifactRef,
            ResidentCountQualityCheckpoint,
            ResidentPairRetirementCheckpoint,
            ResidentPairCrossoverEvidence,
        )
    )


@dataclass(frozen=True)
class AuditContinuation:
    """Durable audit result bound to its unique evaluator claim."""

    nonce: str
    operation_digest: str
    audit_witnesses: tuple[tuple[str, object], ...]
    audit_started: float
    audit_completed: float
    audit_last_completed: float

@dataclass(frozen=True)
class QualityContinuation:
    """Everything downstream grading needs once pristine T is durable."""

    teardown_before: OCIQuiescenceReceipt
    entropy: SelectionEntropyReceipt
    entropy_observed: float
    requests: tuple[ReferenceRequest, ...]
    reference_execution: PristineReferenceExecutionEvidence
    teardown_after: OCIQuiescenceReceipt
    t_nonce: str
    t_operation_digest: str


class QualificationContinuationStore:
    """One private directory of per-cohort continuation records."""

    def __init__(self, root: Path) -> None:
        if not isinstance(root, Path) or not root.is_absolute():
            raise QualificationContinuationError(
                "continuation root must be one absolute private path"
            )
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        root.chmod(0o700)
        self.root = root

    def scope(
        self,
        *,
        request_digest: str | None = None,
        authority_digest: str,
        source_digest: str,
    ) -> "QualificationContinuation":
        return QualificationContinuation(
            self, request_digest, authority_digest, source_digest
        )


class QualificationContinuation:
    """Continuation records for exactly one sealed cohort identity."""

    def __init__(
        self,
        store: QualificationContinuationStore,
        request_digest: str | None,
        authority_digest: str,
        source_digest: str,
    ) -> None:
        if type(store) is not QualificationContinuationStore:
            raise QualificationContinuationError(
                "continuation scope requires an exact store"
            )
        try:
            self.request_digest = require_sha256_hex(
                request_digest, field="authenticated request digest"
            )
            self.authority_digest = require_sha256_hex(
                authority_digest, field="qualification authority digest"
            )
            self.source_digest = require_sha256_hex(
                source_digest, field="sealed source digest"
            )
        except ValueError as exc:
            raise QualificationContinuationError(str(exc)) from None
        self.directory = store.root / (
            f"{self.request_digest}-{self.authority_digest}-{self.source_digest}"
        )
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.directory.chmod(0o700)
        self._codec = _codec()

    # -- closed record file handling ----------------------------------------

    def _record_bytes(self, stage: str, payload: object) -> bytes:
        body = {
            "authority_digest": self.authority_digest,
            "payload": payload,
            "request_digest": self.request_digest,
            "schema": RECORD_SCHEMA,
            "source_digest": self.source_digest,
            "stage": stage,
        }
        record = dict(body)
        record["record_digest"] = canonical_digest(RECORD_SCHEMA, body)
        return canonical_json_bytes(record)

    def _record(self, stage: str, payload: object) -> None:
        if stage not in _STAGES:
            raise QualificationContinuationError(f"unknown continuation stage {stage!r}")
        encoded = self._record_bytes(stage, payload)
        final = self.directory / f"{stage}.json"
        if final.is_symlink() or (final.exists() and not final.is_file()):
            raise QualificationContinuationError(
                f"continuation {stage} record is not a regular file"
            )
        if final.exists():
            if final.read_bytes() == encoded:
                return
            raise QualificationContinuationError(
                f"continuation {stage} record already exists with other content"
            )
        temporary = self.directory / (
            f".{stage}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        )
        descriptor = os.open(
            temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
        )
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temporary, 0o400)
            try:
                os.link(temporary, final, follow_symlinks=False)
            except FileExistsError:
                if final.is_symlink() or not final.is_file():
                    raise QualificationContinuationError(
                        f"continuation {stage} record is not a regular file"
                    ) from None
                if final.read_bytes() != encoded:
                    raise QualificationContinuationError(
                        f"continuation {stage} record already exists with other content"
                    ) from None
            directory_fd = os.open(self.directory, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            if temporary.exists():
                temporary.unlink()

    def _load(self, stage: str) -> object | None:
        if stage not in _STAGES:
            raise QualificationContinuationError(f"unknown continuation stage {stage!r}")
        final = self.directory / f"{stage}.json"
        if final.is_symlink():
            raise QualificationContinuationError(
                f"continuation {stage} record is not a regular file"
            )
        if not final.exists():
            return None
        if not final.is_file():
            raise QualificationContinuationError(
                f"continuation {stage} record is not a regular file"
            )
        import json

        raw = final.read_bytes()
        try:
            record = json.loads(raw.decode("utf-8"))
        except (UnicodeError, ValueError) as exc:
            raise QualificationContinuationError(
                f"continuation {stage} record is malformed: {exc}"
            ) from None
        expected_keys = {
            "authority_digest",
            "payload",
            "record_digest",
            "request_digest",
            "schema",
            "source_digest",
            "stage",
        }
        if type(record) is not dict or set(record) != expected_keys:
            raise QualificationContinuationError(
                f"continuation {stage} record is not closed"
            )
        body = {key: record[key] for key in record if key != "record_digest"}
        if (
            record["schema"] != RECORD_SCHEMA
            or record["stage"] != stage
            or record["request_digest"] != self.request_digest
            or record["authority_digest"] != self.authority_digest
            or record["source_digest"] != self.source_digest
            or record["record_digest"] != canonical_digest(RECORD_SCHEMA, body)
        ):
            raise QualificationContinuationError(
                f"continuation {stage} record names another sealed identity"
            )
        if self._record_bytes(stage, record["payload"]) != raw:
            raise QualificationContinuationError(
                f"continuation {stage} record bytes are not canonical"
            )
        return record["payload"]

    # -- speed ---------------------------------------------------------------

    def record_resident_speed(self, crossover: ResidentCrossoverEvidence) -> None:
        if type(crossover) is not ResidentCrossoverEvidence:
            raise QualificationContinuationError(
                "resident speed continuation requires exact crossover evidence"
            )
        self._record(
            "speed", {"mode": "resident", "crossover": self._codec.encode(crossover)}
        )

    def load_resident_speed(self) -> ResidentCrossoverEvidence | None:
        payload = self._load("speed")
        if payload is None:
            return None
        if (
            type(payload) is not dict
            or set(payload) != {"crossover", "mode"}
            or payload.get("mode") != "resident"
        ):
            raise QualificationContinuationError(
                "speed continuation record is not the resident shape"
            )
        try:
            crossover = self._codec.decode(payload["crossover"])
        except ContinuationCodecError as exc:
            raise QualificationContinuationError(str(exc)) from None
        if type(crossover) is not ResidentCrossoverEvidence:
            raise QualificationContinuationError(
                "speed continuation reopened another evidence type"
            )
        return crossover

    def record_resident_pair_speed(
        self, crossover: ResidentPairCrossoverEvidence
    ) -> None:
        if type(crossover) is not ResidentPairCrossoverEvidence:
            raise QualificationContinuationError(
                "resident pair speed continuation requires exact crossover evidence"
            )
        try:
            encoded = self._codec.encode(crossover)
        except ContinuationCodecError as exc:
            raise QualificationContinuationError(str(exc)) from None
        self._record(
            "speed",
            {"mode": "resident_pair", "crossover": encoded},
        )

    def load_resident_pair_speed(
        self, plan: ResidentPairCrossoverPlan
    ) -> ResidentPairCrossoverEvidence | None:
        if type(plan) is not ResidentPairCrossoverPlan:
            raise QualificationContinuationError(
                "resident pair speed continuation requires the exact crossover plan"
            )
        crossover = self.load_resident_pair_speed_raw()
        if crossover is None:
            return None
        try:
            crossover.regrade(plan)
        except ResidentPairCrossoverError as exc:
            raise QualificationContinuationError(
                f"resident pair speed continuation is invalid: {exc}"
            ) from None
        return crossover

    def load_resident_pair_speed_raw(
        self,
    ) -> ResidentPairCrossoverEvidence | None:
        """Reopen the frozen binding; callers must still regrade with a plan."""

        payload = self._load("speed")
        if payload is None:
            return None
        if (
            type(payload) is not dict
            or set(payload) != {"crossover", "mode"}
            or payload.get("mode") != "resident_pair"
        ):
            raise QualificationContinuationError(
                "speed continuation record is not the resident pair shape"
            )
        try:
            crossover = self._codec.decode(payload["crossover"])
        except ContinuationCodecError as exc:
            raise QualificationContinuationError(str(exc)) from None
        if type(crossover) is not ResidentPairCrossoverEvidence:
            raise QualificationContinuationError(
                "speed continuation reopened another evidence type"
            )
        return crossover

    def record_marginal_speed(self, lifecycle: MarginalLifecycleEvidence) -> None:
        if type(lifecycle) is not MarginalLifecycleEvidence:
            raise QualificationContinuationError(
                "marginal speed continuation requires exact lifecycle evidence"
            )
        encode = self._codec.encode
        self._record(
            "speed",
            {
                "mode": "marginal",
                "baseline_before": encode(lifecycle.baseline_before),
                "candidates": [encode(row.execution) for row in lifecycle.candidates],
                "baseline_after": encode(lifecycle.baseline_after),
                "candidates_repeat": [
                    encode(row.execution) for row in lifecycle.candidates_repeat
                ],
                "baseline_third": (
                    None
                    if lifecycle.baseline_third is None
                    else encode(lifecycle.baseline_third)
                ),
            },
        )

    def load_marginal_speed(
        self, prepared: PreparedMarginalRuntime
    ) -> MarginalLifecycleEvidence | None:
        payload = self._load("speed")
        if payload is None:
            return None
        expected_keys = {
            "baseline_after",
            "baseline_before",
            "baseline_third",
            "candidates",
            "candidates_repeat",
            "mode",
        }
        if (
            type(payload) is not dict
            or set(payload) != expected_keys
            or payload.get("mode") != "marginal"
            or type(payload["candidates"]) is not list
            or type(payload["candidates_repeat"]) is not list
        ):
            raise QualificationContinuationError(
                "speed continuation record is not the marginal shape"
            )

        def execution(encoded: object) -> EngineExecutionEvidence:
            try:
                value = self._codec.decode(encoded)
            except ContinuationCodecError as exc:
                raise QualificationContinuationError(str(exc)) from None
            if type(value) is not EngineExecutionEvidence:
                raise QualificationContinuationError(
                    "speed continuation reopened another evidence type"
                )
            return value

        rows = payload["candidates"]
        repeats = payload["candidates_repeat"]
        if len(rows) != len(prepared.candidates) or (
            repeats and len(repeats) != len(prepared.candidates)
        ):
            raise QualificationContinuationError(
                "speed continuation cardinality differs from the sealed runtime"
            )
        try:
            return MarginalLifecycleEvidence(
                prepared,
                execution(payload["baseline_before"]),
                tuple(
                    CandidateLifecycleEvidence(candidate, execution(encoded))
                    for candidate, encoded in zip(
                        prepared.candidates, rows, strict=True
                    )
                ),
                execution(payload["baseline_after"]),
                (
                    tuple(
                        CandidateLifecycleEvidence(candidate, execution(encoded))
                        for candidate, encoded in zip(
                            prepared.candidates, repeats, strict=True
                        )
                    )
                    if repeats
                    else ()
                ),
                (
                    None
                    if payload["baseline_third"] is None
                    else execution(payload["baseline_third"])
                ),
            )
        except MarginalRuntimeError as exc:
            raise QualificationContinuationError(
                f"speed continuation does not bind the sealed runtime: {exc}"
            ) from None

    # -- resident fixed-stock count quality -----------------------------------

    def _load_quality_stage_without_legacy_count(self) -> object | None:
        """Require explicit migration of the ambiguous old quality-stage shape."""

        legacy_payload = self._load("quality")
        if (
            type(legacy_payload) is dict
            and legacy_payload.get("mode") == "resident_count"
        ):
            raise QualificationContinuationError(
                "legacy quality.json carries resident_count evidence; explicit "
                "migration is required because the continuation shape is ambiguous"
            )
        return legacy_payload

    def record_resident_count_quality(
        self, value: ResidentCountQualityCheckpoint
    ) -> None:
        if type(value) is not ResidentCountQualityCheckpoint:
            raise QualificationContinuationError(
                "resident count quality continuation requires the exact checkpoint type"
            )
        self._load_quality_stage_without_legacy_count()
        self._record(
            "resident_count",
            {
                "mode": "resident_count",
                "schema": RESIDENT_COUNT_QUALITY_PAYLOAD_SCHEMA,
                "checkpoint": self._codec.encode(value),
            },
        )

    def load_resident_count_quality(
        self,
    ) -> ResidentCountQualityCheckpoint | None:
        self._load_quality_stage_without_legacy_count()
        payload = self._load("resident_count")
        if payload is None:
            return None
        if (
            type(payload) is not dict
            or set(payload) != {"checkpoint", "mode", "schema"}
            or payload.get("mode") != "resident_count"
            or payload.get("schema") != RESIDENT_COUNT_QUALITY_PAYLOAD_SCHEMA
        ):
            raise QualificationContinuationError(
                "quality continuation record is not the resident count shape"
            )
        try:
            checkpoint = self._codec.decode(payload["checkpoint"])
        except ContinuationCodecError as exc:
            raise QualificationContinuationError(str(exc)) from None
        if type(checkpoint) is not ResidentCountQualityCheckpoint:
            raise QualificationContinuationError(
                "quality continuation reopened another checkpoint type"
            )
        return checkpoint

    # -- exact resident-pair retirement --------------------------------------

    def record_resident_pair_retirement(
        self, value: ResidentPairRetirementCheckpoint
    ) -> None:
        if type(value) is not ResidentPairRetirementCheckpoint:
            raise QualificationContinuationError(
                "resident pair retirement requires the exact checkpoint type"
            )
        self._record(
            "retirement",
            {
                "mode": "resident_pair",
                "schema": RESIDENT_PAIR_RETIREMENT_PAYLOAD_SCHEMA,
                "checkpoint": self._codec.encode(value),
            },
        )

    def load_resident_pair_retirement(
        self,
    ) -> ResidentPairRetirementCheckpoint | None:
        payload = self._load("retirement")
        if payload is None:
            return None
        if (
            type(payload) is not dict
            or set(payload) != {"checkpoint", "mode", "schema"}
            or payload.get("mode") != "resident_pair"
            or payload.get("schema") != RESIDENT_PAIR_RETIREMENT_PAYLOAD_SCHEMA
        ):
            raise QualificationContinuationError(
                "retirement continuation record is not the resident-pair shape"
            )
        try:
            checkpoint = self._codec.decode(payload["checkpoint"])
        except ContinuationCodecError as exc:
            raise QualificationContinuationError(str(exc)) from None
        if type(checkpoint) is not ResidentPairRetirementCheckpoint:
            raise QualificationContinuationError(
                "retirement continuation reopened another checkpoint type"
            )
        return checkpoint

    # -- at-most-once evaluator claims ---------------------------------------

    def arm_evaluator(self, stage: str, operation_digest: str) -> str:
        """Claim one exact audit or T invocation; an old claim always HOLDs."""

        if stage not in {"audit", "t"}:
            raise QualificationContinuationError("unknown evaluator stage")
        try:
            operation_digest = require_sha256_hex(
                operation_digest, field=f"{stage} operation digest"
            )
        except ValueError as exc:
            raise QualificationContinuationError(str(exc)) from None
        nonce = uuid.uuid4().hex
        self._record(
            f"{stage}_armed",
            {"nonce": nonce, "operation_digest": operation_digest},
        )
        return nonce

    def _require_evaluator_arm(
        self, stage: str, nonce: str, operation_digest: str
    ) -> None:
        payload = self._load(f"{stage}_armed")
        if payload != {"nonce": nonce, "operation_digest": operation_digest}:
            raise QualificationContinuationError(
                f"{stage} completion differs from its unique evaluator claim"
            )

    def record_audit(self, value: AuditContinuation) -> None:
        if type(value) is not AuditContinuation:
            raise QualificationContinuationError("audit checkpoint is not exact")
        self._require_evaluator_arm("audit", value.nonce, value.operation_digest)
        self._record(
            "audit_completed",
            {
                "nonce": value.nonce,
                "operation_digest": value.operation_digest,
                "audit_witnesses": [
                    [digest, self._codec.encode(witness)]
                    for digest, witness in value.audit_witnesses
                ],
                "audit_started": _decimal(value.audit_started, "audit_started"),
                "audit_completed": _decimal(value.audit_completed, "audit_completed"),
                "audit_last_completed": _decimal(
                    value.audit_last_completed, "audit_last_completed"
                ),
            },
        )

    def load_audit(self, operation_digest: str) -> AuditContinuation | None:
        payload = self._load("audit_completed")
        if payload is None:
            return None
        expected = {"nonce", "operation_digest", "audit_witnesses",
                    "audit_started", "audit_completed", "audit_last_completed"}
        if type(payload) is not dict or set(payload) != expected:
            raise QualificationContinuationError("audit checkpoint is not closed")
        self._require_evaluator_arm("audit", payload["nonce"], operation_digest)
        from cacheon.eval.qualification_runner import AuditWitness
        try:
            witnesses = tuple(
                (row[0], self._codec.decode(row[1]))
                for row in payload["audit_witnesses"]
            )
        except (ContinuationCodecError, TypeError, IndexError) as exc:
            raise QualificationContinuationError(f"audit checkpoint is malformed: {exc}") from None
        if any(type(key) is not str or type(witness) is not AuditWitness
               for key, witness in witnesses):
            raise QualificationContinuationError("audit checkpoint witnesses are not exact")
        return AuditContinuation(
            payload["nonce"], operation_digest, witnesses,
            _decimal_value(payload["audit_started"], "audit_started"),
            _decimal_value(payload["audit_completed"], "audit_completed"),
            _decimal_value(payload["audit_last_completed"], "audit_last_completed"),
        )

    # -- pristine T quality ---------------------------------------------------

    def record_quality(self, value: QualityContinuation) -> None:
        if type(value) is not QualityContinuation:
            raise QualificationContinuationError(
                "quality continuation requires the exact checkpoint type"
            )
        self._load_quality_stage_without_legacy_count()
        self._require_evaluator_arm("t", value.t_nonce, value.t_operation_digest)
        encode = self._codec.encode
        self._record(
            "quality",
            {
                "teardown_before": encode(value.teardown_before),
                "entropy": encode(value.entropy),
                "entropy_observed": _decimal(
                    value.entropy_observed, "entropy_observed"
                ),
                "requests": [encode(row) for row in value.requests],
                "reference_execution": encode(value.reference_execution),
                "teardown_after": encode(value.teardown_after),
                "t_nonce": value.t_nonce,
                "t_operation_digest": value.t_operation_digest,
            },
        )

    def load_quality(self) -> QualityContinuation | None:
        payload = self._load_quality_stage_without_legacy_count()
        if payload is None:
            return None
        expected_keys = {
            "entropy",
            "entropy_observed",
            "reference_execution",
            "requests",
            "teardown_after",
            "teardown_before",
            "t_nonce",
            "t_operation_digest",
        }
        if (
            type(payload) is not dict
            or set(payload) != expected_keys
            or type(payload["requests"]) is not list
        ):
            raise QualificationContinuationError(
                "quality continuation record is not closed"
            )

        def decode(encoded: object, expected: type) -> object:
            try:
                value = self._codec.decode(encoded)
            except ContinuationCodecError as exc:
                raise QualificationContinuationError(str(exc)) from None
            if type(value) is not expected:
                raise QualificationContinuationError(
                    "quality continuation reopened another evidence type"
                )
            return value

        self._require_evaluator_arm(
            "t", payload["t_nonce"], payload["t_operation_digest"]
        )

        return QualityContinuation(
            teardown_before=decode(payload["teardown_before"], OCIQuiescenceReceipt),
            entropy=decode(payload["entropy"], SelectionEntropyReceipt),
            entropy_observed=_decimal_value(
                payload["entropy_observed"], "entropy_observed"
            ),
            requests=tuple(
                decode(row, ReferenceRequest) for row in payload["requests"]
            ),
            reference_execution=decode(
                payload["reference_execution"], PristineReferenceExecutionEvidence
            ),
            teardown_after=decode(payload["teardown_after"], OCIQuiescenceReceipt),
            t_nonce=payload["t_nonce"],
            t_operation_digest=payload["t_operation_digest"],
        )

    # -- final product ---------------------------------------------------------

    def record_final(self, reference: EvidenceArtifactRef) -> None:
        if type(reference) is not EvidenceArtifactRef:
            raise QualificationContinuationError(
                "final continuation requires one exact evidence reference"
            )
        self._record("final", {"reference": self._codec.encode(reference)})

    def load_final(self) -> EvidenceArtifactRef | None:
        payload = self._load("final")
        if payload is None:
            return None
        if type(payload) is not dict or set(payload) != {"reference"}:
            raise QualificationContinuationError(
                "final continuation record is not closed"
            )
        try:
            reference = self._codec.decode(payload["reference"])
        except ContinuationCodecError as exc:
            raise QualificationContinuationError(str(exc)) from None
        if type(reference) is not EvidenceArtifactRef:
            raise QualificationContinuationError(
                "final continuation reopened another reference type"
            )
        return reference


__all__ = [
    "AuditContinuation", "QualificationContinuation",
    "QualificationContinuationError", "QualificationContinuationStore",
    "QualityContinuation", "RECORD_SCHEMA",
    "RESIDENT_PAIR_RETIREMENT_CHECKPOINT_SCHEMA",
    "RESIDENT_PAIR_RETIREMENT_PAYLOAD_SCHEMA",
    "RESIDENT_COUNT_QUALITY_CHECKPOINT_SCHEMA",
    "RESIDENT_COUNT_QUALITY_PAYLOAD_SCHEMA", "ResidentCountQualityCheckpoint",
    "ResidentPairLaneRetirement", "ResidentPairRetirementCheckpoint",
]
