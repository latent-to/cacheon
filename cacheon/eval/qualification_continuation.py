"""Durable continuation checkpoints for one causal qualification cohort.

The exact-request fence prevents duplicate evaluations, but full automation
still cannot continue after a crash that happens *after* expensive evidence
exists.  This module adds authenticated durable products at three sealed
boundaries of ``run_causal_qualification``:

- immediately after B/C/B-prime, before grading or quality;
- immediately after pristine T (quality generation), before downstream
  grading/finalization;
- immediately after the final qualification product (attempt or stage exit),
  before CPU import/commit.

Recovery rules (enforced by the runner seams that consume this store):

- speed checkpoint exists -> resume audit/quality/finalization only;
- quality checkpoint exists -> regrade/finalize/import only;
- final product exists -> import/commit only;
- any identity mismatch -> fail closed (HOLD); nothing is silently rerun.

Each record is one closed JSON file bound to the authenticated remote request,
the cohort's qualification authority digest, and its sealed source digest,
carrying a canonical record digest over its own payload.  Payload evidence
round-trips through the exact-typed continuation codec, so every reconstructed
object re-runs its own fail-closed constructor validation, and the speed
evidence additionally re-grades itself against the sealed plan before it is
trusted.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

from cacheon.eval.continuation_codec import (
    ContinuationCodec,
    ContinuationCodecError,
)
from cacheon.eval.crossover_runtime import ResidentCrossoverEvidence
from cacheon.eval.evidence_store import EvidenceArtifactRef
from cacheon.eval.marginal_runtime import (
    CandidateLifecycleEvidence,
    MarginalLifecycleEvidence,
    MarginalRuntimeError,
    PreparedMarginalRuntime,
)
from cacheon.eval.oci_backend import (
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
from cacheon.stack_identity import (
    canonical_digest,
    canonical_json_bytes,
    require_sha256_hex,
)


RECORD_SCHEMA = "cacheon.eval.qualification-continuation-record.v2"
_STAGES = ("speed", "quality", "final")


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
    """Durable bindings for one completed resident count observation."""

    candidate_observation: EvidenceArtifactRef
    candidate_observation_semantic_digest: str
    execution_plan_digest: str
    fixed_stock_authority_digest: str

    def __post_init__(self) -> None:
        if type(self.candidate_observation) is not EvidenceArtifactRef:
            raise QualificationContinuationError(
                "resident count candidate observation is not an exact "
                "evidence reference"
            )
        for field, value in (
            (
                "candidate observation semantic digest",
                self.candidate_observation_semantic_digest,
            ),
            ("resident count execution-plan digest", self.execution_plan_digest),
            ("fixed-stock authority digest", self.fixed_stock_authority_digest),
        ):
            if type(value) is not str:
                raise QualificationContinuationError(f"{field} is not exactly str")
            try:
                require_sha256_hex(value, field=field)
            except ValueError as exc:
                raise QualificationContinuationError(str(exc)) from None


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
            ResidentPairCrossoverEvidence,
        )
    )


@dataclass(frozen=True)
class QualityContinuation:
    """Everything downstream grading needs once pristine T is durable."""

    teardown_before: OCIQuiescenceReceipt
    entropy: SelectionEntropyReceipt
    entropy_observed: float
    requests: tuple[ReferenceRequest, ...]
    reference_execution: PristineReferenceExecutionEvidence
    teardown_after: OCIQuiescenceReceipt
    audit_witnesses: tuple[tuple[str, object], ...]
    audit_started: float
    audit_completed: float


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
        if final.exists():
            if final.read_bytes() == encoded:
                return
            raise QualificationContinuationError(
                f"continuation {stage} record already exists with other content"
            )
        temporary = self.directory / f".{stage}.tmp"
        descriptor = os.open(
            temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
        )
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temporary, 0o400)
            os.replace(temporary, final)
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
        if not final.exists():
            return None
        if final.is_symlink():
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
            if type(crossover) is not ResidentPairCrossoverEvidence:
                raise QualificationContinuationError(
                    "speed continuation reopened another evidence type"
                )
            crossover.regrade(plan)
        except (ContinuationCodecError, ResidentPairCrossoverError) as exc:
            raise QualificationContinuationError(
                f"resident pair speed continuation is invalid: {exc}"
            ) from None
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

    # -- quality ---------------------------------------------------------------

    def record_resident_count_quality(
        self, value: ResidentCountQualityCheckpoint
    ) -> None:
        if type(value) is not ResidentCountQualityCheckpoint:
            raise QualificationContinuationError(
                "resident count quality continuation requires the exact checkpoint type"
            )
        self._record(
            "quality",
            {
                "mode": "resident_count",
                "checkpoint": self._codec.encode(value),
            },
        )

    def load_resident_count_quality(
        self,
    ) -> ResidentCountQualityCheckpoint | None:
        payload = self._load("quality")
        if payload is None:
            return None
        if (
            type(payload) is not dict
            or set(payload) != {"checkpoint", "mode"}
            or payload.get("mode") != "resident_count"
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

    def record_quality(self, value: QualityContinuation) -> None:
        if type(value) is not QualityContinuation:
            raise QualificationContinuationError(
                "quality continuation requires the exact checkpoint type"
            )
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
                "audit_witnesses": [
                    [digest, encode(witness)]
                    for digest, witness in value.audit_witnesses
                ],
                "audit_started": _decimal(value.audit_started, "audit_started"),
                "audit_completed": _decimal(
                    value.audit_completed, "audit_completed"
                ),
            },
        )

    def load_quality(self) -> QualityContinuation | None:
        payload = self._load("quality")
        if payload is None:
            return None
        expected_keys = {
            "audit_completed",
            "audit_started",
            "audit_witnesses",
            "entropy",
            "entropy_observed",
            "reference_execution",
            "requests",
            "teardown_after",
            "teardown_before",
        }
        if (
            type(payload) is not dict
            or set(payload) != expected_keys
            or type(payload["requests"]) is not list
            or type(payload["audit_witnesses"]) is not list
            or any(
                type(row) is not list or len(row) != 2 or type(row[0]) is not str
                for row in payload["audit_witnesses"]
            )
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

        from cacheon.eval.qualification_runner import AuditWitness

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
            audit_witnesses=tuple(
                (digest, decode(encoded, AuditWitness))
                for digest, encoded in payload["audit_witnesses"]
            ),
            audit_started=_decimal_value(payload["audit_started"], "audit_started"),
            audit_completed=_decimal_value(
                payload["audit_completed"], "audit_completed"
            ),
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
    "QualificationContinuation",
    "QualificationContinuationError",
    "QualificationContinuationStore",
    "QualityContinuation",
    "RECORD_SCHEMA",
    "ResidentCountQualityCheckpoint",
]
