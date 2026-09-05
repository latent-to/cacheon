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
from cacheon.eval.evidence_store import EvidenceArtifactRef
from cacheon.eval.oci_backend import (
    EngineExecutionEvidence,
    PristineReferenceExecutionEvidence,
)
from cacheon.eval.oci_process import OCIQuiescenceReceipt
from cacheon.eval.qualification import SelectionEntropyReceipt
from cacheon.eval.reference_protocol import ReferenceRequest
from cacheon.stack_identity import (
    canonical_digest,
    canonical_json_bytes,
    require_sha256_hex,
)


RECORD_SCHEMA = "cacheon.eval.qualification-continuation-record.v2"
_STAGES = ("speed", "audit_armed", "audit_completed", "t_armed", "quality", "final")


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
        payload = self._load("quality")
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
]
