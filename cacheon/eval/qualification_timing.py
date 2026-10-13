"""Recorded qualification timing, shared by production execution and regrading."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from cacheon.eval.evidence_store import EvidenceArtifactRef
from cacheon.eval.qualification import QualificationDecision

if TYPE_CHECKING:
    from cacheon.eval.qualification_runner import AuditWitness, ResidentSpeedWitness

from cacheon.stack_identity import canonical_digest, require_sha256_hex


@dataclass(frozen=True)
class QualificationTimingWitness:
    """Bound executed stages while preserving their original timestamps.

    Validator repair downtime between durable stages is not evaluator work.
    The runner still bounds each live invocation and charges retained stages
    before starting missing work. This allows late recovery without buying a
    new speed measurement or pretending the old measurement happened today.
    """

    policy_digest: str
    speed_evidence_digest: str
    audit_evidence_digest: str
    reference_session_digest: str
    max_qualification_seconds: int
    speed_started_monotonic_s: float
    speed_completed_monotonic_s: float
    audit_started_monotonic_s: float
    audit_completed_monotonic_s: float
    t_started_monotonic_s: float
    t_completed_monotonic_s: float
    qualification_completed_monotonic_s: float
    clock_domains: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        from cacheon.eval.qualification_runner import QualificationRunnerError

        for name in (
            "policy_digest",
            "speed_evidence_digest",
            "audit_evidence_digest",
            "reference_session_digest",
        ):
            object.__setattr__(
                self, name, require_sha256_hex(getattr(self, name), field=name)
            )
        if (
            type(self.max_qualification_seconds) is not int
            or not 60 <= self.max_qualification_seconds <= 14_400
        ):
            raise QualificationRunnerError(
                "qualification timing wall budget is malformed"
            )
        timestamps = tuple(
            getattr(self, name)
            for name in self.__dataclass_fields__
            if name.endswith("_monotonic_s")
        )
        domains = self.clock_domains
        if domains and (type(domains) is not tuple or len(domains) != 3
                        or any(type(row) is not str or not row for row in domains)):
            raise QualificationRunnerError("qualification clock domains are malformed")
        if (
            any(type(row) is not float or not math.isfinite(row) for row in timestamps)
            or not (
                self.speed_started_monotonic_s
                < self.speed_completed_monotonic_s
                and (bool(domains) and domains[0] != domains[1]
                     or self.speed_completed_monotonic_s <= self.audit_started_monotonic_s)
                and self.audit_started_monotonic_s
                < self.audit_completed_monotonic_s
                and (bool(domains) and domains[1] != domains[2]
                     or self.audit_completed_monotonic_s <= self.t_started_monotonic_s)
                and self.t_started_monotonic_s
                < self.t_completed_monotonic_s
                <= self.qualification_completed_monotonic_s
            )
            or (
                self.speed_completed_monotonic_s - self.speed_started_monotonic_s
                + self.audit_completed_monotonic_s - self.audit_started_monotonic_s
                + self.qualification_completed_monotonic_s - self.t_started_monotonic_s
            ) > self.max_qualification_seconds
        ):
            raise QualificationRunnerError(
                "qualification timing order or total wall time is invalid"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            name: (
                format(value, ".17g")
                if name.endswith("_monotonic_s")
                else list(value) if name == "clock_domains" else value
            )
            for name, value in (
                (field, getattr(self, field))
                for field in self.__dataclass_fields__
            )
            if name != "clock_domains" or value
        }

    @classmethod
    def from_dict(cls, value: object) -> "QualificationTimingWitness":
        from cacheon.eval.qualification_runner import QualificationRunnerError, _strict

        # dict(...) copy: _strict returns the caller's mapping, and the float
        # decode below must never mutate a reopened payload in place — the
        # publish/reopen self-check compares to_dict() against that payload.
        raw = dict(_strict({"clock_domains": [], **value}, set(cls.__dataclass_fields__), "qualification timing"))
        raw["clock_domains"] = tuple(raw["clock_domains"])
        for name in cls.__dataclass_fields__:
            if name.endswith("_monotonic_s"):
                encoded = raw[name]
                try:
                    decoded = float(encoded)
                except (TypeError, ValueError) as exc:
                    raise QualificationRunnerError(
                        f"qualification timing {name} is malformed"
                    ) from exc
                if not math.isfinite(decoded) or format(decoded, ".17g") != encoded:
                    raise QualificationRunnerError(
                        f"qualification timing {name} is noncanonical"
                    )
                raw[name] = decoded
        return cls(**raw)  # type: ignore[arg-type]

    @property
    def digest(self) -> str:
        return canonical_digest(
            "cacheon.qualification.operational-timing.v1", self.to_dict()
        )



@dataclass(frozen=True)
class QualificationStageExit:
    """Durable early terminal result carrying every executed stage; T never ran."""

    authority_digest: str
    source_digest: str
    selected_delta_digest: str
    stage: str
    decision: QualificationDecision
    reason: str
    speed_witness: ResidentSpeedWitness
    audit_witness: AuditWitness | None
    audit_started_monotonic_s: float | None
    audit_completed_monotonic_s: float | None
    terminal_quiescence_digest: str | None
    worker_recovery: EvidenceArtifactRef | None = None
    clock_domains: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        from cacheon.eval import qualification_runner as owner

        for name in ("authority_digest", "source_digest", "selected_delta_digest"):
            object.__setattr__(
                self, name, require_sha256_hex(getattr(self, name), field=name)
            )
        object.__setattr__(self, "decision", owner._decision(self.decision))
        if (bool(self.clock_domains) != (self.worker_recovery is not None)
                or self.worker_recovery is not None and (type(self.worker_recovery) is not EvidenceArtifactRef
                    or type(self.clock_domains) is not tuple or len(self.clock_domains) != 2
                    or any(type(row) is not str or not row for row in self.clock_domains))):
            raise owner.QualificationRunnerError("stage exit clocks lack their worker recovery")
        if (
            self.stage not in {"speed", "audit"}
            or type(self.speed_witness) is not owner.ResidentSpeedWitness
            or self.speed_witness.selected_delta_digest
            != self.selected_delta_digest
        ):
            raise owner.QualificationRunnerError("qualification stage exit is malformed")
        if self.terminal_quiescence_digest is not None:
            try:
                terminal_quiescence = require_sha256_hex(
                    self.terminal_quiescence_digest,
                    field="terminal_quiescence_digest",
                )
            except ValueError as exc:
                raise owner.QualificationRunnerError(str(exc)) from None
            object.__setattr__(
                self, "terminal_quiescence_digest", terminal_quiescence
            )
        # "speed_regression" is the retained pre-band coarse vocabulary; new
        # speed failures carry the graded band reason from their witness.
        allowed_reasons = {
            ("speed", QualificationDecision.FAIL): (
                owner.SPEED_FAIL_REASONS | {"speed_regression"}
            ),
            ("speed", QualificationDecision.NO_DECISION): {"speed_noise"},
            **{("audit", row): {reason} for row, reason in owner._AUDIT_REASONS.items()},
        }.get((self.stage, self.decision), {None})
        if self.reason not in allowed_reasons:
            raise owner.QualificationRunnerError("qualification stage-exit reason differs")
        # A speed exit carries an audit only after calibration observation audited it.
        if self.stage == "speed" and self.audit_witness is None:
            if any(
                value is not None
                for value in (
                    self.audit_started_monotonic_s,
                    self.audit_completed_monotonic_s,
                    self.terminal_quiescence_digest,
                )
            ):
                raise owner.QualificationRunnerError(
                    "pre-audit exit contains a later-stage witness"
                )
        else:
            if (
                type(self.audit_witness) is not owner.AuditWitness
                or self.audit_witness.decision
                is not (QualificationDecision.PASS if self.stage == "speed" else self.decision)
                or type(self.audit_started_monotonic_s) is not float
                or type(self.audit_completed_monotonic_s) is not float
                or not math.isfinite(self.audit_started_monotonic_s)
                or not math.isfinite(self.audit_completed_monotonic_s)
                or ((not self.clock_domains or self.clock_domains[0] == self.clock_domains[1])
                    and self.audit_started_monotonic_s < self.speed_witness.completed_monotonic_s)
                or self.audit_completed_monotonic_s
                <= self.audit_started_monotonic_s
                or (self.audit_completed_monotonic_s - self.audit_started_monotonic_s
                    + self.speed_witness.completed_monotonic_s - self.speed_witness.started_monotonic_s
                    if self.clock_domains else self.audit_completed_monotonic_s
                    - self.speed_witness.started_monotonic_s)
                > self.speed_witness.resident_policy.max_qualification_seconds
                or self.terminal_quiescence_digest is None
            ):
                raise owner.QualificationRunnerError("audit stage-exit timing is malformed")

    def to_dict(self) -> dict[str, object]:
        return {
            "audit_completed_monotonic_s": (
                None
                if self.audit_completed_monotonic_s is None
                else format(self.audit_completed_monotonic_s, ".17g")
            ),
            "audit_started_monotonic_s": (
                None
                if self.audit_started_monotonic_s is None
                else format(self.audit_started_monotonic_s, ".17g")
            ),
            "audit_witness": (
                None if self.audit_witness is None else self.audit_witness.to_dict()
            ),
            "authority_digest": self.authority_digest,
            "decision": self.decision.value,
            "reason": self.reason,
            "selected_delta_digest": self.selected_delta_digest,
            "source_digest": self.source_digest,
            "speed_witness": self.speed_witness.to_dict(),
            "stage": self.stage,
            "terminal_quiescence_digest": self.terminal_quiescence_digest,
            **({"worker_recovery": self.worker_recovery.to_dict(),
                "clock_domains": list(self.clock_domains)} if self.worker_recovery is not None else {}),
        }

    @classmethod
    def from_dict(cls, value: object) -> "QualificationStageExit":
        from cacheon.eval import qualification_runner as owner

        if type(value) is not dict:
            raise owner.QualificationRunnerError("qualification stage exit is not an object")
        raw = owner._strict(
            {"worker_recovery": None, "clock_domains": [], **value},
            set(cls.__dataclass_fields__),
            "qualification stage exit",
        )

        def optional_time(name: str) -> float | None:
            encoded = raw[name]
            if encoded is None:
                return None
            try:
                result = float(encoded)
            except (TypeError, ValueError) as exc:
                raise owner.QualificationRunnerError(
                    f"qualification stage-exit {name} is malformed"
                ) from exc
            if not math.isfinite(result) or format(result, ".17g") != encoded:
                raise owner.QualificationRunnerError(
                    f"qualification stage-exit {name} is noncanonical"
                )
            return result

        return cls(
            **{
                **raw,
                "worker_recovery": (None if raw["worker_recovery"] is None else
                                    EvidenceArtifactRef.from_dict(raw["worker_recovery"])),
                "clock_domains": tuple(raw["clock_domains"]),
                "speed_witness": owner.ResidentSpeedWitness.from_dict(raw["speed_witness"]),
                "audit_witness": (
                    None
                    if raw["audit_witness"] is None
                    else owner.AuditWitness.from_dict(raw["audit_witness"])
                ),
                "audit_started_monotonic_s": optional_time(
                    "audit_started_monotonic_s"
                ),
                "audit_completed_monotonic_s": optional_time(
                    "audit_completed_monotonic_s"
                ),
            }
        )  # type: ignore[arg-type]

    @property
    def digest(self) -> str:
        from cacheon.eval import qualification_runner as owner

        return canonical_digest(owner.STAGE_EXIT_SCHEMA, self.to_dict())

