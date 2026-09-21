"""Recorded qualification timing, shared by production execution and regrading."""

from __future__ import annotations

import math
from dataclasses import dataclass

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
        if (
            any(type(row) is not float or not math.isfinite(row) for row in timestamps)
            or not (
                self.speed_started_monotonic_s
                < self.speed_completed_monotonic_s
                <= self.audit_started_monotonic_s
                < self.audit_completed_monotonic_s
                <= self.t_started_monotonic_s
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
                else value
            )
            for name, value in (
                (field, getattr(self, field))
                for field in self.__dataclass_fields__
            )
        }

    @classmethod
    def from_dict(cls, value: object) -> "QualificationTimingWitness":
        from cacheon.eval.qualification_runner import QualificationRunnerError, _strict

        # dict(...) copy: _strict returns the caller's mapping, and the float
        # decode below must never mutate a reopened payload in place — the
        # publish/reopen self-check compares to_dict() against that payload.
        raw = dict(_strict(value, set(cls.__dataclass_fields__), "qualification timing"))
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

