"""Typed scoring and causal regrade records for exact-count quality evidence.

This module owns policy arithmetic only.  It cannot crown, settle, activate an
incentive, or mutate an earlier result.  Operators must reopen the referenced
source and observation artifacts before consuming a regrade record.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from cacheon._strict import require_digest, require_exact_fields, require_int
from cacheon.stack_identity import canonical_digest

COUNT_QUALITY_POLICY_SCHEMA = "cacheon.count-quality-policy.v1"
COUNT_QUALITY_EVIDENCE_SCHEMA = "cacheon.count-quality-evidence.v1"
COUNT_QUALITY_VERDICT_SCHEMA = "cacheon.count-quality-verdict.v1"
COUNT_QUALITY_REGRADE_SCHEMA = "cacheon.count-quality-regrade.v1"
COUNT_QUALITY_DECISIONS = frozenset({"PASS", "FAIL"})
_MAX_TOTAL = 1_000_000


class CountQualityError(ValueError):
    """Exact-count quality evidence or policy is malformed."""


def _strict(value: object, fields: frozenset[str], label: str) -> Mapping[str, object]:
    return require_exact_fields(value, fields=fields, label=label, error=CountQualityError)


def _digest(value: object, field: str) -> str:
    return require_digest(value, field=field, error=CountQualityError)


def _integer(value: object, field: str, *, minimum: int, maximum: int) -> int:
    return require_int(
        value,
        field=field,
        error=CountQualityError,
        minimum=minimum,
        maximum=maximum,
    )


@dataclass(frozen=True)
class CountQualityPolicy:
    """Reject when the observed correct-count drop reaches this threshold."""

    regression_threshold_drop: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "regression_threshold_drop",
            _integer(
                self.regression_threshold_drop,
                "regression threshold drop",
                minimum=1,
                maximum=_MAX_TOTAL,
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {"regression_threshold_drop": self.regression_threshold_drop}

    @classmethod
    def from_dict(cls, value: object) -> "CountQualityPolicy":
        raw = _strict(value, frozenset({"regression_threshold_drop"}), "count quality policy")
        return cls(regression_threshold_drop=raw["regression_threshold_drop"])

    @property
    def digest(self) -> str:
        return canonical_digest(COUNT_QUALITY_POLICY_SCHEMA, self.to_dict())


@dataclass(frozen=True)
class CountQualityEvidence:
    """Authenticated aggregate projection of stock and candidate observations."""

    stock_observation_digest: str
    candidate_observation_digest: str
    stock_correct: int
    candidate_correct: int
    total: int

    def __post_init__(self) -> None:
        for field in ("stock_observation_digest", "candidate_observation_digest"):
            object.__setattr__(self, field, _digest(getattr(self, field), field))
        total = _integer(self.total, "total", minimum=1, maximum=_MAX_TOTAL)
        object.__setattr__(self, "total", total)
        for field in ("stock_correct", "candidate_correct"):
            object.__setattr__(
                self,
                field,
                _integer(getattr(self, field), field, minimum=0, maximum=total),
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_correct": self.candidate_correct,
            "candidate_observation_digest": self.candidate_observation_digest,
            "stock_correct": self.stock_correct,
            "stock_observation_digest": self.stock_observation_digest,
            "total": self.total,
        }

    @classmethod
    def from_dict(cls, value: object) -> "CountQualityEvidence":
        raw = _strict(
            value,
            frozenset(
                {
                    "candidate_correct",
                    "candidate_observation_digest",
                    "stock_correct",
                    "stock_observation_digest",
                    "total",
                }
            ),
            "count quality evidence",
        )
        return cls(**raw)

    @property
    def digest(self) -> str:
        return canonical_digest(COUNT_QUALITY_EVIDENCE_SCHEMA, self.to_dict())


@dataclass(frozen=True)
class CountQualityVerdict:
    decision: str
    observed_drop: int
    regression_threshold_drop: int
    evidence_digest: str
    policy_digest: str

    def __post_init__(self) -> None:
        if self.decision not in COUNT_QUALITY_DECISIONS:
            raise CountQualityError("count quality decision is invalid")
        object.__setattr__(
            self,
            "observed_drop",
            _integer(self.observed_drop, "observed drop", minimum=0, maximum=_MAX_TOTAL),
        )
        object.__setattr__(
            self,
            "regression_threshold_drop",
            _integer(
                self.regression_threshold_drop,
                "regression threshold drop",
                minimum=1,
                maximum=_MAX_TOTAL,
            ),
        )
        for field in ("evidence_digest", "policy_digest"):
            object.__setattr__(self, field, _digest(getattr(self, field), field))
        expected = "FAIL" if self.observed_drop >= self.regression_threshold_drop else "PASS"
        if self.decision != expected:
            raise CountQualityError("count quality decision disagrees with policy arithmetic")

    def to_dict(self) -> dict[str, object]:
        return {
            "decision": self.decision,
            "evidence_digest": self.evidence_digest,
            "observed_drop": self.observed_drop,
            "policy_digest": self.policy_digest,
            "regression_threshold_drop": self.regression_threshold_drop,
        }

    @classmethod
    def from_dict(cls, value: object) -> "CountQualityVerdict":
        raw = _strict(
            value,
            frozenset(
                {
                    "decision",
                    "evidence_digest",
                    "observed_drop",
                    "policy_digest",
                    "regression_threshold_drop",
                }
            ),
            "count quality verdict",
        )
        return cls(**raw)

    @property
    def digest(self) -> str:
        return canonical_digest(COUNT_QUALITY_VERDICT_SCHEMA, self.to_dict())


def score_count_quality(
    evidence: CountQualityEvidence,
    policy: CountQualityPolicy,
) -> CountQualityVerdict:
    """Regrade retained count evidence under one explicit frozen policy."""

    if type(evidence) is not CountQualityEvidence or type(policy) is not CountQualityPolicy:
        raise CountQualityError("count quality evidence and policy must be typed")
    observed_drop = max(0, evidence.stock_correct - evidence.candidate_correct)
    decision = "FAIL" if observed_drop >= policy.regression_threshold_drop else "PASS"
    return CountQualityVerdict(
        decision=decision,
        observed_drop=observed_drop,
        regression_threshold_drop=policy.regression_threshold_drop,
        evidence_digest=evidence.digest,
        policy_digest=policy.digest,
    )


@dataclass(frozen=True)
class CountQualityRegrade:
    """Append-only link from an immutable source result to a recomputed verdict."""

    source_result_digest: str
    evidence: CountQualityEvidence
    policy: CountQualityPolicy
    verdict: CountQualityVerdict

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_result_digest",
            _digest(self.source_result_digest, "source result digest"),
        )
        if (
            type(self.evidence) is not CountQualityEvidence
            or type(self.policy) is not CountQualityPolicy
            or type(self.verdict) is not CountQualityVerdict
        ):
            raise CountQualityError("count quality regrade fields must be typed")
        if self.verdict != score_count_quality(self.evidence, self.policy):
            raise CountQualityError("count quality regrade verdict does not recompute")

    def to_dict(self) -> dict[str, object]:
        return {
            "evidence": self.evidence.to_dict(),
            "policy": self.policy.to_dict(),
            "source_result_digest": self.source_result_digest,
            "verdict": self.verdict.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: object) -> "CountQualityRegrade":
        raw = _strict(
            value,
            frozenset({"evidence", "policy", "source_result_digest", "verdict"}),
            "count quality regrade",
        )
        return cls(
            source_result_digest=raw["source_result_digest"],
            evidence=CountQualityEvidence.from_dict(raw["evidence"]),
            policy=CountQualityPolicy.from_dict(raw["policy"]),
            verdict=CountQualityVerdict.from_dict(raw["verdict"]),
        )

    @property
    def digest(self) -> str:
        return canonical_digest(COUNT_QUALITY_REGRADE_SCHEMA, self.to_dict())


__all__ = [
    "COUNT_QUALITY_DECISIONS",
    "COUNT_QUALITY_EVIDENCE_SCHEMA",
    "COUNT_QUALITY_POLICY_SCHEMA",
    "COUNT_QUALITY_REGRADE_SCHEMA",
    "COUNT_QUALITY_VERDICT_SCHEMA",
    "CountQualityError",
    "CountQualityEvidence",
    "CountQualityPolicy",
    "CountQualityRegrade",
    "CountQualityVerdict",
    "score_count_quality",
]
