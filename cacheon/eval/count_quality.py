"""Append-only count-quality policy and causal regrade (handoff aug6-2-1 §6).

Generic first-failing-drop arithmetic over sealed stock/candidate observation
counts.  No target, reservation, mission path, or score is hardcoded.  A
regrade preserves the old quality product as history; it never reruns the
observation workload
and never awards or publishes settlement/incentive/weight authority by itself.
"""

from __future__ import annotations

from dataclasses import dataclass

from cacheon.stack_identity import (
    canonical_digest,
    canonical_json_bytes,
    require_sha256_hex,
)


REGRADE_SCHEMA = "cacheon.count-quality-regrade.v1"
_POLICY_DOMAIN = "cacheon.count-quality-policy.v1"
_EVIDENCE_DOMAIN = "cacheon.count-quality-evidence.v1"
_REGRADE_DOMAIN = "cacheon.count-quality-regrade.v1"
_DECISIONS = frozenset({"PASS", "FAIL"})


class CountQualityError(ValueError):
    """Count-quality policy, evidence, or regrade is malformed."""


@dataclass(frozen=True)
class CountQualityPolicy:
    """Closed drop-threshold policy. Digests exclude ambient identity."""

    regression_threshold_drop: int

    def __post_init__(self) -> None:
        if (
            type(self.regression_threshold_drop) is not int
            or isinstance(self.regression_threshold_drop, bool)
            or self.regression_threshold_drop < 1
        ):
            raise CountQualityError("regression threshold drop must be a positive int")

    def to_dict(self) -> dict[str, object]:
        return {"regression_threshold_drop": self.regression_threshold_drop}

    @property
    def digest(self) -> str:
        return canonical_digest(_POLICY_DOMAIN, self.to_dict())


@dataclass(frozen=True)
class CountQualityEvidence:
    """Sealed stock/candidate correct-counts bound to observation digests."""

    stock_correct: int
    candidate_correct: int
    total: int
    stock_observation_digest: str
    candidate_observation_digest: str

    def __post_init__(self) -> None:
        for name in ("stock_correct", "candidate_correct", "total"):
            value = getattr(self, name)
            if type(value) is not int or isinstance(value, bool) or value < 0:
                raise CountQualityError(f"{name} must be a nonnegative int")
        if self.total < 1:
            raise CountQualityError("total must be a positive int")
        if self.stock_correct > self.total or self.candidate_correct > self.total:
            raise CountQualityError("correct counts exceed total")
        try:
            object.__setattr__(
                self,
                "stock_observation_digest",
                require_sha256_hex(
                    self.stock_observation_digest, field="stock observation digest"
                ),
            )
            object.__setattr__(
                self,
                "candidate_observation_digest",
                require_sha256_hex(
                    self.candidate_observation_digest,
                    field="candidate observation digest",
                ),
            )
        except ValueError as exc:
            raise CountQualityError(str(exc)) from None

    @property
    def observed_drop(self) -> int:
        """First-failing drop: stock minus candidate (never negative for PASS path)."""

        return self.stock_correct - self.candidate_correct

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_correct": self.candidate_correct,
            "candidate_observation_digest": self.candidate_observation_digest,
            "stock_correct": self.stock_correct,
            "stock_observation_digest": self.stock_observation_digest,
            "total": self.total,
        }

    @property
    def digest(self) -> str:
        return canonical_digest(_EVIDENCE_DOMAIN, self.to_dict())


@dataclass(frozen=True)
class CountQualityVerdict:
    """Policy regrade of sealed counts. Not settlement or weight authority."""

    decision: str
    observed_drop: int
    regression_threshold_drop: int
    policy_digest: str
    evidence_digest: str

    def __post_init__(self) -> None:
        if self.decision not in _DECISIONS:
            raise CountQualityError("count-quality decision is not closed")
        if (
            type(self.observed_drop) is not int
            or isinstance(self.observed_drop, bool)
            or type(self.regression_threshold_drop) is not int
            or isinstance(self.regression_threshold_drop, bool)
            or self.regression_threshold_drop < 1
        ):
            raise CountQualityError("count-quality verdict numbers are malformed")
        try:
            object.__setattr__(
                self,
                "policy_digest",
                require_sha256_hex(self.policy_digest, field="policy digest"),
            )
            object.__setattr__(
                self,
                "evidence_digest",
                require_sha256_hex(self.evidence_digest, field="evidence digest"),
            )
        except ValueError as exc:
            raise CountQualityError(str(exc)) from None

    def to_dict(self) -> dict[str, object]:
        return {
            "decision": self.decision,
            "evidence_digest": self.evidence_digest,
            "observed_drop": self.observed_drop,
            "policy_digest": self.policy_digest,
            "regression_threshold_drop": self.regression_threshold_drop,
        }


@dataclass(frozen=True)
class CountQualityRegrade:
    """Append-only causal regrade envelope bound to one source result digest."""

    source_result_digest: str
    policy: CountQualityPolicy
    evidence: CountQualityEvidence
    verdict: CountQualityVerdict

    def __post_init__(self) -> None:
        try:
            object.__setattr__(
                self,
                "source_result_digest",
                require_sha256_hex(
                    self.source_result_digest, field="source result digest"
                ),
            )
        except ValueError as exc:
            raise CountQualityError(str(exc)) from None
        if type(self.policy) is not CountQualityPolicy:
            raise CountQualityError("regrade policy is not exactly typed")
        if type(self.evidence) is not CountQualityEvidence:
            raise CountQualityError("regrade evidence is not exactly typed")
        if type(self.verdict) is not CountQualityVerdict:
            raise CountQualityError("regrade verdict is not exactly typed")
        expected = grade_count_quality(self.evidence, self.policy)
        if self.verdict != expected:
            raise CountQualityError("regrade verdict does not match sealed evidence")

    def to_dict(self) -> dict[str, object]:
        return {
            "evidence": self.evidence.to_dict(),
            "policy": self.policy.to_dict(),
            "regrade_digest": self.digest,
            "schema": REGRADE_SCHEMA,
            "source_result_digest": self.source_result_digest,
            "verdict": self.verdict.to_dict(),
        }

    @property
    def digest(self) -> str:
        # The registered wire authority omits schema from the digest body.
        return canonical_digest(
            _REGRADE_DOMAIN,
            {
                "evidence": self.evidence.to_dict(),
                "policy": self.policy.to_dict(),
                "source_result_digest": self.source_result_digest,
                "verdict": self.verdict.to_dict(),
            },
        )

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())


def grade_count_quality(
    evidence: CountQualityEvidence, policy: CountQualityPolicy
) -> CountQualityVerdict:
    """Grade sealed counts under the supplied drop-threshold policy."""

    if type(evidence) is not CountQualityEvidence:
        raise CountQualityError("evidence is not exactly typed")
    if type(policy) is not CountQualityPolicy:
        raise CountQualityError("policy is not exactly typed")
    observed = evidence.observed_drop
    # Reject only when the candidate falls threshold-or-more below stock.
    decision = "FAIL" if observed >= policy.regression_threshold_drop else "PASS"
    return CountQualityVerdict(
        decision=decision,
        observed_drop=observed,
        regression_threshold_drop=policy.regression_threshold_drop,
        policy_digest=policy.digest,
        evidence_digest=evidence.digest,
    )


def regrade_count_quality(
    *,
    source_result_digest: str,
    policy: CountQualityPolicy,
    evidence: CountQualityEvidence,
) -> CountQualityRegrade:
    """Build one append-only regrade; preserves source result identity only."""

    return CountQualityRegrade(
        source_result_digest=source_result_digest,
        policy=policy,
        evidence=evidence,
        verdict=grade_count_quality(evidence, policy),
    )


def reopen_count_quality_regrade(payload: object) -> CountQualityRegrade:
    """Reopen one closed regrade receipt; fail closed on any drift."""

    if type(payload) is not dict:
        raise CountQualityError("count-quality regrade payload is not an object")
    expected_keys = {
        "evidence",
        "policy",
        "regrade_digest",
        "schema",
        "source_result_digest",
        "verdict",
    }
    if set(payload) != expected_keys or payload.get("schema") != REGRADE_SCHEMA:
        raise CountQualityError("count-quality regrade payload is not closed")
    policy_raw = payload["policy"]
    evidence_raw = payload["evidence"]
    if type(policy_raw) is not dict or set(policy_raw) != {"regression_threshold_drop"}:
        raise CountQualityError("count-quality policy payload is not closed")
    evidence_keys = {
        "candidate_correct",
        "candidate_observation_digest",
        "stock_correct",
        "stock_observation_digest",
        "total",
    }
    if type(evidence_raw) is not dict or set(evidence_raw) != evidence_keys:
        raise CountQualityError("count-quality evidence payload is not closed")
    policy = CountQualityPolicy(
        regression_threshold_drop=policy_raw["regression_threshold_drop"]
    )
    evidence = CountQualityEvidence(
        stock_correct=evidence_raw["stock_correct"],
        candidate_correct=evidence_raw["candidate_correct"],
        total=evidence_raw["total"],
        stock_observation_digest=evidence_raw["stock_observation_digest"],
        candidate_observation_digest=evidence_raw["candidate_observation_digest"],
    )
    regrade = regrade_count_quality(
        source_result_digest=payload["source_result_digest"],
        policy=policy,
        evidence=evidence,
    )
    if (
        regrade.digest != payload["regrade_digest"]
        or regrade.verdict.to_dict() != payload["verdict"]
        or regrade.canonical_bytes() != canonical_json_bytes(payload)
    ):
        raise CountQualityError("count-quality regrade receipt does not match digests")
    return regrade


__all__ = [
    "CountQualityError",
    "CountQualityEvidence",
    "CountQualityPolicy",
    "CountQualityRegrade",
    "CountQualityVerdict",
    "REGRADE_SCHEMA",
    "grade_count_quality",
    "regrade_count_quality",
    "reopen_count_quality_regrade",
]
