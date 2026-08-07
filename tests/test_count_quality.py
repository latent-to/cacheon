from __future__ import annotations

from copy import deepcopy

import pytest

from cacheon.eval.count_quality import (
    REGRADE_SCHEMA,
    CountQualityError,
    CountQualityEvidence,
    CountQualityPolicy,
    grade_count_quality,
    regrade_count_quality,
    reopen_count_quality_regrade,
)
from cacheon.stack_identity import canonical_digest, canonical_json_bytes


def _d(label: str) -> str:
    import hashlib

    return hashlib.sha256(label.encode()).hexdigest()


def _sealed_policy() -> CountQualityPolicy:
    return CountQualityPolicy(regression_threshold_drop=10)


_SEALED_REGRADE_PAYLOAD = {
    "evidence": {
        "candidate_correct": 59,
        "candidate_observation_digest": (
            "7c33d41d4b04c3f2b90fd4c8a29ebec40bdd233f43e8f86b048b96a157aa92fb"
        ),
        "stock_correct": 62,
        "stock_observation_digest": (
            "72ed69ad365c75c3670d76f7e1aba09c1658b1f6d3621013027d8433838071d9"
        ),
        "total": 64,
    },
    "policy": {"regression_threshold_drop": 10},
    "regrade_digest": (
        "ba603eff284454fa28d3423ac89113c513609f5968d309d72fd021edaf49f42e"
    ),
    "schema": "cacheon.count-quality-regrade.v1",
    "source_result_digest": (
        "8d4be372521845954530dcc28c50abb8a485f8924447c8a33737041438edeb9c"
    ),
    "verdict": {
        "decision": "PASS",
        "evidence_digest": (
            "dde39ab907439cd6960136b2ab31ac17142e13611416deefcac856f8310f3a10"
        ),
        "observed_drop": 3,
        "policy_digest": (
            "ed19caeaf875ef7489f9c152ce8756957c99f0b059858723ccda0fd40ac491cb"
        ),
        "regression_threshold_drop": 10,
    },
}


def test_sealed_policy_digest_matches_registered_wire_policy() -> None:
    assert _sealed_policy().digest == (
        "ed19caeaf875ef7489f9c152ce8756957c99f0b059858723ccda0fd40ac491cb"
    )


def test_sealed_evidence_digest_and_pass_verdict() -> None:
    evidence = CountQualityEvidence(
        stock_correct=62,
        candidate_correct=59,
        total=64,
        stock_observation_digest=(
            "72ed69ad365c75c3670d76f7e1aba09c1658b1f6d3621013027d8433838071d9"
        ),
        candidate_observation_digest=(
            "7c33d41d4b04c3f2b90fd4c8a29ebec40bdd233f43e8f86b048b96a157aa92fb"
        ),
    )
    assert evidence.observed_drop == 3
    assert evidence.digest == (
        "dde39ab907439cd6960136b2ab31ac17142e13611416deefcac856f8310f3a10"
    )
    verdict = grade_count_quality(evidence, _sealed_policy())
    assert verdict.decision == "PASS"
    assert verdict.observed_drop == 3


def test_sealed_regrade_receipt_round_trips_byte_for_byte() -> None:
    payload = deepcopy(_SEALED_REGRADE_PAYLOAD)
    regrade = reopen_count_quality_regrade(payload)
    assert regrade.digest == (
        "ba603eff284454fa28d3423ac89113c513609f5968d309d72fd021edaf49f42e"
    )
    assert regrade.verdict.decision == "PASS"
    assert regrade.canonical_bytes() == canonical_json_bytes(payload)
    assert regrade.to_dict() == payload


def test_regrade_preserves_source_result_without_observation_callbacks() -> None:
    evidence = CountQualityEvidence(
        62,
        59,
        64,
        "72ed69ad365c75c3670d76f7e1aba09c1658b1f6d3621013027d8433838071d9",
        "7c33d41d4b04c3f2b90fd4c8a29ebec40bdd233f43e8f86b048b96a157aa92fb",
    )
    source = "8d4be372521845954530dcc28c50abb8a485f8924447c8a33737041438edeb9c"
    regrade = regrade_count_quality(
        source_result_digest=source,
        policy=_sealed_policy(),
        evidence=evidence,
    )
    assert regrade.source_result_digest == source
    # Regrade is pure over sealed counts — no observation callbacks exist.
    assert regrade.to_dict()["schema"] == REGRADE_SCHEMA


def test_drop_at_threshold_fails() -> None:
    evidence = CountQualityEvidence(60, 50, 64, _d("stock"), _d("candidate"))
    verdict = grade_count_quality(evidence, _sealed_policy())
    assert evidence.observed_drop == 10
    assert verdict.decision == "FAIL"


def test_drop_just_under_threshold_passes() -> None:
    evidence = CountQualityEvidence(60, 51, 64, _d("stock"), _d("candidate"))
    assert grade_count_quality(evidence, _sealed_policy()).decision == "PASS"


def test_candidate_above_stock_passes() -> None:
    evidence = CountQualityEvidence(50, 55, 64, _d("stock"), _d("candidate"))
    verdict = grade_count_quality(evidence, _sealed_policy())
    assert verdict.observed_drop == -5
    assert verdict.decision == "PASS"


def test_policy_rejects_non_positive_threshold() -> None:
    with pytest.raises(CountQualityError, match="positive int"):
        CountQualityPolicy(0)
    with pytest.raises(CountQualityError, match="positive int"):
        CountQualityPolicy(True)  # type: ignore[arg-type]


def test_evidence_rejects_counts_above_total_or_bad_digests() -> None:
    with pytest.raises(CountQualityError, match="exceed total"):
        CountQualityEvidence(65, 1, 64, _d("s"), _d("c"))
    with pytest.raises(CountQualityError):
        CountQualityEvidence(1, 1, 64, "not-a-digest", _d("c"))


def test_reopen_rejects_tampered_regrade_digest() -> None:
    payload = deepcopy(_SEALED_REGRADE_PAYLOAD)
    payload["regrade_digest"] = _d("tampered")
    with pytest.raises(CountQualityError, match="does not match"):
        reopen_count_quality_regrade(payload)


def test_reopen_rejects_foreign_schema_or_open_fields() -> None:
    payload = deepcopy(_SEALED_REGRADE_PAYLOAD)
    payload["schema"] = "cacheon.count-quality-regrade.v0"
    with pytest.raises(CountQualityError, match="not closed"):
        reopen_count_quality_regrade(payload)
    payload = deepcopy(_SEALED_REGRADE_PAYLOAD)
    payload["injected"] = 1
    with pytest.raises(CountQualityError, match="not closed"):
        reopen_count_quality_regrade(payload)


def test_regrade_is_not_settlement_or_weight_authority() -> None:
    payload = deepcopy(_SEALED_REGRADE_PAYLOAD)
    keys = set(payload) | set(payload["verdict"]) | set(payload["evidence"])
    forbidden = {"settlement", "weight", "incentive", "crown", "award", "reservation"}
    assert not {key for key in keys if any(marker in key for marker in forbidden)}
    # Digest domains stay generic — no target/reservation identity in the domain.
    assert canonical_digest(
        "cacheon.count-quality-policy.v1", {"regression_threshold_drop": 10}
    ) == _sealed_policy().digest
