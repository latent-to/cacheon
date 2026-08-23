"""A rejected screen must be able to say which gate stopped it, and why."""

from __future__ import annotations

import re
from pathlib import Path

from cacheon.eval.screen_reason import (
    SCREEN_EVIDENCE_SCHEMA,
    STAGE_REASONS,
    recover_screen_reason,
)
from cacheon.stack_identity import canonical_digest

_AUTHORITY = "a" * 64
_CANDIDATE = "b" * 64
_PUBLICATION = "c" * 64
_SERVICE = "d" * 64


def _digest(stage: str, grade: str, reason: str, facts: dict[str, str], attempt=2):
    return canonical_digest(
        SCREEN_EVIDENCE_SCHEMA,
        {
            "authority_digest": _AUTHORITY,
            "candidate_digest": _CANDIDATE,
            "facts": dict(sorted(facts.items())),
            "grade": grade,
            "publication_digest": _PUBLICATION,
            "reason": reason,
            "screen_attempt": attempt,
            "service_digest": _SERVICE,
            "stage": stage,
        },
    )


def _recover(stage: str, grade: str, digest: str, *, slots=(), attempt=2):
    return recover_screen_reason(
        stage=stage,
        grade=grade,
        evidence_digest=digest,
        authority_digest=_AUTHORITY,
        candidate_digest=_CANDIDATE,
        publication_digest=_PUBLICATION,
        service_digest=_SERVICE,
        screen_attempt=attempt,
        slots=slots,
    )


def test_the_gate_that_rejected_a_bundle_names_itself_and_its_exception():
    digest = _digest(
        "static", "fail", "static_policy", {"exception_type": "_CandidateStaticFailure"}
    )

    found = _recover("static", "fail", digest)

    assert found is not None
    assert (found.stage, found.reason) == ("static", "static_policy")
    assert found.facts == (("exception_type", "_CandidateStaticFailure"),)
    assert found.sentence() == (
        "static: static_policy (exception_type _CandidateStaticFailure)"
    )


def test_a_quant_mismatch_names_the_slot_and_the_quantization_it_needed():
    digest = _digest(
        "static",
        "fail",
        "static_runtime_quant_mismatch",
        {"required_quant": "nvfp4", "slot": "moe.fused_experts"},
    )

    found = _recover("static", "fail", digest, slots=("moe.fused_experts",))

    assert found is not None
    assert found.sentence() == (
        "static: static_runtime_quant_mismatch "
        "(required_quant nvfp4, slot moe.fused_experts)"
    )


def test_every_stage_with_a_vocabulary_recovers_an_infrastructure_fault():
    for stage in ("build", "abi", "graph"):
        digest = _digest(stage, "no_decision", f"{stage}_infrastructure", {"exception_type": "OSError"})

        found = _recover(stage, "no_decision", digest)

        assert found is not None, stage
        assert found.reason == f"{stage}_infrastructure"
        assert found.facts == (("exception_type", "OSError"),)


def test_a_reason_it_cannot_explain_is_absent_rather_than_guessed():
    """No match must never become a confident wrong answer."""

    digest = _digest("static", "fail", "some_reason_nobody_registered", {})

    assert _recover("static", "fail", digest) is None
    # A stage with no vocabulary at all is absent, not an error.
    assert _recover("abbreviated_serving", "fail", digest) is None


def test_recovery_is_bound_to_the_exact_attempt_and_candidate():
    """The digest commits to the whole payload, so a near miss must not match."""

    digest = _digest("static", "fail", "static_policy", {"exception_type": "OSError"})

    assert _recover("static", "fail", digest, attempt=3) is None
    assert _recover("static", "no_decision", digest) is None


def test_the_reason_table_still_covers_what_the_stages_actually_emit():
    """A stage that adds a reason without updating the table is caught here.

    The table is a transcription of call sites in another module, so it goes
    stale silently. This walks the source and fails when it does.
    """

    source = (
        Path(__file__).resolve().parents[1]
        / "cacheon"
        / "eval"
        / "b300_screen_stages.py"
    ).read_text(encoding="utf-8")
    emitted = set(re.findall(r'reason=\("?([a-z_]+)"', source))
    emitted |= set(re.findall(r'else "([a-z_]+)"\),', source))
    emitted |= set(re.findall(r'^\s+"([a-z]+_infrastructure)",$', source, re.MULTILINE))

    known = {reason for reasons in STAGE_REASONS.values() for reason in reasons}
    # Verified reasons carry digests of what the stage built and are outside
    # what enumeration can reach; they are also never the reason a bundle died.
    unexplained = {name for name in emitted - known if not name.endswith("_verified")}

    assert not unexplained, f"screen reasons missing from STAGE_REASONS: {sorted(unexplained)}"
