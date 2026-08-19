"""Replaying a terminal verdict onto byte-identical resubmissions.

Grounded in mainnet data from 2026-08-18: across all 330 lifetime reservations,
23 groups of byte-identical bundles were evaluated more than once under the same
arena digest, and in zero of them did the decision or the reason differ. These
cases pin the conditions under which that observation may be relied on.
"""

from cacheon.chain.duplicate_replay import (
    PriorVerdict,
    decide_replay,
    replay_reason,
)

ARENA = "544b9007119f6409a0c6270506f26c4f77be65ef74d10ec3d8123359c02c01af"
OTHER_ARENA = "a642998323ef1ebfd5ff" + "0" * 44
BYTES_A = "eb124cd83e75364d4716bb53831e90e83a3c996cfce5d14f86105083f9dba4d1"

FAILED = PriorVerdict(
    reservation_id="0ce5ee9716984f05c9732e88d5072023",
    content_hash=BYTES_A,
    arena_service_digest=ARENA,
    decision="FAIL",
    reason="speed_regression",
)


def _decide(**kw):
    base = dict(
        content_hash=BYTES_A,
        arena_service_digest=ARENA,
        screen_lane="primary",
        priors=(FAILED,),
    )
    return decide_replay(**{**base, **kw})


def test_identical_bytes_same_arena_replays_the_fail():
    d = _decide()
    assert d.replay
    assert d.prior_reservation_id == FAILED.reservation_id
    assert "speed_regression" in d.reason


def test_reason_carries_source_and_original_cause():
    assert replay_reason(FAILED) == "duplicate_of:0ce5ee9716984f05:speed_regression"


def test_elapsed_time_is_not_a_condition():
    """Five days or fifty: the rule keys on arena identity, never on time.

    There is deliberately no clock in this module. If a test ever needs to pass
    one in, the design has drifted.
    """
    assert _decide().replay


def test_a_different_arena_must_re_measure():
    """The whole safety argument. A speedup describes one baseline."""
    assert not _decide(arena_service_digest=OTHER_ARENA).replay


def test_different_bytes_never_replay():
    assert not _decide(content_hash="f" * 64).replay


def test_a_pass_is_refused_not_replayed():
    """A first PASS is reproduction_pending; settlement needs a bound pair.

    Replaying it would manufacture the second observation from the first.
    """
    passed = PriorVerdict(
        reservation_id="a07740c3" + "0" * 24,
        content_hash=BYTES_A,
        arena_service_digest=ARENA,
        decision="PASS",
        reason="qualified",
    )
    d = _decide(priors=(passed,))
    assert not d.replay
    assert "independently bound" in d.refused
    assert d.prior_reservation_id == passed.reservation_id


def test_reproduction_lane_is_never_replayed():
    """A reproduction exists to be an independent second measurement."""
    d = _decide(screen_lane="reproduction")
    assert not d.replay
    assert "independent measurement" in d.refused


def test_a_candidate_without_arena_identity_is_not_replayed():
    """107 mainnet rows carry an empty arena digest -- never measured.

    They must never become a replay source or a replay target.
    """
    assert not _decide(arena_service_digest="").replay
    assert not _decide(content_hash="").replay


def test_prior_without_arena_identity_is_ignored():
    unmeasured = PriorVerdict(
        reservation_id="dead" + "0" * 28,
        content_hash=BYTES_A,
        arena_service_digest="",
        decision="FAIL",
        reason="candidate_kernel_does_not_compile",
    )
    assert not _decide(priors=(unmeasured,)).replay


def test_no_priors_at_all_is_a_clean_miss():
    assert not _decide(priors=()).replay
