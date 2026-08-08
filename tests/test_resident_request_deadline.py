"""Absolute-wall contracts shared by resident speed and count quality."""

from __future__ import annotations

import threading
import time
from dataclasses import replace

import pytest

from cacheon.eval.resident_count_quality_execution import (
    ResidentCountQualityExecutionError,
    ResidentCountQualityExecutionHold,
    execute_candidate_count_quality,
)
from cacheon.eval.resident_evaluation_pair import (
    ResidentEvaluationPairError,
    ResidentLaneRequest,
)
from cacheon.eval.resident_request_deadline import (
    ResidentRequestDeadlineError,
    require_resident_request_deadline,
    resolve_resident_request_deadline,
)
from tests.test_resident_count_quality_execution import _fixture as quality_fixture
from tests.test_resident_evaluation_pair import DIGEST_A, DIGEST_B, _pair


@pytest.mark.parametrize(
    "invalid", (True, "12", 10**400, float("nan"), float("inf"))
)
def test_deadline_helper_rejects_inexact_or_nonfinite_outer_wall(invalid) -> None:
    with pytest.raises(ResidentRequestDeadlineError, match="deadline is invalid"):
        resolve_resident_request_deadline(100.0, 10.0, invalid)


def test_deadline_helper_rejects_expiry_and_intersects_configured_wall() -> None:
    with pytest.raises(ResidentRequestDeadlineError, match="has expired"):
        require_resident_request_deadline(100, now=100.0)
    with pytest.raises(ResidentRequestDeadlineError, match="has expired"):
        resolve_resident_request_deadline(100.0, 10.0, 99.0)

    assert resolve_resident_request_deadline(100.0, 10.0, None) == 110.0
    assert resolve_resident_request_deadline(100.0, 10.0, 107) == 107.0
    assert resolve_resident_request_deadline(100.0, 10.0, 120.0) == 110.0


def test_run_lane_and_run_lanes_use_one_outer_bounded_wall(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pair, _, _ = _pair()
    monkeypatch.setattr(pair, "_clock", lambda: 100.0)
    admitted: list[float] = []
    original_accept = pair._accept  # type: ignore[attr-defined]
    original_accept_pair = pair._accept_pair  # type: ignore[attr-defined]

    def observe_one(lane, work):
        admitted.append(work.deadline)
        return original_accept(lane, work)

    def observe_pair(work_a, work_b):
        admitted.extend((work_a.deadline, work_b.deadline))
        return original_accept_pair(work_a, work_b)

    monkeypatch.setattr(pair, "_accept", observe_one)
    monkeypatch.setattr(pair, "_accept_pair", observe_pair)

    one = pair.run_lane(
        "A",
        DIGEST_A,
        lambda handle: handle.identity,
        expected_batch_count=0,
        expected_swap_count=0,
        deadline=103.0,
    )
    both = pair.run_lanes(
        ResidentLaneRequest(DIGEST_A, lambda handle: handle.identity, 0, 0),
        ResidentLaneRequest(DIGEST_B, lambda handle: handle.identity, 0, 0),
        deadline=110.0,
    )

    assert one.ok and all(result.ok for result in both)
    assert admitted == [103.0, 105.0, 105.0]
    pair.close()


def test_expired_paired_wall_admits_nothing_and_executes_no_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pair, _, _ = _pair()
    monkeypatch.setattr(pair, "_clock", lambda: 100.0)
    invoked: list[str] = []
    admissions = 0

    def forbidden(*_args) -> None:
        nonlocal admissions
        admissions += 1

    monkeypatch.setattr(pair, "_accept_pair", forbidden)

    with pytest.raises(ResidentEvaluationPairError, match="has expired"):
        pair.run_lanes(
            ResidentLaneRequest(DIGEST_A, lambda _handle: invoked.append("A"), 0, 0),
            ResidentLaneRequest(DIGEST_B, lambda _handle: invoked.append("B"), 0, 0),
            deadline=100.0,
        )

    assert invoked == []
    assert admissions == 0
    assert pair.request_history == ()
    assert pair.fatal_error is None
    pair.close()


def test_legacy_pair_call_without_outer_wall_still_uses_configured_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pair, _, _ = _pair()
    monkeypatch.setattr(pair, "_clock", lambda: 100.0)
    admitted: list[float] = []
    original = pair._accept_pair  # type: ignore[attr-defined]

    def observe(work_a, work_b):
        admitted.extend((work_a.deadline, work_b.deadline))
        return original(work_a, work_b)

    monkeypatch.setattr(pair, "_accept_pair", observe)
    results = pair.run_lanes(
        ResidentLaneRequest(DIGEST_A, lambda handle: handle.identity, 0, 0),
        ResidentLaneRequest(DIGEST_B, lambda handle: handle.identity, 0, 0),
    )

    assert all(result.ok for result in results)
    assert admitted == [105.0, 105.0]
    pair.close()


def test_candidate_quality_requires_and_forwards_exact_wall_for_two_digests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base, judge, pair, _, _ = quality_fixture(4, barrier=False)
    forwarded: list[float] = []
    original = pair.run_lanes

    def observe(lane_a, lane_b, *, deadline=None):
        forwarded.append(deadline)
        return original(lane_a, lane_b, deadline=deadline)

    monkeypatch.setattr(pair, "run_lanes", observe)
    walls = (time.monotonic() + 30.0, time.monotonic() + 31.0)
    plans = (
        replace(base, candidate_bundle_digest=DIGEST_A),
        replace(base, candidate_bundle_digest=DIGEST_B),
    )
    observations = tuple(
        execute_candidate_count_quality(
            plan, pair=pair, judge=judge, deadline=wall
        )
        for plan, wall in zip(plans, walls, strict=True)
    )

    assert forwarded == list(walls)
    assert (
        observations[0].execution_evidence_digest
        != observations[1].execution_evidence_digest
    )
    pair.close()


def test_candidate_expired_wall_is_api_error_before_pair_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, judge, pair, _, _ = quality_fixture(4, barrier=False)
    calls = 0

    def forbidden(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("pair admission must not run")

    monkeypatch.setattr(pair, "run_lanes", forbidden)
    with pytest.raises(TypeError, match="deadline"):
        execute_candidate_count_quality(plan, pair=pair, judge=judge)  # type: ignore[call-arg]
    with pytest.raises(ResidentCountQualityExecutionError, match="has expired"):
        execute_candidate_count_quality(
            plan, pair=pair, judge=judge, deadline=time.monotonic()
        )
    assert calls == 0
    pair.close()


def test_candidate_timeout_after_pair_admission_is_typed_hold() -> None:
    plan, judge, pair, factory_a, factory_b = quality_fixture(4, barrier=False)
    release = threading.Event()
    entered = (threading.Event(), threading.Event())

    for event, session in zip(
        entered, (factory_a.sessions[0], factory_b.sessions[0]), strict=True
    ):
        original = session.execute_batch_with_shape

        def blocked(prompts, *, shape, canary=False, event=event, original=original):
            event.set()
            assert release.wait(2.0)
            return original(prompts, shape=shape, canary=canary)

        session.execute_batch_with_shape = blocked

    with pytest.raises(ResidentCountQualityExecutionHold, match="timed out") as raised:
        execute_candidate_count_quality(
            plan, pair=pair, judge=judge, deadline=time.monotonic() + 0.1
        )
    assert raised.value.decision == "HOLD"
    assert all(event.is_set() for event in entered)
    release.set()
    pair.close()
