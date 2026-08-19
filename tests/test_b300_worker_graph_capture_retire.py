"""Planning-stage graph-capture starvation must retire a released pair.

2026-08-14, epoch 8afcf1ed: a resident pair left loaded by a completed
qualification starved every later target-switch capture.  The retire-and-
retry existed only around ``work.factory.build()``, but the capture hold
surfaces one stage earlier, in ``plan_qualification`` — so the retire never
fired, three attempts burned per reservation, and the hold breaker parked
the lane.  These tests pin the planning-stage retire path.
"""

from __future__ import annotations

from pathlib import Path

import cacheon.eval.b300_mainnet_worker as worker_module
import tests.test_b300_mainnet_worker as worker_tests
from cacheon.chain.remote_qualification_hold import RemoteQualificationHoldReason
from cacheon.eval.b300_mainnet_worker import (
    B300MainnetWorker,
    B300QualificationGraphGateHold,
)
from cacheon.eval.qualification_continuation import QualificationContinuationStore

executor_factory = worker_tests.executor_factory


def _worker_under_test(tmp_path: Path, executor_factory):
    authorities, _resident, _builder = worker_tests._authorities(
        tmp_path, executor_factory
    )
    manifest = worker_tests._manifest(authorities)
    readiness = worker_tests._readiness(manifest, authorities)
    claim = worker_tests._qualification_claim(tmp_path / "cohort", manifest)
    continuation = QualificationContinuationStore(tmp_path / "continuation")
    worker = B300MainnetWorker(manifest, authorities, readiness)
    return worker, claim, continuation


def test_planning_hold_retires_released_pair_and_replans(
    tmp_path: Path,
    executor_factory,
    monkeypatch,
) -> None:
    worker, claim, continuation = _worker_under_test(tmp_path, executor_factory)
    request_digest = worker_tests._h("remote-request")

    original_plan = worker.service.plan_qualification
    plan_calls: list[int] = []

    def flaky_plan(candidates, screen_receipts, *, state=None):
        plan_calls.append(1)
        if len(plan_calls) == 1:
            raise worker_module.B300QualificationGraphEvidenceHold(
                "capture devices busy"
            )
        return original_plan(candidates, screen_receipts, state=state)

    monkeypatch.setattr(worker.service, "plan_qualification", flaky_plan)
    retires: list[bool] = []
    monkeypatch.setattr(
        worker._resident_pair_factory,
        "retire_released_pair",
        lambda: retires.append(True) or True,
        raising=False,
    )
    try:
        result = worker.run_remote_qualification(
            claim.lease,
            claim.candidates,
            claim.screen_receipts,
            screen_lane="primary",
            continuation_store=continuation,
            request_digest=request_digest,
        )

        # Planning succeeded on the post-retirement retry; the run then held
        # at the unbound graph-gate root, which is downstream of planning.
        assert type(result) is B300QualificationGraphGateHold
        assert (
            result.reason
            is RemoteQualificationHoldReason.GRAPH_EVIDENCE_UNAVAILABLE
        )
        assert len(plan_calls) == 2
        assert retires == [True]
    finally:
        worker.close()


def test_planning_hold_without_retirable_pair_holds_without_replanning(
    tmp_path: Path,
    executor_factory,
    monkeypatch,
) -> None:
    worker, claim, continuation = _worker_under_test(tmp_path, executor_factory)
    request_digest = worker_tests._h("remote-request")

    plan_calls: list[int] = []

    def busy_plan(candidates, screen_receipts, *, state=None):
        plan_calls.append(1)
        raise worker_module.B300QualificationGraphEvidenceHold(
            "capture devices busy"
        )

    monkeypatch.setattr(worker.service, "plan_qualification", busy_plan)
    retires: list[bool] = []
    monkeypatch.setattr(
        worker._resident_pair_factory,
        "retire_released_pair",
        lambda: retires.append(False) or False,
        raising=False,
    )
    try:
        result = worker.run_remote_qualification(
            claim.lease,
            claim.candidates,
            claim.screen_receipts,
            screen_lane="primary",
            continuation_store=continuation,
            request_digest=request_digest,
        )

        assert type(result) is B300QualificationGraphGateHold
        assert (
            result.reason
            is RemoteQualificationHoldReason.GRAPH_EVIDENCE_UNAVAILABLE
        )
        assert len(plan_calls) == 1
        assert retires == [False]
    finally:
        worker.close()


def test_planning_hold_that_survives_retirement_holds_after_one_replan(
    tmp_path: Path,
    executor_factory,
    monkeypatch,
) -> None:
    worker, claim, continuation = _worker_under_test(tmp_path, executor_factory)
    request_digest = worker_tests._h("remote-request")

    plan_calls: list[int] = []

    def busy_plan(candidates, screen_receipts, *, state=None):
        plan_calls.append(1)
        raise worker_module.B300QualificationGraphEvidenceHold(
            "capture devices busy"
        )

    monkeypatch.setattr(worker.service, "plan_qualification", busy_plan)
    retires: list[bool] = []
    monkeypatch.setattr(
        worker._resident_pair_factory,
        "retire_released_pair",
        lambda: retires.append(True) or True,
        raising=False,
    )
    try:
        result = worker.run_remote_qualification(
            claim.lease,
            claim.candidates,
            claim.screen_receipts,
            screen_lane="primary",
            continuation_store=continuation,
            request_digest=request_digest,
        )

        assert type(result) is B300QualificationGraphGateHold
        assert (
            result.reason
            is RemoteQualificationHoldReason.GRAPH_EVIDENCE_UNAVAILABLE
        )
        assert len(plan_calls) == 2
        assert retires == [True]
    finally:
        worker.close()
