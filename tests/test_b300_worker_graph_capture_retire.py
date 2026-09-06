"""A planning-stage graph-capture hold is an authenticated hold.

2026-08-14, epoch 8afcf1ed: a resident pair left loaded by a completed
qualification starved every later target-switch capture, and the
retire-and-retry that existed only around ``work.factory.build()`` never
fired for a hold raised one stage earlier in ``plan_qualification``. The
resident pair lane is gone (every candidate now boots its own two-process
crossover), so nothing holds devices between requests and there is nothing
to retire: a capture hold at planning surfaces as the hold itself, once,
without a replan.
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


def test_planning_hold_holds_without_replanning(
    tmp_path: Path,
    executor_factory,
    monkeypatch,
) -> None:
    authorities, _resident, _builder = worker_tests._authorities(
        tmp_path, executor_factory
    )
    manifest = worker_tests._manifest(authorities)
    readiness = worker_tests._readiness(manifest, authorities)
    claim = worker_tests._qualification_claim(tmp_path / "cohort", manifest)
    continuation = QualificationContinuationStore(tmp_path / "continuation")
    worker = B300MainnetWorker(manifest, authorities, readiness)
    request_digest = worker_tests._h("remote-request")
    plan_calls: list[int] = []

    def busy_plan(candidates, screen_receipts, *, state=None):
        plan_calls.append(1)
        raise worker_module.B300QualificationGraphEvidenceHold(
            "capture devices busy"
        )

    monkeypatch.setattr(worker.service, "plan_qualification", busy_plan)
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
    finally:
        worker.close()
