"""Replay speed fixtures: a sealed slice, its replay window, and the v17 policy grading it.

Every speed workload is a sealed session replay (speed policies 16 and 17). These
helpers turn a batch transport fixture into the replay workload the evaluator
accepts, so tests of shared orchestration exercise the only speed path there is.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from cacheon.eval.agent_replay import AgentReplayPlan
from cacheon.eval.goodput_runtime import GoodputPolicy
from cacheon.eval.service_capacity import ServiceContract
from tests.test_agent_slice import _session, _write_slice

CONTRACT = ServiceContract(0.001, 1000.0, 0.8)
# Policy 17 grades elapsed work statistically: required 1, error rate 1%.
GOODPUT = GoodputPolicy(CONTRACT, 1.0, 0.003, 0.0, 0.0, 0.01, 0.001)


def replay_plan(root: Path, *, windows: int = 5) -> AgentReplayPlan:
    """One sealed single-session slice under ``root``, read at load 1."""

    root.mkdir(parents=True, exist_ok=True)
    manifest = _write_slice(root, [_session(1, k=3, inner=0)])
    return AgentReplayPlan(manifest, (1,), Path("/aiperf"), Path("/model"), root / "out",
                           CONTRACT, "incumbent", 1, "lane", windows=windows, elapsed_cost=True)


def replay_session(session, replay: AgentReplayPlan):
    """The same engine and warmup, measuring the replay instead of timed batches."""

    return replace(session, replay=replay)
