"""The full lane's CPU plan also serves its one-engine audit and pristine roles."""

import json
from types import SimpleNamespace

import pytest

from cacheon.eval import engine_worker

# (lane engines' first GPUs, plan engine of the first running one, running engines as (companion, first GPU, TP, DP))
LAYOUTS = {
    "one engine": ((0,), 0, ((None, 0, 4, 2),)),
    "pair audit": ((0, 4), 0, ((None, 0, 4, 2),)),
    "pair pristine": ((0, 4), 1, ((None, 4, 4, 2),)),
    "pair timed": ((0, 4), 0, ((200, 0, 4, 2), (None, 4, 4, 2))),
    "pair timed, dp 1 decode": ((0, 4), 0, ((200, 0, 4, 2), (None, 4, 4, 1))),
    "two TP2 prefill servers": ((0, 2, 4), 0, ((200, 0, 2, 1), (300, 2, 2, 1), (None, 4, 4, 1))),
    "two-server pristine": ((0, 2, 4), 2, ((None, 4, 2, 1),)),
}


@pytest.mark.parametrize("name", LAYOUTS)
def test_engine_role_pins_follow_the_running_engines(tmp_path, monkeypatch, name):
    """Each running engine's ranks take its own GPUs' schedulers and its plan engine's vCPUs; an engine with
    dp 1 (a GLM-5.3-Flash role) runs no DP controller."""
    lane, first, running = LAYOUTS[name]
    plan = {"schedulers": list(range(10, 18)), "pool": [80, 81],
            "engines": [[90 + 2 * index, 91 + 2 * index] for index in range(len(lane))]}
    processes, expected, trees = {}, {}, {}
    for side, (companion, gpu, tp, dp) in enumerate(running):
        root = companion or 100
        rows = {root: "python", **({root + 1: "sglang::data_parallel_controller"} if dp > 1 else {})}
        rows.update({root + 10 + rank: f"sglang::scheduler_DP0_TP{rank}" for rank in range(tp)})
        processes.update(rows)
        trees[companion] = list(rows)
        expected[root] = {plan["engines"][first + side][1]}
        if dp > 1:
            expected[root + 1] = {plan["engines"][first + side][0]}
        expected.update({root + 10 + rank: {plan["schedulers"][gpu + rank]} for rank in range(tp)})
    processes[999], expected[999] = "sglang::detokenizer", {80, 81}
    for pid, title in processes.items():
        (tmp_path / str(pid)).mkdir()
        (tmp_path / str(pid) / "cmdline").write_bytes(title.encode() + b"\0")
    applied = {}
    monkeypatch.setattr(engine_worker, "Path", lambda path: tmp_path / path.removeprefix("/proc/"))
    monkeypatch.setattr(engine_worker, "_process_tree", lambda root: trees[root] if root != 100 else list(processes))
    monkeypatch.setattr(engine_worker, "os", SimpleNamespace(
        getpid=lambda: 100, listdir=lambda path: [path.split("/")[2]],
        sched_setaffinity=lambda pid, cpus: applied.__setitem__(pid, cpus),
    ))
    engine_worker._pin_engine_processes(json.dumps(plan), pinned=8, lane_engines=len(lane), first=first,
                                        engines=running)
    assert applied == expected


@pytest.mark.parametrize("lane_engines,declared_engines", [(1, 2), (2, 1), (2, 3), (3, 2)])
def test_engine_role_pins_reject_a_different_lane(lane_engines, declared_engines):
    """Running only the first engine does not excuse an incomplete full-lane role plan."""
    plan = {"schedulers": list(range(8)), "pool": [80],
            "engines": [[90 + 2 * side, 91 + 2 * side] for side in range(declared_engines)]}
    with pytest.raises(RuntimeError, match="CPU pin plan names"):
        engine_worker._pin_engine_processes(json.dumps(plan), pinned=8, lane_engines=lane_engines,
                                            engines=((None, 0, 4, 2),))
