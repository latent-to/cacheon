"""The full lane's CPU plan also serves its one-engine audit and pristine roles."""

import json
from types import SimpleNamespace

import pytest

from cacheon.eval import engine_worker


@pytest.mark.parametrize("tp", [2, 4])
@pytest.mark.parametrize("lane_engines,live_engines,decode_dp", [(1, 1, 2), (2, 1, 2), (2, 2, 2), (2, 2, 1)])
def test_engine_role_pins_follow_the_running_half(tmp_path, monkeypatch, tp, lane_engines, live_engines, decode_dp):
    """Timed pairs keep both role groups; audit and pristine T use the prefill group. A dp 1 decode role
    (GLM-5.3-Flash) runs no DP controller."""
    plan = {"schedulers": list(range(10, 10 + tp * lane_engines)), "pool": [80, 81],
            "engines": [[90, 91], [92, 93]][:lane_engines]}
    companion = 200 if live_engines == 2 else None
    processes, expected, prefill = {}, {}, []
    dp_sizes = (2, decode_dp) if companion else (decode_dp,)
    for side, root in enumerate(([200, 100] if companion else [100])):
        rows = {root: "python", **({root + 1: "sglang::data_parallel_controller"} if dp_sizes[side] > 1 else {})}
        rows.update({root + 10 + rank: f"sglang::scheduler_DP0_TP{rank}" for rank in range(tp)})
        processes.update(rows)
        if root == companion:
            prefill = list(rows)
        expected[root] = {plan["engines"][side][1]}
        if dp_sizes[side] > 1:
            expected[root + 1] = {plan["engines"][side][0]}
        expected.update({root + 10 + rank: {plan["schedulers"][side * tp + rank]} for rank in range(tp)})
    processes[999], expected[999] = "sglang::detokenizer", {80, 81}
    for pid, title in processes.items():
        process = tmp_path / str(pid)
        process.mkdir()
        (process / "cmdline").write_bytes(title.encode() + b"\0")
    applied = {}
    monkeypatch.setattr(engine_worker, "Path", lambda path: tmp_path / path.removeprefix("/proc/"))
    monkeypatch.setattr(engine_worker, "_process_tree", lambda root: prefill if root == companion else list(processes))
    monkeypatch.setattr(engine_worker, "os", SimpleNamespace(
        getpid=lambda: 100, listdir=lambda path: [path.split("/")[2]],
        sched_setaffinity=lambda pid, cpus: applied.__setitem__(pid, cpus),
    ))
    engine_worker._pin_engine_processes(json.dumps(plan), schedulers=tp,
                                       pinned=tp * lane_engines, companion=companion, dp_sizes=dp_sizes)
    assert applied == expected


@pytest.mark.parametrize("lane_engines,declared_engines", [(1, 2), (2, 1), (2, 3)])
def test_engine_role_pins_reject_a_different_lane(lane_engines, declared_engines):
    """Running only the first engine does not excuse an incomplete full-lane role plan."""
    plan = {"schedulers": list(range(4 * lane_engines)), "pool": [80],
            "engines": [[90 + 2 * side, 91 + 2 * side] for side in range(declared_engines)]}
    with pytest.raises(RuntimeError, match="CPU pin plan names"):
        engine_worker._pin_engine_processes(json.dumps(plan), schedulers=4, pinned=4 * lane_engines, dp_sizes=(2,))
