from __future__ import annotations

import json
import os
import sys
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from cacheon import receipts
from cacheon.eval import engine_worker


def _active(pid: int, rank: int, slots=("a", "b"), world_size=2):
    return {
        "pid": pid,
        "rank": rank,
        "world_size": world_size,
        "slots": list(slots),
    }


def _write(root, kind, payload, index):
    root.mkdir(parents=True, exist_ok=True)
    (root / f"{kind}.{index}.json").write_text(json.dumps(payload))


def test_teardown_summary_retains_failed_receipts(tmp_path, capsys):
    _write(
        tmp_path,
        "failed",
        {
            "error": "boom",
            "error_type": "RuntimeError",
            "line": 17,
            "phase": "entry",
            "rank": 1,
            "slot": "slot.a",
            "source": "kernels/fail.py",
        },
        0,
    )
    engine_worker._emit_execution_summary(str(tmp_path))
    summary = json.loads(capsys.readouterr().err.split(engine_worker.EXECUTION_SUMMARY_PREFIX)[1])
    assert summary["failed"][0]["source"] == "kernels/fail.py"
    assert summary["failed"][0]["line"] == 17


def test_only_candidate_owned_receipts_type_the_engine_failure(tmp_path):
    candidate = tmp_path / "candidate"
    runtime = tmp_path / "runtime"
    _write(candidate, "failed", {"slot": "a", "error": "boom"}, 0)
    _write(
        runtime,
        "failed",
        {"slot": "a", "error": "permission", "failure_owner": "validator_runtime"},
        0,
    )
    assert "boom" in engine_worker._candidate_receipt_failure(str(candidate), receipts)
    assert engine_worker._candidate_receipt_failure(str(runtime), receipts) == ""


@pytest.mark.parametrize("phase", ("launch", "running"))
@pytest.mark.parametrize("tp_size", (2, 4))
def test_isolated_engine_reports_child_death_before_parent_exit(
    monkeypatch, phase, tp_size
):
    monkeypatch.setenv("CACHEON_EXTERNAL_NO_EGRESS", "1")
    monkeypatch.setenv("CACHEON_ENGINE_WORKER", "1")
    for name in (
        "_loopback_is_up", "_network_namespace_is_loopback_only",
        "_egress_is_blocked", "_process_sandbox_is_hardened",
    ):
        monkeypatch.setattr(engine_worker, name, lambda: True)
    monkeypatch.setattr(
        engine_worker, "engine_kwargs",
        lambda cfg, **_: {"tp_size": cfg.tp_size, "disable_cuda_graph": False},
    )
    events = []

    class Signals:
        def __init__(self, manager):
            self.manager = manager

        def sigterm_handler(self):
            events.append("drain")

        def running_phase_sigquit_handler(self):
            pytest.fail("SGLang must not kill the reporter before its error frame")

    class Engine:
        def __init__(self, **kwargs):
            assert kwargs["tp_size"] == tp_size
            assert kwargs["disable_cuda_graph"] is False
            assert kwargs["custom_sigquit_handler"] is engine_worker._engine_child_failed
            self.tokenizer_manager = SimpleNamespace(signal_handler_class=Signals)
            if phase == "launch":
                kwargs["custom_sigquit_handler"]()

        def shutdown(self):
            events.append("shutdown")

    monkeypatch.setitem(sys.modules, "sglang", SimpleNamespace(Engine=Engine))
    with pytest.raises(SystemExit, match="SGLang child process failed"):
        with engine_worker.isolated_engine_session(
            SimpleNamespace(tp_size=tp_size), bundle_path="", active=False,
            framework_mode=False, install_seams=False,
        ) as handle:
            manager = handle.engine.tokenizer_manager
            handler = manager.signal_handler_class(manager)
            handler.sigterm_handler()
            handler.running_phase_sigquit_handler()
    assert events == (["drain", "shutdown"] if phase == "running" else [])


def test_child_failure_escapes_an_asyncio_signal_callback():
    import asyncio

    loop = asyncio.new_event_loop()
    pending = loop.create_future()
    loop.call_soon(engine_worker._engine_child_failed)
    try:
        with pytest.raises(SystemExit, match="SGLang child process failed"):
            loop.run_until_complete(pending)
    finally:
        pending.cancel()
        loop.close()


def _distributed_receipt_worker(rank, world_size, store_path, receipt_dir):
    import torch.distributed as dist

    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method=f"file://{store_path}",
        rank=rank,
        world_size=world_size,
    )
    try:
        os.environ["CACHEON_SEAM_RECEIPT_DIR"] = receipt_dir
        receipts.write("active", {"slots": ["slot.a"]})
        receipts.completed("slot.a")
        dist.barrier()
    finally:
        dist.destroy_process_group()


def test_active_members_require_exact_count_and_identical_slot_set():
    active = [_active(10, 0), _active(11, 1)]
    assert engine_worker._active_execution_members(
        active, expected_member_count=2
    ) == ["a", "b"]

    with pytest.raises(RuntimeError, match="1/2"):
        engine_worker._active_execution_members(active[:1], expected_member_count=2)
    with pytest.raises(RuntimeError, match="3/2"):
        engine_worker._active_execution_members(
            [*active, _active(12, 2, world_size=3)], expected_member_count=2
        )
    with pytest.raises(RuntimeError, match="duplicate"):
        engine_worker._active_execution_members(
            [active[0], {**active[1], "pid": 10}], expected_member_count=2
        )
    with pytest.raises(RuntimeError, match="disagree"):
        engine_worker._active_execution_members(
            [active[0], _active(11, 1, slots=("a",))], expected_member_count=2
        )


def test_fired_without_completed_fails_execution_gate(tmp_path):
    active = [_active(10, 0, slots=("a",), world_size=1)]
    _write(
        tmp_path,
        "fired",
        {"slot": "a", "pid": 10, "rank": 0, "world_size": 1},
        0,
    )
    _write(
        tmp_path,
        "aot_loaded",
        {"slot": "a", "pid": 10, "rank": 0, "world_size": 1},
        0,
    )
    with pytest.raises(RuntimeError, match="aot_loaded:1,aot_invoked:0"):
        engine_worker._require_execution_completion(
            str(tmp_path),
            active_receipts=active,
            expected_slots=["a"],
            expected_member_count=1,
        )


def test_every_member_must_complete_every_slot(tmp_path):
    active = [_active(10, 0), _active(11, 1)]
    index = 0
    for rank, pid in enumerate((10, 11)):
        for slot in ("a", "b"):
            _write(
                tmp_path,
                "completed",
                {"slot": slot, "pid": pid, "rank": rank, "world_size": 2},
                index,
            )
            index += 1
    detail = engine_worker._require_execution_completion(
        str(tmp_path),
        active_receipts=active,
        expected_slots=["a", "b"],
        expected_member_count=2,
    )
    assert "4/4" in detail


def test_sealed_aot_requires_load_and_use_on_every_active_member(tmp_path):
    active = [_active(10, 0, slots=("a",)), _active(11, 1, slots=("a",))]
    for index, (rank, pid) in enumerate(((0, 10), (1, 11))):
        identity = {"slot": "a", "pid": pid, "rank": rank, "world_size": 2}
        _write(tmp_path, "completed", identity, index)
        _write(tmp_path, "aot_loaded", identity, index)
    _write(
        tmp_path,
        "aot_invoked",
        {"slot": "a", "pid": 10, "rank": 0, "world_size": 2},
        0,
    )

    with pytest.raises(RuntimeError, match="failed sealed CuTe AOT use coverage"):
        engine_worker._require_execution_completion(
            str(tmp_path),
            active_receipts=active,
            expected_slots=["a"],
            expected_member_count=2,
        )

    _write(
        tmp_path,
        "aot_invoked",
        {"slot": "a", "pid": 11, "rank": 1, "world_size": 2},
        1,
    )
    detail = engine_worker._require_execution_completion(
        str(tmp_path),
        active_receipts=active,
        expected_slots=["a"],
        expected_member_count=2,
    )
    assert "sealed CuTe AOT" in detail


def test_sealed_aot_rejects_use_without_load_or_inactive_slot(tmp_path):
    active = [_active(10, 0, slots=("a",), world_size=1)]
    complete = {"slot": "a", "pid": 10, "rank": 0, "world_size": 1}
    _write(tmp_path, "completed", complete, 0)
    _write(tmp_path, "aot_invoked", complete, 0)
    with pytest.raises(RuntimeError, match="without matching load evidence"):
        engine_worker._require_execution_completion(
            str(tmp_path),
            active_receipts=active,
            expected_slots=["a"],
            expected_member_count=1,
        )

    (tmp_path / "aot_invoked.0.json").unlink()
    _write(
        tmp_path,
        "aot_loaded",
        {"slot": "other", "pid": 10, "rank": 0, "world_size": 1},
        0,
    )
    with pytest.raises(RuntimeError, match="inactive slot"):
        engine_worker._require_execution_completion(
            str(tmp_path),
            active_receipts=active,
            expected_slots=["a"],
            expected_member_count=1,
        )


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="requires two CUDA GPUs",
)
def test_real_distributed_receipts_cover_every_nccl_member(tmp_path):
    import torch.multiprocessing as mp

    receipt_dir = tmp_path / "receipts"
    store_path = tmp_path / "nccl_store"
    mp.spawn(
        _distributed_receipt_worker,
        args=(2, str(store_path), str(receipt_dir)),
        nprocs=2,
        join=True,
    )
    active = receipts.collect(receipt_dir, "active")
    completed = receipts.collect(receipt_dir, "completed")
    assert {row["rank"] for row in active} == {0, 1}
    assert len({row["pid"] for row in active}) == 2
    ok, detail = receipts.completed_gate(
        completed,
        expected_slots=("slot.a",),
        member_receipts=active,
        expected_member_count=2,
    )
    assert ok, detail


def test_coverage_failure_names_the_field_that_kept_the_candidate_off(tmp_path):
    """A run where nothing executed must say WHY, not just that nothing did."""
    active = [_active(10, 0, slots=("a",), world_size=1)]
    _write(
        tmp_path,
        "not_selected",
        {
            "slot": "a",
            "reasons": [
                {
                    "outcome": "out_of_domain",
                    "fields": ["num_tokens"],
                    "mismatches": [
                        {
                            "field": "num_tokens",
                            "reason": "outside_domain",
                            "expected": "in [1, 2048]",
                        }
                    ],
                }
            ],
        },
        0,
    )
    with pytest.raises(engine_worker.CandidateNeverExecutedError) as caught:
        engine_worker._require_execution_completion(
            str(tmp_path),
            active_receipts=active,
            expected_slots=["a"],
            expected_member_count=1,
        )
    assert "not_selected=a:out_of_domain(num_tokens)" in str(caught.value)


def test_total_silence_is_typed_as_the_candidate_never_executing(tmp_path):
    """The one coverage shape that is provably the candidate's own defect.

    ``cacheon/seam.py`` writes the ``active`` receipt from the candidate's own
    process, through this module, into this same root -- so an active-member
    check that passes is positive proof the receipt path works. When every
    execution kind is then empty, the seam loaded, registered the slot, and
    the candidate never dispatched. Two mainnet bundles hit exactly this on
    2026-08-18, rode the generic worker-error path, burned three attempts of
    exclusive 8-GPU time each, and parked with no verdict at all.
    """

    active = [_active(10, 0, slots=("a",), world_size=1)]
    with pytest.raises(engine_worker.CandidateNeverExecutedError) as caught:
        engine_worker._require_execution_completion(
            str(tmp_path),
            active_receipts=active,
            expected_slots=["a"],
            expected_member_count=1,
        )
    # The worker runs in a sealed OCI lifetime, so only the text crosses the
    # boundary. The marker is the wire contract; a class name or a prose
    # fragment would be silently broken by any later edit.
    assert engine_worker.CANDIDATE_NEVER_EXECUTED_MARKER in str(caught.value)
    assert "completed:0,aot_loaded:0,aot_invoked:0" in str(caught.value)


def test_partial_coverage_stays_infrastructure(tmp_path):
    """Some receipts but not all may never be charged to a candidate.

    A member that died mid-run and a read that raced a peer's write both look
    like partial coverage. Only total silence is unambiguous, so anything
    partial keeps the generic type and stays on the retry/hold path.
    """

    active = [_active(10, 0), _active(11, 1)]
    _write(
        tmp_path,
        "completed",
        {"slot": "a", "pid": 10, "rank": 0, "world_size": 2},
        0,
    )
    with pytest.raises(engine_worker.CandidateExecutionCoverageError) as caught:
        engine_worker._require_execution_completion(
            str(tmp_path),
            active_receipts=active,
            expected_slots=["a"],
            expected_member_count=2,
        )
    assert type(caught.value) is engine_worker.CandidateExecutionCoverageError
    assert engine_worker.CANDIDATE_NEVER_EXECUTED_MARKER not in str(caught.value)
