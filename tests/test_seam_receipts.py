"""Seam-activation receipts — the anti-phantom-pass gate (cacheon/receipts.py).

Pins the failure mode hit for real on 2026-07-07: a candidate engine that comes up
WITHOUT the seam (missing .pth bootstrap / bundle load fallback) produced
bit-identical logits, KL exactly 0.0, and a PASS verdict. The eval driver must
demand positive evidence from the ranks, and the diagnosis must distinguish
"no bootstrap at all" from "bundle load fell back to baseline".
"""

from __future__ import annotations

import os

import pytest

from cacheon import receipts
from cacheon.capabilities import CallDescriptor
from cacheon.registry import Eligibility, KernelImpl, KernelRegistry


@pytest.fixture()
def receipt_dir(tmp_path, monkeypatch):
    rdir = tmp_path / "receipts"
    monkeypatch.setenv("CACHEON_SEAM_RECEIPT_DIR", str(rdir))
    monkeypatch.setattr(receipts, "_ONCE", set())
    monkeypatch.setattr(receipts, "_CALLS", {})
    monkeypatch.setattr(receipts, "_GRAPH_PROBE", None)
    monkeypatch.setattr(receipts, "_IDENTITY", None)
    monkeypatch.setattr(receipts, "_NOT_SELECTED", {})
    return rdir


def test_no_env_is_a_silent_noop(tmp_path, monkeypatch):
    monkeypatch.delenv("CACHEON_SEAM_RECEIPT_DIR", raising=False)
    receipts.write("active", {"bundle": "x"})  # must not raise, must not create files
    receipts.completed("norm.rmsnorm")
    assert list(tmp_path.iterdir()) == []


def test_exit_flush_keeps_the_identity_the_execution_actually_had(
    receipt_dir, monkeypatch
):
    """Regression: the counts flush runs at ``atexit``, after every scheduler rank
    has destroyed its process group. Re-detecting identity there yields ``-1`` and
    no longer matches the ``active`` receipt, so the coverage check would call a
    perfectly good run's execution evidence malformed. Found on 4x B300, not here.
    """
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    receipts.write("active", {"slots": ["slot.a"]})
    receipts.completed("slot.a")

    # The group goes away; so does every way to read the rank back.
    monkeypatch.delenv("RANK")
    monkeypatch.delenv("WORLD_SIZE")
    receipts.flush_calls()

    active = receipts.collect(receipt_dir, "active")[0]
    done = receipts.collect(receipt_dir, "completed")[0]
    assert (done["rank"], done["world_size"]) == (0, 1)
    assert (done["rank"], done["world_size"]) == (
        active["rank"], active["world_size"]
    )
    ok, detail = receipts.completed_gate(
        [done],
        expected_slots=("slot.a",),
        member_receipts=[active],
        expected_member_count=1,
    )
    assert ok, detail


def test_write_and_collect_roundtrip(receipt_dir):
    receipts.write("active", {"bundle": "b", "slots": ["s"]})
    receipts.write("completed", {"slot": "collective.all_reduce"},
                   tag="collective.all_reduce")
    active = receipts.collect(receipt_dir, "active")
    assert active[0]["bundle"] == "b" and active[0]["slots"] == ["s"]
    assert active[0]["pid"] == os.getpid()
    done = receipts.collect(receipt_dir, "completed")
    assert done[0]["slot"] == "collective.all_reduce"
    assert {"pid", "rank", "world_size"} <= done[0].keys()
    # tag is sanitized into the filename; pid keeps concurrent ranks from colliding
    names = [p.name for p in receipt_dir.iterdir()]
    assert any(n.startswith("completed.collective.all_reduce") for n in names)
    assert all(str(os.getpid()) in n for n in names)


def test_require_passes_with_receipt(receipt_dir):
    receipts.write("active", {"bundle": "b"})
    got = receipts.require(receipt_dir, "active", context="test")
    assert got and got[0]["bundle"] == "b"


def test_require_diagnoses_missing_bootstrap(receipt_dir):
    receipt_dir.mkdir(parents=True)
    with pytest.raises(RuntimeError, match="WITHOUT the miner kernel"):
        receipts.require(receipt_dir, "active", context="test")


def test_require_diagnoses_loud_bundle_activation_failure(receipt_dir):
    receipts.write("load_failed", {"bundle": "b", "reason": "exception during load"})
    with pytest.raises(RuntimeError, match="failed bundle activation"):
        receipts.require(receipt_dir, "active", context="test")


def test_resolving_an_impl_writes_nothing(receipt_dir):
    reg = KernelRegistry()
    reg.register(KernelImpl(slot="activation.silu_and_mul", bundle_id="t",
                            entry=lambda *a: None, eligibility=Eligibility()))
    reg.enable()
    for _ in range(3):
        assert reg.select(
            "activation.silu_and_mul",
            CallDescriptor.from_legacy(
                dtype_name="bfloat16", last_dim=128, arch=None, num_tokens=2
            ),
        ).impl is not None
    assert receipts.collect(receipt_dir, "completed") == []


def test_registry_miss_writes_nothing(receipt_dir, monkeypatch):
    reg = KernelRegistry()
    reg.register(KernelImpl(slot="norm.rmsnorm", bundle_id="t", entry=lambda *a: None,
                            eligibility=Eligibility(dtypes=frozenset({"float16"}))))
    reg.enable()
    # Ineligible (dtype mismatch) -> no selection -> no COMPLETED receipt, but the
    # reason is recorded so "registered and never ran" is never a mystery.
    assert reg.select(
        "norm.rmsnorm",
        CallDescriptor.from_legacy(
            dtype_name="bfloat16", last_dim=128, arch=None, num_tokens=2
        ),
    ).impl is None
    assert receipts.collect(receipt_dir, "completed") == []
    why = receipts.collect(receipt_dir, "not_selected")
    assert [row["slot"] for row in why] == ["norm.rmsnorm"]
    assert why[0]["reasons"][0]["fields"] == ["dtype"]


def test_no_env_does_not_consume_execution_once_guard(tmp_path, monkeypatch):
    monkeypatch.setattr(receipts, "_ONCE", set())
    monkeypatch.delenv("CACHEON_SEAM_RECEIPT_DIR", raising=False)
    receipts.completed("norm.rmsnorm")
    monkeypatch.setenv("CACHEON_SEAM_RECEIPT_DIR", str(tmp_path))
    receipts.completed("norm.rmsnorm")
    assert len(receipts.collect(tmp_path, "completed")) == 1


def test_execution_once_guard_is_scoped_to_resolved_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(receipts, "_ONCE", set())
    first = tmp_path / "first"
    second = tmp_path / "second"
    monkeypatch.setenv("CACHEON_SEAM_RECEIPT_DIR", str(first))
    receipts.completed("norm.rmsnorm")
    receipts.completed("norm.rmsnorm")
    monkeypatch.setenv(
        "CACHEON_SEAM_RECEIPT_DIR", f"{first}/../{first.name}"
    )
    receipts.completed("norm.rmsnorm")
    monkeypatch.setenv("CACHEON_SEAM_RECEIPT_DIR", str(second))
    receipts.completed("norm.rmsnorm")
    assert len(receipts.collect(first, "completed")) == 1
    assert len(receipts.collect(second, "completed")) == 1


def test_failed_execution_receipt_write_does_not_consume_guard(tmp_path, monkeypatch):
    monkeypatch.setattr(receipts, "_ONCE", set())
    monkeypatch.setenv("CACHEON_SEAM_RECEIPT_DIR", str(tmp_path))
    outcomes = iter((False, True))
    calls = []

    def fake_write(*_args, **_kwargs):
        calls.append(1)
        return next(outcomes)

    monkeypatch.setattr(receipts, "_write_to", fake_write)
    receipts.completed("slot.a")
    receipts.completed("slot.a")
    receipts.completed("slot.a")
    assert len(calls) == 2


def test_completed_receipt_carries_the_real_invocation_count(receipt_dir):
    # The file is written once; the count it carries is not frozen at one. Before
    # this, "8 candidate executions" meant eight receipt FILES and was identical
    # for a one-request and a twelve-request launch.
    for _ in range(12):
        receipts.completed("norm.rmsnorm")
    receipts.flush_calls()
    rows = receipts.collect(receipt_dir, "completed")
    assert len(rows) == 1 and rows[0]["calls"] == 12


def test_capture_is_recorded_when_the_entry_runs_inside_a_graph(receipt_dir):
    # The scored windows replay the captured graph and never re-enter Python. Being
    # invoked during capture is what puts the candidate in the path that gets timed.
    capturing = iter((False, False, True, False))
    receipts.set_graph_probe(lambda: next(capturing, False))
    for _ in range(4):
        receipts.completed("norm.rmsnorm")
    receipts.flush_calls()
    row = receipts.collect(receipt_dir, "completed")[0]
    assert row["calls"] == 4 and row["captured"] is True


def test_first_completed_receipt_survives_without_exit_flush(receipt_dir):
    """A killed scheduler still preserves its first invocation/capture fact."""

    receipts.set_graph_probe(lambda: True)
    receipts.completed("norm.rmsnorm")
    row = receipts.collect(receipt_dir, "completed")[0]
    assert row["calls"] == 1 and row["captured"] is True


def test_capture_transition_rewrites_eager_receipt_before_exit(receipt_dir):
    capturing = iter((False, True))
    receipts.set_graph_probe(lambda: next(capturing, True))
    receipts.completed("norm.rmsnorm")
    assert receipts.collect(receipt_dir, "completed")[0]["captured"] is False
    receipts.completed("norm.rmsnorm")
    row = receipts.collect(receipt_dir, "completed")[0]
    assert row["calls"] == 2 and row["captured"] is True


def test_a_candidate_skipped_at_capture_is_visible_as_such(receipt_dir):
    # The phantom pass: not graph-safe, so capture bakes stock and every replay
    # serves stock — while `completed` was already written during eager warmup.
    receipts.set_graph_probe(lambda: False)
    for _ in range(6):
        receipts.completed("norm.rmsnorm")
    receipts.flush_calls()
    row = receipts.collect(receipt_dir, "completed")[0]
    assert row["calls"] == 6 and row["captured"] is False


def test_without_a_probe_capture_is_unknown_not_false(receipt_dir):
    receipts.completed("norm.rmsnorm")
    receipts.flush_calls()
    row = receipts.collect(receipt_dir, "completed")[0]
    assert row["calls"] == 1 and "captured" not in row


def test_detected_identity_overrides_payload(receipt_dir, monkeypatch):
    monkeypatch.setattr(
        receipts, "identity", lambda: {"pid": 7, "rank": 1, "world_size": 2}
    )
    receipts.write(
        "completed",
        {"slot": "slot.a", "pid": 999, "rank": 999, "world_size": 999},
    )
    got = receipts.collect(receipt_dir, "completed")[0]
    assert (got["pid"], got["rank"], got["world_size"]) == (7, 1, 2)


def test_completed_is_written_once_per_slot(receipt_dir):
    for _ in range(3):
        receipts.completed("norm.rmsnorm")
    assert len(receipts.collect(receipt_dir, "completed")) == 1


@pytest.mark.parametrize(
    ("environment", "expected_phase"),
    (
        ({}, "all"),
        (
            {
                "CACHEON_ENGINE_WORKER": "1",
                "CACHEON_PREBUILT_ARTIFACTS": "1",
            },
            "load",
        ),
    ),
)
def test_scheduler_bundle_rebuild_phase_matches_launch_authority(
    monkeypatch, environment, expected_phase
):
    """Production is load-only; explicit direct eval reuses its dev cache."""
    from cacheon import manifest, rebuild, sandbox
    from cacheon.registry import REGISTRY
    from cacheon.seam import _load_bundle_into_registry

    class EmptyManifest:
        ops = ()

    class CleanTree:
        ok = True
        violations = ()

    calls = []
    monkeypatch.setattr(manifest, "load_manifest", lambda _bundle: EmptyManifest())
    monkeypatch.setattr(
        manifest, "all_declared_cuda_sources", lambda _bundle, _manifest: ()
    )
    monkeypatch.setattr(sandbox, "scan_tree", lambda *_args, **_kwargs: CleanTree())
    monkeypatch.setattr(
        rebuild,
        "apply_rebuild_plan",
        lambda bundle, *, phase: calls.append((bundle, phase)),
    )
    for name in ("CACHEON_ENGINE_WORKER", "CACHEON_PREBUILT_ARTIFACTS"):
        monkeypatch.delenv(name, raising=False)
    for name, value in environment.items():
        monkeypatch.setenv(name, value)

    REGISTRY.clear()
    try:
        _load_bundle_into_registry("/sealed/candidate-tree")
    finally:
        REGISTRY.clear()

    assert calls == [("/sealed/candidate-tree", expected_phase)]


@pytest.mark.parametrize(
    "environment",
    (
        {"CACHEON_ENGINE_WORKER": "1"},
        {"CACHEON_PREBUILT_ARTIFACTS": "1"},
    ),
)
def test_scheduler_rejects_partial_native_artifact_authority(
    monkeypatch, environment
):
    from cacheon import manifest, sandbox
    from cacheon.registry import REGISTRY
    from cacheon.seam import _load_bundle_into_registry

    class EmptyManifest:
        ops = ()

    class CleanTree:
        ok = True
        violations = ()

    monkeypatch.setattr(manifest, "load_manifest", lambda _bundle: EmptyManifest())
    monkeypatch.setattr(manifest, "all_declared_cuda_sources", lambda *_args: ())
    monkeypatch.setattr(sandbox, "scan_tree", lambda *_args, **_kwargs: CleanTree())
    for name in ("CACHEON_ENGINE_WORKER", "CACHEON_PREBUILT_ARTIFACTS"):
        monkeypatch.delenv(name, raising=False)
    for name, value in environment.items():
        monkeypatch.setenv(name, value)

    REGISTRY.clear()
    try:
        with pytest.raises(RuntimeError, match="incomplete native-artifact authority"):
            _load_bundle_into_registry("/sealed/candidate-tree")
    finally:
        REGISTRY.clear()


@pytest.mark.parametrize("payload", ("{", "[]"))
def test_strict_collection_rejects_malformed_receipts(receipt_dir, payload):
    receipt_dir.mkdir(parents=True, exist_ok=True)
    (receipt_dir / "completed.bad.json").write_text(payload)
    with pytest.raises(receipts.ReceiptFormatError):
        receipts.collect(receipt_dir, "completed")


def _active_members(count=2):
    return [
        {"pid": 10 + rank, "rank": rank, "world_size": count, "slots": ["a", "b"]}
        for rank in range(count)
    ]


def _completed_members(count=2):
    return [
        {"slot": slot, "pid": 10 + rank, "rank": rank, "world_size": count}
        for rank in range(count)
        for slot in ("a", "b")
    ]


def test_completed_gate_requires_exact_slot_member_cross_product():
    active = _active_members()
    completed = _completed_members()
    ok, desc = receipts.completed_gate(
        completed,
        expected_slots=("a", "b"),
        member_receipts=active,
        expected_member_count=2,
    )
    assert ok and "4/4" in desc

    ok, desc = receipts.completed_gate(
        completed[:-1],
        expected_slots=("a", "b"),
        member_receipts=active,
        expected_member_count=2,
    )
    assert not ok and "pid:11" in desc and "slot" in desc


@pytest.mark.parametrize("active", (_active_members(1), _active_members(3)))
def test_completed_gate_rejects_wrong_active_member_count(active):
    completed = [
        {"slot": "a", "pid": row["pid"], "rank": row["rank"],
         "world_size": row["world_size"]}
        for row in active
    ]
    ok, desc = receipts.completed_gate(
        completed,
        expected_slots=("a",),
        member_receipts=active,
        expected_member_count=2,
    )
    assert not ok and "members=" in desc


def test_completed_gate_rejects_invalid_distributed_active_membership():
    duplicate_rank = [
        {"pid": 10, "rank": 0, "world_size": 2, "slots": ["a"]},
        {"pid": 11, "rank": 0, "world_size": 2, "slots": ["a"]},
    ]
    wrong_world = [
        {"pid": 10, "rank": 0, "world_size": 8, "slots": ["a"]},
        {"pid": 11, "rank": 1, "world_size": 8, "slots": ["a"]},
    ]
    for active in (duplicate_rank, wrong_world):
        completed = [
            {"slot": "a", "pid": row["pid"], "rank": row["rank"],
             "world_size": row["world_size"]}
            for row in active
        ]
        ok, desc = receipts.completed_gate(
            completed,
            expected_slots=("a",),
            member_receipts=active,
            expected_member_count=2,
        )
        assert not ok and "malformed" in desc


def test_completed_gate_rejects_identity_change_for_active_pid():
    active = _active_members(2)
    completed = _completed_members(2)
    completed[0] = {**completed[0], "rank": 1}
    ok, desc = receipts.completed_gate(
        completed,
        expected_slots=("a", "b"),
        member_receipts=active,
        expected_member_count=2,
    )
    assert not ok and "malformed" in desc


def test_unknown_active_tp_members_require_completion_rank_proof():
    active = [
        {"pid": 10, "rank": -1, "world_size": -1, "slots": ["a"]},
        {"pid": 11, "rank": -1, "world_size": -1, "slots": ["a"]},
    ]

    def completed(identities):
        return [
            {"slot": "a", "pid": 10 + index, "rank": rank, "world_size": world}
            for index, (rank, world) in enumerate(identities)
        ]

    for invalid in (
        completed(((0, 2), (0, 2))),
        completed(((0, 8), (1, 8))),
        completed(((-1, -1), (-1, -1))),
    ):
        ok, desc = receipts.completed_gate(
            invalid,
            expected_slots=("a",),
            member_receipts=active,
            expected_member_count=2,
        )
        assert not ok and "malformed" in desc

    ok, desc = receipts.completed_gate(
        completed(((0, 2), (1, 2))),
        expected_slots=("a",),
        member_receipts=active,
        expected_member_count=2,
    )
    assert ok and "2/2" in desc


def test_coverage_needs_a_roster_and_never_infers_one_from_completions():
    # The roster is the `active` receipts. Inferring it from the completions lets a
    # rank that silently stopped reporting shrink the roster to the ranks that did
    # report, so short coverage would read as full coverage.
    assert not receipts.coverage_matrix(
        [{"slot": "a", "pid": 11, "rank": 1, "world_size": 2}],
        expected_slots=("a",),
        member_receipts=(),
        expected_member_count=2,
    )["ok"]
    conflicting = receipts.coverage_matrix(
        [{"slot": "a", "pid": 10, "rank": 0, "world_size": 1}],
        expected_slots=("a",),
        member_receipts=[
            {"pid": 10, "rank": 0, "world_size": 1},
            {"pid": 11, "rank": 1, "world_size": 2},
        ],
    )
    assert not conflicting["ok"] and conflicting["malformed"]


@pytest.mark.parametrize(
    "bad",
    (
        {"slot": "a", "pid": True, "rank": 0, "world_size": 1},
        {"slot": "a", "pid": "10", "rank": 0, "world_size": 1},
        {"slot": "a", "pid": 10, "rank": 2, "world_size": 2},
    ),
)
def test_coverage_rejects_non_exact_or_incoherent_identity(bad):
    detail = receipts.coverage_matrix(
        [bad], expected_slots=("a",), member_receipts=_active_members(1)
    )
    assert not detail["ok"] and detail["malformed"]


def test_coverage_rejects_duplicate_and_unexpected_completion():
    active = _active_members(1)
    duplicate = [
        {"slot": "a", "pid": 10, "rank": 0, "world_size": 1},
        {"slot": "a", "pid": 10, "rank": 0, "world_size": 1},
    ]
    unexpected = [
        {"slot": "other", "pid": 10, "rank": 0, "world_size": 1}
    ]
    assert not receipts.coverage_matrix(
        duplicate, expected_slots=("a",), member_receipts=active
    )["ok"]
    assert not receipts.coverage_matrix(
        unexpected, expected_slots=("a",), member_receipts=active
    )["ok"]


def test_a_candidate_raise_is_receipted_and_blamed_all_the_way_out(
    tmp_path, monkeypatch
):
    """The path that used to crash the lane with nothing to show for it.

    A raise inside the entry is receipted under the driver's receipt root,
    reduces to a rank that is not clean, and is named by the session worker
    when the engine dies under it.
    """

    from cacheon.eval.resident_execution_evidence import RankExecution

    monkeypatch.setattr(receipts, "_ONCE", set())
    control = tmp_path / "ctl"
    root = control / "receipts" / "4"
    monkeypatch.setenv("CACHEON_SEAM_RECEIPT_DIR", str(root))

    def entry(_x):
        raise RuntimeError("CUDA error: illegal memory access\nat line 3")

    with pytest.raises(RuntimeError, match="illegal memory access"):
        receipts.invoke("attention.msa_block_score", entry, object())
    # Once per slot per process: a second raise does not churn the receipt.
    with pytest.raises(RuntimeError):
        receipts.invoke("attention.msa_block_score", entry, object())

    rows = {"failed": receipts.collect(root, "failed")}
    (failed,) = rows["failed"]
    assert failed["slot"] == "attention.msa_block_score"
    assert failed["error_type"] == "RuntimeError"
    assert failed["pid"] == os.getpid()

    reduced = RankExecution.from_receipts(0, rows)
    assert reduced.slots[0].error.startswith("RuntimeError: CUDA error: illegal memory access")



def test_validator_library_permission_is_not_blamed_on_candidate(
    tmp_path, monkeypatch
):

    monkeypatch.setattr(receipts, "_ONCE", set())
    control = tmp_path / "ctl"
    root = control / "receipts" / "5"
    monkeypatch.setenv("CACHEON_SEAM_RECEIPT_DIR", str(root))

    def entry(_x):
        raise PermissionError(
            13,
            "Permission denied",
            "/usr/local/lib/python3.12/dist-packages/library/jit-cache",
        )

    with pytest.raises(PermissionError):
        receipts.invoke("moe.fused_experts", entry, object())
    rows = {"failed": receipts.collect(root, "failed")}
    (failed,) = rows["failed"]
    assert failed["failure_owner"] == "validator_runtime"
