"""Heterogeneous, recovery, and deadline regressions for the graph store."""

from __future__ import annotations

import concurrent.futures
import hashlib
import inspect
import os
import threading
import time
from pathlib import Path

import pytest

import cacheon.eval.b300_qualification_graph_evidence_store as store_module
from cacheon.eval.b300_qualification_capabilities import (
    StructuredGraphShapeRecord,
    StructuredGraphVariantRecord,
)
from cacheon.eval.b300_qualification_graph_evidence_store import (
    B300QualificationGraphAttemptToken,
    B300QualificationGraphEvidenceHold,
    B300QualificationGraphEvidenceStore,
    B300QualificationGraphEvidenceStoreError,
    B300QualificationGraphGenerationOutput,
)
from cacheon.eval.b300_qualification_graph_provider import (
    B300QualificationGraphArtifact,
    B300QualificationGraphBinding,
)


def _h(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _deadline(seconds: float = 5.0) -> float:
    return time.monotonic() + seconds


def _binding(label: str) -> B300QualificationGraphBinding:
    member = f"slot.{label}"
    return B300QualificationGraphBinding(
        reservation_digest=_h(label + ":reservation"),
        reservation_identity_digest=_h(label + ":reservation-identity"),
        candidate_binding_digest=_h(label + ":candidate"),
        screen_attempt=1,
        target_id=f"target.{label}",
        target_members=(member,),
        target_spec_digest=_h(label + ":spec"),
        selected_delta_digest=_h(label + ":delta"),
        publication_content_hash=_h(label + ":content"),
        publication_address_digest=_h(label + ":address"),
        publication_digest=_h(label + ":publication"),
        publication_receipt_digest=_h(label + ":receipt"),
        prepared_arm_digest=_h(label + ":arm"),
        prepared_contribution_digest=_h(label + ":contribution"),
        prepared_launch_digest=_h(label + ":launch"),
        materialized_stack_digest=_h(label + ":stack"),
        materialized_tree_digest=_h(label + ":tree"),
        trusted_tree_identity_digest=_h(label + ":trusted-tree"),
        native_build_spec_digest=_h(label + ":native-build"),
    )


def _output(
    binding: B300QualificationGraphBinding,
    policy: str,
    token: B300QualificationGraphAttemptToken,
) -> B300QualificationGraphGenerationOutput:
    member = binding.target_members[0]
    artifact = B300QualificationGraphArtifact(
        binding,
        policy,
        3,
        (
            StructuredGraphVariantRecord(
                member,
                "commissioned",
                True,
                True,
                (
                    StructuredGraphShapeRecord(
                        _h(binding.digest + ":shape"),
                        True,
                        True,
                        True,
                        3,
                        True,
                        True,
                        False,
                    ),
                ),
            ),
        ),
    )
    return B300QualificationGraphGenerationOutput(token, artifact)


def _producer(policy: str):
    def produce(
        binding: B300QualificationGraphBinding,
        token: B300QualificationGraphAttemptToken,
        _bound: float,
    ) -> B300QualificationGraphGenerationOutput:
        return _output(binding, policy, token)

    return produce


def test_one_store_handles_many_heterogeneous_bindings_concurrently(
    tmp_path: Path,
) -> None:
    policy = _h("heterogeneous-policy")
    store = B300QualificationGraphEvidenceStore(tmp_path / "store", policy)
    bindings = tuple(_binding(f"profile-{index:02d}") for index in range(40))

    def run(binding: B300QualificationGraphBinding):
        return store.probe_once(binding, _producer(policy), deadline=_deadline(10))

    with concurrent.futures.ThreadPoolExecutor(max_workers=40) as pool:
        references = tuple(pool.map(run, bindings))

    assert len({reference.sha256 for reference in references}) == len(bindings)
    assert tuple(
        store.reopen(binding, deadline=_deadline()) for binding in bindings
    ) == references


def test_live_producer_cannot_be_raced_by_a_synthetic_recoverer(
    tmp_path: Path,
) -> None:
    policy = _h("producer-race-policy")
    binding = _binding("producer-race")
    store = B300QualificationGraphEvidenceStore(tmp_path / "store", policy)
    entered = threading.Event()
    release = threading.Event()
    first_result: list[object] = []
    second_called = False

    def first_producer(exact, token, _bound):
        entered.set()
        assert release.wait(5)
        return _output(exact, policy, token)

    def run_first() -> None:
        first_result.append(
            store.probe_once(binding, first_producer, deadline=_deadline(10))
        )

    thread = threading.Thread(target=run_first)
    thread.start()
    assert entered.wait(5)

    def second_producer(*_args):
        nonlocal second_called
        second_called = True
        raise AssertionError("second producer entered")

    with pytest.raises(B300QualificationGraphEvidenceHold, match="armed"):
        store.probe_once(binding, second_producer, deadline=_deadline())
    assert "recoverer" not in inspect.signature(store.probe_once).parameters
    assert not second_called
    release.set()
    thread.join(5)
    assert not thread.is_alive() and len(first_result) == 1


def test_index_without_attempt_history_is_rejected(
    tmp_path: Path,
) -> None:
    policy = _h("index-only-policy")
    binding = _binding("index-only")
    root = tmp_path / "store"
    store = B300QualificationGraphEvidenceStore(root, policy)
    store.probe_once(binding, _producer(policy), deadline=_deadline())
    attempt = root / "attempts" / policy / binding.digest
    for record in attempt.iterdir():
        record.unlink()
    called = False

    def must_not_run(*_args):
        nonlocal called
        called = True
        raise AssertionError("producer entered on impossible index-only state")

    with pytest.raises(
        B300QualificationGraphEvidenceStoreError,
        match="no authenticated terminal",
    ):
        B300QualificationGraphEvidenceStore(root, policy).probe_once(
            binding, must_not_run, deadline=_deadline()
        )
    assert not called


@pytest.mark.parametrize("shape", ("mode600", "extra-link"))
def test_unexpected_partial_index_is_not_silently_repaired(
    tmp_path: Path,
    shape: str,
) -> None:
    policy = _h("partial-index-policy:" + shape)
    binding = _binding("partial-index-" + shape)
    root = tmp_path / "store"
    store = B300QualificationGraphEvidenceStore(root, policy)
    store.probe_once(binding, _producer(policy), deadline=_deadline())
    index = root / "indexes" / policy / f"{binding.digest}.json"
    if shape == "mode600":
        index.chmod(0o600)
        index.write_bytes(b"partial")
    else:
        os.link(index, index.parent / "unexpected-hardlink")
    with pytest.raises(B300QualificationGraphEvidenceStoreError, match="unsafe"):
        store.reopen(binding, deadline=_deadline())


def test_late_arm_publication_returns_hold_not_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = _h("late-arm-policy")
    binding = _binding("late-arm")

    def late_boundary(kind: str, phase: str) -> None:
        if (kind, phase) == ("arm", "parents_fsynced"):
            time.sleep(0.05)

    monkeypatch.setattr(store_module, "_publication_boundary", late_boundary)
    store = B300QualificationGraphEvidenceStore(tmp_path / "store", policy)
    with pytest.raises(B300QualificationGraphEvidenceHold, match="deadline expired"):
        store.arm(binding, deadline=_deadline(0.02))
    with pytest.raises(B300QualificationGraphEvidenceHold, match="armed"):
        store.probe_once(binding, _producer(policy), deadline=_deadline())


def test_artifact_publication_receives_same_absolute_deadline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = _h("artifact-deadline-policy")
    binding = _binding("artifact-deadline")
    deadline = _deadline()
    original = store_module.publish_evidence
    observed: list[float | None] = []

    def publish(*args, **kwargs):
        observed.append(kwargs.get("deadline"))
        return original(*args, **kwargs)

    monkeypatch.setattr(store_module, "publish_evidence", publish)
    B300QualificationGraphEvidenceStore(tmp_path / "store", policy).probe_once(
        binding, _producer(policy), deadline=deadline
    )
    assert observed == [deadline]
