"""Durability and fail-closed contracts for the graph evidence store."""

from __future__ import annotations

import fcntl
import hashlib
import multiprocessing
import os
import stat
import time
from pathlib import Path

import pytest

import cacheon.eval.b300_qualification_graph_evidence_store as graph_store_module
from cacheon.eval.b300_qualification_capabilities import (
    StructuredGraphShapeRecord,
    StructuredGraphVariantRecord,
)
from cacheon.eval.b300_qualification_graph_evidence_store import (
    ATTEMPT_SCHEMA,
    B300QualificationGraphAttemptToken,
    B300QualificationGraphEvidenceHold,
    B300QualificationGraphEvidenceStore,
    B300QualificationGraphEvidenceStoreError,
    B300QualificationGraphGenerationOutput,
    B300QualificationGraphPreEntryFailure,
)
from cacheon.eval.b300_qualification_graph_provider import (
    ARTIFACT_DOMAIN,
    ARTIFACT_MEDIA_TYPE,
    ARTIFACT_SCHEMA,
    B300QualificationGraphArtifact,
    B300QualificationGraphBinding,
)
from cacheon.eval.evidence_store import EvidenceArtifactRef, publish_evidence, reopen_evidence
from cacheon.stack_identity import canonical_json_bytes


def _h(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _deadline(seconds: float = 5.0) -> float:
    return time.monotonic() + seconds


def _binding(label: str, members: tuple[str, ...]) -> B300QualificationGraphBinding:
    return B300QualificationGraphBinding(
        reservation_digest=_h(label + ":reservation"),
        reservation_identity_digest=_h(label + ":reservation-identity"),
        candidate_binding_digest=_h(label + ":candidate"),
        screen_attempt=1,
        target_id=label,
        target_members=members,
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


def _artifact(
    binding: B300QualificationGraphBinding,
    policy: str,
    *,
    suffix: str = "base",
) -> B300QualificationGraphArtifact:
    variants = tuple(
        StructuredGraphVariantRecord(
            member,
            "commissioned",
            True,
            True,
            (
                StructuredGraphShapeRecord(
                    _h(f"{binding.digest}:{member}:{suffix}"),
                    True,
                    True,
                    True,
                    3,
                    True,
                    True,
                    False,
                ),
            ),
        )
        for member in binding.target_members
    )
    return B300QualificationGraphArtifact(binding, policy, 3, variants)


def _output(
    binding: B300QualificationGraphBinding,
    policy: str,
    token: B300QualificationGraphAttemptToken,
    *,
    suffix: str = "base",
) -> B300QualificationGraphGenerationOutput:
    return B300QualificationGraphGenerationOutput(token, _artifact(binding, policy, suffix=suffix))


def _index_path(root: Path, policy: str, binding: B300QualificationGraphBinding) -> Path:
    return root / "indexes" / policy / f"{binding.digest}.json"


def _attempt_dir(root: Path, policy: str, binding: B300QualificationGraphBinding) -> Path:
    return root / "attempts" / policy / binding.digest


def _staging_dir(root: Path, policy: str) -> Path:
    return root / "staging" / policy


def _attempt_row(
    policy: str,
    binding: B300QualificationGraphBinding,
    generation: int,
    nonce: str,
    record_type: str,
) -> dict[str, object]:
    return {
        "binding": binding.to_dict(),
        "binding_digest": binding.digest,
        "generation": generation,
        "generation_nonce": nonce,
        "record_type": record_type,
        "schema": ATTEMPT_SCHEMA,
        "verification_policy_digest": policy,
    }


def _write_sealed(path: Path, payload: bytes) -> None:
    path.write_bytes(payload)
    path.chmod(0o400)


def _write_attempt(
    directory: Path,
    generation: int,
    record_type: str,
    value: object,
) -> None:
    _write_sealed(
        directory / f"{generation:016d}.{record_type}.json",
        canonical_json_bytes(value),
    )


def _lock_is_free(root: Path, policy: str, binding: B300QualificationGraphBinding) -> bool:
    path = root / "locks" / policy / f"{binding.digest}.lock"
    descriptor = os.open(path, os.O_RDWR)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        return True
    finally:
        os.close(descriptor)


def _process_probe(
    root: str,
    policy: str,
    binding: B300QualificationGraphBinding,
    marker: str,
    start: multiprocessing.synchronize.Event,
    results: multiprocessing.queues.Queue,
) -> None:
    start.wait(10)
    store = B300QualificationGraphEvidenceStore(Path(root), policy)

    def produce(
        exact: B300QualificationGraphBinding,
        token: B300QualificationGraphAttemptToken,
        deadline: float,
    ) -> B300QualificationGraphGenerationOutput:
        assert deadline > time.monotonic()
        assert _lock_is_free(Path(root), policy, exact)
        descriptor = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(descriptor, b"produced\n")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        time.sleep(0.05)
        return _output(exact, policy, token)

    try:
        reference = store.probe_once(binding, produce, deadline=_deadline(10))
        results.put(("ok", reference.to_dict()))
    except B300QualificationGraphEvidenceHold as exc:
        results.put(("hold", str(exc)))
    except Exception as exc:  # pragma: no cover - surfaced in the parent assertion
        results.put(("error", repr(exc)))


def _hold_key_lock(
    root: str,
    policy: str,
    binding: B300QualificationGraphBinding,
    ready: multiprocessing.synchronize.Event,
    release: multiprocessing.synchronize.Event,
) -> None:
    B300QualificationGraphEvidenceStore(Path(root), policy)
    path = Path(root) / "locks" / policy / f"{binding.digest}.lock"
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
        os.fsync(descriptor)
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        ready.set()
        release.wait(10)
    finally:
        os.close(descriptor)


def _crash_publication(
    root: str,
    policy: str,
    binding: B300QualificationGraphBinding,
    kind: str,
    phase: str,
) -> None:
    def crash(found_kind: str, found_phase: str) -> None:
        if (found_kind, found_phase) == (kind, phase):
            os._exit(31)

    graph_store_module._publication_boundary = crash
    store = B300QualificationGraphEvidenceStore(Path(root), policy)

    def produce(
        exact: B300QualificationGraphBinding,
        token: B300QualificationGraphAttemptToken,
        _deadline_value: float,
    ) -> B300QualificationGraphGenerationOutput:
        return _output(exact, policy, token)

    store.probe_once(binding, produce, deadline=_deadline(10))


@pytest.fixture
def exact_inputs() -> tuple[str, B300QualificationGraphBinding]:
    return _h("policy:primary"), _binding("profile.alpha", ("slot.alpha", "slot.beta"))


def test_publish_reopen_restart_and_same_binding_are_idempotent(
    tmp_path: Path,
    exact_inputs: tuple[str, B300QualificationGraphBinding],
) -> None:
    policy, binding = exact_inputs
    root = tmp_path / "store"
    calls: list[B300QualificationGraphAttemptToken] = []

    def produce(
        exact: B300QualificationGraphBinding,
        token: B300QualificationGraphAttemptToken,
        deadline: float,
    ) -> B300QualificationGraphGenerationOutput:
        assert deadline > time.monotonic()
        calls.append(token)
        return _output(exact, policy, token)

    store = B300QualificationGraphEvidenceStore(root, policy)
    first = store.probe_once(binding, produce, deadline=_deadline())
    second = store.probe_once(binding, produce, deadline=_deadline())
    restarted = B300QualificationGraphEvidenceStore(root, policy)
    third = restarted.probe_once(binding, produce, deadline=_deadline())
    expected = _artifact(binding, policy).canonical_bytes

    assert len(calls) == 1
    assert first == second == third == restarted.reopen(binding, deadline=_deadline())
    assert first == EvidenceArtifactRef(
        ARTIFACT_DOMAIN,
        hashlib.sha256(expected).hexdigest(),
        len(expected),
        ARTIFACT_MEDIA_TYPE,
        ARTIFACT_SCHEMA,
    )
    assert reopen_evidence(restarted.evidence_root, first) == expected
    index = _index_path(root, policy, binding)
    records = sorted(_attempt_dir(root, policy, binding).iterdir())
    assert [path.name for path in records] == [
        "0000000000000001.armed.json",
        "0000000000000001.output.json",
        "0000000000000001.terminal.json",
    ]
    assert stat.S_IMODE(index.stat().st_mode) == 0o400 and index.stat().st_nlink == 1
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o400 and path.stat().st_nlink == 1 for path in records)
    assert not list(_staging_dir(root, policy).iterdir())


def test_direct_arm_returns_nonce_and_finalize_requires_exact_token(
    tmp_path: Path,
    exact_inputs: tuple[str, B300QualificationGraphBinding],
) -> None:
    policy, binding = exact_inputs
    store = B300QualificationGraphEvidenceStore(tmp_path / "store", policy)
    token = store.arm(binding, deadline=_deadline())
    assert token.generation == 1
    assert token.binding_digest == binding.digest
    assert token.verification_policy_digest == policy
    assert len(token.nonce) == 64

    wrong = B300QualificationGraphAttemptToken(1, _h("wrong"), binding.digest, policy)
    with pytest.raises(B300QualificationGraphEvidenceStoreError, match="token differs"):
        store.finalize(binding, _output(binding, policy, wrong), deadline=_deadline())
    reference = store.finalize(binding, _output(binding, policy, token), deadline=_deadline())
    assert store.reopen(binding, deadline=_deadline()) == reference


def test_callbacks_receive_exact_token_and_deadline_outside_lock(
    tmp_path: Path,
    exact_inputs: tuple[str, B300QualificationGraphBinding],
) -> None:
    policy, binding = exact_inputs
    root = tmp_path / "store"
    deadline = _deadline()
    seen: list[tuple[B300QualificationGraphAttemptToken, float]] = []

    def produce(
        exact: B300QualificationGraphBinding,
        token: B300QualificationGraphAttemptToken,
        exact_deadline: float,
    ) -> B300QualificationGraphGenerationOutput:
        assert exact == binding
        assert exact_deadline is deadline
        assert _lock_is_free(root, policy, binding)
        seen.append((token, exact_deadline))
        return _output(exact, policy, token)

    B300QualificationGraphEvidenceStore(root, policy).probe_once(binding, produce, deadline=deadline)
    assert len(seen) == 1


def test_arbitrary_pre_entry_digest_never_authorizes_retry(
    tmp_path: Path,
    exact_inputs: tuple[str, B300QualificationGraphBinding],
) -> None:
    policy, binding = exact_inputs
    root = tmp_path / "store"
    calls = 0

    def untrusted_claim(
        _binding_value: B300QualificationGraphBinding,
        _token: B300QualificationGraphAttemptToken,
        _deadline_value: float,
    ) -> B300QualificationGraphGenerationOutput:
        nonlocal calls
        calls += 1
        raise B300QualificationGraphPreEntryFailure(_h("arbitrary-proof"))

    with pytest.raises(B300QualificationGraphEvidenceHold, match="ambiguous"):
        B300QualificationGraphEvidenceStore(root, policy).probe_once(
            binding, untrusted_claim, deadline=_deadline()
        )

    def would_succeed(
        exact: B300QualificationGraphBinding,
        token: B300QualificationGraphAttemptToken,
        _deadline_value: float,
    ) -> B300QualificationGraphGenerationOutput:
        nonlocal calls
        calls += 1
        return _output(exact, policy, token)

    with pytest.raises(B300QualificationGraphEvidenceHold, match="armed"):
        B300QualificationGraphEvidenceStore(root, policy).probe_once(
            binding, would_succeed, deadline=_deadline()
        )
    assert calls == 1
    assert sorted(path.name for path in _attempt_dir(root, policy, binding).iterdir()) == [
        "0000000000000001.armed.json"
    ]


def test_prearm_validation_failure_creates_no_generation(
    tmp_path: Path,
    exact_inputs: tuple[str, B300QualificationGraphBinding],
) -> None:
    policy, binding = exact_inputs
    root = tmp_path / "store"
    store = B300QualificationGraphEvidenceStore(root, policy)
    with pytest.raises(B300QualificationGraphEvidenceStoreError, match="callable"):
        store.probe_once(binding, None, deadline=_deadline())  # type: ignore[arg-type]
    assert not os.path.lexists(_attempt_dir(root, policy, binding))


def test_armed_without_terminal_never_invokes_producer(
    tmp_path: Path,
    exact_inputs: tuple[str, B300QualificationGraphBinding],
) -> None:
    policy, binding = exact_inputs
    store = B300QualificationGraphEvidenceStore(tmp_path / "store", policy)
    store.arm(binding, deadline=_deadline())
    called = False

    def must_not_run(*_args: object) -> B300QualificationGraphGenerationOutput:
        nonlocal called
        called = True
        raise AssertionError("producer re-entered")

    with pytest.raises(B300QualificationGraphEvidenceHold, match="armed"):
        store.probe_once(binding, must_not_run, deadline=_deadline())
    assert not called


def test_callback_result_returned_after_deadline_remains_hold(
    tmp_path: Path,
    exact_inputs: tuple[str, B300QualificationGraphBinding],
) -> None:
    policy, binding = exact_inputs
    calls = 0

    def late(
        exact: B300QualificationGraphBinding,
        token: B300QualificationGraphAttemptToken,
        _deadline_value: float,
    ) -> B300QualificationGraphGenerationOutput:
        nonlocal calls
        calls += 1
        time.sleep(0.04)
        return _output(exact, policy, token)

    store = B300QualificationGraphEvidenceStore(tmp_path / "store", policy)
    with pytest.raises(B300QualificationGraphEvidenceHold, match="deadline expired"):
        store.probe_once(binding, late, deadline=_deadline(0.01))
    with pytest.raises(B300QualificationGraphEvidenceHold, match="armed"):
        store.probe_once(binding, late, deadline=_deadline())
    assert calls == 1


def test_lock_acquisition_obeys_absolute_deadline(
    tmp_path: Path,
    exact_inputs: tuple[str, B300QualificationGraphBinding],
) -> None:
    policy, binding = exact_inputs
    root = tmp_path / "store"
    B300QualificationGraphEvidenceStore(root, policy)
    context = multiprocessing.get_context("fork")
    ready = context.Event()
    release = context.Event()
    process = context.Process(target=_hold_key_lock, args=(str(root), policy, binding, ready, release))
    process.start()
    assert ready.wait(5)
    started = time.monotonic()
    with pytest.raises(B300QualificationGraphEvidenceHold, match="lock deadline"):
        B300QualificationGraphEvidenceStore(root, policy).arm(binding, deadline=_deadline(0.05))
    elapsed = time.monotonic() - started
    release.set()
    process.join(5)
    assert process.exitcode == 0
    assert 0.04 <= elapsed < 0.5


def test_two_processes_create_one_arm_and_invoke_one_producer(
    tmp_path: Path,
    exact_inputs: tuple[str, B300QualificationGraphBinding],
) -> None:
    policy, binding = exact_inputs
    root = tmp_path / "store"
    marker = tmp_path / "producer-calls"
    context = multiprocessing.get_context("fork")
    start = context.Event()
    results = context.Queue()
    processes = [
        context.Process(
            target=_process_probe,
            args=(str(root), policy, binding, str(marker), start, results),
        )
        for _ in range(2)
    ]
    for process in processes:
        process.start()
    start.set()
    for process in processes:
        process.join(15)
        assert not process.is_alive()
        assert process.exitcode == 0
    observed = [results.get(timeout=2) for _ in processes]
    assert sorted(status for status, _ in observed) in (["hold", "ok"], ["ok", "ok"])
    assert marker.read_bytes() == b"produced\n"
    assert sorted(path.name for path in _attempt_dir(root, policy, binding).iterdir()) == [
        "0000000000000001.armed.json",
        "0000000000000001.output.json",
        "0000000000000001.terminal.json",
    ]
    B300QualificationGraphEvidenceStore(root, policy).reopen(binding, deadline=_deadline())


def test_terminal_success_repairs_absent_index_without_producer(
    tmp_path: Path,
    exact_inputs: tuple[str, B300QualificationGraphBinding],
) -> None:
    policy, binding = exact_inputs
    root = tmp_path / "store"
    store = B300QualificationGraphEvidenceStore(root, policy)
    reference = store.probe_once(
        binding,
        lambda exact, token, _deadline_value: _output(exact, policy, token),
        deadline=_deadline(),
    )
    index = _index_path(root, policy, binding)
    index.unlink()
    called = False

    def must_not_run(*_args: object) -> B300QualificationGraphGenerationOutput:
        nonlocal called
        called = True
        raise AssertionError("producer re-entered")

    repaired = B300QualificationGraphEvidenceStore(root, policy).probe_once(
        binding, must_not_run, deadline=_deadline()
    )
    assert not called
    assert repaired == reference
    assert index.stat().st_nlink == 1 and stat.S_IMODE(index.stat().st_mode) == 0o400


@pytest.mark.parametrize("kind", ("arm", "output", "terminal", "index"))
@pytest.mark.parametrize("phase", ("temp_created", "renamed", "parents_fsynced"))
def test_restart_is_safe_at_each_owned_publication_crash_boundary(
    tmp_path: Path,
    exact_inputs: tuple[str, B300QualificationGraphBinding],
    kind: str,
    phase: str,
) -> None:
    policy, binding = exact_inputs
    root = tmp_path / "store"
    context = multiprocessing.get_context("fork")
    process = context.Process(
        target=_crash_publication,
        args=(str(root), policy, binding, kind, phase),
    )
    process.start()
    process.join(15)
    assert not process.is_alive()
    assert process.exitcode == 31

    called = False

    def producer(
        exact: B300QualificationGraphBinding,
        token: B300QualificationGraphAttemptToken,
        _deadline_value: float,
    ) -> B300QualificationGraphGenerationOutput:
        nonlocal called
        called = True
        return _output(exact, policy, token)

    restarted = B300QualificationGraphEvidenceStore(root, policy)
    if kind == "arm" and phase == "temp_created":
        restarted.probe_once(binding, producer, deadline=_deadline())
        assert called
    elif kind == "arm":
        with pytest.raises(B300QualificationGraphEvidenceHold, match="armed"):
            restarted.probe_once(binding, producer, deadline=_deadline())
        assert not called
    elif kind == "output" and phase == "temp_created":
        with pytest.raises(B300QualificationGraphEvidenceHold, match="armed"):
            restarted.probe_once(binding, producer, deadline=_deadline())
        assert not called
    else:
        restarted.probe_once(binding, producer, deadline=_deadline())
        assert not called

    attempt = _attempt_dir(root, policy, binding)
    assert all(not path.name.startswith(".") and path.stat().st_nlink == 1 for path in attempt.iterdir())
    assert all(path.parent == _staging_dir(root, policy) for path in _staging_dir(root, policy).iterdir())


def test_abandoned_staging_file_is_harmless(
    tmp_path: Path,
    exact_inputs: tuple[str, B300QualificationGraphBinding],
) -> None:
    policy, binding = exact_inputs
    root = tmp_path / "store"
    store = B300QualificationGraphEvidenceStore(root, policy)
    abandoned = _staging_dir(root, policy) / ".arm.abandoned.tmp"
    _write_sealed(abandoned, b"not-authoritative")
    reference = store.probe_once(
        binding,
        lambda exact, token, _deadline_value: _output(exact, policy, token),
        deadline=_deadline(),
    )
    assert store.reopen(binding, deadline=_deadline()) == reference
    assert abandoned.read_bytes() == b"not-authoritative"


@pytest.mark.parametrize(
    "scenario",
    ("unexpected", "malformed", "gap", "terminal-only", "nonce-mismatch", "retryable", "second-generation"),
)
def test_malformed_gapped_or_reordered_history_fails_closed(
    tmp_path: Path,
    exact_inputs: tuple[str, B300QualificationGraphBinding],
    scenario: str,
) -> None:
    policy, binding = exact_inputs
    root = tmp_path / "store"
    store = B300QualificationGraphEvidenceStore(root, policy)
    directory = _attempt_dir(root, policy, binding)
    directory.mkdir(mode=0o700)
    nonce = _h("nonce")
    arm = _attempt_row(policy, binding, 1, nonce, "armed")
    artifact = _artifact(binding, policy)
    reference = publish_evidence(
        store.evidence_root,
        artifact.canonical_bytes,
        domain=ARTIFACT_DOMAIN,
        media_type=ARTIFACT_MEDIA_TYPE,
        schema=ARTIFACT_SCHEMA,
    )
    success = {
        **_attempt_row(policy, binding, 1, nonce, "terminal"),
        "artifact_reference": reference.to_dict(),
        "outcome": "success",
    }
    if scenario == "unexpected":
        _write_sealed(directory / ".temporary", b"x")
    elif scenario == "malformed":
        _write_attempt(directory, 1, "armed", {})
    elif scenario == "gap":
        _write_attempt(directory, 2, "armed", _attempt_row(policy, binding, 2, nonce, "armed"))
    elif scenario == "terminal-only":
        _write_attempt(directory, 1, "terminal", success)
    elif scenario == "nonce-mismatch":
        _write_attempt(directory, 1, "armed", arm)
        _write_attempt(
            directory,
            1,
            "terminal",
            {**success, "generation_nonce": _h("other-nonce")},
        )
    elif scenario == "retryable":
        _write_attempt(directory, 1, "armed", arm)
        _write_attempt(
            directory,
            1,
            "terminal",
            {
                **_attempt_row(policy, binding, 1, nonce, "terminal"),
                "outcome": "retryable",
                "proof_digest": _h("arbitrary"),
            },
        )
    else:
        _write_attempt(directory, 1, "armed", arm)
        _write_attempt(directory, 1, "terminal", success)
        _write_attempt(directory, 2, "armed", _attempt_row(policy, binding, 2, _h("next"), "armed"))
    called = False

    def must_not_run(*_args: object) -> B300QualificationGraphGenerationOutput:
        nonlocal called
        called = True
        raise AssertionError("producer ran on malformed history")

    with pytest.raises(
        B300QualificationGraphEvidenceStoreError,
        match="attempt|generation|terminal|output",
    ):
        store.probe_once(binding, must_not_run, deadline=_deadline())
    assert not called


def test_two_distinct_target_bindings_never_cross_authorize(
    tmp_path: Path,
    exact_inputs: tuple[str, B300QualificationGraphBinding],
) -> None:
    policy, first = exact_inputs
    second = _binding("profile.omega", ("slot.omega",))
    store = B300QualificationGraphEvidenceStore(tmp_path / "store", policy)
    calls: list[str] = []

    def produce(
        exact: B300QualificationGraphBinding,
        token: B300QualificationGraphAttemptToken,
        _deadline_value: float,
    ) -> B300QualificationGraphGenerationOutput:
        calls.append(exact.digest)
        return _output(exact, policy, token)

    first_ref = store.probe_once(first, produce, deadline=_deadline())
    second_ref = store.probe_once(second, produce, deadline=_deadline())
    assert calls == [first.digest, second.digest]
    assert first_ref != second_ref
    assert store.reopen(first, deadline=_deadline()) == first_ref
    assert store.reopen(second, deadline=_deadline()) == second_ref


@pytest.mark.parametrize("kind", ("wrong-binding", "wrong-policy"))
def test_generation_bound_output_still_requires_exact_binding_and_policy(
    tmp_path: Path,
    kind: str,
) -> None:
    policy = _h("policy:callback-mismatch")
    binding = _binding("profile.callback-mismatch", ("slot.callback-mismatch",))
    other = _binding("profile.callback-other", ("slot.callback-other",))
    store = B300QualificationGraphEvidenceStore(tmp_path / "store", policy)

    def mismatch(
        exact: B300QualificationGraphBinding,
        token: B300QualificationGraphAttemptToken,
        _deadline_value: float,
    ) -> B300QualificationGraphGenerationOutput:
        artifact = (
            _artifact(other, policy)
            if kind == "wrong-binding"
            else _artifact(exact, _h("wrong-policy"))
        )
        return B300QualificationGraphGenerationOutput(token, artifact)

    with pytest.raises(B300QualificationGraphEvidenceHold, match="binding|policy"):
        store.probe_once(binding, mismatch, deadline=_deadline())
    assert not os.path.lexists(_index_path(tmp_path / "store", policy, binding))
    assert [path.name for path in _attempt_dir(tmp_path / "store", policy, binding).iterdir()] == [
        "0000000000000001.armed.json"
    ]


def test_unsafe_existing_key_lock_fails_before_producer(
    tmp_path: Path,
    exact_inputs: tuple[str, B300QualificationGraphBinding],
) -> None:
    policy, binding = exact_inputs
    root = tmp_path / "store"
    store = B300QualificationGraphEvidenceStore(root, policy)
    lock = root / "locks" / policy / f"{binding.digest}.lock"
    lock.write_bytes(b"")
    lock.chmod(0o644)
    called = False

    def must_not_run(*_args: object) -> B300QualificationGraphGenerationOutput:
        nonlocal called
        called = True
        raise AssertionError("producer ran with an unsafe lock")

    with pytest.raises(B300QualificationGraphEvidenceStoreError, match="key lock is unsafe"):
        store.probe_once(binding, must_not_run, deadline=_deadline())
    assert not called
    assert stat.S_IMODE(lock.stat().st_mode) == 0o644


def test_content_artifact_tampering_is_rejected(
    tmp_path: Path,
    exact_inputs: tuple[str, B300QualificationGraphBinding],
) -> None:
    policy, binding = exact_inputs
    store = B300QualificationGraphEvidenceStore(tmp_path / "store", policy)
    reference = store.probe_once(
        binding,
        lambda exact, token, _deadline_value: _output(exact, policy, token),
        deadline=_deadline(),
    )
    target = store.evidence_root / reference.domain / reference.sha256[:2] / reference.sha256
    target.chmod(0o600)
    target.write_bytes(b"X" * reference.size)
    target.chmod(0o400)
    with pytest.raises(B300QualificationGraphEvidenceStoreError, match="did not reopen exactly"):
        store.reopen(binding, deadline=_deadline())


@pytest.mark.parametrize("kind", ("relative", "symlink", "mode"))
def test_rejects_unsafe_root(tmp_path: Path, kind: str) -> None:
    policy = _h("policy")
    if kind == "relative":
        root = Path("relative/store")
        match = "canonical and absolute"
    elif kind == "symlink":
        real = tmp_path / "real"
        real.mkdir(mode=0o700)
        root = tmp_path / "alias"
        root.symlink_to(real, target_is_directory=True)
        match = "canonical, nonsymlink"
    else:
        root = tmp_path / "unsafe"
        root.mkdir(mode=0o755)
        match = "mode 0700"
    with pytest.raises(B300QualificationGraphEvidenceStoreError, match=match):
        B300QualificationGraphEvidenceStore(root, policy)


def test_sealed_divergent_index_is_not_silently_repaired(
    tmp_path: Path,
    exact_inputs: tuple[str, B300QualificationGraphBinding],
) -> None:
    policy, binding = exact_inputs
    root = tmp_path / "store"
    store = B300QualificationGraphEvidenceStore(root, policy)
    store.probe_once(
        binding,
        lambda exact, token, _deadline_value: _output(exact, policy, token),
        deadline=_deadline(),
    )
    index = _index_path(root, policy, binding)
    index.chmod(0o600)
    index.write_bytes(canonical_json_bytes({"sealed": "divergent"}))
    index.chmod(0o400)
    with pytest.raises(B300QualificationGraphEvidenceStoreError, match="divergent|closed schema"):
        store.reopen(binding, deadline=_deadline())
