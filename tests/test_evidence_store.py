from __future__ import annotations

import hashlib
import multiprocessing
import os
import stat
import time
from dataclasses import replace
from pathlib import Path

import pytest

from cacheon.eval.evidence_store import (
    HARD_MAX_EVIDENCE_BYTES,
    EvidenceArtifactRef,
    EvidenceStoreError,
    prepare_evidence_root,
    publish_canonical_json_evidence,
    publish_evidence,
    reopen_evidence,
)
from cacheon.stack_identity import canonical_json_bytes


DOMAIN = "qualification.raw"
SCHEMA = "cacheon.qualification.raw.v1"
MEDIA = "application/vnd.cacheon.qualification+json"


def _target(root: Path, reference: EvidenceArtifactRef) -> Path:
    return root / reference.domain / reference.sha256[:2] / reference.sha256


def _publish(
    root: Path,
    payload: bytes = b"sealed evidence",
    *,
    deadline: float | None = None,
) -> EvidenceArtifactRef:
    return publish_evidence(
        root,
        payload,
        domain=DOMAIN,
        media_type=MEDIA,
        schema=SCHEMA,
        deadline=deadline,
    )


def _crash_publish(root: str, payload: bytes, phase: str) -> None:
    import cacheon.eval.evidence_store as store

    def crash(found: str) -> None:
        if found == phase:
            os._exit(31)

    store._publication_boundary = crash
    _publish(Path(root), payload)


def _concurrent_publish(root: str, payload: bytes, start, results) -> None:
    try:
        if not start.wait(10):
            raise RuntimeError("publication start gate timed out")
        reference = _publish(Path(root), payload)
        results.put(("ok", reference.to_dict()))
    except BaseException as exc:
        results.put(("error", repr(exc)))


def _hold_publication_lock(root: str, payload: bytes, ready, release) -> None:
    import cacheon.eval.evidence_store as store

    reference = EvidenceArtifactRef(
        DOMAIN, hashlib.sha256(payload).hexdigest(), len(payload), MEDIA, SCHEMA
    )
    evidence_root = prepare_evidence_root(Path(root))
    store._target(evidence_root, reference, create=True)
    lock, _stage, _staging = store._staging_paths(evidence_root, reference)
    with store._publication_lock(lock, deadline=None):
        ready.set()
        release.wait(10)


def test_reference_is_strict_and_round_trips() -> None:
    payload = b"abc"
    reference = EvidenceArtifactRef(
        DOMAIN, hashlib.sha256(payload).hexdigest(), len(payload), MEDIA, SCHEMA
    )
    assert EvidenceArtifactRef.from_dict(reference.to_dict()) == reference
    with pytest.raises(EvidenceStoreError, match="not closed"):
        EvidenceArtifactRef.from_dict({**reference.to_dict(), "headline": 9})
    with pytest.raises(EvidenceStoreError, match="not closed"):
        EvidenceArtifactRef.from_dict([reference.to_dict()])


@pytest.mark.parametrize(
    "changes",
    (
        {"domain": "../escape"},
        {"sha256": "A" * 64},
        {"size": True},
        {"size": HARD_MAX_EVIDENCE_BYTES + 1},
        {"media_type": "application/json; charset=utf-8"},
        {"schema": "../schema"},
    ),
)
def test_reference_rejects_noncanonical_fields(changes) -> None:
    values = dict(
        domain=DOMAIN,
        sha256="a" * 64,
        size=1,
        media_type=MEDIA,
        schema=SCHEMA,
    )
    values.update(changes)
    with pytest.raises(EvidenceStoreError):
        EvidenceArtifactRef(**values)


def test_prepare_root_is_private_owned_and_idempotent(tmp_path: Path) -> None:
    root = prepare_evidence_root(tmp_path / "evidence")
    assert root == prepare_evidence_root(root)
    info = root.lstat()
    assert stat.S_ISDIR(info.st_mode)
    assert stat.S_IMODE(info.st_mode) == 0o700
    assert info.st_uid == os.geteuid()


def test_prepare_rejects_relative_symlink_and_unsafe_roots(tmp_path: Path) -> None:
    with pytest.raises(EvidenceStoreError, match="canonical and absolute"):
        prepare_evidence_root(Path("relative/evidence"))

    real = tmp_path / "real"
    real.mkdir(mode=0o700)
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(EvidenceStoreError, match="unsafe|symlink"):
        prepare_evidence_root(link)

    unsafe = tmp_path / "unsafe"
    unsafe.mkdir()
    unsafe.chmod(0o755)
    with pytest.raises(EvidenceStoreError, match="owner or mode"):
        prepare_evidence_root(unsafe)


def test_publish_reopen_and_exact_duplicate_are_content_addressed(tmp_path: Path) -> None:
    root = tmp_path / "evidence"
    payload = b"opaque\x00bytes"
    first = _publish(root, payload)
    target = _target(root, first)
    before = target.lstat()
    second = _publish(root, payload)
    after = target.lstat()

    assert first == second
    assert first.sha256 == hashlib.sha256(payload).hexdigest()
    assert first.size == len(payload)
    assert reopen_evidence(root, first) == payload
    assert before.st_ino == after.st_ino
    assert after.st_nlink == 1
    assert after.st_uid == os.geteuid()
    assert stat.S_IMODE(after.st_mode) == 0o400
    assert not list(target.parent.glob(".*.tmp.*"))


def test_publish_enforces_type_and_size_caps(tmp_path: Path) -> None:
    root = tmp_path / "evidence"
    with pytest.raises(EvidenceStoreError, match="exact bytes"):
        publish_evidence(
            root, bytearray(b"x"), domain=DOMAIN, media_type=MEDIA, schema=SCHEMA
        )
    with pytest.raises(EvidenceStoreError, match="size limit"):
        publish_evidence(
            root, b"xx", domain=DOMAIN, media_type=MEDIA, schema=SCHEMA, max_bytes=1
        )
    reference = _publish(root, b"xx")
    with pytest.raises(EvidenceStoreError, match="reference exceeds"):
        reopen_evidence(root, reference, max_bytes=1)


def test_publish_deadline_is_exact_finite_and_future(tmp_path: Path) -> None:
    root = tmp_path / "evidence"
    with pytest.raises(EvidenceStoreError, match="finite absolute monotonic float"):
        _publish(root, deadline=1)  # type: ignore[arg-type]
    with pytest.raises(EvidenceStoreError, match="deadline expired"):
        _publish(root, deadline=time.monotonic() - 1.0)


def test_held_publication_lock_respects_absolute_deadline(tmp_path: Path) -> None:
    root = tmp_path / "evidence"
    payload = b"deadline-bound evidence"
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    release = context.Event()
    process = context.Process(
        target=_hold_publication_lock,
        args=(str(root), payload, ready, release),
    )
    process.start()
    try:
        assert ready.wait(10)
        started = time.monotonic()
        with pytest.raises(EvidenceStoreError, match="deadline expired"):
            _publish(root, payload, deadline=started + 0.12)
        elapsed = time.monotonic() - started
        assert 0.10 <= elapsed < 0.40
    finally:
        release.set()
        process.join(10)
    assert process.exitcode == 0
    reference = _publish(root, payload, deadline=time.monotonic() + 1.0)
    assert reopen_evidence(root, reference) == payload


def test_canonical_json_is_convenience_not_semantic_authority(tmp_path: Path) -> None:
    root = tmp_path / "evidence"
    reference = publish_canonical_json_evidence(
        root,
        {"z": [2, 1], "a": "value"},
        domain=DOMAIN,
        schema=SCHEMA,
    )
    assert reopen_evidence(root, reference) == canonical_json_bytes(
        {"a": "value", "z": [2, 1]}
    )

    # A JSON media label does not make the byte store parse or approve JSON.
    opaque = publish_evidence(
        root,
        b"not json",
        domain=DOMAIN,
        media_type="application/json",
        schema=SCHEMA,
    )
    assert reopen_evidence(root, opaque) == b"not json"


def test_reopen_rejects_symlink_and_nonregular_artifacts(tmp_path: Path) -> None:
    root = tmp_path / "evidence"
    reference = _publish(root)
    target = _target(root, reference)
    target.chmod(0o600)
    target.unlink()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.write_bytes(b"sealed evidence")
    target.symlink_to(elsewhere)
    with pytest.raises(EvidenceStoreError, match="unsafe shape"):
        reopen_evidence(root, reference)

    target.unlink()
    target.mkdir(mode=0o400)
    with pytest.raises(EvidenceStoreError, match="unsafe shape"):
        reopen_evidence(root, reference)


def test_reopen_rejects_hardlinks_and_unsafe_file_mode(tmp_path: Path) -> None:
    root = tmp_path / "evidence"
    reference = _publish(root)
    target = _target(root, reference)
    peer = tmp_path / "peer"
    os.link(target, peer)
    with pytest.raises(EvidenceStoreError, match="unsafe shape"):
        reopen_evidence(root, reference)
    peer.unlink()

    target.chmod(0o600)
    with pytest.raises(EvidenceStoreError, match="unsafe shape"):
        reopen_evidence(root, reference)


def test_reopen_rejects_truncation_growth_and_digest_mismatch(tmp_path: Path) -> None:
    root = tmp_path / "evidence"
    payload = b"sealed evidence"
    reference = _publish(root, payload)
    target = _target(root, reference)

    target.chmod(0o600)
    target.write_bytes(payload[:-1])
    target.chmod(0o400)
    with pytest.raises(EvidenceStoreError, match="size"):
        reopen_evidence(root, reference)

    target.chmod(0o600)
    target.write_bytes(b"X" * len(payload))
    target.chmod(0o400)
    with pytest.raises(EvidenceStoreError, match="digest mismatch"):
        reopen_evidence(root, reference)

    wrong_size = replace(reference, size=reference.size - 1)
    with pytest.raises(EvidenceStoreError, match="size"):
        reopen_evidence(root, wrong_size)


def test_reopen_rejects_unsafe_directory_and_root_owner(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "evidence"
    reference = _publish(root)
    domain = root / DOMAIN
    domain.chmod(0o755)
    with pytest.raises(EvidenceStoreError, match="owner or mode"):
        reopen_evidence(root, reference)
    domain.chmod(0o700)

    monkeypatch.setattr(os, "geteuid", lambda: root.lstat().st_uid + 1)
    with pytest.raises(EvidenceStoreError, match="owner or mode"):
        reopen_evidence(root, reference)


def test_reopen_rejects_path_escape_through_store_symlink(tmp_path: Path) -> None:
    real = prepare_evidence_root(tmp_path / "real")
    reference = _publish(real)
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    with pytest.raises(EvidenceStoreError, match="unsafe|symlink"):
        reopen_evidence(alias, reference)


def test_atomic_publish_failure_preserves_only_reusable_non_authoritative_stage(
    tmp_path: Path, monkeypatch
) -> None:
    import cacheon.eval.evidence_store as store

    root = tmp_path / "evidence"
    payload = b"sealed evidence"
    real_rename = store._atomic_rename_noreplace
    calls = 0

    def fail_target_rename(source: Path, target: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected rename failure")
        real_rename(source, target)

    monkeypatch.setattr(store, "_atomic_rename_noreplace", fail_target_rename)
    with pytest.raises(EvidenceStoreError, match="cannot publish"):
        _publish(root, payload)
    reference = EvidenceArtifactRef(
        DOMAIN, hashlib.sha256(payload).hexdigest(), len(payload), MEDIA, SCHEMA
    )
    assert not os.path.lexists(_target(root, reference))
    stages = list((root / ".staging" / DOMAIN).glob("*.stage"))
    assert len(stages) == 1 and stages[0].read_bytes() == payload

    assert _publish(root, payload) == reference
    assert reopen_evidence(root, reference) == payload
    assert not list((root / ".staging" / DOMAIN).glob("*.stage"))


def test_staging_cleanup_error_does_not_mask_publication_error(
    tmp_path: Path, monkeypatch
) -> None:
    import cacheon.eval.evidence_store as store

    root = tmp_path / "evidence"
    real_unlink = Path.unlink

    def fail_rename(_source: Path, _target_path: Path) -> None:
        raise OSError("primary rename failure")

    def fail_work_cleanup(path: Path, *args, **kwargs) -> None:
        if ".tmp." in path.name:
            raise OSError("secondary cleanup failure")
        real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(store, "_atomic_rename_noreplace", fail_rename)
    monkeypatch.setattr(Path, "unlink", fail_work_cleanup)
    with pytest.raises(EvidenceStoreError, match="primary rename failure"):
        _publish(root)


def test_descriptor_close_error_does_not_mask_staging_error(
    tmp_path: Path, monkeypatch
) -> None:
    import cacheon.eval.evidence_store as store

    payload = b"close-error evidence"
    reference = EvidenceArtifactRef(
        DOMAIN, hashlib.sha256(payload).hexdigest(), len(payload), MEDIA, SCHEMA
    )
    stage = tmp_path / "stage"
    real_close = store._close_descriptor

    def fail_write(_descriptor: int, _view) -> int:
        raise OSError("primary write failure")

    def close_then_fail(descriptor: int) -> None:
        real_close(descriptor)
        raise OSError("secondary close failure")

    monkeypatch.setattr(store.os, "write", fail_write)
    monkeypatch.setattr(store, "_close_descriptor", close_then_fail)
    with pytest.raises(EvidenceStoreError, match="primary write failure"):
        store._write_stage(stage, payload, reference)


@pytest.mark.parametrize(
    "phase",
    (
        "staged_temp_created",
        "rename_complete_before_directory_fsync",
        "directory_fsync_complete",
    ),
)
def test_real_process_crash_boundaries_republish_without_cleanup(
    tmp_path: Path, phase: str
) -> None:
    root = tmp_path / phase
    payload = f"crash:{phase}".encode()
    context = multiprocessing.get_context("spawn")
    process = context.Process(
        target=_crash_publish,
        args=(str(root), payload, phase),
    )
    process.start()
    process.join(15)
    assert process.exitcode == 31

    reference = _publish(root, payload)
    assert reopen_evidence(root, reference) == payload
    target = _target(root, reference)
    assert target.lstat().st_nlink == 1
    assert stat.S_IMODE(target.lstat().st_mode) == 0o400
    assert not list((root / ".staging" / DOMAIN).glob("*.stage"))


def test_concurrent_same_payload_publishers_converge(tmp_path: Path) -> None:
    root = tmp_path / "evidence"
    payload = b"same exact concurrent evidence"
    context = multiprocessing.get_context("spawn")
    start = context.Event()
    results = context.Queue()
    processes = [
        context.Process(
            target=_concurrent_publish,
            args=(str(root), payload, start, results),
        )
        for _ in range(4)
    ]
    for process in processes:
        process.start()
    start.set()
    for process in processes:
        process.join(15)
    assert [process.exitcode for process in processes] == [0, 0, 0, 0]
    rows = [results.get(timeout=2) for _ in processes]
    assert {status for status, _value in rows} == {"ok"}
    references = [EvidenceArtifactRef.from_dict(value) for _status, value in rows]
    assert len(set(references)) == 1
    reference = references[0]
    assert reopen_evidence(root, reference) == payload
    assert _target(root, reference).lstat().st_nlink == 1
    assert not list((root / ".staging" / DOMAIN).glob("*.stage"))


def test_publish_rejects_divergent_or_unsafe_existing_target(tmp_path: Path) -> None:
    root = tmp_path / "evidence"
    payload = b"expected bytes"
    reference = _publish(root, payload)
    target = _target(root, reference)
    target.chmod(0o600)
    target.write_bytes(b"different byte")
    target.chmod(0o400)
    with pytest.raises(EvidenceStoreError, match="digest mismatch"):
        _publish(root, payload)
    assert target.read_bytes() == b"different byte"

    target.chmod(0o600)
    target.unlink()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.write_bytes(payload)
    target.symlink_to(elsewhere)
    with pytest.raises(EvidenceStoreError, match="unsafe shape"):
        _publish(root, payload)
    assert target.is_symlink()


def test_unsafe_non_authoritative_stage_cannot_poison_canonical_target(
    tmp_path: Path,
) -> None:
    root = tmp_path / "evidence"
    payload = b"canonical evidence wins"
    reference = _publish(root, payload)
    stage = root / ".staging" / DOMAIN / f"{reference.sha256}.stage"
    elsewhere = tmp_path / "untrusted-stage"
    elsewhere.write_bytes(b"not authoritative")
    stage.symlink_to(elsewhere)

    assert _publish(root, payload) == reference
    assert reopen_evidence(root, reference) == payload
    assert stage.is_symlink()


def test_hardlinked_existing_artifact_fails_closed(tmp_path: Path) -> None:
    # The former link-before-unlink publication window was drained from every
    # retained evidence root; a multiply-linked target is now simply unsafe.
    root = tmp_path / "evidence"
    payload = b"legacy exact bytes"
    reference = _publish(root, payload)
    target = _target(root, reference)
    peer = target.with_name(f".{target.name}.tmp.1234.{'a' * 32}")
    os.link(target, peer)
    assert target.lstat().st_nlink == 2
    with pytest.raises(EvidenceStoreError, match="unsafe shape"):
        _publish(root, payload)
    assert target.lstat().st_nlink == 2
    assert peer.exists()


def test_same_payload_in_second_domain_has_independent_path(tmp_path: Path) -> None:
    root = tmp_path / "evidence"
    payload = b"domain-neutral bytes"
    first = _publish(root, payload)
    second = publish_evidence(
        root,
        payload,
        domain="quality.audit",
        media_type=MEDIA,
        schema=SCHEMA,
    )
    assert first.sha256 == second.sha256
    assert first.domain != second.domain
    assert reopen_evidence(root, first) == reopen_evidence(root, second) == payload


def test_reopen_requires_exact_reference_type(tmp_path: Path) -> None:
    root = tmp_path / "evidence"
    reference = _publish(root)
    with pytest.raises(EvidenceStoreError, match="exact and typed"):
        reopen_evidence(root, reference.to_dict())  # type: ignore[arg-type]
