from __future__ import annotations

import importlib.util
import io
import json
import stat
import sys
from pathlib import Path

import pytest

from cacheon.chain import remote_worker_spool as spool
from cacheon.chain.publication import reopen_worker_bundle
from cacheon.eval import b300_remote_worker_adapter as adapter


def _spool_fixtures():
    path = Path(__file__).with_name("test_remote_worker_spool.py")
    specification = importlib.util.spec_from_file_location(
        "cacheon_remote_spool_test_fixtures_for_adapter", path
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def _adapter_paths(tmp_path: Path) -> adapter.AdapterPaths:
    return adapter.AdapterPaths(
        registration=tmp_path / "registration.json",
        ready_receipt=tmp_path / "ready-receipt.json",
        credential=tmp_path / "credential.secret",
        publication_root=tmp_path / "publications",
        processing_root=tmp_path / "processing",
        results_root=tmp_path / "results",
    )


def test_publication_transport_reconstructs_reopenable_immutable_tree(
    tmp_path: Path,
) -> None:
    fixtures = _spool_fixtures()
    (
        coordinator,
        claim,
        _service,
        credential,
        identity,
        registration,
        _request,
        _request_id,
        job_dir,
    ) = fixtures._screen_authority(tmp_path)
    try:
        outer = spool.verify_request(
            spool.load_json(job_dir / "request.json"),
            job_dir,
            registration,
            identity=identity,
            credential=credential,
        )
        archive_path = spool.artifact_for_role(
            outer, job_dir, "candidate_publication"
        )
        import tarfile

        with tarfile.open(archive_path, "r:") as archive:
            assert "bundle/.cacheon-native-artifact.json" in archive.getnames()

        publication_root = tmp_path / "pod-publications"
        reconstructed = adapter.safe_publication(
            archive_path, claim.publication.to_dict(), publication_root
        )

        assert reconstructed.to_dict() == claim.publication.to_dict()
        assert stat.S_IMODE(reconstructed.root.stat().st_mode) == 0o555
        assert (
            stat.S_IMODE(
                (reconstructed.root / ".cacheon-native-artifact.json").stat().st_mode
            )
            == 0o444
        )
        for logical in reconstructed.directories:
            assert (
                stat.S_IMODE(
                    reconstructed.root.joinpath(*Path(logical).parts).stat().st_mode
                )
                == 0o555
            )
        for row in reconstructed.files:
            assert (
                stat.S_IMODE(
                    reconstructed.root.joinpath(*Path(row.path).parts).stat().st_mode
                )
                == 0o444
            )

        reopened = reopen_worker_bundle(
            reconstructed.root,
            claim.publication.content_hash,
            expected_publication_digest=claim.publication.publication_digest,
            expected_receipt_digest=claim.publication.digest,
        )
        assert reopened.to_dict() == claim.publication.to_dict()
        reused = adapter.safe_publication(
            archive_path, claim.publication.to_dict(), publication_root
        )
        assert reused.to_dict() == claim.publication.to_dict()
    finally:
        coordinator._release(claim.lease, reason="test_cleanup")


def test_adapter_runtime_refuses_changed_commission_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _adapter_paths(tmp_path)
    monkeypatch.setattr(
        adapter,
        "load_json",
        lambda path: (
            {"identity": "changed"}
            if path == paths.registration
            else {"identity": "ready"}
        ),
    )
    monkeypatch.setattr(adapter, "verify_registration", lambda value: value)
    monkeypatch.setattr(adapter, "verify_ready_receipt", lambda value: value)
    runtime = object.__new__(adapter.AdapterRuntime)
    runtime.paths = paths
    runtime.registration = {"identity": "original"}
    runtime.ready = {"identity": "ready"}
    with pytest.raises(adapter.AdapterError, match="authority changed"):
        runtime.verify_current()


class _FakeWorker:
    def __init__(self) -> None:
        self.calls = 0

    def run_remote_screen(self, _lease, _candidate):
        self.calls += 1
        raise AssertionError("pre-resident failure must not call worker")


def _runtime_shell(paths: adapter.AdapterPaths) -> adapter.AdapterRuntime:
    runtime = object.__new__(adapter.AdapterRuntime)
    runtime.paths = paths
    runtime.registration = {}
    runtime.ready = {}
    runtime.credential = object()
    runtime.identity = object()
    runtime.worker = _FakeWorker()
    runtime.closed = False
    runtime.verify_current = lambda: None
    return runtime


def test_adapter_pre_resident_carrier_failure_never_calls_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _adapter_paths(tmp_path)
    runtime = _runtime_shell(paths)
    monkeypatch.setattr(adapter, "load_json", lambda _path: {})

    def rejecting_verify(_value, _root, _registration, *, identity, credential):
        del identity, credential
        raise spool.RemoteWorkerError("malformed request carrier")

    monkeypatch.setattr(adapter, "verify_request", rejecting_verify)
    with pytest.raises(adapter.AdapterRequestFailed) as captured:
        adapter.run_with_runtime(tmp_path / "request", tmp_path / "result", runtime)
    assert isinstance(captured.value.__cause__, spool.RemoteWorkerError)
    assert runtime.worker.calls == 0


def test_qualification_requests_are_refused_before_resident_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _adapter_paths(tmp_path)
    runtime = _runtime_shell(paths)
    monkeypatch.setattr(adapter, "load_json", lambda _path: {})
    monkeypatch.setattr(
        adapter,
        "verify_request",
        lambda _value, _root, _registration, *, identity, credential: {
            "lease": {"stage": "qualification"}
        },
    )
    with pytest.raises(adapter.AdapterRequestFailed) as captured:
        adapter.run_with_runtime(tmp_path / "request", tmp_path / "result", runtime)
    assert "qualification execution authority" in str(captured.value.__cause__)
    assert runtime.worker.calls == 0


def test_adapter_request_failure_continues_on_same_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _adapter_paths(tmp_path)
    request_ids = ("1" * 64, "2" * 64)
    frames_by_raw = {
        b"first\n": (
            request_ids[0],
            tmp_path / request_ids[0],
            tmp_path / f".{request_ids[0]}.1",
        ),
        b"second\n": (
            request_ids[1],
            tmp_path / request_ids[1],
            tmp_path / f".{request_ids[1]}.1",
        ),
    }
    runtime = object()
    seen: list[tuple[object, str]] = []

    monkeypatch.setattr(
        adapter,
        "validated_command_paths",
        lambda raw, _paths: frames_by_raw[raw],
    )

    def run_with_runtime(request_dir, _result_dir, observed_runtime):
        request_id = request_dir.name
        seen.append((observed_runtime, request_id))
        if request_id == request_ids[0]:
            raise adapter.AdapterRequestFailed("bad carrier")

    monkeypatch.setattr(adapter, "run_with_runtime", run_with_runtime)
    controls = io.BytesIO()

    assert adapter.serve_runtime(runtime, paths, iter(frames_by_raw), controls) == 0
    frames = tuple(json.loads(row) for row in controls.getvalue().splitlines())

    assert seen == [(runtime, request_ids[0]), (runtime, request_ids[1])]
    assert frames == (
        {"schema": spool.SCHEMA_ADAPTER_CONTROL, "state": "ready"},
        {
            "request_id": request_ids[0],
            "schema": spool.SCHEMA_ADAPTER_CONTROL,
            "state": "request_failed",
        },
        {
            "request_id": request_ids[1],
            "schema": spool.SCHEMA_ADAPTER_CONTROL,
            "state": "completed",
        },
    )


def test_adapter_epoch_failure_exits_before_next_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _adapter_paths(tmp_path)
    request_ids = ("3" * 64, "4" * 64)
    frames_by_raw = {
        b"first\n": (
            request_ids[0],
            tmp_path / request_ids[0],
            tmp_path / f".{request_ids[0]}.1",
        ),
        b"second\n": (
            request_ids[1],
            tmp_path / request_ids[1],
            tmp_path / f".{request_ids[1]}.1",
        ),
    }
    runtime = object()
    seen: list[tuple[object, str]] = []

    monkeypatch.setattr(
        adapter,
        "validated_command_paths",
        lambda raw, _paths: frames_by_raw[raw],
    )

    def run_with_runtime(request_dir, _result_dir, observed_runtime):
        seen.append((observed_runtime, request_dir.name))
        raise adapter.AdapterEpochFailed("resident lifetime failed")

    monkeypatch.setattr(adapter, "run_with_runtime", run_with_runtime)
    controls = io.BytesIO()

    assert adapter.serve_runtime(runtime, paths, iter(frames_by_raw), controls) == 2
    frames = tuple(json.loads(row) for row in controls.getvalue().splitlines())

    assert seen == [(runtime, request_ids[0])]
    assert frames == (
        {"schema": spool.SCHEMA_ADAPTER_CONTROL, "state": "ready"},
        {
            "request_id": request_ids[0],
            "schema": spool.SCHEMA_ADAPTER_CONTROL,
            "state": "epoch_failed",
        },
    )
