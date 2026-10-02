from __future__ import annotations

import io
import json
import stat
import subprocess
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from cacheon.bundle_hash import content_hash
from cacheon.chain import remote_worker_spool as spool
from cacheon.chain.publication import publish_worker_bundle, reopen_worker_bundle
from cacheon.chain.remote_worker_artifact_recovery import publication_archive
from cacheon.eval import b300_publication_intake as publication_intake
from cacheon.eval import b300_remote_worker_adapter as adapter
from cacheon.eval.qualification_continuation import QualificationContinuationStore
from tests.support.b300 import (
    qualification_capabilities as _qualification_capabilities,
)


def _adapter_paths(tmp_path: Path) -> adapter.AdapterPaths:
    return adapter.AdapterPaths(
        registration=tmp_path / "registration.json",
        ready_receipt=tmp_path / "ready-receipt.json",
        credential=tmp_path / "credential.secret",
        publication_root=tmp_path / "publications",
        processing_root=tmp_path / "processing",
        results_root=tmp_path / "results",
        continuation_root=tmp_path / "continuation",
    )


def _published(tmp_path: Path, name: str):
    """Publish one bundle and archive it with the CPU transport's own writer."""

    source = tmp_path / name
    (source / "kernels").mkdir(parents=True)
    files = (source / "manifest.toml", source / "kernels" / "op.py")
    files[0].write_text(f"bundle_id = '{name}'\n", encoding="utf-8")
    files[1].write_text("VALUE = 1\n", encoding="utf-8")
    for path in files:
        path.chmod(0o600)
    for path in (source / "kernels", source):
        path.chmod(0o700)
    publication = publish_worker_bundle(
        source, tmp_path / f"{name}-publications", content_hash(source)
    )
    archive = tmp_path / f"{name}.tar"
    publication_archive(publication, archive)
    return publication, archive


def test_native_child_stdout_cannot_corrupt_adapter_control() -> None:
    code = r"""
import subprocess
import sys
from cacheon.eval.b300_remote_worker_adapter import (
    _emit_control,
    _isolate_control_output,
)

control = _isolate_control_output()
try:
    subprocess.run(
        [sys.executable, "-c", "import os; os.write(1, b'native-output\\n')"],
        check=True,
    )
    _emit_control("ready", output=control)
finally:
    control.close()
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
    )

    assert result.stdout == (
        b'{"schema":"cacheon-b300-adapter-control-v1","state":"ready"}\n'
    )
    assert result.stderr == b"native-output\n"


def test_continuation_root_is_an_explicit_absolute_adapter_path(
    tmp_path: Path,
) -> None:
    paths = _adapter_paths(tmp_path)
    assert paths.continuation_root == tmp_path / "continuation"
    with pytest.raises(adapter.AdapterError, match="continuation_root.*absolute"):
        adapter.AdapterPaths(
            paths.registration,
            paths.ready_receipt,
            paths.credential,
            paths.publication_root,
            paths.processing_root,
            paths.results_root,
            Path("relative-continuation"),
        )


def test_publication_transport_reconstructs_reopenable_immutable_tree(
    tmp_path: Path,
) -> None:
    publication, archive_path = _published(tmp_path, "candidate")
    with tarfile.open(archive_path, "r:") as archive:
        assert "bundle/.cacheon-native-artifact.json" in archive.getnames()

    publication_root = tmp_path / "pod-publications"
    reconstructed = publication_intake.safe_publication(
        archive_path, publication.to_dict(), publication_root
    )

    assert reconstructed.to_dict() == publication.to_dict()
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
        publication.content_hash,
        expected_publication_digest=publication.publication_digest,
        expected_receipt_digest=publication.digest,
    )
    assert reopened.to_dict() == publication.to_dict()
    reused = publication_intake.safe_publication(
        archive_path, publication.to_dict(), publication_root
    )
    assert reused.to_dict() == publication.to_dict()


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

    def run_remote_qualification(self, *_args, **_kwargs):
        self.calls += 1
        raise AssertionError("pre-resident failure must not call worker")


def _runtime_shell(
    paths: adapter.AdapterPaths, *, qualification_commission=None
) -> adapter.AdapterRuntime:
    runtime = object.__new__(adapter.AdapterRuntime)
    runtime.paths = paths
    runtime.registration = {}
    runtime.ready = {}
    runtime.credential = object()
    runtime.identity = object()
    runtime.worker = _FakeWorker()
    runtime.qualification_commission = qualification_commission
    runtime.qualification_continuation_store = QualificationContinuationStore(
        paths.continuation_root
    )
    runtime._commissioned_service = None
    runtime.closed = False
    runtime.verify_current = lambda: None
    return runtime


def _patch_authenticated_carrier(
    monkeypatch: pytest.MonkeyPatch,
    *,
    stage: str,
    wire: object,
    lease: dict[str, object] | None = None,
) -> None:
    from cacheon.chain import remote_evaluation_dispatcher as dispatcher

    if lease is None:
        lease = {
            "claimed_block": 10,
            "expires_block": 20,
            "generation": 1,
            "initial_expires_block": 20,
            "lease_id": "1" * 64,
            "members": [
                {"prior_status": "published", "reservation_id": "2" * 64}
            ],
            "owner": "operator-a",
            "stage": stage,
        }
    outer = {
        "artifacts": [
            {"role": f"{stage}_payload", "sha256": "3" * 64, "size": 1}
        ],
        "lease": lease,
        "ready_receipt_digest": "4" * 64,
        "request_id": "5" * 64,
        "schema": spool.SCHEMA_REQUEST,
        "service_identity": "6" * 64,
        "worker_epoch": "7" * 32,
        "worker_readiness_digest": "8" * 64,
    }
    monkeypatch.setattr(adapter, "load_json", lambda _path, **_kwargs: {})
    monkeypatch.setattr(
        adapter,
        "verify_request",
        lambda _value, _root, _registration, *, identity, credential: outer,
    )
    monkeypatch.setattr(
        adapter,
        "artifact_for_role",
        lambda _outer, root, role: root / role,
    )
    monkeypatch.setattr(
        dispatcher.RemoteEvaluationRequest,
        "from_dict",
        classmethod(lambda _cls, _value: wire),
    )
    monkeypatch.setattr(
        dispatcher,
        "verify_remote_request",
        lambda observed, _identity, _credential: (
            None if observed is wire else pytest.fail("decoded wrong request")
        ),
    )


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
        adapter.run_with_runtime(tmp_path / ("1" * 64), tmp_path / "result", runtime)
    assert isinstance(captured.value.__cause__, spool.RemoteWorkerError)
    assert runtime.worker.calls == 0


def test_runtime_without_qualification_authority_refuses_before_reading_seals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _adapter_paths(tmp_path)
    monkeypatch.setattr(
        adapter, "load_json", lambda path: pytest.fail(f"read sealed {path}")
    )
    with pytest.raises(
        adapter.AdapterError, match="qualification authority is required"
    ):
        adapter.AdapterRuntime(paths)
    assert not paths.continuation_root.exists()


def test_non_qualification_lease_is_refused_before_resident_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = _runtime_shell(_adapter_paths(tmp_path))
    _patch_authenticated_carrier(monkeypatch, stage="screen", wire=object())
    result_dir = tmp_path / "result"
    with pytest.raises(adapter.AdapterRequestFailed) as captured:
        adapter.run_with_runtime(tmp_path / ("1" * 64), result_dir, runtime)
    assert "only qualification leases" in str(captured.value.__cause__)
    assert runtime.worker.calls == 0
    assert not (result_dir / "RESIDENT_ENTRY_ARMED.json").exists()


def test_adapter_runtime_rejects_untyped_qualification_commission(
    tmp_path: Path,
) -> None:
    with pytest.raises(adapter.AdapterError, match="authorities.*typed"):
        adapter.B300RemoteQualificationCommission(object(), object(), object())
    with pytest.raises(adapter.AdapterError, match="qualification commission.*typed"):
        adapter.AdapterRuntime(
            _adapter_paths(tmp_path), qualification_commission=object()
        )


def test_commission_materializes_and_resolves_each_fifo_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cacheon.chain import remote_evaluation_dispatcher as dispatcher
    from cacheon.eval import b300_remote_qualification_adapter as qualification_adapter

    paths = _adapter_paths(tmp_path)
    commission = object.__new__(adapter.B300RemoteQualificationCommission)
    fixed_authorities = (object(), SimpleNamespace(
        incumbent_stack=SimpleNamespace(digest="a" * 64),
        incumbent_tree_digest="b" * 64,
    ), object())
    object.__setattr__(commission, "deployment", fixed_authorities[0])
    object.__setattr__(commission, "construction", fixed_authorities[1])
    object.__setattr__(commission, "readiness", fixed_authorities[2])
    runtime = _runtime_shell(paths, qualification_commission=commission)
    run_calls: list[tuple[object, object]] = []
    built: list[PerRequestAdapter] = []

    class PerRequestAdapter:
        def __init__(
            self, deployment, construction, readiness, resolver, continuation_store
        ) -> None:
            assert (deployment, construction, readiness) == fixed_authorities
            assert continuation_store is runtime.qualification_continuation_store
            assert len(resolver.publications) == 1
            self.publication = resolver.publications[0]
            assert resolver.resolve(self.publication.to_dict()) == self.publication
            self.closed = 0
            built.append(self)

        def run(self, observed_wire):
            run_calls.append((self.publication, observed_wire))
            return object()

        def close(self) -> None:
            self.closed += 1

    monkeypatch.setattr(
        qualification_adapter, "B300RemoteQualificationAdapter", PerRequestAdapter
    )
    monkeypatch.setattr(
        dispatcher,
        "seal_remote_response",
        lambda _wire, _payload, identity, credential: (
            SimpleNamespace(
                to_dict=lambda: {
                    "schema": "sealed-response",
                    "stage": "qualification",
                }
            )
            if identity is runtime.identity and credential is runtime.credential
            else pytest.fail("sealed response used changed authority")
        ),
    )
    expected = []
    wires = []
    for index, name in enumerate(("first", "second")):
        publication, archive = _published(tmp_path, name)
        wire = SimpleNamespace(
            body={
                "candidates": [{"publication": publication.to_dict()}],
                "screen_lane": "primary",
                "incumbent_stack_digest": "a" * 64,
                "incumbent_tree_digest": "b" * 64,
            }
        )
        wires.append(wire)
        _patch_authenticated_carrier(monkeypatch, stage="qualification", wire=wire)
        monkeypatch.setattr(
            publication_intake,
            "artifacts_for_role",
            lambda _outer, _root, _role, archive=archive: (archive,),
        )
        result_dir = tmp_path / f"result-{index}"
        result_dir.mkdir(mode=0o700)

        adapter.run_with_runtime(tmp_path / ("1" * 64), result_dir, runtime)

        materialized = run_calls[-1][0]
        expected.append(materialized)
        assert materialized.to_dict() == publication.to_dict()
        assert materialized.root != publication.root
        response = result_dir / "response.json"
        assert response.is_file()
        assert stat.S_IMODE(response.stat().st_mode) == 0o400

    assert len(expected) == 2
    assert expected[0].digest != expected[1].digest
    assert run_calls == list(zip(expected, wires))
    # Each request's adapter is closed exactly once behind its result.
    assert [row.closed for row in built] == [1, 1]


def test_qualification_archive_mismatch_never_builds_or_runs_adapter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cacheon.eval.b300_remote_qualification_adapter import (
        B300RemoteQualificationAdapter,
    )

    _publication, archive = _published(tmp_path, "source")
    commission = object.__new__(adapter.B300RemoteQualificationCommission)
    runtime = _runtime_shell(
        _adapter_paths(tmp_path), qualification_commission=commission
    )
    object.__setattr__(commission, "construction", SimpleNamespace(
        incumbent_stack=SimpleNamespace(digest="a" * 64),
        incumbent_tree_digest="b" * 64,
    ))
    wire = SimpleNamespace(body={
        "candidates": [{"publication": {"changed": "wire"}}],
        "screen_lane": "primary", "incumbent_stack_digest": "a" * 64,
        "incumbent_tree_digest": "b" * 64,
    })
    _patch_authenticated_carrier(
        monkeypatch, stage="qualification", wire=wire
    )
    monkeypatch.setattr(
        publication_intake,
        "artifacts_for_role",
        lambda _outer, _root, _role: (archive,),
    )
    factory_calls: list[object] = []
    resident_calls: list[object] = []
    monkeypatch.setattr(
        adapter.B300RemoteQualificationCommission,
        "adapter_for",
        lambda _self, publication, _store: factory_calls.append(publication),
    )
    monkeypatch.setattr(
        B300RemoteQualificationAdapter,
        "run",
        lambda _self, request: resident_calls.append(request),
    )
    with pytest.raises(adapter.AdapterRequestFailed) as captured:
        adapter.run_with_runtime(
            tmp_path / ("1" * 64), tmp_path / "result", runtime
        )
    assert "changed wire authority" in str(captured.value.__cause__)
    assert factory_calls == []
    assert resident_calls == []
    assert runtime.worker.calls == 0


def test_qualification_execution_failure_is_epoch_fatal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cacheon.eval.b300_remote_qualification_adapter import (
        B300RemoteQualificationAdapter,
    )
    from cacheon.eval.qualification_continuation import (
        QualificationContinuationError,
    )

    commission = object.__new__(adapter.B300RemoteQualificationCommission)
    commissioned = object.__new__(B300RemoteQualificationAdapter)
    object.__setattr__(commission, "construction", SimpleNamespace(
        incumbent_stack=SimpleNamespace(digest="a" * 64),
        incumbent_tree_digest="b" * 64,
    ))
    runtime = _runtime_shell(
        _adapter_paths(tmp_path), qualification_commission=commission
    )
    wire = SimpleNamespace(
        body={
            "candidates": [{"publication": {"candidate": "one"}}],
            "screen_lane": "primary",
            "incumbent_stack_digest": "a" * 64,
            "incumbent_tree_digest": "b" * 64,
        }
    )
    _patch_authenticated_carrier(
        monkeypatch, stage="qualification", wire=wire
    )
    monkeypatch.setattr(
        adapter, "resolve_cohort_publications", lambda *_args: (object(),)
    )
    monkeypatch.setattr(
        adapter.B300RemoteQualificationCommission,
        "adapter_for",
        lambda _self, _publication, _store: commissioned,
    )

    def fail_after_entry(_self, observed):
        assert observed is wire
        raise QualificationContinuationError("continuation identity changed")

    closed: list[object] = []
    monkeypatch.setattr(B300RemoteQualificationAdapter, "run", fail_after_entry)
    monkeypatch.setattr(
        B300RemoteQualificationAdapter, "close", lambda self: closed.append(self)
    )
    result_dir = tmp_path / "result"
    result_dir.mkdir(mode=0o700)
    with pytest.raises(adapter.AdapterEpochFailed) as captured:
        adapter.run_with_runtime(tmp_path / ("1" * 64), result_dir, runtime)
    assert isinstance(captured.value.__cause__, QualificationContinuationError)
    assert closed == [commissioned]
    assert (result_dir / "RESIDENT_ENTRY_ARMED.json").is_file()


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
    runtime = _ServingRuntime()
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
    assert runtime.closed == 1
    assert frames == (
        {"schema": spool.SCHEMA_ADAPTER_CONTROL, "state": "ready"},
        {
            "request_id": request_ids[0],
            "schema": spool.SCHEMA_ADAPTER_CONTROL,
            "state": "request_failed",
        },
        {
            "request_id": request_ids[1],
            "retired": True,
            "schema": spool.SCHEMA_ADAPTER_CONTROL,
            "state": "completed",
        },
    )


class _ServingRuntime:
    """The serve loop's only use of a runtime is closing it at retirement."""

    def __init__(self) -> None:
        self.closed = 0

    def close(self) -> None:
        self.closed += 1


def test_completed_qualification_retires_adapter_behind_its_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _adapter_paths(tmp_path)
    request_ids = ("7" * 64, "8" * 64)
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
    runtime = _ServingRuntime()
    seen: list[str] = []
    monkeypatch.setattr(
        adapter,
        "validated_command_paths",
        lambda raw, _paths: frames_by_raw[raw],
    )

    def run_with_runtime(request_dir, _result_dir, _runtime):
        # A qualification completes normally, but its lane containers would
        # hold the GPUs until the next request's idle drain timed out.
        seen.append(request_dir.name)

    monkeypatch.setattr(adapter, "run_with_runtime", run_with_runtime)
    controls = io.BytesIO()

    assert adapter.serve_runtime(runtime, paths, iter(frames_by_raw), controls) == 0
    frames = tuple(json.loads(row) for row in controls.getvalue().splitlines())

    assert seen == [request_ids[0]]
    assert runtime.closed == 1
    assert frames == (
        {"schema": spool.SCHEMA_ADAPTER_CONTROL, "state": "ready"},
        {
            "request_id": request_ids[0],
            "retired": True,
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


def test_adapter_runtime_rejects_untyped_or_doubled_qualification_authority(
    tmp_path: Path,
) -> None:
    paths = _adapter_paths(tmp_path)
    # Both checks fail closed before any sealed file is read.
    with pytest.raises(adapter.AdapterError, match="capabilities.*typed"):
        adapter.AdapterRuntime(paths, qualification_capabilities=object())
    commission = object.__new__(adapter.B300RemoteQualificationCommission)
    with pytest.raises(adapter.AdapterError, match="mutually exclusive"):
        adapter.AdapterRuntime(
            paths,
            qualification_commission=commission,
            qualification_capabilities=_qualification_capabilities(),
        )


def test_capabilities_commission_one_qualification_service(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import cacheon.eval.b300_qualification_commission as commission_module

    paths = _adapter_paths(tmp_path)
    registration = {"ready_receipt_digest": "d" * 64}
    ready = {"receipt_digest": "d" * 64}
    monkeypatch.setattr(
        adapter,
        "load_json",
        lambda path: registration if path == paths.registration else ready,
    )
    monkeypatch.setattr(adapter, "verify_registration", lambda value: value)
    monkeypatch.setattr(adapter, "verify_ready_receipt", lambda value: value)
    monkeypatch.setattr(
        adapter, "registration_credential", lambda _registration, _path: object()
    )
    monkeypatch.setattr(
        adapter, "registration_transport_identity", lambda _registration: object()
    )

    worker = _FakeWorker()
    commission = object.__new__(adapter.B300RemoteQualificationCommission)
    closed: list[str] = []
    service = SimpleNamespace(
        worker=worker,
        commission=commission,
        close=lambda: closed.append("service"),
    )
    observed: list[tuple[object, object, object]] = []

    def build(observed_registration, observed_ready, observed_capabilities, *, commissioned_root=None):
        observed.append(
            (observed_registration, observed_ready, observed_capabilities)
        )
        return service

    monkeypatch.setattr(
        commission_module,
        "build_commissioned_b300_qualification_service",
        build,
    )

    capabilities = _qualification_capabilities()
    runtime = adapter.AdapterRuntime(paths, qualification_capabilities=capabilities)
    assert observed == [(registration, ready, capabilities)]
    assert runtime.worker is worker
    assert runtime.qualification_commission is commission
    assert runtime.qualification_continuation_store.root == paths.continuation_root

    runtime.close()
    runtime.close()
    assert closed == ["service"]
    assert worker.calls == 0


def test_load_qualification_capabilities_names_one_reviewed_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cacheon.eval import qualification_capability_loader as capability_loader

    receipt = object()
    observed: list[tuple[str, str]] = []

    def load(specifier: str, source_sha256: str):
        observed.append((specifier, source_sha256))
        return receipt

    monkeypatch.setattr(capability_loader, "load_qualification_capabilities", load)
    assert (
        adapter._load_qualification_capabilities("private_factory:build", "a" * 64)
        is receipt
    )
    assert observed == [("private_factory:build", "a" * 64)]

    def refuse(_specifier: str, _source_sha256: str):
        raise capability_loader.QualificationCapabilityLoadError("source differs")

    monkeypatch.setattr(capability_loader, "load_qualification_capabilities", refuse)
    with pytest.raises(adapter.AdapterError, match="source differs"):
        adapter._load_qualification_capabilities("private_factory:build", "b" * 64)


def _required_argv(paths: adapter.AdapterPaths) -> list[str]:
    return [
        "--registration",
        str(paths.registration),
        "--ready-receipt",
        str(paths.ready_receipt),
        "--credential",
        str(paths.credential),
        "--publication-root",
        str(paths.publication_root),
        "--processing-root",
        str(paths.processing_root),
        "--results-root",
        str(paths.results_root),
    ]


@pytest.mark.parametrize(
    "mode",
    ((), ("--serve", "--request-dir", "/request", "--result-dir", "/result")),
    ids=("without-serve", "one-shot-directories"),
)
def test_cli_serve_is_the_only_mode(tmp_path: Path, mode: tuple[str, ...]) -> None:
    paths = _adapter_paths(tmp_path)
    argv = [
        *mode,
        *_required_argv(paths),
        "--continuation-root",
        str(paths.continuation_root),
    ]
    with pytest.raises(SystemExit) as captured:
        adapter.main(argv)
    assert captured.value.code == 2


def test_cli_requires_explicit_continuation_root(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as captured:
        adapter.main(["--serve", *_required_argv(_adapter_paths(tmp_path))])
    assert captured.value.code == 2


def test_result_carrier_tolerates_only_the_shared_run_journal(
    tmp_path: Path,
) -> None:
    paths = _adapter_paths(tmp_path)
    request_id = "5" * 64
    request_dir = paths.processing_root / request_id
    request_dir.mkdir(parents=True)
    result_dir = paths.results_root / f".{request_id}.1"
    result_dir.mkdir(parents=True)
    raw = (
        adapter.spool_canonical_json(
            {
                "schema": adapter.SCHEMA_ADAPTER_COMMAND,
                "operation": "evaluate",
                "request_id": request_id,
                "request_dir": str(request_dir),
                "result_dir": str(result_dir),
            }
        )
        + b"\n"
    )

    # The pod runner journals its own lifecycle rows into the shared
    # per-request journal before delegating to the adapter.
    (result_dir / adapter.JOURNAL_NAME).write_text('{"event":"started"}\n')
    assert adapter.validated_command_paths(raw, paths) == (
        request_id,
        request_dir,
        result_dir,
    )

    (result_dir / "stale-result.json").write_text("{}\n")
    with pytest.raises(adapter.AdapterRequestFailed, match="carrier state"):
        adapter.validated_command_paths(raw, paths)
