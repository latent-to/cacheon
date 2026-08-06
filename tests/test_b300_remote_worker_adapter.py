from __future__ import annotations

import importlib.util
import io
import json
import stat
import sys
from pathlib import Path
from types import SimpleNamespace

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
                {
                    "prior_status": "promoted" if stage == "qualification" else "published",
                    "reservation_id": "2" * 64,
                }
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
        adapter.run_with_runtime(tmp_path / "request", tmp_path / "result", runtime)
    assert isinstance(captured.value.__cause__, spool.RemoteWorkerError)
    assert runtime.worker.calls == 0


def test_qualification_requests_are_refused_before_resident_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _adapter_paths(tmp_path)
    runtime = _runtime_shell(paths)
    _patch_authenticated_carrier(
        monkeypatch, stage="qualification", wire=object()
    )
    with pytest.raises(adapter.AdapterRequestFailed) as captured:
        adapter.run_with_runtime(tmp_path / "request", tmp_path / "result", runtime)
    assert "qualification execution authority" in str(captured.value.__cause__)
    assert runtime.worker.calls == 0


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
    from cacheon.bundle_hash import content_hash
    from cacheon.chain import remote_evaluation_dispatcher as dispatcher
    from cacheon.chain.publication import publish_worker_bundle
    from cacheon.eval import b300_remote_qualification_adapter as qualification_adapter

    fixtures = _spool_fixtures()
    authorities = []
    for name in ("first", "second"):
        source = tmp_path / name
        source.mkdir()
        authorities.append(fixtures._screen_authority(source))
    distinct_source = tmp_path / "distinct-source"
    distinct_source.mkdir()
    distinct_manifest = distinct_source / "manifest.toml"
    distinct_manifest.write_text(
        "bundle_id = 'second-fifo-publication'\n", encoding="utf-8"
    )
    distinct_source.chmod(0o700)
    distinct_manifest.chmod(0o600)
    distinct_publication = publish_worker_bundle(
        distinct_source,
        tmp_path / "distinct-publications",
        content_hash(distinct_source),
    )
    distinct_archive = tmp_path / "distinct-publication.tar"
    fixtures._publication_tar(distinct_publication, distinct_archive)
    paths = _adapter_paths(tmp_path)
    commission = object.__new__(adapter.B300RemoteQualificationCommission)
    fixed_authorities = (object(), object(), object())
    object.__setattr__(commission, "deployment", fixed_authorities[0])
    object.__setattr__(commission, "construction", fixed_authorities[1])
    object.__setattr__(commission, "readiness", fixed_authorities[2])
    runtime = _runtime_shell(paths, qualification_commission=commission)
    run_calls: list[tuple[object, object]] = []

    class PerRequestAdapter:
        def __init__(self, deployment, construction, readiness, resolver) -> None:
            assert (deployment, construction, readiness) == fixed_authorities
            assert len(resolver.publications) == 1
            self.publication = resolver.publications[0]
            assert resolver.resolve(self.publication.to_dict()) == self.publication

        def run(self, observed_wire):
            run_calls.append((self.publication, observed_wire))
            return object()

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
    try:
        expected = []
        wires = []
        for index, authority in enumerate(authorities):
            coordinator, claim, *_prefix, job_dir = authority
            if index == 0:
                expected_publication = claim.publication
                outer = spool.load_json(job_dir / "request.json")
                archive = spool.artifact_for_role(
                    outer, job_dir, "candidate_publication"
                )
            else:
                expected_publication = distinct_publication
                archive = distinct_archive
            wire = SimpleNamespace(
                body={
                    "candidates": [
                        {"publication": expected_publication.to_dict()}
                    ]
                }
            )
            wires.append(wire)
            _patch_authenticated_carrier(
                monkeypatch, stage="qualification", wire=wire
            )
            monkeypatch.setattr(
                adapter,
                "artifact_for_role",
                lambda _outer, root, role, archive=archive: (
                    archive if role == "candidate_publication" else root / role
                ),
            )
            result_dir = tmp_path / f"result-{index}"
            result_dir.mkdir(mode=0o700)

            adapter.run_with_runtime(job_dir, result_dir, runtime)

            materialized = run_calls[-1][0]
            expected.append(materialized)
            assert materialized.to_dict() == expected_publication.to_dict()
            assert materialized.root != expected_publication.root
            response = result_dir / "response.json"
            assert response.is_file()
            assert stat.S_IMODE(response.stat().st_mode) == 0o400

        assert len(expected) == 2
        assert expected[0].digest != expected[1].digest
        assert run_calls == list(zip(expected, wires))
    finally:
        for coordinator, claim, *_rest in authorities:
            coordinator._release(claim.lease, reason="test_cleanup")


def test_qualification_archive_mismatch_never_builds_or_runs_adapter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cacheon.eval.b300_remote_qualification_adapter import (
        B300RemoteQualificationAdapter,
    )

    fixtures = _spool_fixtures()
    source = tmp_path / "source"
    source.mkdir()
    authority = fixtures._screen_authority(source)
    coordinator, claim, *_prefix, job_dir = authority
    commission = object.__new__(adapter.B300RemoteQualificationCommission)
    runtime = _runtime_shell(
        _adapter_paths(tmp_path), qualification_commission=commission
    )
    outer = spool.load_json(job_dir / "request.json")
    archive = spool.artifact_for_role(outer, job_dir, "candidate_publication")
    wire = SimpleNamespace(
        body={"candidates": [{"publication": {"changed": "wire"}}]}
    )
    _patch_authenticated_carrier(
        monkeypatch, stage="qualification", wire=wire
    )
    monkeypatch.setattr(
        adapter,
        "artifact_for_role",
        lambda _outer, root, role: (
            archive if role == "candidate_publication" else root / role
        ),
    )
    factory_calls: list[object] = []
    resident_calls: list[object] = []
    monkeypatch.setattr(
        adapter.B300RemoteQualificationCommission,
        "adapter_for",
        lambda _self, publication: factory_calls.append(publication),
    )
    monkeypatch.setattr(
        B300RemoteQualificationAdapter,
        "run",
        lambda _self, request: resident_calls.append(request),
    )
    try:
        with pytest.raises(adapter.AdapterRequestFailed) as captured:
            adapter.run_with_runtime(job_dir, tmp_path / "result", runtime)
        assert "changed wire authority" in str(captured.value.__cause__)
        assert factory_calls == []
        assert resident_calls == []
        assert runtime.worker.calls == 0
    finally:
        coordinator._release(claim.lease, reason="test_cleanup")


def test_qualification_execution_failure_is_epoch_fatal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cacheon.eval.b300_remote_qualification_adapter import (
        B300RemoteQualificationAdapter,
    )

    commission = object.__new__(adapter.B300RemoteQualificationCommission)
    commissioned = object.__new__(B300RemoteQualificationAdapter)
    runtime = _runtime_shell(
        _adapter_paths(tmp_path), qualification_commission=commission
    )
    wire = SimpleNamespace(
        body={"candidates": [{"publication": {"candidate": "one"}}]}
    )
    _patch_authenticated_carrier(
        monkeypatch, stage="qualification", wire=wire
    )
    monkeypatch.setattr(adapter, "safe_publication", lambda *_args: object())
    monkeypatch.setattr(
        adapter.B300RemoteQualificationCommission,
        "adapter_for",
        lambda _self, _publication: commissioned,
    )

    def fail_after_entry(_self, observed):
        assert observed is wire
        raise RuntimeError("resident qualification failed")

    monkeypatch.setattr(B300RemoteQualificationAdapter, "run", fail_after_entry)
    result_dir = tmp_path / "result"
    result_dir.mkdir(mode=0o700)
    with pytest.raises(adapter.AdapterEpochFailed) as captured:
        adapter.run_with_runtime(tmp_path / "request", result_dir, runtime)
    assert isinstance(captured.value.__cause__, RuntimeError)
    assert (result_dir / "RESIDENT_ENTRY_ARMED.json").is_file()


def test_screen_requests_still_use_only_screen_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cacheon import arena_service
    from cacheon.chain import evaluation_leases
    from cacheon.chain import remote_evaluation_dispatcher as dispatcher
    from cacheon.eval import qualification_intake

    paths = _adapter_paths(tmp_path)
    runtime = _runtime_shell(paths)
    reservation_id = "1" * 64
    lease_value = {
        "claimed_block": 10,
        "expires_block": 20,
        "generation": 1,
        "initial_expires_block": 20,
        "lease_id": "2" * 64,
        "members": [
            {"prior_status": "published", "reservation_id": reservation_id}
        ],
        "owner": "operator-a",
        "stage": "screen",
    }
    wire = SimpleNamespace(
        body={
            "candidate_digest": "3" * 64,
            "publication": {},
            "reservation": {},
            "screen_attempt": 1,
        }
    )
    _patch_authenticated_carrier(
        monkeypatch, stage="screen", wire=wire, lease=lease_value
    )
    lease = SimpleNamespace(
        lease_id=lease_value["lease_id"], reservation_ids=(reservation_id,)
    )
    reservation = SimpleNamespace(reservation_digest=reservation_id)
    candidate = SimpleNamespace(digest=wire.body["candidate_digest"])
    receipt = arena_service.ArenaScreenReceipt(
        "4" * 64,
        candidate.digest,
        1,
        (
            arena_service.ScreenStageResult(
                arena_service.SCREEN_STAGES[0],
                arena_service.ScreenGrade.NO_DECISION,
                "5" * 64,
                1,
            ),
        ),
        arena_service.PromotionDecision.RETRY,
    )
    evaluation = SimpleNamespace(
        lease=lease,
        disposition="completed",
        envelope=SimpleNamespace(
            lease_id=lease.lease_id, payload_digest=receipt.digest
        ),
        payload=receipt,
    )
    screen_calls: list[tuple[object, object]] = []
    runtime.worker.run_remote_screen = lambda observed_lease, observed_candidate: (
        screen_calls.append((observed_lease, observed_candidate)) or evaluation
        if observed_lease is lease and observed_candidate is candidate
        else pytest.fail("screen worker received changed inputs")
    )
    monkeypatch.setattr(
        evaluation_leases, "EvaluationLeaseMember", lambda **_row: object()
    )
    monkeypatch.setattr(evaluation_leases, "EvaluationLease", lambda *_args: lease)
    monkeypatch.setattr(
        qualification_intake.QualificationReservation,
        "from_dict",
        classmethod(lambda _cls, _value: reservation),
    )
    monkeypatch.setattr(
        arena_service, "ArenaCandidateBinding", lambda *_args: candidate
    )
    monkeypatch.setattr(adapter, "safe_publication", lambda *_args: object())
    monkeypatch.setattr(
        dispatcher,
        "seal_remote_response",
        lambda *_args: SimpleNamespace(
            to_dict=lambda: {"schema": "sealed-response", "stage": "screen"}
        ),
    )
    monkeypatch.setattr(adapter, "publish_resident_entry", lambda *_args: {})
    result_dir = tmp_path / "result"
    result_dir.mkdir(mode=0o700)

    adapter.run_with_runtime(tmp_path / "request", result_dir, runtime)

    assert (result_dir / "response.json").is_file()
    assert screen_calls == [(lease, candidate)]


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
