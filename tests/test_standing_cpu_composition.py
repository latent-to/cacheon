from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from cacheon.arena_service import ArenaService
from cacheon.chain import remote_worker_spool as spool
from cacheon.chain.evaluation_coordinator import EvaluationRun
from cacheon.chain.mainnet_screen_dispatcher import (
    make_qualification_publication_resolver,
)
from cacheon.chain.recoverable_intake import RecoverableFinalizedIntakeStore
from cacheon.chain.recoverable_qualification_dispatcher import (
    RecoverableQualificationDispatcher,
)
from cacheon.chain.remote_evaluation_dispatcher import (
    AuthenticatedRemoteEvaluationResponse,
    RemoteEvaluationDispatcher,
    RemoteEvaluationRequest,
    RemoteWorkerCredential,
    seal_remote_response,
    verify_remote_request,
)
from cacheon.chain.remote_worker_registration import verify_registration
from cacheon.chain.ssh_worker_transport import (
    DurableSpoolAuthenticatedWorkerTransport,
)
from cacheon.chain.standing_cpu_supervisor import (
    StandingCpuSupervisor,
    StandingCpuSupervisorError,
    SupervisorPhase,
)


def _test_module(filename: str, module_name: str):
    path = Path(__file__).with_name(filename)
    specification = importlib.util.spec_from_file_location(module_name, path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


class _ComposedTransport:
    """Fake execution edge over the real durable qualification spool protocol."""

    def __init__(
        self,
        *,
        root: Path,
        coordinator,
        credential: RemoteWorkerCredential,
        delegate: DurableSpoolAuthenticatedWorkerTransport,
        registration: dict[str, object],
        fixtures,
        plan_fixtures,
        hold_after_publish: bool,
    ) -> None:
        self.root = root
        self.coordinator = coordinator
        self.credential = credential
        self.delegate = delegate
        self.registration = registration
        self.fixtures = fixtures
        self.plan_fixtures = plan_fixtures
        self.hold_after_publish = hold_after_publish
        self.identity = delegate.identity
        self.screen_requests: list[RemoteEvaluationRequest] = []
        self.plan = None
        self.plans = 0
        self.materializations = 0
        self.publications = 0
        self.resumes = 0
        self.resume_request_ids: list[str] = []
        self.fail_next_resume = not hold_after_publish

    @property
    def outbox(self) -> Path:
        return self.root / "spool" / "outbox"

    @property
    def results(self) -> Path:
        return self.root / "spool" / "results"

    def run_screen(self, request, *, job):
        parsed = RemoteEvaluationRequest.from_dict(request.to_dict())
        verify_remote_request(parsed, self.identity, self.credential)
        assert parsed.lease_id == job.lease.lease_id
        self.screen_requests.append(parsed)
        receipt = self.coordinator.service.screen(job.candidate)
        return AuthenticatedRemoteEvaluationResponse.from_dict(
            seal_remote_response(
                parsed, receipt, self.identity, self.credential
            ).to_dict()
        )

    def run_qualification(self, _request, *, job):
        """Satisfy the closed transport protocol without bypassing recovery.

        The composed qualification path below must use the durable
        plan/materialize/publish/resume methods.  A direct legacy transport call
        would skip the exactly-once request fence, so make that mistake loud.
        """

        raise AssertionError(
            f"direct qualification transport bypassed recovery for {job!r}"
        )

    def plan_qualification_request(self, request):
        self.plans += 1
        self.plan = self.delegate.plan_qualification_request(request)
        return self.plan

    def materialize_planned_qualification(self, plan, request):
        self.materializations += 1
        return self.delegate.materialize_planned_qualification(plan, request)

    def inspect_planned_qualification(self, plan):
        return self.delegate.inspect_planned_qualification(plan)

    def prove_planned_qualification_prepublication(self, plan):
        return self.delegate.prove_planned_qualification_prepublication(plan)

    def publish_planned_qualification(self, plan):
        self.publications += 1
        observed = self.delegate.publish_planned_qualification(plan)
        if self.hold_after_publish:
            spool.write_local_no_decision(
                self.results,
                plan.request_dict(),
                "adapter_start_failed",
            )
        return observed

    def resume_planned_qualification(self, plan):
        self.resumes += 1
        self.resume_request_ids.append(plan.request_id)
        if self.fail_next_resume:
            self.fail_next_resume = False
            raise TimeoutError("simulated waiter interruption")
        return self.delegate.resume_planned_qualification(plan)

    def complete(self) -> None:
        assert self.plan is not None
        authority = SimpleNamespace(
            root=self.root,
            coordinator=self.coordinator,
            service=self.coordinator.service,
            credential=self.credential,
            identity=self.identity,
            registration=self.registration,
            request=self.plan.remote_request,
            fixtures=self.fixtures,
            outbox=self.outbox,
            results=self.results,
        )
        self.plan_fixtures._write_completed_result(authority, self.plan)


def _registration(root: Path, coordinator, credential, identity):
    secret = root / "credential.secret"
    secret.write_bytes(b"q" * 32)
    secret.chmod(0o400)
    known_hosts = root / "known_hosts"
    known_hosts.write_text("pinned-host-key\n", encoding="utf-8")
    known_hosts.chmod(0o600)
    row: dict[str, object] = {
        "adapter_sha256": "a" * 64,
        "created_at_unix": 1,
        "credential_digest": credential.digest,
        "credential_file_sha256": spool.file_sha256(secret),
        "credential_id": credential.credential_id,
        "credential_path": str(secret),
        "known_hosts_path": str(known_hosts),
        "known_hosts_sha256": spool.file_sha256(known_hosts),
        "lane_devices": list(range(coordinator.readiness.gpu_count)),
        "lane_digest": "e" * 64,
        "pod_host": "pod.example",
        "pod_port": 22,
        "pod_user": "root",
        "python_executable": sys.executable,
        "python_executable_sha256": spool.file_sha256(Path(sys.executable).resolve()),
        "ready_receipt_digest": coordinator.readiness.ready_receipt_digest,
        "ready_receipt_file_sha256": "b" * 64,
        "remote_service_sha256": "c" * 64,
        "schema": spool.SCHEMA_REGISTRATION,
        "service_identity": coordinator.service.manifest.service_id,
        "transport_identity": identity.to_dict(),
        "transport_identity_digest": identity.digest,
        "worker_epoch": "d" * 32,
        "worker_readiness": coordinator.readiness.to_dict(),
        "worker_readiness_digest": coordinator.readiness.digest,
    }
    row["registration_digest"] = spool.spool_digest(
        spool.DOMAIN_REGISTRATION, row
    )
    verify_registration(row)
    path = root / "registration.json"
    spool.atomic_json(path, row, mode=0o400)
    return row, path, secret


def _harness(tmp_path: Path, *, profile: str, hold_after_publish: bool):
    fixtures = _test_module(
        "test_remote_evaluation_dispatcher.py",
        f"cacheon_composed_dispatch_fixtures_{profile}",
    )
    plan_fixtures = _test_module(
        "test_remote_worker_request_plan.py",
        f"cacheon_composed_plan_fixtures_{profile}",
    )
    plan_fixtures._published_profile(fixtures, tmp_path, profile)
    service = ArenaService(fixtures._manifest(), fixtures._Provider())
    cursor = fixtures._Cursor((fixtures.BLOCK, fixtures._block_hash(fixtures.BLOCK)))
    coordinator = fixtures._coordinator(
        tmp_path,
        service,
        cursor,
        qualification_max_members=1,
        store_factory=RecoverableFinalizedIntakeStore,
    )
    credential = RemoteWorkerCredential("composed-key-v1", b"q" * 32)
    identity = fixtures._transport_identity(
        coordinator, credential, endpoint=f"endpoint-{profile}"
    )
    registration, registration_path, secret = _registration(
        tmp_path, coordinator, credential, identity
    )
    spool_root = tmp_path / "spool"
    spool_root.mkdir(mode=0o700)
    resolved: list[tuple[dict[str, object], ...]] = []
    resolver = make_qualification_publication_resolver(
        intake_db=fixtures._db_path(tmp_path),
        policy=fixtures.POLICY,
        scope=fixtures.SCOPE,
        store_factory=RecoverableFinalizedIntakeStore,
    )

    def observed_resolver(request):
        publications = resolver(request)
        resolved.append(tuple(item.to_dict() for item in publications))
        return publications

    delegate = DurableSpoolAuthenticatedWorkerTransport(
        registration_path=registration_path,
        spool_root=spool_root,
        credential_path=secret,
        qualification_publication_resolver=observed_resolver,
        response_timeout_seconds=1,
        poll_seconds=1,
    )
    transport = _ComposedTransport(
        root=tmp_path,
        coordinator=coordinator,
        credential=credential,
        delegate=delegate,
        registration=registration,
        fixtures=fixtures,
        plan_fixtures=plan_fixtures,
        hold_after_publish=hold_after_publish,
    )
    screen = RemoteEvaluationDispatcher(
        coordinator=coordinator,
        transport=transport,
        credential=credential,
    )
    incumbent = fixtures._incumbent(service)
    return SimpleNamespace(
        coordinator=coordinator,
        credential=credential,
        fixtures=fixtures,
        incumbent=incumbent,
        resolved=resolved,
        screen=screen,
        transport=transport,
    )


def _supervisor(harness, screen_runs: list[EvaluationRun]) -> StandingCpuSupervisor:
    qualification = RecoverableQualificationDispatcher(
        coordinator=harness.coordinator,
        transport=harness.transport,
        credential=harness.credential,
        qualification_evidence_root=harness.transport.root / "cpu-evidence",
        qualification_incumbent_stack=harness.incumbent,
        qualification_incumbent_tree_digest=harness.fixtures._h("incumbent-tree"),
    )

    def screen_once():
        result = harness.screen.dispatch_screen_once()
        if result is not None:
            screen_runs.append(result)
        return result

    return StandingCpuSupervisor(
        screen_once=screen_once,
        qualification_once=qualification.dispatch_once,
    )


@pytest.mark.parametrize("profile", ["collective-alpha", "block-beta"])
def test_screen_to_qualification_restart_reuses_one_request(
    tmp_path: Path,
    profile: str,
) -> None:
    harness = _harness(tmp_path, profile=profile, hold_after_publish=False)
    screen_runs: list[EvaluationRun] = []
    supervisor = _supervisor(harness, screen_runs)

    assert supervisor.weights_once is None
    assert supervisor.tick().phase is SupervisorPhase.SCREEN
    assert len(screen_runs) == 1
    reservation_id = screen_runs[0].lease.reservation_ids[0]
    with RecoverableFinalizedIntakeStore(
        harness.fixtures._db_path(tmp_path),
        harness.fixtures.POLICY,
        scope=harness.fixtures.SCOPE,
    ) as store:
        assert store.get(reservation_id).status == "promoted"

    with pytest.raises(StandingCpuSupervisorError, match="same-request"):
        supervisor.tick()
    plan = harness.transport.plan
    assert plan is not None
    request_id = plan.request_id
    harness.transport.complete()

    restarted = _supervisor(harness, screen_runs)
    assert restarted.weights_once is None
    assert restarted.tick().phase is SupervisorPhase.QUALIFICATION
    assert harness.transport.plan.request_id == request_id
    assert (
        harness.transport.plans,
        harness.transport.materializations,
        harness.transport.publications,
    ) == (1, 1, 1)
    # One interrupted wait plus one reopen at each durable downstream phase;
    # every read must address the same already-published request carrier.
    assert harness.transport.resumes == 3
    assert harness.transport.resume_request_ids == [request_id] * 3
    # Planning and materialization independently reopen the same typed
    # publication; neither lookup may select a different candidate root.
    assert len(harness.resolved) == 2
    assert harness.resolved[0] == harness.resolved[1]
    assert harness.resolved[0][0] == plan.remote_request.body["candidates"][0][
        "publication"
    ]

    assert restarted.tick().phase is SupervisorPhase.IDLE
    assert len(harness.transport.screen_requests) == 1
    assert len(screen_runs) == 1
    assert len([path for path in harness.transport.outbox.iterdir() if path.is_dir()]) == 1
    with RecoverableFinalizedIntakeStore(
        harness.fixtures._db_path(tmp_path),
        harness.fixtures.POLICY,
        scope=harness.fixtures.SCOPE,
    ) as store:
        assert store.get(reservation_id).status == "failed"
        assert store.pending_qualification_recovery() is None


def test_postpublication_infrastructure_results_requeue_fresh_until_capped(
    tmp_path: Path,
) -> None:
    # Owner ruling 2026-08-10: a worker infrastructure result retires its dead
    # request and requeues a fresh one instead of parking HELD; the systemic
    # release cap bounds the retries and parks the reservation visibly for the
    # operator, leaving the queue free.
    harness = _harness(
        tmp_path,
        profile="hold-profile",
        hold_after_publish=True,
    )
    screen_runs: list[EvaluationRun] = []
    supervisor = _supervisor(harness, screen_runs)

    assert supervisor.tick().phase is SupervisorPhase.SCREEN
    request_ids: list[str] = []
    for attempt in range(3):
        status = supervisor.tick()
        assert status.phase is SupervisorPhase.QUALIFICATION
        assert status.last_disposition == "requeue"
        request_ids.append(harness.transport.plan.request_id)
    assert len(set(request_ids)) == 3, "each retry must mint a fresh request"
    assert (
        harness.transport.plans,
        harness.transport.materializations,
        harness.transport.publications,
    ) == (3, 3, 3)
    assert len(harness.transport.screen_requests) == 1
    assert len(screen_runs) == 1

    idle = supervisor.tick()
    assert idle.phase is SupervisorPhase.IDLE

    reservation_id = screen_runs[0].lease.reservation_ids[0]
    with RecoverableFinalizedIntakeStore(
        harness.fixtures._db_path(tmp_path),
        harness.fixtures.POLICY,
        scope=harness.fixtures.SCOPE,
    ) as store:
        parked = store.get(reservation_id)
        assert parked.status == "held"
        assert parked.decision == "NO_DECISION"
        assert parked.reason.startswith("systemic_release_cap:")
        assert store.pending_qualification_recovery() is None
