from __future__ import annotations

import concurrent.futures
import hashlib
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

import tests.test_b300_qualification_deployment as deployment_fixtures
import tests.test_b300_remote_qualification_adapter as remote_fixtures
import tests.test_oci_backend as backend_fixtures
import tests.test_resident_pair_crossover as pair_fixtures
from cacheon.chain.evaluation_coordinator import WorkerReadiness
from cacheon.engine_tree import MaterializedEngineTree
from cacheon.eval import b300_resident_pair_factory as pair_factory
from cacheon.eval.b300_qualification_lanes import (
    B300QualificationLanePair,
    B300QualificationLanePolicy,
)
from cacheon.eval.engine_launch import (
    EngineLaunchSpec,
    LogicalHardwareSpec,
    NativeBuildSpec,
    PhysicalHardwareBinding,
    TrustedLaunchBinding,
    native_compiler_policy_digest,
    native_patcher_digest,
    native_toolchain_digest,
)
from cacheon.eval.oci_backend import (
    OCIEngineExecutor,
    TrustedArenaModelMountReceipt,
    expected_runtime_preflight,
    runtime_identity_from_preflight,
)
from cacheon.eval.oci_outer_session import SessionExecutionPlan
from cacheon.eval.oci_resident_session import ResidentSessionPlan
from cacheon.eval.oci_session_protocol import EngineSessionConfig
from cacheon.eval.resident_evaluation_pair import (
    ResidentEvaluationRetirementEvidence,
)


def _h(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


@dataclass
class _LifetimeHarness:
    monkeypatch: pytest.MonkeyPatch

    def __post_init__(self) -> None:
        self.clock = pair_fixtures._Clock()
        self.activity = pair_fixtures._Activity()
        self.templates: dict[int, SessionExecutionPlan] = {}
        self.plans: dict[int, pair_factory.B300ResidentStockLanePlan] = {}
        self.calls: list[dict[str, object]] = []
        self.factories: list[pair_fixtures._Factory] = []
        self.monkeypatch.setattr(
            pair_factory,
            "make_backend_lifetime_factory",
            self.make,
        )

    def register(self, plan: pair_factory.B300ResidentStockLanePlan) -> None:
        self.templates[id(plan.executor)] = plan.speed_workload
        self.plans[id(plan.executor)] = plan

    def make(
        self,
        executor,
        launch,
        binding,
        mount,
        resident_plan,
        *,
        swap_intake_root,
        deadline_provider,
    ):
        plan = self.plans[id(executor)]
        assert launch is plan.stock_launch
        assert binding is plan.stock_binding
        assert resident_plan is plan.resident_plan
        assert Path(swap_intake_root).name == "resident-intake"
        deadline = deadline_provider()
        self.calls.append(
            {
                "deadline": deadline,
                "executor": executor,
                "launch": launch,
                "mount": mount,
            }
        )
        lane = plan.lane_policy.lane_id
        session_id = f"{len(self.calls):032x}"
        lifetime = pair_fixtures._Factory(
            session_id,
            self.templates[id(executor)],
            (1.0,),
            self.clock,
            self.activity,
        )
        lifetime.lane_id = lane
        self.factories.append(lifetime)
        return lifetime


@dataclass
class _Commissioned:
    factory: pair_factory.B300CommissionedResidentPairFactory
    plans: tuple[
        pair_factory.B300ResidentStockLanePlan,
        pair_factory.B300ResidentStockLanePlan,
    ]
    executors: tuple[OCIEngineExecutor, OCIEngineExecutor]
    readiness: WorkerReadiness
    model_mount: TrustedArenaModelMountReceipt


@pytest.fixture
def lifetimes(monkeypatch: pytest.MonkeyPatch) -> _LifetimeHarness:
    return _LifetimeHarness(monkeypatch)


@pytest.fixture
def managed_executors():
    rows: list[OCIEngineExecutor] = []
    yield rows
    for executor in rows:
        executor.manager.close()


def _executor(
    tmp_path: Path,
    *,
    lane_id: str,
    allocation_offset: int,
    identity: str,
) -> OCIEngineExecutor:
    role = f"request-pair-{identity}-{lane_id.lower()}"
    base = remote_fixtures._executor(
        tmp_path,
        role=role,
        lane=lane_id,
    )
    if allocation_offset == 0:
        return base
    first = allocation_offset + (0 if lane_id == "A" else 4)
    policy = replace(
        base.device_policy,
        expected_gpus=tuple(
            deployment_fixtures._gpu(index) for index in range(first, first + 4)
        ),
    )
    config = base.config
    base.manager.close()
    return OCIEngineExecutor(config, policy)


def _native(
    *,
    tree_digest: str,
    image_digest: str,
    platform_digest: str,
    worker_digest: str,
    executor: OCIEngineExecutor,
) -> NativeBuildSpec:
    dependency = executor.config.prebuild.policy.dependency_policy_digest
    return NativeBuildSpec(
        tree_digest=tree_digest,
        image_digest=image_digest,
        platform_digest=platform_digest,
        worker_distribution_digest=worker_digest,
        toolchain_digest=native_toolchain_digest(
            image_digest=image_digest,
            platform_digest=platform_digest,
        ),
        patcher_digest=native_patcher_digest(
            worker_distribution_digest=worker_digest
        ),
        compiler_flags_digest=native_compiler_policy_digest(
            image_digest=image_digest,
            worker_distribution_digest=worker_digest,
            dependency_policy_digest=dependency,
            target_architecture="sm103",
        ),
        target_architecture="sm103",
        dependency_policy_digest=dependency,
    )


def _commissioned(
    tmp_path: Path,
    lifetimes: _LifetimeHarness,
    managed_executors: list[OCIEngineExecutor],
    *,
    identity: str = "one",
    allocation_offset: int = 0,
) -> _Commissioned:
    executors = tuple(
        _executor(
            tmp_path / f"executor-{lane_id}",
            lane_id=lane_id,
            allocation_offset=allocation_offset,
            identity=identity,
        )
        for lane_id in ("A", "B")
    )
    managed_executors.extend(executors)
    runtime = executors[0].config.runtime
    image, platform, worker = (
        _h(f"{identity}:{field}") for field in ("image", "platform", "worker")
    )
    preflight = backend_fixtures._preflight(
        image=image,
        platform=platform,
        worker=worker,
        runtime=runtime,
    )
    runtime_identity = runtime_identity_from_preflight(preflight)
    service = _h(f"{identity}:service")
    topology = _h(f"{identity}:topology")
    stack_digest = _h(f"{identity}:stock-stack")
    tree_digest = _h(f"{identity}:stock-tree")
    tree_root = tmp_path / "stock-tree"
    tree_root.mkdir(parents=True)
    tree = MaterializedEngineTree(
        tree_root,
        stack_digest,
        tree_digest,
        (),
        None,
    )
    model_root = tmp_path / "model"
    model_root.mkdir()
    engine_config = EngineSessionConfig(
        model_path="/cacheon/input/model",
        dtype="bfloat16",
        deterministic=False,
        attention_backend="flashinfer",
        disable_cuda_graph=False,
        mem_fraction_static=0.82,
        log_level="error",
        max_running_requests=64,
        tp_size=4,
        moe_runner_backend="flashinfer_trtllm",
        disable_custom_all_reduce=False,
        engine_kwargs={},
    )
    model_mount = TrustedArenaModelMountReceipt.capture(
        model_root,
        arena_digest=service,
        model_revision_digest=_h(f"{identity}:model-revision"),
        model_manifest_digest=_h(f"{identity}:model-manifest"),
        model_content_digest=_h(f"{identity}:model-content"),
    )
    plans = []
    for lane_id, executor in zip(("A", "B"), executors):
        lane_policy = B300QualificationLanePolicy.from_device_policy(
            lane_id, executor.device_policy
        )
        hardware = LogicalHardwareSpec(
            visible_gpu_count=4,
            architecture="sm103",
            topology_class="nvlink",
            topology_digest=topology,
            tp_size=4,
            ep_size=1,
            dp_size=1,
            device_policy_digest=lane_policy.device_policy_digest,
        )
        physical = PhysicalHardwareBinding(
            tuple(str(value) for value in lane_policy.physical_gpu_ids),
            architecture="sm103",
            topology_class="nvlink",
            topology_digest=topology,
            tp_size=4,
            ep_size=1,
            dp_size=1,
            device_policy_digest=lane_policy.device_policy_digest,
        )
        native = _native(
            tree_digest=tree_digest,
            image_digest=image,
            platform_digest=platform,
            worker_digest=worker,
            executor=executor,
        )
        launch = EngineLaunchSpec(
            runtime_digest=runtime_identity.runtime_digest,
            base_engine_digest=runtime_identity.base_engine_digest,
            arena_digest=service,
            stack_digest=stack_digest,
            tree_digest=tree_digest,
            image_digest=image,
            platform_digest=platform,
            controller_distribution_digest=_h(f"{identity}:controller"),
            worker_distribution_digest=worker,
            model_revision_digest=model_mount.model_revision_digest,
            model_manifest_digest=model_mount.model_manifest_digest,
            model_content_digest=model_mount.model_content_digest,
            validator_overlay_digest=runtime_identity.validator_overlay_digest,
            engine_config_digest=engine_config.digest,
            seccomp_policy_digest=_h(f"{identity}:seccomp"),
            resource_policy_digest=(
                executor.config.prebuild.policy.resource_policy_digest
            ),
            native_build_spec_digest=native.digest,
            hardware=hardware,
        )
        binding = TrustedLaunchBinding(
            tree.root,
            launch.controller_distribution_digest,
            native,
            preflight,
            physical,
        )
        expected = expected_runtime_preflight(launch, preflight)
        resident_plan = ResidentSessionPlan(
            launch.digest,
            engine_config.digest,
            engine_config,
            expected,
            100,
            100,
            1,
            1,
            0.0,
        )
        speed = SessionExecutionPlan(
            launch.digest,
            engine_config.digest,
            engine_config,
            expected,
            (("generic warmup",), ("generic timed",)),
            1,
            1,
            1,
            1,
            0.0,
        )
        plans.append(
            pair_factory.B300ResidentStockLanePlan(
                lane_policy,
                tree,
                launch,
                binding,
                resident_plan,
                speed,
                executor,
            )
        )
    typed_plans = tuple(plans)
    lane_pair = B300QualificationLanePair(
        typed_plans[0].lane_policy,
        typed_plans[1].lane_policy,
    )
    readiness = WorkerReadiness(
        ready_receipt_digest=_h(f"{identity}:ready-receipt"),
        ready_epoch=7,
        service_digest=service,
        arena_id=f"arena-{identity}",
        provider_digest=_h(f"{identity}:provider"),
        runtime_digest=runtime_identity.runtime_digest,
        worker_distribution_digest=worker,
        model_revision_digest=model_mount.model_revision_digest,
        model_manifest_digest=model_mount.model_manifest_digest,
        model_content_digest=model_mount.model_content_digest,
        target_architecture="sm103",
        topology_class="nvlink",
        topology_digest=topology,
        gpu_count=4,
        tensor_parallel_size=4,
        workload_digest=_h(f"{identity}:workload"),
        qualification_policy_digest=_h(f"{identity}:qualification-policy"),
    )
    swap_root = tmp_path / "resident-intake"
    swap_root.mkdir()
    for plan in typed_plans:
        lifetimes.register(plan)
    commissioned = pair_factory.B300CommissionedResidentPairFactory(
        service_digest=service,
        readiness=readiness,
        lane_pair=lane_pair,
        # Role/executor order is not authority; the constructor canonicalizes A/B.
        lane_plans=tuple(reversed(typed_plans)),
        model_mount=model_mount,
        swap_intake_root=swap_root,
        start_timeout_s=2.0,
        request_timeout_s=2.0,
        close_timeout_s=2.0,
        clock=lifetimes.clock,
    )
    return _Commissioned(
        commissioned,
        commissioned.lane_plans,
        executors,
        readiness,
        model_mount,
    )


def _authority(label: str) -> pair_factory.B300ResidentPairRequestAuthority:
    return pair_factory.B300ResidentPairRequestAuthority(
        _h(f"{label}:request"),
        _h(f"{label}:qualification-authority"),
        _h(f"{label}:target-profile"),
    )


def test_two_sequential_requests_create_fresh_pairs_and_four_lifetimes(
    tmp_path: Path,
    lifetimes: _LifetimeHarness,
    managed_executors,
) -> None:
    commissioned = _commissioned(tmp_path, lifetimes, managed_executors)
    first_authority, second_authority = _authority("profile-one"), _authority(
        "profile-two"
    )
    first = commissioned.factory.open_request(first_authority, deadline=100.0)
    assert lifetimes.calls == []
    assert lifetimes.factories == []

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        borrows = tuple(
            pool.map(lambda _index: first.borrow(first_authority), range(2))
        )
    assert borrows[0] is borrows[1]
    first_borrow = borrows[0]
    assert len(lifetimes.calls) == 2
    assert len(lifetimes.factories[0].sessions) == 1
    assert len(lifetimes.factories[1].sessions) == 1
    assert all(
        session.finish_calls == 0
        for lifetime in lifetimes.factories[:2]
        for session in lifetime.sessions
    )
    with pytest.raises(pair_factory.B300ResidentPairFactoryError, match="pair is live"):
        first.complete(first_authority, first_borrow.binding)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        retirements = tuple(pool.map(lambda _index: first.close(), range(2)))
    first_retirement = retirements[0]
    assert type(first_retirement) is ResidentEvaluationRetirementEvidence
    assert retirements[1] is first_retirement
    assert first.close() is first_retirement
    assert first.complete(first_authority, first_borrow.binding) is first_retirement
    assert all(
        session.finish_calls == 1
        for lifetime in lifetimes.factories[:2]
        for session in lifetime.sessions
    )

    second = commissioned.factory.open_request(second_authority, deadline=200.0)
    second_borrow = second.borrow(second_authority)
    assert second_borrow.pair is not first_borrow.pair
    assert len(lifetimes.calls) == 4
    assert sum(len(row.sessions) for row in lifetimes.factories) == 4
    assert (
        first_borrow.binding.service_epoch_digest
        == second_borrow.binding.service_epoch_digest
        == commissioned.factory.commissioned_epoch_digest
    )
    assert first.request_epoch_digest != second.request_epoch_digest
    assert first_borrow.binding.digest != second_borrow.binding.digest
    assert set(first_borrow.binding.identities).isdisjoint(
        second_borrow.binding.identities
    )
    second_retirement = second.close()
    assert type(second_retirement) is ResidentEvaluationRetirementEvidence
    assert all(
        session.finish_calls == 1
        for lifetime in lifetimes.factories
        for session in lifetime.sessions
    )


def test_binding_freezes_exact_stock_lane_and_request_authorities(
    tmp_path: Path,
    lifetimes: _LifetimeHarness,
    managed_executors,
) -> None:
    commissioned = _commissioned(tmp_path, lifetimes, managed_executors)
    authority = _authority("generic")
    owner = commissioned.factory.open_request(authority, deadline=100.0)
    borrowed = owner.borrow(authority)

    assert tuple(row.lane_id for row in borrowed.binding.lanes) == ("A", "B")
    for row, plan in zip(borrowed.binding.lanes, commissioned.plans):
        assert row.stock_launch_digest == plan.stock_launch.digest
        assert row.lane_digest == plan.lane_authority_digest
        assert row.allocation_digest == plan.lane_policy.digest
        assert row.executor_namespace_digest == plan.executor.manager.namespace_digest
    assert borrowed.binding.identities == borrowed.pair.identities
    assert authority.target_profile_digest not in repr(borrowed.binding)
    owner.close()


def test_foreign_request_or_runtime_binding_fails_without_new_work(
    tmp_path: Path,
    lifetimes: _LifetimeHarness,
    managed_executors,
) -> None:
    commissioned = _commissioned(tmp_path, lifetimes, managed_executors)
    authority, foreign = _authority("first"), _authority("foreign")
    owner = commissioned.factory.open_request(authority, deadline=100.0)
    with pytest.raises(pair_factory.B300ResidentPairFactoryError, match="stale or foreign"):
        owner.borrow(foreign)
    assert not any(row.sessions for row in lifetimes.factories)

    borrowed = owner.borrow(authority)
    changed = replace(
        borrowed.binding,
        service_epoch_digest=_h("foreign-service-epoch"),
    )
    history = borrowed.pair.request_history
    with pytest.raises(pair_factory.B300ResidentPairFactoryError, match="stale or foreign"):
        owner.require_binding(authority, changed)
    with pytest.raises(pair_factory.B300ResidentPairFactoryError, match="stale or foreign"):
        owner.require_binding(foreign, borrowed.binding)
    assert borrowed.pair.request_history == history
    owner.close()


def test_delayed_borrow_expires_before_lifetimes_or_bounds_start_wall(
    tmp_path: Path,
    lifetimes: _LifetimeHarness,
    managed_executors,
) -> None:
    commissioned = _commissioned(tmp_path, lifetimes, managed_executors)
    expired = commissioned.factory.open_request(
        _authority("expired"), deadline=10.0
    )
    lifetimes.clock.span(9.0)  # 1.0 -> exact request deadline.
    with pytest.raises(
        pair_factory.B300ResidentPairFactoryError,
        match="expired before pair start",
    ):
        expired.borrow(expired.authority)
    assert lifetimes.calls == []
    assert lifetimes.factories == []

    bounded = commissioned.factory.open_request(
        _authority("bounded"), deadline=11.0
    )
    lifetimes.clock.span(0.75)
    borrowed = bounded.borrow(bounded.authority)
    assert borrowed.pair._timeouts[0] == pytest.approx(0.25)
    assert borrowed.pair._timeouts[2] == pytest.approx(2.0)
    assert {row["deadline"] for row in lifetimes.calls} == {11.0}
    bounded.close()


def test_two_target_profiles_and_physical_allocations_are_generic(
    tmp_path: Path,
    lifetimes: _LifetimeHarness,
    managed_executors,
) -> None:
    first = _commissioned(
        tmp_path / "allocation-one",
        lifetimes,
        managed_executors,
        identity="first",
        allocation_offset=0,
    )
    second = _commissioned(
        tmp_path / "allocation-two",
        lifetimes,
        managed_executors,
        identity="second",
        allocation_offset=8,
    )
    first_owner = first.factory.open_request(_authority("shape-one"), deadline=100.0)
    second_owner = second.factory.open_request(
        _authority("shape-two"), deadline=100.0
    )
    first_borrow = first_owner.borrow(first_owner.authority)
    first_owner.close()
    second_borrow = second_owner.borrow(second_owner.authority)
    second_owner.close()

    assert {
        row.allocation_digest for row in first_borrow.binding.lanes
    }.isdisjoint({row.allocation_digest for row in second_borrow.binding.lanes})
    assert first_owner.authority.target_profile_digest != (
        second_owner.authority.target_profile_digest
    )
    source = Path(pair_factory.__file__).read_text(encoding="utf-8").lower()
    for target_name in ("arnorm", "msa", "all_reduce", "moe_epilogue"):
        assert target_name not in source


def test_factory_failure_starts_no_lifetime(
    tmp_path: Path,
    lifetimes: _LifetimeHarness,
    managed_executors,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commissioned = _commissioned(tmp_path, lifetimes, managed_executors)
    calls = []

    def fail_second(*args, **kwargs):
        calls.append((args, kwargs))
        if len(calls) == 2:
            raise RuntimeError("second backend factory failed")
        return lambda _driver: pytest.fail("partial factory must never start")

    monkeypatch.setattr(pair_factory, "make_backend_lifetime_factory", fail_second)
    owner = commissioned.factory.open_request(
        _authority("factory-fail"), deadline=100.0
    )
    with pytest.raises(
        pair_factory.B300ResidentPairFactoryError,
        match="second backend factory",
    ):
        owner.borrow(owner.authority)
    assert len(calls) == 2


def test_partial_start_closes_the_started_lane_once_and_latches_failure(
    tmp_path: Path,
    lifetimes: _LifetimeHarness,
    managed_executors,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commissioned = _commissioned(tmp_path, lifetimes, managed_executors)
    good = pair_fixtures._Factory(
        "a" * 32,
        commissioned.plans[0].speed_workload,
        (1.0,),
        lifetimes.clock,
        lifetimes.activity,
    )
    calls = 0

    def partial(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return good

        def fail(_driver):
            raise RuntimeError("lane B failed before READY")

        return fail

    monkeypatch.setattr(pair_factory, "make_backend_lifetime_factory", partial)
    authority = _authority("start-fail")
    owner = commissioned.factory.open_request(authority, deadline=100.0)
    with pytest.raises(pair_factory.B300ResidentPairFactoryError, match="failed to start"):
        owner.borrow(authority)
    assert len(good.sessions) == 1
    assert good.sessions[0].finish_calls == 1
    with pytest.raises(pair_factory.B300ResidentPairFactoryError, match="permanently failed"):
        owner.borrow(authority)
    with pytest.raises(pair_factory.B300ResidentPairFactoryError, match="previously failed"):
        owner.close()
    assert good.sessions[0].finish_calls == 1


def test_binding_freeze_failure_retires_both_started_lanes_once(
    tmp_path: Path,
    lifetimes: _LifetimeHarness,
    managed_executors,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commissioned = _commissioned(tmp_path, lifetimes, managed_executors)
    rows = []

    def duplicate_session_factory(executor, *_args, **_kwargs):
        plan = next(row for row in commissioned.plans if row.executor is executor)
        lifetime = pair_fixtures._Factory(
            "a" * 32,
            plan.speed_workload,
            (1.0,),
            lifetimes.clock,
            lifetimes.activity,
        )
        rows.append(lifetime)
        return lifetime

    monkeypatch.setattr(
        pair_factory,
        "make_backend_lifetime_factory",
        duplicate_session_factory,
    )
    authority = _authority("duplicate-session")
    owner = commissioned.factory.open_request(authority, deadline=100.0)
    with pytest.raises(
        pair_factory.B300ResidentPairFactoryError,
        match="failed to freeze",
    ):
        owner.borrow(authority)
    assert len(rows) == 2
    assert all(row.sessions[0].finish_calls == 1 for row in rows)
    with pytest.raises(pair_factory.B300ResidentPairFactoryError):
        owner.close()
    assert all(row.sessions[0].finish_calls == 1 for row in rows)


def test_stock_pair_rejects_candidate_tree_foreign_lane_and_ready_width(
    tmp_path: Path,
    lifetimes: _LifetimeHarness,
    managed_executors,
) -> None:
    commissioned = _commissioned(tmp_path, lifetimes, managed_executors)
    left, right = commissioned.plans
    with pytest.raises(pair_factory.B300ResidentPairFactoryError, match="stock session"):
        replace(
            left,
            stock_tree=replace(left.stock_tree, runtime_manifest="manifest.toml"),
        )

    foreign_pair = B300QualificationLanePair(
        left.lane_policy,
        replace(
            right.lane_policy,
            device_configuration_digest=_h("foreign-allocation"),
        ),
    )
    with pytest.raises(pair_factory.B300ResidentPairFactoryError, match="sealed lane pair"):
        pair_factory.B300CommissionedResidentPairFactory(
            service_digest=commissioned.factory.service_digest,
            readiness=commissioned.readiness,
            lane_pair=foreign_pair,
            lane_plans=commissioned.plans,
            model_mount=commissioned.model_mount,
            swap_intake_root=commissioned.factory.swap_intake_root,
            clock=lifetimes.clock,
        )

    with pytest.raises(pair_factory.B300ResidentPairFactoryError, match="READY"):
        pair_factory.B300CommissionedResidentPairFactory(
            service_digest=commissioned.factory.service_digest,
            readiness=replace(commissioned.readiness, gpu_count=8),
            lane_pair=commissioned.factory.lane_pair,
            lane_plans=commissioned.plans,
            model_mount=commissioned.model_mount,
            swap_intake_root=commissioned.factory.swap_intake_root,
            clock=lifetimes.clock,
        )


def test_owner_has_no_partial_close_or_audit_launch_surface(
    tmp_path: Path,
    lifetimes: _LifetimeHarness,
    managed_executors,
) -> None:
    commissioned = _commissioned(tmp_path, lifetimes, managed_executors)
    owner = commissioned.factory.open_request(_authority("surface"), deadline=100.0)
    for forbidden in (
        "close_lane",
        "close_a",
        "close_b",
        "launch_audit",
        "run_audit",
        "reuse",
    ):
        assert not hasattr(owner, forbidden)
    assert owner.close() is None
    assert owner.close() is None
