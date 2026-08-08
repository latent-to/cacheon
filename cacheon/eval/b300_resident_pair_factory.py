"""Request-scoped commissioned B300 resident-pair construction.

The eight-device qualification deployment can host one physical TP4 pair at a
time.  This module therefore commissions reusable *stock launch authorities*,
but creates a fresh :class:`ResidentEvaluationPair` for every authenticated
qualification request.  A request owner starts its two lifetimes lazily,
freezes the actual session identities, and must retire both lanes before the
request can be considered complete.

No candidate path, target name, or audit launch is accepted here.  Candidate
work is admitted later through the pair-native request capabilities.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from cacheon.chain.evaluation_coordinator import WorkerReadiness
from cacheon.engine_tree import MaterializedEngineTree
from cacheon.eval.b300_qualification_lanes import (
    B300QualificationLanePair,
    B300QualificationLanePolicy,
)
from cacheon.eval.engine_launch import (
    EngineLaunchSpec,
    TrustedLaunchBinding,
    validate_native_build_spec,
)
from cacheon.eval.oci_backend import (
    OCIEngineExecutor,
    TrustedArenaModelMountReceipt,
    expected_runtime_preflight,
)
from cacheon.eval.oci_outer_session import SessionExecutionPlan
from cacheon.eval.oci_resident_session import ResidentSessionPlan
from cacheon.eval.resident_evaluation_pair import (
    ResidentEvaluationPair,
    ResidentEvaluationRetirementEvidence,
)
from cacheon.eval.resident_pair_binding import (
    ResidentPairLaneBinding,
    ResidentPairRuntimeBinding,
)
from cacheon.eval.resident_screen_lane import make_backend_lifetime_factory
from cacheon.eval.scoring import marginal_workload_digest
from cacheon.stack_identity import canonical_digest, require_sha256_hex


PAIR_START_TIMEOUT_SECONDS = 1_800.0
PAIR_REQUEST_TIMEOUT_SECONDS = 3_600.0
PAIR_CLOSE_TIMEOUT_SECONDS = 1_800.0
_OWNER_CONSTRUCTION_TOKEN = object()


class B300ResidentPairFactoryError(RuntimeError):
    """Commissioned request-pair construction or lifecycle failed closed."""


def _digest(value: object, field: str) -> str:
    try:
        return require_sha256_hex(value, field=field)
    except (TypeError, ValueError) as exc:
        raise B300ResidentPairFactoryError(str(exc)) from None


def _positive_seconds(value: object, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0 < float(value) <= 86_400
    ):
        raise B300ResidentPairFactoryError(f"{field} is invalid")
    return float(value)


def _absolute_deadline(value: object, clock: Callable[[], float]) -> float:
    try:
        now = float(clock())
    except BaseException as exc:
        raise B300ResidentPairFactoryError(
            f"resident pair host clock failed: {exc}"
        ) from None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not math.isfinite(now)
        or float(value) <= now
    ):
        raise B300ResidentPairFactoryError(
            "resident pair requires a future absolute deadline"
        )
    return float(value)


def _physical_ids(policy: B300QualificationLanePolicy) -> tuple[str, ...]:
    return tuple(str(value) for value in policy.physical_gpu_ids)


def _workload_shape(plan: SessionExecutionPlan) -> tuple[object, ...]:
    return (
        plan.engine_config,
        plan.prompt_batches,
        plan.warmup_count,
        plan.conditioning_count,
        plan.max_new_tokens,
        plan.top_logprobs_num,
        plan.temperature,
        plan.expected_discovery_overlay_identity_digest,
        plan.audit_policy,
    )


def _resident_shape(plan: ResidentSessionPlan) -> tuple[object, ...]:
    return (
        plan.expected_engine_config_digest,
        plan.engine_config,
        plan.max_swaps,
        plan.max_batches,
        plan.max_new_tokens,
        plan.top_logprobs_num,
        plan.temperature,
    )


@dataclass(frozen=True)
class B300ResidentPairRequestAuthority:
    """Target-neutral identity of one authenticated qualification request."""

    authenticated_request_digest: str
    qualification_authority_digest: str
    target_profile_digest: str

    def __post_init__(self) -> None:
        for field in (
            "authenticated_request_digest",
            "qualification_authority_digest",
            "target_profile_digest",
        ):
            object.__setattr__(
                self,
                field,
                _digest(getattr(self, field), field.replace("_", " ")),
            )

    @property
    def digest(self) -> str:
        return canonical_digest(
            "cacheon.eval.b300-resident-pair-request-authority.v1",
            {
                "authenticated_request": self.authenticated_request_digest,
                "qualification_authority": self.qualification_authority_digest,
                "target_profile": self.target_profile_digest,
            },
        )


@dataclass(frozen=True)
class B300ResidentStockLanePlan:
    """One sealed physical lane and its genuine stock launch authority."""

    lane_policy: B300QualificationLanePolicy
    stock_tree: MaterializedEngineTree
    stock_launch: EngineLaunchSpec
    stock_binding: TrustedLaunchBinding
    resident_plan: ResidentSessionPlan
    speed_workload: SessionExecutionPlan
    executor: OCIEngineExecutor

    def __post_init__(self) -> None:
        if (
            type(self.lane_policy) is not B300QualificationLanePolicy
            or type(self.stock_tree) is not MaterializedEngineTree
            or type(self.stock_launch) is not EngineLaunchSpec
            or type(self.stock_binding) is not TrustedLaunchBinding
            or type(self.resident_plan) is not ResidentSessionPlan
            or type(self.speed_workload) is not SessionExecutionPlan
            or type(self.executor) is not OCIEngineExecutor
        ):
            raise B300ResidentPairFactoryError(
                "resident stock lane authorities are not exactly typed"
            )
        lane = self.lane_policy
        policy = self.executor.device_policy
        expected_gpus = tuple(policy.expected_gpus)
        hardware = self.stock_launch.hardware
        physical = self.stock_binding.physical_hardware
        try:
            validate_native_build_spec(
                self.stock_launch, self.stock_binding.native_build_spec
            )
            expected_preflight = expected_runtime_preflight(
                self.stock_launch,
                self.stock_binding.runtime_preflight_receipt,
            )
            physical.validate_against(hardware)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise B300ResidentPairFactoryError(
                f"resident stock launch is invalid: {exc}"
            ) from None
        if (
            self.stock_tree.runtime_manifest is not None
            or self.stock_tree.stack_digest != self.stock_launch.stack_digest
            or self.stock_tree.tree_digest != self.stock_launch.tree_digest
            or self.stock_tree.root != self.stock_binding.materialized_tree_root
            or self.stock_binding.controller_distribution_digest
            != self.stock_launch.controller_distribution_digest
            or self.resident_plan.launch_digest != self.stock_launch.digest
            or self.resident_plan.expected_preflight != expected_preflight
            or self.speed_workload.launch_digest != self.stock_launch.digest
            or self.speed_workload.expected_preflight != expected_preflight
            or self.speed_workload.expected_engine_config_digest
            != self.resident_plan.expected_engine_config_digest
            or self.speed_workload.engine_config != self.resident_plan.engine_config
            or self.speed_workload.audit_policy is not None
            or self.resident_plan.engine_config.disable_cuda_graph
            or self.speed_workload.engine_config.disable_cuda_graph
        ):
            raise B300ResidentPairFactoryError(
                "resident lane is not one exact graph-on stock session plan"
            )
        if (
            tuple(gpu.physical_id for gpu in expected_gpus)
            != lane.physical_gpu_ids
            or tuple(gpu.uuid for gpu in expected_gpus) != lane.gpu_uuids
            or policy.configuration_sha256 != lane.device_configuration_digest
            or policy.policy_sha256 != lane.device_policy_digest
            or physical.physical_gpu_ids != _physical_ids(lane)
            or physical.device_policy_digest != lane.device_policy_digest
            or hardware.device_policy_digest != lane.device_policy_digest
            or hardware.visible_gpu_count != len(lane.physical_gpu_ids)
            or hardware.tp_size != physical.tp_size
            or hardware.ep_size != physical.ep_size
            or hardware.dp_size != physical.dp_size
            or hardware.architecture != physical.architecture
            or hardware.topology_class != physical.topology_class
            or hardware.topology_digest != physical.topology_digest
        ):
            raise B300ResidentPairFactoryError(
                "resident stock launch differs from its sealed physical lane"
            )
        if (
            self.executor.config.prebuild.policy.resource_policy_digest
            != self.stock_launch.resource_policy_digest
            or self.executor.device_policy.configuration_sha256
            != lane.device_configuration_digest
        ):
            raise B300ResidentPairFactoryError(
                "resident stock launch differs from its executor policies"
            )

    @property
    def allocation_digest(self) -> str:
        return self.lane_policy.digest

    @property
    def lane_authority_digest(self) -> str:
        physical = self.stock_binding.physical_hardware
        return canonical_digest(
            "cacheon.qualification.resident-lane",
            {
                "configuration": self.lane_policy.device_configuration_digest,
                "namespace": self.executor.manager.namespace_digest,
                "physical_gpu_ids": list(physical.physical_gpu_ids),
                "policy": self.lane_policy.device_policy_digest,
                "launch_resource_policy": self.stock_launch.resource_policy_digest,
                "runtime_policy": self.executor.config.runtime.digest,
                "topology": self.stock_launch.hardware.topology_digest,
            },
        )

    @property
    def commissioning_digest(self) -> str:
        return canonical_digest(
            "cacheon.eval.b300-resident-stock-lane-plan.v1",
            {
                "allocation": self.allocation_digest,
                "engine_config": self.resident_plan.expected_engine_config_digest,
                "executor_namespace": self.executor.manager.namespace_digest,
                "lane": self.lane_policy.lane_id,
                "lane_authority": self.lane_authority_digest,
                "resident_limits": {
                    "batches": self.resident_plan.max_batches,
                    "swaps": self.resident_plan.max_swaps,
                },
                "stock_launch": self.stock_launch.digest,
                "workload": canonical_digest(
                    "cacheon.eval.b300-resident-speed-workload-binding.v1",
                    {
                        "digest": marginal_workload_digest(self.speed_workload),
                    },
                ),
            },
        )


@dataclass(frozen=True)
class B300ResidentRequestPair:
    """Non-owning request-scoped pair capability and its frozen binding."""

    authority: B300ResidentPairRequestAuthority
    pair: ResidentEvaluationPair
    binding: ResidentPairRuntimeBinding

    def __post_init__(self) -> None:
        if (
            type(self.authority) is not B300ResidentPairRequestAuthority
            or type(self.pair) is not ResidentEvaluationPair
            or type(self.binding) is not ResidentPairRuntimeBinding
            or self.pair.identities != self.binding.identities
        ):
            raise B300ResidentPairFactoryError(
                "resident request pair is not exactly bound"
            )


class B300ResidentPairRequestOwner:
    """Own one fresh pair from lazy start through all-or-nothing retirement."""

    def __init__(
        self,
        *,
        authority: B300ResidentPairRequestAuthority,
        commissioned_epoch_digest: str,
        lane_plans: tuple[B300ResidentStockLanePlan, B300ResidentStockLanePlan],
        lifetime_factories: Callable[[], tuple[Callable, Callable]],
        deadline: float,
        start_timeout_s: float,
        request_timeout_s: float,
        close_timeout_s: float,
        clock: Callable[[], float],
        _construction_token: object | None = None,
    ) -> None:
        if (
            _construction_token is not _OWNER_CONSTRUCTION_TOKEN
            or type(authority) is not B300ResidentPairRequestAuthority
            or type(lane_plans) is not tuple
            or len(lane_plans) != 2
            or any(type(row) is not B300ResidentStockLanePlan for row in lane_plans)
            or tuple(row.lane_policy.lane_id for row in lane_plans) != ("A", "B")
            or not callable(lifetime_factories)
            or not callable(clock)
        ):
            raise B300ResidentPairFactoryError(
                "resident request owner authorities are not exact"
            )
        self.authority = authority
        self.commissioned_epoch_digest = _digest(
            commissioned_epoch_digest, "commissioned resident epoch digest"
        )
        self._plans = lane_plans
        self._lifetime_factories = lifetime_factories
        self._deadline = _absolute_deadline(deadline, clock)
        self._start_timeout_s = _positive_seconds(
            start_timeout_s, "pair start timeout"
        )
        self._request_timeout_s = _positive_seconds(
            request_timeout_s, "pair request timeout"
        )
        self._close_timeout_s = _positive_seconds(
            close_timeout_s, "pair close timeout"
        )
        self._clock = clock
        self._pair: ResidentEvaluationPair | None = None
        self._lock = threading.RLock()
        self._borrow: B300ResidentRequestPair | None = None
        self._retirement: ResidentEvaluationRetirementEvidence | None = None
        self._start_attempted = False
        self._close_attempted = False
        self._failure: BaseException | None = None

    @property
    def request_epoch_digest(self) -> str:
        return canonical_digest(
            "cacheon.eval.b300-resident-request-pair-epoch.v1",
            {
                "commissioned_epoch": self.commissioned_epoch_digest,
                "request_authority": self.authority.digest,
            },
        )

    @property
    def retirement(self) -> ResidentEvaluationRetirementEvidence | None:
        with self._lock:
            return self._retirement

    def _require_authority(
        self, authority: B300ResidentPairRequestAuthority
    ) -> None:
        if (
            type(authority) is not B300ResidentPairRequestAuthority
            or authority != self.authority
        ):
            raise B300ResidentPairFactoryError(
                "resident pair request authority is stale or foreign"
            )

    def _retire_after_start_failure(self) -> None:
        self._close_attempted = True
        if self._pair is None:
            return
        try:
            self._pair.close()
        except BaseException:
            # The original start failure is authoritative. ResidentEvaluationPair
            # has still issued the one terminal close to every started lane.
            pass

    def borrow(
        self, authority: B300ResidentPairRequestAuthority
    ) -> B300ResidentRequestPair:
        """Start this request's pair once and freeze actual A/B sessions."""

        with self._lock:
            self._require_authority(authority)
            if self._failure is not None:
                raise B300ResidentPairFactoryError(
                    "resident request pair is permanently failed"
                ) from self._failure
            if self._close_attempted:
                raise B300ResidentPairFactoryError(
                    "resident request pair is already retired"
                )
            if self._borrow is not None:
                return self._borrow
            self._start_attempted = True
            try:
                remaining = self._deadline - float(self._clock())
            except BaseException as exc:
                self._failure = exc
                self._close_attempted = True
                raise B300ResidentPairFactoryError(
                    "resident pair host clock failed before start"
                ) from exc
            if not math.isfinite(remaining) or remaining <= 0:
                failure = B300ResidentPairFactoryError(
                    "resident request deadline expired before pair start"
                )
                self._failure = failure
                self._close_attempted = True
                raise failure
            try:
                factories = self._lifetime_factories()
            except BaseException as exc:
                self._failure = exc
                self._close_attempted = True
                raise B300ResidentPairFactoryError(
                    f"resident backend lifetime factory failed: {exc}"
                ) from exc
            if (
                type(factories) is not tuple
                or len(factories) != 2
                or any(not callable(row) for row in factories)
            ):
                failure = B300ResidentPairFactoryError(
                    "resident backend lifetime factories are not exact"
                )
                self._failure = failure
                self._close_attempted = True
                raise failure
            try:
                remaining = self._deadline - float(self._clock())
            except BaseException as exc:
                self._failure = exc
                self._close_attempted = True
                raise B300ResidentPairFactoryError(
                    "resident pair host clock failed before start"
                ) from exc
            if not math.isfinite(remaining) or remaining <= 0:
                failure = B300ResidentPairFactoryError(
                    "resident request deadline expired before pair start"
                )
                self._failure = failure
                self._close_attempted = True
                raise failure
            try:
                self._pair = ResidentEvaluationPair(
                    factories[0],
                    factories[1],
                    start_timeout_s=min(self._start_timeout_s, remaining),
                    request_timeout_s=self._request_timeout_s,
                    close_timeout_s=self._close_timeout_s,
                    clock=self._clock,
                )
            except BaseException as exc:
                self._failure = exc
                self._close_attempted = True
                raise B300ResidentPairFactoryError(
                    "resident request pair failed to construct"
                ) from exc
            try:
                identities = self._pair.start()
            except BaseException as exc:
                self._failure = exc
                self._retire_after_start_failure()
                raise B300ResidentPairFactoryError(
                    "resident request pair failed to start"
                ) from exc
            if tuple(row.lane_id for row in identities) != ("A", "B"):
                failure = B300ResidentPairFactoryError(
                    "resident request pair returned reordered sessions"
                )
                self._failure = failure
                self._retire_after_start_failure()
                raise failure
            try:
                binding = ResidentPairRuntimeBinding(
                    self.commissioned_epoch_digest,
                    tuple(
                        ResidentPairLaneBinding(
                            identity.lane_id,
                            identity.session_id,
                            plan.stock_launch.digest,
                            plan.lane_authority_digest,
                            plan.allocation_digest,
                            plan.executor.manager.namespace_digest,
                        )
                        for identity, plan in zip(identities, self._plans)
                    ),
                )
                self._borrow = B300ResidentRequestPair(
                    self.authority, self._pair, binding
                )
            except BaseException as exc:
                self._failure = exc
                self._retire_after_start_failure()
                raise B300ResidentPairFactoryError(
                    "resident request pair failed to freeze its runtime binding"
                ) from exc
            return self._borrow

    def require_binding(
        self,
        authority: B300ResidentPairRequestAuthority,
        binding: ResidentPairRuntimeBinding,
    ) -> B300ResidentRequestPair:
        """Reopen only the exact binding frozen by this request owner."""

        with self._lock:
            self._require_authority(authority)
            if (
                type(binding) is not ResidentPairRuntimeBinding
                or self._borrow is None
                or binding != self._borrow.binding
            ):
                raise B300ResidentPairFactoryError(
                    "resident pair runtime binding is stale or foreign"
                )
            return self._borrow

    def close(self) -> ResidentEvaluationRetirementEvidence | None:
        """Retire both lanes once; a started pair must yield exact retirement."""

        with self._lock:
            if self._retirement is not None:
                return self._retirement
            if self._close_attempted:
                if self._failure is not None:
                    raise B300ResidentPairFactoryError(
                        "resident request pair retirement previously failed"
                    ) from self._failure
                return None
            self._close_attempted = True
            if self._pair is None:
                return None
            try:
                retirement = self._pair.close()
            except BaseException as exc:
                if self._failure is None:
                    self._failure = exc
                raise B300ResidentPairFactoryError(
                    "resident request pair retirement failed"
                ) from exc
            if (
                self._start_attempted
                and type(retirement) is not ResidentEvaluationRetirementEvidence
            ):
                failure = B300ResidentPairFactoryError(
                    "started resident request pair produced no exact retirement"
                )
                self._failure = failure
                raise failure
            if retirement is not None and self._borrow is not None:
                if (
                    retirement.lane_a.identity
                    != self._borrow.binding.identities[0]
                    or retirement.lane_b.identity
                    != self._borrow.binding.identities[1]
                ):
                    failure = B300ResidentPairFactoryError(
                        "resident retirement changed the frozen A/B sessions"
                    )
                    self._failure = failure
                    raise failure
            self._retirement = retirement
            return retirement

    def complete(
        self,
        authority: B300ResidentPairRequestAuthority,
        binding: ResidentPairRuntimeBinding,
    ) -> ResidentEvaluationRetirementEvidence:
        """Return request-complete proof only after both lanes retired."""

        with self._lock:
            self.require_binding(authority, binding)
            if self._retirement is None:
                raise B300ResidentPairFactoryError(
                    "qualification request cannot complete while its pair is live"
                )
            return self._retirement


class B300CommissionedResidentPairFactory:
    """Commissioned stock authority which creates one new pair per request."""

    def __init__(
        self,
        *,
        service_digest: str,
        readiness: WorkerReadiness,
        lane_pair: B300QualificationLanePair,
        lane_plans: tuple[B300ResidentStockLanePlan, B300ResidentStockLanePlan],
        model_mount: TrustedArenaModelMountReceipt,
        swap_intake_root: str | Path,
        start_timeout_s: float = PAIR_START_TIMEOUT_SECONDS,
        request_timeout_s: float = PAIR_REQUEST_TIMEOUT_SECONDS,
        close_timeout_s: float = PAIR_CLOSE_TIMEOUT_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        service = _digest(service_digest, "resident pair service digest")
        if (
            type(readiness) is not WorkerReadiness
            or type(lane_pair) is not B300QualificationLanePair
            or type(lane_plans) is not tuple
            or len(lane_plans) != 2
            or any(type(row) is not B300ResidentStockLanePlan for row in lane_plans)
            or type(model_mount) is not TrustedArenaModelMountReceipt
            or not callable(clock)
        ):
            raise B300ResidentPairFactoryError(
                "commissioned resident pair authorities are not exact"
            )
        by_lane = {row.lane_policy.lane_id: row for row in lane_plans}
        if set(by_lane) != {"A", "B"}:
            raise B300ResidentPairFactoryError(
                "commissioned resident pair requires canonical physical A/B plans"
            )
        plans = (by_lane["A"], by_lane["B"])
        if (
            plans[0].lane_policy != lane_pair.lane_a
            or plans[1].lane_policy != lane_pair.lane_b
        ):
            raise B300ResidentPairFactoryError(
                "resident stock plans differ from the sealed lane pair"
            )
        root = Path(swap_intake_root)
        if not root.is_absolute() or not root.is_dir() or root.is_symlink():
            raise B300ResidentPairFactoryError(
                "resident swap intake root is not one existing absolute directory"
            )
        self._validate_common_authority(service, readiness, plans, model_mount)
        self.service_digest = service
        self.readiness = readiness
        self.lane_pair = lane_pair
        self.lane_plans = plans
        self.model_mount = model_mount
        self.swap_intake_root = root
        self.start_timeout_s = _positive_seconds(start_timeout_s, "pair start timeout")
        self.request_timeout_s = _positive_seconds(
            request_timeout_s, "pair request timeout"
        )
        self.close_timeout_s = _positive_seconds(close_timeout_s, "pair close timeout")
        self.clock = clock
        self.commissioned_epoch_digest = canonical_digest(
            "cacheon.eval.b300-commissioned-resident-pair-epoch.v1",
            {
                "lane_pair": lane_pair.digest,
                "lanes": [row.commissioning_digest for row in plans],
                "model_mount": model_mount.digest,
                "readiness": {
                    "digest": readiness.digest,
                    "ready_epoch": readiness.ready_epoch,
                    "ready_receipt": readiness.ready_receipt_digest,
                },
                "service": service,
            },
        )

    @staticmethod
    def _validate_common_authority(
        service: str,
        readiness: WorkerReadiness,
        plans: tuple[B300ResidentStockLanePlan, B300ResidentStockLanePlan],
        model_mount: TrustedArenaModelMountReceipt,
    ) -> None:
        left, right = plans
        left_launch, right_launch = left.stock_launch, right.stock_launch
        common_launch_fields = set(left_launch.__dataclass_fields__) - {
            "hardware",
            "resource_policy_digest",
        }
        common_binding_fields = (
            "materialized_tree_root",
            "controller_distribution_digest",
            "native_build_spec",
            "runtime_preflight_receipt",
            "native_compile_profile",
        )
        if (
            readiness.service_digest != service
            or left_launch.arena_digest != service
            or right_launch.arena_digest != service
            or left.stock_tree != right.stock_tree
            or any(
                getattr(left_launch, field) != getattr(right_launch, field)
                for field in common_launch_fields
            )
            or any(
                getattr(left.stock_binding, field)
                != getattr(right.stock_binding, field)
                for field in common_binding_fields
            )
            or _resident_shape(left.resident_plan)
            != _resident_shape(right.resident_plan)
            or _workload_shape(left.speed_workload)
            != _workload_shape(right.speed_workload)
            or left.stock_launch.digest == right.stock_launch.digest
            or left.executor.manager is right.executor.manager
            or left.executor.manager.namespace_digest
            == right.executor.manager.namespace_digest
        ):
            raise B300ResidentPairFactoryError(
                "commissioned lanes do not share one immutable distinct stock pair"
            )
        for plan in plans:
            launch = plan.stock_launch
            if (
                launch.runtime_digest != readiness.runtime_digest
                or launch.model_revision_digest != readiness.model_revision_digest
                or launch.model_manifest_digest != readiness.model_manifest_digest
                or launch.model_content_digest != readiness.model_content_digest
                or launch.hardware.architecture != readiness.target_architecture
                or launch.hardware.topology_class != readiness.topology_class
                or launch.hardware.topology_digest != readiness.topology_digest
                or launch.hardware.visible_gpu_count != readiness.gpu_count
                or launch.hardware.tp_size != readiness.tensor_parallel_size
                or model_mount.arena_digest != service
                or model_mount.model_revision_digest != launch.model_revision_digest
                or model_mount.model_manifest_digest != launch.model_manifest_digest
                or model_mount.model_content_digest != launch.model_content_digest
            ):
                raise B300ResidentPairFactoryError(
                    "resident stock pair differs from service READY or model authority"
                )

    def open_request(
        self,
        authority: B300ResidentPairRequestAuthority,
        *,
        deadline: float,
    ) -> B300ResidentPairRequestOwner:
        """Create a fresh unstarted pair for exactly one authenticated request."""

        if type(authority) is not B300ResidentPairRequestAuthority:
            raise B300ResidentPairFactoryError(
                "resident pair request authority is not exact"
            )
        absolute = _absolute_deadline(deadline, self.clock)
        def lifetime_factories() -> tuple[Callable, Callable]:
            factories = []
            try:
                for plan in self.lane_plans:
                    factories.append(
                        make_backend_lifetime_factory(
                            plan.executor,
                            plan.stock_launch,
                            plan.stock_binding,
                            self.model_mount,
                            plan.resident_plan,
                            swap_intake_root=self.swap_intake_root,
                            deadline_provider=lambda absolute=absolute: absolute,
                        )
                    )
            except Exception as exc:
                raise B300ResidentPairFactoryError(
                    f"resident backend lifetime factory failed: {exc}"
                ) from exc
            return factories[0], factories[1]

        return B300ResidentPairRequestOwner(
            authority=authority,
            commissioned_epoch_digest=self.commissioned_epoch_digest,
            lane_plans=self.lane_plans,
            lifetime_factories=lifetime_factories,
            deadline=absolute,
            start_timeout_s=self.start_timeout_s,
            request_timeout_s=self.request_timeout_s,
            close_timeout_s=self.close_timeout_s,
            clock=self.clock,
            _construction_token=_OWNER_CONSTRUCTION_TOKEN,
        )


__all__ = [
    "B300CommissionedResidentPairFactory",
    "B300ResidentPairFactoryError",
    "B300ResidentPairRequestAuthority",
    "B300ResidentPairRequestOwner",
    "B300ResidentRequestPair",
    "B300ResidentStockLanePlan",
]
