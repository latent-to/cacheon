"""Closed B300/TP4 composition root for arena screening and qualification.

The generic arena service deliberately leaves deployment assembly out of the
consensus-facing types.  This module supplies that assembly without resolving
module names, entry points, commands, or candidate-controlled configuration.
Every executable authority is supplied in process, paired with a deployment
identity digest, and captured before the provider accepts work.

There is no generic build/ABI/graph screen API in the core package.  The four
non-serving stages therefore use exact :class:`B300ScreenStageHandler` values.
Those handlers must return real ``ScreenStageResult`` evidence; this provider
never turns a boolean or an exception into ``PASS``.  The serving stage is the
existing :class:`ResidentServingScreenStage`, owned through a bounded lifetime
so it can be released before qualification takes the deployment executors.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, fields
from typing import Callable, Protocol

from cacheon.arena_service import (
    SCREEN_STAGES,
    ArenaCandidateBinding,
    ArenaQualificationRequest,
    ArenaQualificationWork,
    ArenaRuntimeIdentity,
    ArenaServiceManifest,
    ScreenGrade,
    ScreenStagePolicy,
    ScreenStageResult,
)
from cacheon.eval.device_state import DeviceStatePolicy
from cacheon.eval.b300_qualification_lanes import (
    B300_GPU_COUNT,
    QUALIFICATION_LANE_PAIR_SCHEMA,
    QUALIFICATION_LANE_SCHEMA,
    QUALIFICATION_ROLE_SWAP_SCHEMA,
    B300ArenaProviderError,
    B300QualificationLaneOrientation,
    B300QualificationLanePair,
    B300QualificationLanePolicy,
    _digest,
)
from cacheon.eval.oci_backend import OCIBackendConfig, OCIEngineExecutor
from cacheon.eval.b300_qualification_graph_store_io import (
    B300QualificationGraphEvidenceHold,
)
from cacheon.eval.qualification_intake import QualificationPlanFactory
from cacheon.eval.qualification_runner import HiddenJudgeBinding
from cacheon.eval.resident_screen_lane import (
    ResidentScreenLifetimeFailed,
    ResidentServingScreenStage,
)
from cacheon.stack_identity import canonical_digest


PROVIDER_SCHEMA = "cacheon.eval.b300-arena-provider.v2"
SCREEN_EXCEPTION_SCHEMA = "cacheon.eval.b300-screen-exception.v1"
B300_ARCHITECTURE = "sm103"
B300_TENSOR_PARALLEL_SIZE = 4
_NON_SERVING_STAGES = SCREEN_STAGES[:-1]
_SERVING_STAGE = SCREEN_STAGES[-1]


def _resource_ids(value: object, field: str, *, allow_empty: bool) -> tuple[str, ...]:
    if type(value) is not tuple:
        raise B300ArenaProviderError(f"{field} must be an exact tuple")
    rows = tuple(value)
    if (
        (not allow_empty and not rows)
        or rows != tuple(sorted(set(rows)))
        or any(
            not isinstance(row, str)
            or not row
            or row.strip() != row
            or len(row) > 128
            or any(character in row for character in "\x00\r\n")
            for row in rows
        )
    ):
        raise B300ArenaProviderError(f"{field} are not canonical")
    return rows


class ScreenStageRunner(Protocol):
    """Closed call shape for a deployment-owned non-serving screen."""

    def __call__(
        self,
        manifest: ArenaServiceManifest,
        policy: ScreenStagePolicy,
        candidate: ArenaCandidateBinding,
    ) -> ScreenStageResult: ...


@dataclass(frozen=True)
class B300ScreenStageHandler:
    """Sealed identity and callable for one real non-serving screen stage."""

    stage: str
    identity_digest: str
    resource_ids: tuple[str, ...]
    runner: ScreenStageRunner

    def __post_init__(self) -> None:
        if self.stage not in _NON_SERVING_STAGES or not callable(self.runner):
            raise B300ArenaProviderError("screen handler stage or runner is invalid")
        object.__setattr__(
            self,
            "identity_digest",
            _digest(self.identity_digest, f"{self.stage} handler identity"),
        )
        object.__setattr__(
            self,
            "resource_ids",
            _resource_ids(
                self.resource_ids,
                f"{self.stage} handler resources",
                allow_empty=True,
            ),
        )

    def run_screen(
        self,
        manifest: ArenaServiceManifest,
        policy: ScreenStagePolicy,
        candidate: ArenaCandidateBinding,
    ) -> ScreenStageResult:
        return self.runner(manifest, policy, candidate)


@dataclass(frozen=True)
class B300ResidentScreenLifetime:
    """One exact resident screen stage plus its deployment-owned teardown."""

    screen_stage: ResidentServingScreenStage
    closer: Callable[[], None]

    def __post_init__(self) -> None:
        if type(self.screen_stage) is not ResidentServingScreenStage or not callable(
            self.closer
        ):
            raise B300ArenaProviderError("resident screen lifetime is not exact")

    def close(self) -> None:
        self.closer()


@dataclass(frozen=True)
class B300ResidentScreenFactory:
    """Sealed construction authority for a lazily opened resident stage."""

    identity_digest: str
    resource_ids: tuple[str, ...]
    create: Callable[[], B300ResidentScreenLifetime]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "identity_digest",
            _digest(self.identity_digest, "resident screen factory identity"),
        )
        object.__setattr__(
            self,
            "resource_ids",
            _resource_ids(
                self.resource_ids,
                "resident screen factory resources",
                allow_empty=False,
            ),
        )
        if not callable(self.create):
            raise B300ArenaProviderError("resident screen factory is not callable")


QualificationFactoryBuilder = Callable[
    [ArenaQualificationRequest, object | None], QualificationPlanFactory
]
DeadlineProvider = Callable[[ArenaQualificationRequest, object | None], float]


@dataclass(frozen=True)
class B300DeclaredQualificationAuthorities:
    """Path-free qualification identities retained by screen-only workers.

    A commissioned screen worker must identify the later qualification worker
    without pretending to possess its private judge, entropy, executors, or
    factory.  These declarations are therefore sufficient for provider/service
    identity, but grant no qualification capability.
    """

    qualification_policy_digest: str
    qualification_builder_digest: str
    candidate_executor_policy_digest: str
    resident_baseline_executor_policy_digest: str
    lane_pair: B300QualificationLanePair
    entropy_provider_digest: str
    hidden_judge_binding_digest: str
    deadline_policy_digest: str

    def __post_init__(self) -> None:
        for field in (
            "qualification_policy_digest",
            "qualification_builder_digest",
            "candidate_executor_policy_digest",
            "resident_baseline_executor_policy_digest",
            "entropy_provider_digest",
            "hidden_judge_binding_digest",
            "deadline_policy_digest",
        ):
            object.__setattr__(self, field, _digest(getattr(self, field), field))
        if type(self.lane_pair) is not B300QualificationLanePair:
            raise B300ArenaProviderError("qualification lane pair is not exact")

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_executor_policy_digest": (
                self.candidate_executor_policy_digest
            ),
            "deadline_policy_digest": self.deadline_policy_digest,
            "entropy_provider_digest": self.entropy_provider_digest,
            "hidden_judge_binding_digest": self.hidden_judge_binding_digest,
            "lane_pair": self.lane_pair.to_dict(),
            "qualification_builder_digest": self.qualification_builder_digest,
            "qualification_policy_digest": self.qualification_policy_digest,
            "resident_baseline_executor_policy_digest": (
                self.resident_baseline_executor_policy_digest
            ),
        }


def _validate_screen_authorities(
    runtime_identity: ArenaRuntimeIdentity,
    screen_handlers: tuple[B300ScreenStageHandler, ...],
    resident_screen_factory: B300ResidentScreenFactory,
) -> tuple[B300ScreenStageHandler, ...]:
    if type(runtime_identity) is not ArenaRuntimeIdentity:
        raise B300ArenaProviderError("runtime identity is not exact")
    _validate_b300_runtime(runtime_identity)
    handlers = tuple(screen_handlers)
    if (
        type(screen_handlers) is not tuple
        or any(type(row) is not B300ScreenStageHandler for row in handlers)
        or tuple(row.stage for row in handlers) != _NON_SERVING_STAGES
    ):
        raise B300ArenaProviderError(
            "screen handlers differ from the canonical stage order"
        )
    if type(resident_screen_factory) is not B300ResidentScreenFactory:
        raise B300ArenaProviderError("resident screen factory is not exact")
    resident_resources = set(resident_screen_factory.resource_ids)
    non_serving_resources = {
        resource for handler in handlers for resource in handler.resource_ids
    }
    if resident_resources.intersection(non_serving_resources):
        raise B300ArenaProviderError(
            "resident screen resources overlap a retained non-serving handler"
        )
    return handlers


@dataclass(frozen=True)
class B300ScreenDeploymentAuthorities:
    """Executable screen authority with declared, but unavailable, qualification."""

    runtime_identity: ArenaRuntimeIdentity
    screen_handlers: tuple[B300ScreenStageHandler, ...]
    resident_screen_factory: B300ResidentScreenFactory
    qualification: B300DeclaredQualificationAuthorities

    def __post_init__(self) -> None:
        handlers = _validate_screen_authorities(
            self.runtime_identity,
            self.screen_handlers,
            self.resident_screen_factory,
        )
        object.__setattr__(self, "screen_handlers", handlers)
        if type(self.qualification) is not B300DeclaredQualificationAuthorities:
            raise B300ArenaProviderError(
                "declared qualification authority is not exact"
            )

    @property
    def qualification_policy_digest(self) -> str:
        return self.qualification.qualification_policy_digest


@dataclass(frozen=True)
class B300DeploymentAuthorities:
    """All non-public authorities needed by one B300/TP4 service.

    The public arena manifest cannot and does not recreate these values.  In
    particular, private selection-secret lookup, plan construction, entropy,
    hidden judging, executor lifecycles, and the absolute deadline remain
    explicit deployment inputs.
    """

    runtime_identity: ArenaRuntimeIdentity
    screen_handlers: tuple[B300ScreenStageHandler, ...]
    resident_screen_factory: B300ResidentScreenFactory
    qualification_policy_digest: str
    qualification_builder_digest: str
    qualification_factory_builder: QualificationFactoryBuilder
    executor: OCIEngineExecutor
    resident_baseline_executor: OCIEngineExecutor
    entropy_provider_digest: str
    entropy_provider: object
    hidden_judge: object
    deadline_policy_digest: str
    deadline_provider: DeadlineProvider
    qualification_lane_pair: B300QualificationLanePair
    qualification_stage: str
    resident_pair_factory: object
    resident_count_quality: object

    def __post_init__(self) -> None:
        from cacheon.eval.b300_resident_pair_factory import (
            B300CommissionedResidentPairFactory,
        )
        from cacheon.eval.registered_resident_count_quality import (
            B300ResidentCountQualityCapability,
        )

        handlers = _validate_screen_authorities(
            self.runtime_identity,
            self.screen_handlers,
            self.resident_screen_factory,
        )
        object.__setattr__(self, "screen_handlers", handlers)
        for field in (
            "qualification_policy_digest",
            "qualification_builder_digest",
            "entropy_provider_digest",
            "deadline_policy_digest",
        ):
            object.__setattr__(self, field, _digest(getattr(self, field), field))
        if not callable(self.qualification_factory_builder):
            raise B300ArenaProviderError("qualification factory builder is not callable")
        if type(self.qualification_lane_pair) is not B300QualificationLanePair:
            raise B300ArenaProviderError("qualification lane pair is not exact")
        if (
            type(self.resident_pair_factory) is not B300CommissionedResidentPairFactory
            or type(self.resident_count_quality)
            is not B300ResidentCountQualityCapability
        ):
            raise B300ArenaProviderError(
                "resident pair or count authority is not exactly commissioned"
            )
        orientation = self.qualification_lane_pair.orientation(
            self.qualification_stage
        )
        if (
            type(self.executor) is not OCIEngineExecutor
            or type(self.resident_baseline_executor) is not OCIEngineExecutor
            or self.executor is self.resident_baseline_executor
            or self.executor.manager is self.resident_baseline_executor.manager
        ):
            raise B300ArenaProviderError("qualification executors are not exact and distinct")
        _executor_identity(self.executor, role="candidate")
        _executor_identity(self.resident_baseline_executor, role="resident_baseline")
        for role, executor in (
            ("candidate", self.executor),
            ("resident baseline", self.resident_baseline_executor),
        ):
            gpus = executor.device_policy.expected_gpus
            if len(gpus) != B300_GPU_COUNT or any(
                "B300" not in gpu.name.upper() for gpu in gpus
            ):
                raise B300ArenaProviderError(
                    f"{role} executor does not bind exactly four B300 devices"
                )
        _validate_executor_lane(
            self.executor,
            orientation.candidate,
            role="candidate",
        )
        _validate_executor_lane(
            self.resident_baseline_executor,
            orientation.resident_baseline,
            role="resident_baseline",
        )
        if not callable(self.entropy_provider) or not callable(self.hidden_judge):
            raise B300ArenaProviderError("entropy or hidden-judge authority is not callable")
        if type(getattr(self.hidden_judge, "binding", None)) is not HiddenJudgeBinding:
            raise B300ArenaProviderError("hidden judge has no exact sealed binding")
        if not callable(self.deadline_provider):
            raise B300ArenaProviderError("deadline provider is not callable")

    @property
    def qualification(self) -> B300DeclaredQualificationAuthorities:
        binding = getattr(self.hidden_judge, "binding", None)
        if type(binding) is not HiddenJudgeBinding:
            raise B300ArenaProviderError("hidden judge binding changed or is untyped")
        orientation = self.qualification_orientation
        _validate_executor_lane(
            self.executor,
            orientation.candidate,
            role="candidate",
        )
        _validate_executor_lane(
            self.resident_baseline_executor,
            orientation.resident_baseline,
            role="resident_baseline",
        )
        return B300DeclaredQualificationAuthorities(
            self.qualification_policy_digest,
            self.qualification_builder_digest,
            _executor_role_policy_identity(self.executor, role="candidate"),
            _executor_role_policy_identity(
                self.resident_baseline_executor,
                role="resident_baseline",
            ),
            self.qualification_lane_pair,
            self.entropy_provider_digest,
            binding.digest,
            self.deadline_policy_digest,
        )

    @property
    def qualification_orientation(self) -> B300QualificationLaneOrientation:
        return self.qualification_lane_pair.orientation(self.qualification_stage)


def _validate_b300_runtime(runtime: ArenaRuntimeIdentity) -> None:
    if (
        runtime.target_architecture != B300_ARCHITECTURE
        or runtime.gpu_count != B300_GPU_COUNT
        or runtime.tensor_parallel_size != B300_TENSOR_PARALLEL_SIZE
    ):
        raise B300ArenaProviderError(
            "runtime must be an exact sm103, four-GPU, TP4 authority"
        )


def _native_limits_payload(config: OCIBackendConfig) -> dict[str, int]:
    return {
        field.name: getattr(config.native_limits, field.name)
        for field in fields(config.native_limits)
    }


def _executor_config(
    executor: OCIEngineExecutor,
    *,
    role: str,
) -> tuple[OCIBackendConfig, DeviceStatePolicy]:
    if type(executor) is not OCIEngineExecutor:
        raise B300ArenaProviderError(f"{role} executor is not exact")
    config = getattr(executor, "config", None)
    device_policy = getattr(executor, "device_policy", None)
    manager = getattr(executor, "manager", None)
    if (
        type(config) is not OCIBackendConfig
        or type(device_policy) is not DeviceStatePolicy
        or manager is None
        or getattr(manager, "executor_id", None) != config.prebuild.executor_id
    ):
        raise B300ArenaProviderError(f"{role} executor configuration is inconsistent")
    return config, device_policy


def b300_executor_role_policy_digest(
    config: OCIBackendConfig,
    *,
    role: str,
) -> str:
    """Bind an executor role without binding it to one physical TP4 lane."""

    if type(config) is not OCIBackendConfig or role not in {
        "candidate",
        "resident_baseline",
    }:
        raise B300ArenaProviderError("qualification executor role policy is invalid")
    return canonical_digest(
        "cacheon.eval.b300-oci-executor-role-policy.v1",
        {
            "dependency_policy_digest": config.prebuild.policy.dependency_policy_digest,
            "executor_id": config.prebuild.executor_id,
            "native_limits": _native_limits_payload(config),
            "resource_policy_digest": config.prebuild.policy.resource_policy_digest,
            "role": role,
            "runtime_policy_digest": config.runtime.digest,
        },
    )


def _executor_role_policy_identity(executor: OCIEngineExecutor, *, role: str) -> str:
    config, _device_policy = _executor_config(executor, role=role)
    return b300_executor_role_policy_digest(config, role=role)


def _executor_identity(executor: OCIEngineExecutor, *, role: str) -> str:
    config, device_policy = _executor_config(executor, role=role)
    return canonical_digest(
        "cacheon.eval.b300-oci-executor-policy.v1",
        {
            "device_configuration_digest": device_policy.configuration_sha256,
            "device_policy_digest": device_policy.policy_sha256,
            "role_policy_digest": b300_executor_role_policy_digest(
                config, role=role
            ),
        },
    )


def _validate_executor_lane(
    executor: OCIEngineExecutor,
    expected: B300QualificationLanePolicy,
    *,
    role: str,
) -> None:
    _config, policy = _executor_config(executor, role=role)
    observed = B300QualificationLanePolicy.from_device_policy(
        expected.lane_id,
        policy,
    )
    if observed != expected:
        raise B300ArenaProviderError(
            f"{role} executor differs from its selected physical TP4 lane"
        )


_AuthorityBundle = B300DeploymentAuthorities | B300ScreenDeploymentAuthorities


def b300_arena_provider_digest(authorities: _AuthorityBundle) -> str:
    """Return the path-free provider identity before a manifest is constructed."""

    if type(authorities) not in {
        B300DeploymentAuthorities,
        B300ScreenDeploymentAuthorities,
    }:
        raise B300ArenaProviderError("deployment authorities are not exact")
    handlers = authorities.screen_handlers
    resident_resources = set(authorities.resident_screen_factory.resource_ids)
    non_serving_resources = {
        resource for handler in handlers for resource in handler.resource_ids
    }
    if resident_resources.intersection(non_serving_resources):
        raise B300ArenaProviderError(
            "resident screen resources overlap a retained non-serving handler"
        )
    qualification = authorities.qualification
    return canonical_digest(
        PROVIDER_SCHEMA,
        {
            "implementation": {
                "architecture": B300_ARCHITECTURE,
                "gpu_count": B300_GPU_COUNT,
                "tensor_parallel_size": B300_TENSOR_PARALLEL_SIZE,
            },
            "qualification": {
                "builder_digest": qualification.qualification_builder_digest,
                "candidate_executor_policy_digest": (
                    qualification.candidate_executor_policy_digest
                ),
                "deadline_policy_digest": qualification.deadline_policy_digest,
                "entropy_provider_digest": qualification.entropy_provider_digest,
                "hidden_judge_binding_digest": (
                    qualification.hidden_judge_binding_digest
                ),
                "lane_pair": qualification.lane_pair.to_dict(),
                "policy_digest": qualification.qualification_policy_digest,
                "resident_baseline_executor_policy_digest": (
                    qualification.resident_baseline_executor_policy_digest
                ),
            },
            "resident_screen": {
                "factory_digest": authorities.resident_screen_factory.identity_digest,
                "resource_ids": list(authorities.resident_screen_factory.resource_ids),
            },
            "runtime": authorities.runtime_identity.to_dict(),
            "screen_handlers": [
                {
                    "identity_digest": handler.identity_digest,
                    "resource_ids": list(handler.resource_ids),
                    "stage": handler.stage,
                }
                for handler in handlers
            ],
        },
    )


class B300ArenaServiceProvider:
    """Production provider assembled only from exact deployment authorities."""

    def __init__(
        self,
        manifest: ArenaServiceManifest,
        authorities: _AuthorityBundle,
    ) -> None:
        if type(manifest) is not ArenaServiceManifest:
            raise B300ArenaProviderError("arena service manifest is not exact")
        if type(authorities) not in {
            B300DeploymentAuthorities,
            B300ScreenDeploymentAuthorities,
        }:
            raise B300ArenaProviderError("deployment authorities are not exact")
        observed = b300_arena_provider_digest(authorities)
        if manifest.provider_digest != observed:
            raise B300ArenaProviderError("provider identity differs from the manifest")
        if manifest.runtime != authorities.runtime_identity:
            raise B300ArenaProviderError(
                "runtime, model, topology, or TP4 identity differs from the manifest"
            )
        if (
            manifest.qualification_policy_digest
            != authorities.qualification_policy_digest
        ):
            raise B300ArenaProviderError(
                "qualification policy differs from the manifest"
            )
        self.manifest = manifest
        self.provider_digest = observed
        self._authorities = authorities
        self._handlers = {
            handler.stage: (handler, handler.runner)
            for handler in authorities.screen_handlers
        }
        self._create_resident = authorities.resident_screen_factory.create
        capabilities = (
            authorities if type(authorities) is B300DeploymentAuthorities else None
        )
        self._qualification_capabilities = capabilities
        self.qualification_stage = (
            capabilities.qualification_stage if capabilities is not None else None
        )
        self._resident_lifetime: B300ResidentScreenLifetime | None = None
        self._resident_failed = False
        self._resident_teardown_failed = False
        self._closed = False
        self._lock = threading.RLock()

    @property
    def resident_screen_latched(self) -> bool:
        """True once only an adapter restart can clear the resident lifetime."""

        with self._lock:
            return self._resident_failed or self._resident_teardown_failed

    def retire_resident_screen(self) -> None:
        """Release the retained screen lifetime before qualification owns its lane."""

        with self._lock:
            if not self._closed and self._resident_teardown_failed:
                self._release_resident()
            self._require_open_and_current()
            self._release_resident()

    def run_screen(
        self,
        manifest: ArenaServiceManifest,
        stage: ScreenStagePolicy,
        candidate: ArenaCandidateBinding,
    ) -> ScreenStageResult:
        if (
            type(manifest) is not ArenaServiceManifest
            or manifest != self.manifest
            or type(stage) is not ScreenStagePolicy
            or type(candidate) is not ArenaCandidateBinding
        ):
            raise B300ArenaProviderError("screen request differs from deployment authority")
        expected_stage = next(
            (row for row in manifest.screens.stages if row.stage == stage.stage), None
        )
        if expected_stage != stage:
            raise B300ArenaProviderError("screen stage policy was substituted")
        with self._lock:
            self._require_open_and_current()
            started = time.monotonic()
            if stage.stage == _SERVING_STAGE:
                lifetime = self._resident_lifetime
                if lifetime is None:
                    try:
                        created = self._create_resident()
                    except Exception as exc:
                        raise B300ArenaProviderError(
                            "resident screen start failed"
                        ) from exc
                    if type(created) is not B300ResidentScreenLifetime:
                        raise B300ArenaProviderError(
                            "resident factory returned an untyped lifetime"
                        )
                    lifetime = created
                    self._resident_lifetime = lifetime
                try:
                    result = lifetime.screen_stage.run_screen(candidate)
                except ResidentScreenLifetimeFailed as exc:
                    self._resident_failed = True
                    raise B300ArenaProviderError(
                        "resident screen lifetime failed; epoch restart required"
                    ) from exc
                except Exception as exc:
                    # The request may retry while the healthy resident remains,
                    # but the exception is not a candidate screen verdict.
                    raise B300ArenaProviderError(
                        "resident screen request failed"
                    ) from exc
                if lifetime.screen_stage.lifetime_failed:
                    self._resident_failed = True
            else:
                configured = self._handlers.get(stage.stage)
                if configured is None:
                    raise B300ArenaProviderError("screen stage is not configured")
                _handler, runner = configured
                try:
                    result = runner(manifest, stage, candidate)
                except Exception as exc:
                    return self._exception_result(manifest, stage, candidate, exc, started)
            if type(result) is not ScreenStageResult or result.stage != stage.stage:
                raise B300ArenaProviderError(
                    "screen handler changed the requested stage or evidence type"
                )
            return result

    def build_qualification(
        self,
        request: ArenaQualificationRequest,
        state: object | None = None,
    ) -> ArenaQualificationWork:
        if (
            type(request) is not ArenaQualificationRequest
            or request.service_digest != self.manifest.digest
            or request.qualification_policy_digest
            != self.manifest.qualification_policy_digest
        ):
            raise B300ArenaProviderError(
                "qualification request differs from deployment authority"
            )
        with self._lock:
            capabilities = self._qualification_capabilities
            if capabilities is None:
                raise B300ArenaProviderError(
                    "qualification capabilities are unavailable on this screen worker"
                )
            self.retire_resident_screen()
            try:
                factory = capabilities.qualification_factory_builder(request, state)
            except B300QualificationGraphEvidenceHold:
                raise
            except Exception as exc:
                raise B300ArenaProviderError(
                    "qualification factory construction failed"
                ) from exc
            if type(factory) is not QualificationPlanFactory:
                raise B300ArenaProviderError(
                    "qualification builder returned an untyped plan factory"
                )
            expected_reservations = tuple(
                candidate.reservation for candidate in request.candidates
            )
            if factory.manifest.reservations != expected_reservations:
                raise B300ArenaProviderError(
                    "qualification builder changed finalized cohort order"
                )
            try:
                deadline = capabilities.deadline_provider(request, state)
            except Exception as exc:
                raise B300ArenaProviderError("deadline authority failed") from exc
            if (
                isinstance(deadline, bool)
                or not isinstance(deadline, (int, float))
                or not math.isfinite(float(deadline))
                or float(deadline) <= float(capabilities.executor.manager.clock())
            ):
                raise B300ArenaProviderError(
                    "deadline authority returned no future absolute deadline"
                )
            return ArenaQualificationWork(
                factory=factory,
                executor=capabilities.executor,
                entropy_provider=capabilities.entropy_provider,
                hidden_judge=capabilities.hidden_judge,
                deadline=float(deadline),
                qualification_policy_digest=request.qualification_policy_digest,
                resident_baseline_executor=(
                    capabilities.resident_baseline_executor
                ),
            )

    def close(self) -> None:
        """Permanently close the provider and its resident screen lifetime."""

        with self._lock:
            if self._closed and self._resident_lifetime is None:
                return
            self._closed = True
            self._release_resident()

    def _exception_result(
        self,
        manifest: ArenaServiceManifest,
        stage: ScreenStagePolicy,
        candidate: ArenaCandidateBinding,
        exc: Exception,
        started: float,
    ) -> ScreenStageResult:
        handler_digest = (
            self._authorities.resident_screen_factory.identity_digest
            if stage.stage == _SERVING_STAGE
            else self._handlers[stage.stage][0].identity_digest
        )
        evidence = canonical_digest(
            SCREEN_EXCEPTION_SCHEMA,
            {
                "candidate_digest": candidate.digest,
                "exception_type": type(exc).__name__,
                "handler_digest": handler_digest,
                "provider_digest": self.provider_digest,
                "service_digest": manifest.digest,
                "stage": stage.stage,
            },
        )
        elapsed_ms = max(1, round((time.monotonic() - started) * 1_000))
        return ScreenStageResult(
            stage.stage,
            ScreenGrade.PASS,
            evidence,
            elapsed_ms,
        )

    def _release_resident(self) -> None:
        lifetime = self._resident_lifetime
        if lifetime is None:
            self._resident_teardown_failed = False
            return
        try:
            lifetime.close()
        except BaseException as exc:
            self._resident_teardown_failed = True
            raise B300ArenaProviderError("resident screen teardown failed") from exc
        self._resident_lifetime = None
        self._resident_teardown_failed = False

    def _require_open_and_current(self) -> None:
        if self._closed:
            raise B300ArenaProviderError("arena provider is closed")
        if self._resident_failed:
            raise B300ArenaProviderError(
                "resident screen lifetime failed; epoch restart required"
            )
        if self._resident_teardown_failed:
            raise B300ArenaProviderError("resident screen teardown is unresolved")
        if b300_arena_provider_digest(self._authorities) != self.provider_digest:
            raise B300ArenaProviderError(
                "deployment authority identity changed after construction"
            )


__all__ = [
    "B300ArenaProviderError",
    "B300ArenaServiceProvider",
    "B300DeclaredQualificationAuthorities",
    "B300DeploymentAuthorities",
    "B300QualificationLaneOrientation",
    "B300QualificationLanePair",
    "B300QualificationLanePolicy",
    "B300ResidentScreenFactory",
    "B300ResidentScreenLifetime",
    "B300ScreenStageHandler",
    "B300ScreenDeploymentAuthorities",
    "PROVIDER_SCHEMA",
    "QUALIFICATION_LANE_PAIR_SCHEMA",
    "QUALIFICATION_LANE_SCHEMA",
    "QUALIFICATION_ROLE_SWAP_SCHEMA",
    "SCREEN_EXCEPTION_SCHEMA",
    "b300_executor_role_policy_digest",
    "b300_arena_provider_digest",
]
