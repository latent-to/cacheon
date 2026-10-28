"""Closed B300 composition root for arena qualification.

The generic arena service deliberately leaves deployment assembly out of the
consensus-facing types.  This module supplies that assembly without resolving
module names, entry points, commands, or candidate-controlled configuration.
Every executable authority is supplied in process, paired with a deployment
identity digest, and captured before the provider accepts work.

A commissioning run seals only :class:`B300DeclaredAuthorities` into the
manifest; the full :class:`B300DeploymentAuthorities` with executor, judge,
entropy, and deadline exist only inside the qualification worker process.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, fields
from typing import Callable

from cacheon.arena_service import (
    ArenaQualificationRequest,
    ArenaQualificationWork,
    ArenaRuntimeIdentity,
    ArenaServiceManifest,
)
from cacheon.eval.device_state import DeviceStatePolicy
from cacheon.eval.b300_qualification_lanes import (
    QUALIFICATION_LANE_SCHEMA,
    B300ArenaProviderError,
    B300QualificationLanePolicy,
    _digest,
)
from cacheon.eval.oci_backend import OCIBackendConfig, OCIEngineExecutor
from cacheon.eval.qualification_intake import QualificationPlanFactory
from cacheon.eval.qualification_runner import HiddenJudgeBinding
from cacheon.stack_identity import canonical_digest


PROVIDER_SCHEMA = "cacheon.eval.b300-arena-provider.v4"


QualificationFactoryBuilder = Callable[
    [ArenaQualificationRequest, object | None], QualificationPlanFactory
]
DeadlineProvider = Callable[[ArenaQualificationRequest, object | None], float]


@dataclass(frozen=True)
class B300DeclaredQualificationAuthorities:
    """Path-free qualification identities sealed before any executor exists.

    A commissioning run must identify the qualification worker without
    pretending to possess its private judge, entropy, executor, or factory.
    These declarations are therefore sufficient for provider/service identity,
    but grant no qualification capability.
    """

    qualification_policy_digest: str
    qualification_builder_digest: str
    executor_policy_digest: str
    lane: B300QualificationLanePolicy
    entropy_provider_digest: str
    hidden_judge_binding_digest: str
    deadline_policy_digest: str

    def __post_init__(self) -> None:
        for field in (
            "qualification_policy_digest",
            "qualification_builder_digest",
            "executor_policy_digest",
            "entropy_provider_digest",
            "hidden_judge_binding_digest",
            "deadline_policy_digest",
        ):
            object.__setattr__(self, field, _digest(getattr(self, field), field))
        if type(self.lane) is not B300QualificationLanePolicy:
            raise B300ArenaProviderError("qualification lane is not exact")

    def to_dict(self) -> dict[str, object]:
        return {
            "deadline_policy_digest": self.deadline_policy_digest,
            "entropy_provider_digest": self.entropy_provider_digest,
            "executor_policy_digest": self.executor_policy_digest,
            "hidden_judge_binding_digest": self.hidden_judge_binding_digest,
            "lane": self.lane.to_dict(),
            "qualification_builder_digest": self.qualification_builder_digest,
            "qualification_policy_digest": self.qualification_policy_digest,
        }


@dataclass(frozen=True)
class B300DeclaredAuthorities:
    """Runtime identity plus declared qualification: the manifest's provider input.

    This is what a commissioning run seals into the arena manifest.  It can
    validate readiness and identity but cannot build qualification work.
    """

    runtime_identity: ArenaRuntimeIdentity
    qualification: B300DeclaredQualificationAuthorities

    def __post_init__(self) -> None:
        if type(self.runtime_identity) is not ArenaRuntimeIdentity:
            raise B300ArenaProviderError("runtime identity is not exact")
        if type(self.qualification) is not B300DeclaredQualificationAuthorities:
            raise B300ArenaProviderError(
                "declared qualification authority is not exact"
            )
        self.qualification.lane.validate_runtime(self.runtime_identity)

    @property
    def qualification_policy_digest(self) -> str:
        return self.qualification.qualification_policy_digest


@dataclass(frozen=True)
class B300DeploymentAuthorities:
    """All non-public authorities needed by one B300 qualification service.

    The public arena manifest cannot and does not recreate these values.  In
    particular, private selection-secret lookup, plan construction, entropy,
    hidden judging, the executor lifecycle, and the absolute deadline remain
    explicit deployment inputs.  One executor owns the whole lane: it runs the
    stock reference, the candidate, and the stock bracket in turn.
    """

    runtime_identity: ArenaRuntimeIdentity
    qualification_policy_digest: str
    qualification_builder_digest: str
    qualification_factory_builder: QualificationFactoryBuilder
    executor: OCIEngineExecutor
    entropy_provider_digest: str
    entropy_provider: object
    hidden_judge: object
    deadline_policy_digest: str
    deadline_provider: DeadlineProvider
    qualification_lane: B300QualificationLanePolicy

    def __post_init__(self) -> None:
        if type(self.runtime_identity) is not ArenaRuntimeIdentity:
            raise B300ArenaProviderError("runtime identity is not exact")
        for field in (
            "qualification_policy_digest",
            "qualification_builder_digest",
            "entropy_provider_digest",
            "deadline_policy_digest",
        ):
            object.__setattr__(self, field, _digest(getattr(self, field), field))
        if not callable(self.qualification_factory_builder):
            raise B300ArenaProviderError("qualification factory builder is not callable")
        if type(self.qualification_lane) is not B300QualificationLanePolicy:
            raise B300ArenaProviderError("qualification lane is not exact")
        self.qualification_lane.validate_runtime(self.runtime_identity)
        _validate_executor_lane(self.executor, self.qualification_lane)
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
        config, _policy = _validate_executor_lane(self.executor, self.qualification_lane)
        return B300DeclaredQualificationAuthorities(
            self.qualification_policy_digest,
            self.qualification_builder_digest,
            b300_executor_policy_digest(config),
            self.qualification_lane,
            self.entropy_provider_digest,
            binding.digest,
            self.deadline_policy_digest,
        )


def _native_limits_payload(config: OCIBackendConfig) -> dict[str, int]:
    return {
        field.name: getattr(config.native_limits, field.name)
        for field in fields(config.native_limits)
    }


def b300_executor_policy_digest(config: OCIBackendConfig) -> str:
    """Bind the executor policy without binding it to one physical lane."""

    if type(config) is not OCIBackendConfig:
        raise B300ArenaProviderError("qualification executor policy is invalid")
    return canonical_digest(
        "cacheon.eval.b300-oci-executor-role-policy.v2",
        {
            "dependency_policy_digest": config.prebuild.policy.dependency_policy_digest,
            "executor_id": config.prebuild.executor_id,
            "native_limits": _native_limits_payload(config),
            "resource_policy_digest": config.prebuild.policy.resource_policy_digest,
            "runtime_policy_digest": config.runtime.digest,
        },
    )


def _validate_executor_lane(
    executor: OCIEngineExecutor,
    expected: B300QualificationLanePolicy,
) -> tuple[OCIBackendConfig, DeviceStatePolicy]:
    if type(executor) is not OCIEngineExecutor:
        raise B300ArenaProviderError("qualification executor is not exact")
    config = getattr(executor, "config", None)
    policy = getattr(executor, "device_policy", None)
    manager = getattr(executor, "manager", None)
    if (
        type(config) is not OCIBackendConfig
        or type(policy) is not DeviceStatePolicy
        or manager is None
        or getattr(manager, "executor_id", None) != config.prebuild.executor_id
    ):
        raise B300ArenaProviderError("qualification executor configuration is inconsistent")
    if B300QualificationLanePolicy.from_device_policy(expected.lane_id, policy) != expected:
        raise B300ArenaProviderError(
            "qualification executor differs from its physical lane"
        )
    return config, policy


_AuthorityBundle = B300DeploymentAuthorities | B300DeclaredAuthorities


def b300_arena_provider_digest(authorities: _AuthorityBundle) -> str:
    """Return the path-free provider identity before a manifest is constructed."""

    if type(authorities) not in {
        B300DeploymentAuthorities,
        B300DeclaredAuthorities,
    }:
        raise B300ArenaProviderError("deployment authorities are not exact")
    qualification = authorities.qualification
    return canonical_digest(
        PROVIDER_SCHEMA,
        {
            "implementation": {
                "architecture": authorities.runtime_identity.target_architecture,
                "gpu_count": authorities.runtime_identity.gpu_count,
                "tensor_parallel_size": authorities.runtime_identity.tensor_parallel_size,
            },
            "qualification": {
                "builder_digest": qualification.qualification_builder_digest,
                "deadline_policy_digest": qualification.deadline_policy_digest,
                "entropy_provider_digest": qualification.entropy_provider_digest,
                "executor_policy_digest": qualification.executor_policy_digest,
                "hidden_judge_binding_digest": (
                    qualification.hidden_judge_binding_digest
                ),
                "lane": qualification.lane.service_policy(),
                "policy_digest": qualification.qualification_policy_digest,
            },
            "runtime": authorities.runtime_identity.to_dict(),
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
            B300DeclaredAuthorities,
        }:
            raise B300ArenaProviderError("deployment authorities are not exact")
        observed = b300_arena_provider_digest(authorities)
        if manifest.provider_digest != observed:
            raise B300ArenaProviderError("provider identity differs from the manifest")
        if manifest.runtime != authorities.runtime_identity:
            raise B300ArenaProviderError(
                "runtime, model, topology, or TP identity differs from the manifest"
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
        capabilities = (
            authorities if type(authorities) is B300DeploymentAuthorities else None
        )
        self._qualification_capabilities = capabilities
        self._closed = False
        self._lock = threading.RLock()

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
                    "qualification capabilities are unavailable on a declared-only provider"
                )
            self._require_open_and_current()
            try:
                factory = capabilities.qualification_factory_builder(request, state)
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
            )

    def close(self) -> None:
        """Permanently close the provider."""

        with self._lock:
            self._closed = True

    def _require_open_and_current(self) -> None:
        if self._closed:
            raise B300ArenaProviderError("arena provider is closed")
        if b300_arena_provider_digest(self._authorities) != self.provider_digest:
            raise B300ArenaProviderError(
                "deployment authority identity changed after construction"
            )


__all__ = [
    "B300ArenaProviderError",
    "B300ArenaServiceProvider",
    "B300DeclaredAuthorities",
    "B300DeclaredQualificationAuthorities",
    "B300DeploymentAuthorities",
    "B300QualificationLanePolicy",
    "PROVIDER_SCHEMA",
    "QUALIFICATION_LANE_SCHEMA",
    "b300_executor_policy_digest",
    "b300_arena_provider_digest",
]
