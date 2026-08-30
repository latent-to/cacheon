"""Path-free authority for one fresh graph-disabled resident-v3 audit role.

The charged candidate runtime and the untimed audit runtime execute the same
materialized candidate tree on the same physical allocation.  They are not the
same engine lifetime: the audit launch is derived with CUDA graphs disabled and
is started only after the charged resident pair has retired.

This module binds that derivation without owning an executor or importing any
candidate code.  Runtime code must additionally call :meth:`require_executor`
immediately before launch so a stale or foreign commissioned lane fails before
any expensive work starts.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from cacheon.eval.engine_launch import EngineLaunchSpec, TrustedLaunchBinding
from cacheon.eval.oci_backend import expected_runtime_preflight
from cacheon.eval.oci_outer_session import SessionExecutionPlan
from cacheon.eval.oci_session_protocol import SlotAuditPolicy
from cacheon.stack_identity import canonical_digest, require_sha256_hex


RESIDENT_AUDIT_AUTHORITY_SCHEMA = (
    "cacheon.eval.resident-audit-execution-authority.v1"
)
RESIDENT_AUDIT_ALLOCATION_SCHEMA = "cacheon.eval.resident-audit-allocation.v1"
RESIDENT_AUDIT_BINDING_SCHEMA = "cacheon.eval.resident-audit-launch-binding.v1"
RESIDENT_AUDIT_PLAN_SCHEMA = "cacheon.eval.resident-audit-session-plan.v1"


class ResidentAuditAuthorityError(ValueError):
    """A resident audit derivation or commissioned executor is not exact."""


def _digest(value: object, field: str) -> str:
    try:
        return require_sha256_hex(value, field=field)
    except (TypeError, ValueError) as exc:
        raise ResidentAuditAuthorityError(str(exc)) from None


def _same_except(left: object, right: object, allowed: frozenset[str]) -> bool:
    fields = getattr(left, "__dataclass_fields__", None)
    return (
        type(left) is type(right)
        and isinstance(fields, dict)
        and fields == getattr(right, "__dataclass_fields__", None)
        and all(
            getattr(left, name) == getattr(right, name)
            for name in fields
            if name not in allowed
        )
    )


def resident_audit_allocation_digest(
    binding: TrustedLaunchBinding,
    *,
    device_configuration_digest: str,
) -> str:
    """Bind one physical allocation without serializing validator host paths."""

    if type(binding) is not TrustedLaunchBinding:
        raise ResidentAuditAuthorityError("resident audit binding is not exact")
    configuration = _digest(
        device_configuration_digest, "resident audit device configuration"
    )
    physical = binding.physical_hardware
    return canonical_digest(
        RESIDENT_AUDIT_ALLOCATION_SCHEMA,
        {
            "architecture": physical.architecture,
            "device_configuration": configuration,
            "device_policy": physical.device_policy_digest,
            "degrees": {
                "dp": physical.dp_size,
                "ep": physical.ep_size,
                "tp": physical.tp_size,
            },
            "physical_gpu_ids": list(physical.physical_gpu_ids),
            "topology_class": physical.topology_class,
            "topology_digest": physical.topology_digest,
        },
    )


def _binding_digest(binding: TrustedLaunchBinding) -> str:
    receipt = getattr(binding.runtime_preflight_receipt, "sha256", None)
    compile_profile = binding.native_compile_profile
    return canonical_digest(
        RESIDENT_AUDIT_BINDING_SCHEMA,
        {
            "controller_distribution": binding.controller_distribution_digest,
            "native_build": binding.native_build_spec.digest,
            "native_compile_profile": (
                None if compile_profile is None else compile_profile.digest
            ),
            "physical_hardware": {
                "architecture": binding.physical_hardware.architecture,
                "device_policy": binding.physical_hardware.device_policy_digest,
                "dp": binding.physical_hardware.dp_size,
                "ep": binding.physical_hardware.ep_size,
                "physical_gpu_ids": list(
                    binding.physical_hardware.physical_gpu_ids
                ),
                "topology_class": binding.physical_hardware.topology_class,
                "topology_digest": binding.physical_hardware.topology_digest,
                "tp": binding.physical_hardware.tp_size,
            },
            "runtime_preflight_receipt": _digest(
                receipt, "resident audit runtime preflight receipt"
            ),
        },
    )


def resident_audit_session_plan_digest(plan: SessionExecutionPlan) -> str:
    """Bind every sealed, host-owned input of one audit-only session."""

    if type(plan) is not SessionExecutionPlan or type(
        plan.audit_policy
    ) is not SlotAuditPolicy:
        raise ResidentAuditAuthorityError(
            "resident audit plan is not exact and armed"
        )
    payload = {
            "audit_policy": plan.audit_policy.digest,
            "conditioning_count": plan.conditioning_count,
            "discovery_overlay_identity": (
                plan.expected_discovery_overlay_identity_digest
            ),
            "engine_config": plan.expected_engine_config_digest,
            "launch": plan.launch_digest,
            "max_new_tokens": plan.max_new_tokens,
            "preflight": plan.expected_preflight.digest,
            "prompt_batches": plan.prompt_batches,
            "temperature": format(plan.temperature, ".17g"),
            "top_logprobs_num": plan.top_logprobs_num,
            "warmup_count": plan.warmup_count,
        }
    if plan.batch_max_new_tokens or plan.quality_max_new_tokens is not None:
        payload["batch_request_geometry"] = [
            [tokens, prompt_tokens]
            for tokens, prompt_tokens in zip(
                plan.batch_max_new_tokens,
                plan.batch_expected_prompt_tokens,
                strict=True,
            )
        ]
        payload["quality_max_new_tokens"] = plan.quality_tokens_per_prompt
    return canonical_digest(RESIDENT_AUDIT_PLAN_SCHEMA, payload)


@dataclass(frozen=True)
class ResidentAuditExecutionAuthority:
    """Charged-to-eager derivation plus one commissioned physical execution lane."""

    charged_launch: EngineLaunchSpec
    charged_binding: TrustedLaunchBinding
    charged_plan: SessionExecutionPlan
    launch: EngineLaunchSpec
    binding: TrustedLaunchBinding
    plan: SessionExecutionPlan
    executor_namespace_digest: str
    runtime_resource_policy_digest: str
    device_configuration_digest: str
    physical_allocation_digest: str

    def __post_init__(self) -> None:
        if (
            type(self.charged_launch) is not EngineLaunchSpec
            or type(self.charged_binding) is not TrustedLaunchBinding
            or type(self.charged_plan) is not SessionExecutionPlan
            or type(self.launch) is not EngineLaunchSpec
            or type(self.binding) is not TrustedLaunchBinding
            or type(self.plan) is not SessionExecutionPlan
        ):
            raise ResidentAuditAuthorityError(
                "resident audit launch, binding, and plan must be exact"
            )
        for field in (
            "executor_namespace_digest",
            "runtime_resource_policy_digest",
            "device_configuration_digest",
            "physical_allocation_digest",
        ):
            object.__setattr__(
                self,
                field,
                _digest(getattr(self, field), field.replace("_", " ")),
            )
        charged = self.charged_plan
        eager = self.plan
        try:
            charged_preflight = expected_runtime_preflight(
                self.charged_launch,
                self.charged_binding.runtime_preflight_receipt,
            )
        except (AttributeError, TypeError, ValueError) as exc:
            raise ResidentAuditAuthorityError(
                f"charged runtime preflight cannot be derived: {exc}"
            ) from None
        if (
            charged.launch_digest != self.charged_launch.digest
            or charged.expected_engine_config_digest
            != self.charged_launch.engine_config_digest
            or charged.engine_config.disable_cuda_graph
            or charged.audit_policy is not None
            or charged.expected_preflight != charged_preflight
        ):
            raise ResidentAuditAuthorityError(
                "charged speed plan is not one graph-on audit-free authority"
            )
        if (
            eager.launch_digest != self.launch.digest
            or eager.expected_engine_config_digest != self.launch.engine_config_digest
            or not eager.engine_config.disable_cuda_graph
            or type(eager.audit_policy) is not SlotAuditPolicy
            or eager.warmup_count != 1
            or eager.conditioning_count != 1
            or len(eager.prompt_batches) != eager.audit_policy.minimum_calls + 1
            or any(len(batch) != 1 for batch in eager.prompt_batches)
            or len({batch[0] for batch in eager.prompt_batches}) != 1
        ):
            raise ResidentAuditAuthorityError(
                "resident audit plan is not one exact eager audit role"
            )
        if (
            self.launch.digest == self.charged_launch.digest
            or self.launch.engine_config_digest
            == self.charged_launch.engine_config_digest
            or self.binding != self.charged_binding
            or not _same_except(
                eager.engine_config,
                charged.engine_config,
                frozenset({"disable_cuda_graph"}),
            )
            or not _same_except(
                self.launch,
                self.charged_launch,
                frozenset({"engine_config_digest"}),
            )
            or not _same_except(
                eager.expected_preflight,
                charged.expected_preflight,
                frozenset({"launch_digest", "engine_config_digest"}),
            )
            or eager.temperature != charged.temperature
            or eager.expected_discovery_overlay_identity_digest
            != charged.expected_discovery_overlay_identity_digest
        ):
            raise ResidentAuditAuthorityError(
                "eager audit drifted outside graph mode and audit workload"
            )
        try:
            self.binding.physical_hardware.validate_against(self.launch.hardware)
        except (TypeError, ValueError) as exc:
            raise ResidentAuditAuthorityError(
                f"resident audit physical binding is invalid: {exc}"
            ) from None
        expected_allocation = resident_audit_allocation_digest(
            self.binding,
            device_configuration_digest=self.device_configuration_digest,
        )
        if self.physical_allocation_digest != expected_allocation:
            raise ResidentAuditAuthorityError(
                "resident audit physical allocation digest is foreign"
            )

    @classmethod
    def derive(
        cls,
        charged_launch: EngineLaunchSpec,
        charged_binding: TrustedLaunchBinding,
        charged_plan: SessionExecutionPlan,
        *,
        audit_policy: SlotAuditPolicy,
        prompt_batches: tuple[tuple[str, ...], ...],
        max_new_tokens: int,
        top_logprobs_num: int,
        executor_namespace_digest: str,
        runtime_resource_policy_digest: str,
        device_configuration_digest: str,
    ) -> "ResidentAuditExecutionAuthority":
        """Derive the only permitted eager launch/config/preflight mutation."""

        if type(audit_policy) is not SlotAuditPolicy:
            raise ResidentAuditAuthorityError("resident audit policy is not exact")
        eager_config = replace(charged_plan.engine_config, disable_cuda_graph=True)
        eager_launch = replace(
            charged_launch, engine_config_digest=eager_config.digest
        )
        eager_preflight = replace(
            charged_plan.expected_preflight,
            launch_digest=eager_launch.digest,
            engine_config_digest=eager_config.digest,
        )
        eager_plan = replace(
            charged_plan,
            launch_digest=eager_launch.digest,
            expected_engine_config_digest=eager_config.digest,
            engine_config=eager_config,
            expected_preflight=eager_preflight,
            prompt_batches=prompt_batches,
            warmup_count=1,
            conditioning_count=1,
            max_new_tokens=max_new_tokens,
            top_logprobs_num=top_logprobs_num,
            audit_policy=audit_policy,
            batch_max_new_tokens=(),
            batch_expected_prompt_tokens=(),
            quality_max_new_tokens=None,
        )
        allocation = resident_audit_allocation_digest(
            charged_binding,
            device_configuration_digest=device_configuration_digest,
        )
        return cls(
            charged_launch,
            charged_binding,
            charged_plan,
            eager_launch,
            charged_binding,
            eager_plan,
            executor_namespace_digest,
            runtime_resource_policy_digest,
            device_configuration_digest,
            allocation,
        )

    @property
    def audit_policy(self) -> SlotAuditPolicy:
        policy = self.plan.audit_policy
        assert type(policy) is SlotAuditPolicy
        return policy

    @staticmethod
    def audit_launch_matches_role(
        audit_launch_digest: str,
        charged_launch_digest: str,
        *,
        resident: bool,
    ) -> bool:
        """Require a distinct resident eager launch; retain legacy equality."""

        if type(resident) is not bool:
            raise ResidentAuditAuthorityError(
                "resident audit launch relation mode is not exact"
            )
        audit = _digest(audit_launch_digest, "audit launch digest")
        charged = _digest(charged_launch_digest, "charged launch digest")
        return audit != charged if resident else audit == charged

    @property
    def digest(self) -> str:
        return canonical_digest(
            RESIDENT_AUDIT_AUTHORITY_SCHEMA,
            {
                "audit": {
                    "binding": _binding_digest(self.binding),
                    "launch": self.launch.digest,
                    "plan": resident_audit_session_plan_digest(self.plan),
                },
                "charged": {
                    "binding": _binding_digest(self.charged_binding),
                    "launch": self.charged_launch.digest,
                    "plan": canonical_digest(
                        "cacheon.eval.resident-audit-charged-plan.v1",
                        {
                            "engine_config": (
                                self.charged_plan.expected_engine_config_digest
                            ),
                            "launch": self.charged_plan.launch_digest,
                            "preflight": self.charged_plan.expected_preflight.digest,
                            "workload": {
                                "conditioning_count": (
                                    self.charged_plan.conditioning_count
                                ),
                                "max_new_tokens": self.charged_plan.max_new_tokens,
                                "prompt_batches": self.charged_plan.prompt_batches,
                                "temperature": format(
                                    self.charged_plan.temperature, ".17g"
                                ),
                                "top_logprobs_num": (
                                    self.charged_plan.top_logprobs_num
                                ),
                                "warmup_count": self.charged_plan.warmup_count,
                            },
                        },
                    ),
                },
                "execution": {
                    "device_configuration": self.device_configuration_digest,
                    "executor_namespace": self.executor_namespace_digest,
                    "physical_allocation": self.physical_allocation_digest,
                    "runtime_resource_policy": (
                        self.runtime_resource_policy_digest
                    ),
                },
            },
        )

    def require_executor(self, executor: object) -> None:
        """Fail before launch unless a live executor is the sealed audit lane."""

        try:
            namespace = executor.manager.namespace_digest  # type: ignore[attr-defined]
            runtime = executor.config.runtime.digest  # type: ignore[attr-defined]
            launch_policy = (  # type: ignore[attr-defined]
                executor.config.prebuild.policy.resource_policy_digest
            )
            device_policy = executor.device_policy  # type: ignore[attr-defined]
            physical_ids = tuple(map(str, device_policy.physical_gpu_ids))
            configuration = device_policy.configuration_sha256
            policy = device_policy.policy_sha256
        except (AttributeError, TypeError, ValueError) as exc:
            raise ResidentAuditAuthorityError(
                "resident audit executor lacks exact commissioned identities"
            ) from exc
        physical = self.binding.physical_hardware
        if (
            namespace != self.executor_namespace_digest
            or runtime != self.runtime_resource_policy_digest
            or launch_policy != self.launch.resource_policy_digest
            or configuration != self.device_configuration_digest
            or policy != physical.device_policy_digest
            or physical_ids != physical.physical_gpu_ids
        ):
            raise ResidentAuditAuthorityError(
                "resident audit executor differs from commissioned authority"
            )


__all__ = [
    "RESIDENT_AUDIT_AUTHORITY_SCHEMA",
    "ResidentAuditAuthorityError",
    "ResidentAuditExecutionAuthority",
    "resident_audit_allocation_digest",
    "resident_audit_session_plan_digest",
]
