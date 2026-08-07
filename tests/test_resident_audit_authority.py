"""CPU-only contracts for the fresh eager resident-v3 audit authority."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from types import SimpleNamespace

import pytest

from cacheon.eval.resident_audit_authority import (
    ResidentAuditAuthorityError,
    ResidentAuditExecutionAuthority,
    resident_audit_allocation_digest,
)
from cacheon.eval.oci_session_protocol import SlotAuditPolicy
from tests.test_marginal_runtime import _case


def _h(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _authority(tmp_path) -> ResidentAuditExecutionAuthority:
    case = _case(tmp_path)
    policy = SlotAuditPolicy(
        "1" * 32,
        1_000_000,
        2,
        ("attention.msa_prefill_block_score",),
        case.session.engine_config.tp_size,
    )
    prompt = case.session.prompt_batches[0][0]
    return ResidentAuditExecutionAuthority.derive(
        case.launch,
        case.baseline_binding.launch_binding,
        case.session,
        audit_policy=policy,
        prompt_batches=((prompt,), (prompt,), (prompt,)),
        max_new_tokens=2,
        top_logprobs_num=1,
        executor_namespace_digest=_h("audit-executor-namespace"),
        runtime_resource_policy_digest=_h("audit-runtime-policy"),
        device_configuration_digest=_h("audit-device-configuration"),
    )


def _executor(authority: ResidentAuditExecutionAuthority) -> SimpleNamespace:
    physical = authority.binding.physical_hardware
    return SimpleNamespace(
        manager=SimpleNamespace(
            namespace_digest=authority.executor_namespace_digest
        ),
        config=SimpleNamespace(
            runtime=SimpleNamespace(
                digest=authority.runtime_resource_policy_digest
            ),
            prebuild=SimpleNamespace(
                policy=SimpleNamespace(
                    resource_policy_digest=authority.launch.resource_policy_digest
                )
            ),
        ),
        device_policy=SimpleNamespace(
            physical_gpu_ids=tuple(map(int, physical.physical_gpu_ids)),
            configuration_sha256=authority.device_configuration_digest,
            policy_sha256=physical.device_policy_digest,
        ),
    )


def test_derivation_changes_only_graph_config_launch_and_preflight(tmp_path) -> None:
    authority = _authority(tmp_path)
    charged = authority.charged_plan
    eager = authority.plan

    assert not charged.engine_config.disable_cuda_graph
    assert eager.engine_config.disable_cuda_graph
    assert authority.launch.digest != authority.charged_launch.digest
    assert authority.binding == authority.charged_binding
    assert authority.binding.materialized_tree_root == (
        authority.charged_binding.materialized_tree_root
    )
    for name in charged.engine_config.__dataclass_fields__:
        if name != "disable_cuda_graph":
            assert getattr(eager.engine_config, name) == getattr(
                charged.engine_config, name
            )
    for name in authority.launch.__dataclass_fields__:
        if name != "engine_config_digest":
            assert getattr(authority.launch, name) == getattr(
                authority.charged_launch, name
            )
    for name in charged.expected_preflight.__dataclass_fields__:
        if name not in {"launch_digest", "engine_config_digest"}:
            assert getattr(eager.expected_preflight, name) == getattr(
                charged.expected_preflight, name
            )
    assert authority.physical_allocation_digest == (
        resident_audit_allocation_digest(
            authority.binding,
            device_configuration_digest=authority.device_configuration_digest,
        )
    )
    assert authority.digest == _authority(tmp_path / "repeat").digest


def test_rejects_graph_on_or_identical_audit_launch(tmp_path) -> None:
    authority = _authority(tmp_path)
    charged = authority.charged_plan
    graph_on_audit = replace(
        charged,
        prompt_batches=authority.plan.prompt_batches,
        warmup_count=1,
        conditioning_count=1,
        max_new_tokens=authority.plan.max_new_tokens,
        top_logprobs_num=authority.plan.top_logprobs_num,
        audit_policy=authority.audit_policy,
    )
    with pytest.raises(ResidentAuditAuthorityError, match="eager audit role"):
        replace(
            authority,
            launch=authority.charged_launch,
            plan=graph_on_audit,
        )


def test_rejects_foreign_binding_allocation_and_speed_policy_leakage(
    tmp_path,
) -> None:
    authority = _authority(tmp_path)
    foreign_physical = replace(
        authority.binding.physical_hardware,
        physical_gpu_ids=("9",),
    )
    foreign_binding = replace(
        authority.binding, physical_hardware=foreign_physical
    )
    with pytest.raises(ResidentAuditAuthorityError, match="drifted"):
        replace(authority, binding=foreign_binding)
    with pytest.raises(ResidentAuditAuthorityError, match="allocation digest"):
        replace(authority, physical_allocation_digest=_h("foreign-allocation"))
    with pytest.raises(ResidentAuditAuthorityError, match="audit-free"):
        replace(
            authority,
            charged_plan=replace(
                authority.charged_plan, audit_policy=authority.audit_policy
            ),
        )


def test_rejects_foreign_policy_preflight_and_unrelated_config_or_launch_drift(
    tmp_path,
) -> None:
    authority = _authority(tmp_path)
    with pytest.raises(ResidentAuditAuthorityError, match="allocation digest"):
        replace(authority, device_configuration_digest=_h("foreign-device"))

    foreign_preflight = replace(
        authority.plan.expected_preflight,
        runtime_digest=_h("foreign-runtime"),
    )
    with pytest.raises(ResidentAuditAuthorityError, match="drifted"):
        replace(
            authority,
            plan=replace(authority.plan, expected_preflight=foreign_preflight),
        )

    foreign_config = replace(
        authority.plan.engine_config,
        mem_fraction_static=authority.plan.engine_config.mem_fraction_static / 2,
    )
    foreign_launch = replace(
        authority.launch, engine_config_digest=foreign_config.digest
    )
    foreign_preflight = replace(
        authority.plan.expected_preflight,
        launch_digest=foreign_launch.digest,
        engine_config_digest=foreign_config.digest,
    )
    with pytest.raises(ResidentAuditAuthorityError, match="drifted"):
        replace(
            authority,
            launch=foreign_launch,
            plan=replace(
                authority.plan,
                launch_digest=foreign_launch.digest,
                expected_engine_config_digest=foreign_config.digest,
                engine_config=foreign_config,
                expected_preflight=foreign_preflight,
            ),
        )

    foreign_launch = replace(
        authority.launch,
        validator_overlay_digest=_h("foreign-overlay"),
    )
    with pytest.raises(ResidentAuditAuthorityError, match="drifted"):
        replace(
            authority,
            launch=foreign_launch,
            plan=replace(
                authority.plan,
                launch_digest=foreign_launch.digest,
                expected_preflight=replace(
                    authority.plan.expected_preflight,
                    launch_digest=foreign_launch.digest,
                ),
            ),
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("namespace_digest", _h("foreign-namespace")),
        ("runtime_digest", _h("foreign-runtime")),
        ("launch_resource_policy", _h("foreign-launch-policy")),
        ("configuration_sha256", _h("foreign-configuration")),
        ("device_policy_sha256", _h("foreign-device-policy")),
        ("physical_gpu_ids", (99,)),
    ),
)
def test_executor_identity_mismatch_is_rejected(
    tmp_path,
    field: str,
    value: object,
) -> None:
    authority = _authority(tmp_path)
    executor = _executor(authority)
    if field == "namespace_digest":
        executor.manager.namespace_digest = value
    elif field == "runtime_digest":
        executor.config.runtime.digest = value
    elif field == "launch_resource_policy":
        executor.config.prebuild.policy.resource_policy_digest = value
    elif field == "device_policy_sha256":
        executor.device_policy.policy_sha256 = value
    else:
        setattr(executor.device_policy, field, value)

    with pytest.raises(ResidentAuditAuthorityError, match="commissioned"):
        authority.require_executor(executor)


def test_exact_executor_is_accepted(tmp_path) -> None:
    authority = _authority(tmp_path)
    authority.require_executor(_executor(authority))


def test_causal_authority_binds_eager_plan_and_rejects_old_fake(tmp_path) -> None:
    import cacheon.eval.qualification_runner as runner
    from tests.test_qualification_runner import _typed_resident_qualification_input

    value = _typed_resident_qualification_input(tmp_path)
    audit = value.resident_audit_plan
    assert audit is not None
    changed_plan = replace(
        audit.plan,
        prompt_batches=tuple(
            ("changed audit prompt",) for _ in audit.plan.prompt_batches
        ),
    )
    changed = replace(value, resident_audit_plan=replace(audit, plan=changed_plan))
    assert runner.qualification_authority_digest(changed) != (
        runner.qualification_authority_digest(value)
    )
    with pytest.raises(
        runner.QualificationRunnerError,
        match="resident audit plan coverage differs",
    ):
        replace(
            value,
            resident_audit_plan=replace(
                value.prepared.candidates[0].session_plan,
                audit_policy=value.audit_policies[0],
            ),
        )


def test_causal_authority_rejects_foreign_execution_binding_and_policy(
    tmp_path,
) -> None:
    import cacheon.eval.qualification_runner as runner
    from tests.test_qualification_runner import _typed_resident_qualification_input

    value = _typed_resident_qualification_input(tmp_path)
    audit = value.resident_audit_plan
    assert audit is not None

    foreign_physical = replace(
        audit.binding.physical_hardware, physical_gpu_ids=("9",)
    )
    foreign_binding = replace(audit.binding, physical_hardware=foreign_physical)
    foreign_device = _h("causal-foreign-device")
    rows = (
        replace(audit, executor_namespace_digest=_h("causal-foreign-namespace")),
        replace(audit, runtime_resource_policy_digest=_h("causal-foreign-runtime")),
        replace(
            audit,
            device_configuration_digest=foreign_device,
            physical_allocation_digest=resident_audit_allocation_digest(
                audit.binding, device_configuration_digest=foreign_device
            ),
        ),
        replace(
            audit,
            charged_binding=foreign_binding,
            binding=foreign_binding,
            physical_allocation_digest=resident_audit_allocation_digest(
                foreign_binding,
                device_configuration_digest=audit.device_configuration_digest,
            ),
        ),
        replace(
            audit,
            plan=replace(
                audit.plan,
                audit_policy=replace(
                    audit.audit_policy, validator_seed="2" * 32
                ),
            ),
        ),
    )
    for foreign in rows:
        with pytest.raises(
            runner.QualificationRunnerError,
            match="resident audit plan differs",
        ):
            replace(value, resident_audit_plan=foreign)


def test_audit_authority_does_not_leak_into_speed_or_pristine_t(
    monkeypatch,
) -> None:
    from cacheon.eval.qualification import QualificationDecision
    from tests.test_qualification_runner import (
        _Harness,
        _install_resident_runner_path,
        _run_resident_harness,
    )

    harness = _Harness(
        monkeypatch,
        graph=(QualificationDecision.PASS,),
        speed=(QualificationDecision.PASS,),
        quality=(QualificationDecision.PASS,),
    )
    baseline, _, _ = _install_resident_runner_path(
        monkeypatch,
        harness,
        speed_decision=QualificationDecision.PASS,
        escalated=False,
    )
    audit = harness.value.resident_audit_plan
    _run_resident_harness(harness, baseline)
    assert harness.resident_speed_plans == [harness.value.resident_speed_plan]
    assert audit not in harness.resident_speed_plans
    assert len(harness.reference_session_plans) == 1
    assert harness.reference_session_plans[0] is not audit
