"""Qualification composition and authority checks for GLM/TP4 and Qwen/TP1."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from cacheon.arena_service import Workload, WorkloadCell

import pytest
import tests.test_calibration as calibration_fixtures
import tests.test_oci_backend as oci_backend_fixtures

from cacheon.eval import b300_qualification_commission as commission
from cacheon.eval.b300_qualification_deployment import B300RegisteredProfileAuthority
from cacheon.eval.b300_qualification_lanes import (
    B300QualificationLanePair,
    B300QualificationLanePolicy,
)
from cacheon.eval.b300_sealed_qualification_commission import (
    QUALIFICATION_DEADLINE_MAXIMUM_SECONDS,
)
from cacheon.eval.calibration import CalibrationEvidenceSet, derive_calibration_manifest
from cacheon.eval.qualification_runner import HiddenJudgeBinding
from cacheon.target_catalog import default_target_catalog
from tests.support.b300 import (
    M3_REGISTERED_TARGET_IDS,
    StubHiddenJudge as _Judge,
    gpu as _gpu,
    qualification_capabilities as _capabilities,
)


def _h(seed: str) -> str:
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()


class _DeferredJudge:
    def __init__(self, binding: HiddenJudgeBinding, tokenizer_digest: str) -> None:
        self.binding = binding
        self.tokenizer_digest = tokenizer_digest
        self.calls: list[dict[str, object]] = []

    def bind_prompt_plan(self, **kwargs: object) -> _Judge:
        self.calls.append(dict(kwargs))
        result = _Judge()
        result.binding = self.binding
        return result


def test_capabilities_seal_exact_callables_and_identities() -> None:
    capabilities = _capabilities()
    assert capabilities.source_resolver_digest == _h("source-resolver")

    with pytest.raises(commission.B300QualificationCommissionError):
        _capabilities(secret_loader=object())
    with pytest.raises(commission.B300QualificationCommissionError):
        _capabilities(source_resolver=object())
    with pytest.raises(commission.B300QualificationCommissionError):
        _capabilities(hidden_judge=lambda **_kwargs: None)
    with pytest.raises(commission.B300QualificationCommissionError):
        _capabilities(source_resolver_digest="not-a-digest")
    with pytest.raises(commission.B300QualificationCommissionError):
        _capabilities(source_resolver_digest=_h("upper").upper())
    with pytest.raises(commission.B300QualificationCommissionError):
        _capabilities(incumbent_entries=[("moe.fused_experts", object())])
    with pytest.raises(commission.B300QualificationCommissionError):
        _capabilities(incumbent_entries={"moe.fused_experts": object()})


def test_deferred_hidden_judge_binds_only_the_exact_composed_plan() -> None:
    binding = HiddenJudgeBinding(
        _h("hidden-corpus"), _h("hidden-judge"), _h("hidden-policy")
    )
    tokenizer = _h("tokenizer")
    deferred = _DeferredJudge(binding, tokenizer)
    capabilities = _capabilities(hidden_judge=deferred)
    assert capabilities.hidden_judge is deferred

    bound = commission._bind_hidden_judge(
        deferred,
        binding=binding,
        tokenizer_digest=tokenizer,
        prompt_batches=(("prompt",),),
        workload_digest=_h("workload"),
        hidden_tasks_per_prompt=1,
    )
    assert callable(bound)
    assert deferred.calls == [
        {
            "hidden_tasks_per_prompt": 1,
            "prompt_batches": (("prompt",),),
            "workload_digest": _h("workload"),
        }
    ]

    with pytest.raises(
        commission.B300QualificationCommissionError,
        match="tokenizer differs",
    ):
        commission._bind_hidden_judge(
            deferred,
            binding=binding,
            tokenizer_digest=_h("other-tokenizer"),
            prompt_batches=(("prompt",),),
            workload_digest=_h("workload"),
            hidden_tasks_per_prompt=1,
        )


def test_tracked_deadline_is_lease_bounded_monotonic() -> None:
    deadline = commission._tracked_deadline_provider(clock=lambda: 1000.0)
    assert deadline(object()) == 1000.0 + QUALIFICATION_DEADLINE_MAXIMUM_SECONDS
    # The policy never depends on the cohort it is asked about.
    assert deadline(None) == deadline(object())


def test_pristine_reference_is_the_baseline_at_genesis_and_stock_after_a_crown(
    tmp_path: Path,
) -> None:
    case = oci_backend_fixtures._case(tmp_path)
    incumbent_launch, incumbent_plan = case.launch, case.plan

    # Genesis: the declared incumbent is the empty stock stack, so the pristine
    # tree/native identities coincide with the incumbent's and T is that launch.
    # A node arena binds no seam selection that could tell the two apart, and
    # requiring a difference here refused every genesis commission.
    pristine_launch, pristine_plan = commission._pristine_reference_authority(
        incumbent_launch,
        incumbent_plan,
        case.preflight,
        pristine_tree=SimpleNamespace(
            stack_digest=incumbent_launch.stack_digest,
            tree_digest=incumbent_launch.tree_digest,
        ),
        pristine_native=SimpleNamespace(
            digest=incumbent_launch.native_build_spec_digest
        ),
    )

    assert pristine_launch.digest == incumbent_launch.digest
    assert pristine_launch.engine_config_digest == pristine_plan.engine_config.digest
    assert pristine_plan.launch_digest == pristine_launch.digest
    assert pristine_plan.expected_preflight.engine_config_digest == (
        pristine_plan.engine_config.digest
    )

    # Post-crown: pristine T stays anchored to the empty stock tree even when
    # the incumbent baseline carries crowned contributions.
    divergent, _ = commission._pristine_reference_authority(
        incumbent_launch,
        incumbent_plan,
        case.preflight,
        pristine_tree=SimpleNamespace(
            stack_digest=_h("stock-stack"), tree_digest=_h("stock-tree")
        ),
        pristine_native=SimpleNamespace(digest=_h("stock-native")),
    )
    assert (
        divergent.stack_digest,
        divergent.tree_digest,
        divergent.native_build_spec_digest,
    ) == (_h("stock-stack"), _h("stock-tree"), _h("stock-native"))


def test_commission_rejects_an_eleven_row_factory_registry_before_runtime() -> None:
    catalog = default_target_catalog()
    profiles = tuple(
        B300RegisteredProfileAuthority(
            target_id,
            catalog.target_spec_digest(target_id),
            _h(f"resolver:{target_id}"),
            lambda _candidate, _prepared: object(),
        )
        for target_id in M3_REGISTERED_TARGET_IDS
    )

    assert tuple(
        commission._require_complete_factory_profiles(
            profiles, M3_REGISTERED_TARGET_IDS
        )
    ) == (
        M3_REGISTERED_TARGET_IDS
    )
    with pytest.raises(
        commission.B300QualificationCommissionError,
        match="full catalog",
    ):
        commission._require_complete_factory_profiles(
            profiles[:-1], M3_REGISTERED_TARGET_IDS
        )


def test_lane_policies_reopen_exact_canonical_pair() -> None:
    eight = tuple(_gpu(index) for index in range(8))
    policy_a = commission.screen_deployment._device_policy(eight[:4])
    policy_b = commission.screen_deployment._device_policy(eight[4:])
    lanes = B300QualificationLanePair(
        B300QualificationLanePolicy.from_device_policy("A", policy_a),
        B300QualificationLanePolicy.from_device_policy("B", policy_b),
    )
    inputs = SimpleNamespace(
        qualification_gpus=eight,
        qualification_lane_pair=lanes,
    )
    observed_a, observed_b = commission._lane_policies(inputs)
    assert observed_a == policy_a
    assert observed_b == policy_b

    missing = SimpleNamespace(
        qualification_gpus=eight[:7],
        qualification_lane_pair=lanes,
    )
    with pytest.raises(commission.B300QualificationCommissionError):
        commission._lane_policies(missing)


def _calibration_inputs(tmp_path: Path, payload: object) -> SimpleNamespace:
    path = tmp_path / "calibration-package.json"
    raw = json.dumps(payload).encode("utf-8")
    path.write_bytes(raw)
    return SimpleNamespace(
        authority_refs={
            "calibration_package": {
                "path": str(path),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        }
    )


def _calibration_context(stage: str):
    return replace(
        calibration_fixtures._context(
            {"primary": "a", "reproduction": "b"}[stage]
        ),
        logical_hardware_digest=_h(f"{stage}:hardware"),
    )


def _calibration_record(stage: str) -> dict[str, object]:
    context = _calibration_context(stage)
    threshold = replace(
        calibration_fixtures._threshold_policy(),
        context=context,
    )
    observations = calibration_fixtures._observations()
    if stage == "reproduction":
        observations = (
            replace(observations[0], seed_digest=_h("reproduction:negative")),
            *observations[1:],
        )
    manifest = derive_calibration_manifest(threshold, observations)
    evidence = CalibrationEvidenceSet.create(threshold, observations)
    return {
        "evidence": evidence.to_dict(),
        "manifest": manifest.to_dict(),
        "measurement_authority": {
            "context_digest": context.digest,
            "logical_hardware_digest": context.logical_hardware_digest,
            "projection_sha256": _h(f"{stage}:projection"),
            "raw_quality_artifact_sha256": _h(f"{stage}:raw-quality"),
            "raw_quality_binding_digest": _h(f"{stage}:raw-binding"),
            "report_digest": _h(f"{stage}:report"),
            "source_attempt_digest": _h(f"{stage}:attempt"),
            "source_attempt_ref_sha256": _h(f"{stage}:attempt-ref"),
            "transform": "validator-owned-teacher-nll-fail.v1",
        },
        "threshold_policy": threshold.to_dict(),
    }


def _calibration_package() -> dict[str, object]:
    return {
        "schema": commission.CALIBRATION_PACKAGE_SCHEMA,
        "stages": {
            stage: _calibration_record(stage)
            for stage in ("primary", "reproduction")
        },
    }


def test_sealed_calibration_rejects_reference_drift(tmp_path: Path) -> None:
    inputs = _calibration_inputs(
        tmp_path,
        {
            "schema": commission.CALIBRATION_PACKAGE_SCHEMA,
            "stages": {},
        },
    )
    inputs.authority_refs["calibration_package"]["sha256"] = _h("other-bytes")
    with pytest.raises(commission.B300QualificationCommissionError) as captured:
        commission._sealed_calibration(
            inputs, calibration_fixtures._context(), "primary"
        )
    assert "deployment reference" in str(captured.value)


def test_sealed_calibration_rejects_open_or_foreign_packages(
    tmp_path: Path,
) -> None:
    for payload in (
        ["not", "a", "package"],
        {"schema": "cacheon-private-b300-calibration-package-v0"},
        {
            "schema": commission.CALIBRATION_PACKAGE_SCHEMA,
            "stages": {},
            "operator_note": "no",
        },
        {
            "schema": commission.CALIBRATION_PACKAGE_SCHEMA,
            "stages": [],
        },
    ):
        inputs = _calibration_inputs(tmp_path, payload)
        with pytest.raises(commission.B300QualificationCommissionError):
            commission._sealed_calibration(
                inputs, calibration_fixtures._context(), "primary"
            )
        (tmp_path / "calibration-package.json").unlink()


def test_sealed_calibration_rejects_invalid_frozen_authorities(
    tmp_path: Path,
) -> None:
    package = _calibration_package()
    package["stages"]["primary"]["threshold_policy"] = {"not": "a policy"}
    inputs = _calibration_inputs(
        tmp_path,
        package,
    )
    with pytest.raises(commission.B300QualificationCommissionError) as captured:
        commission._sealed_calibration(
            inputs, calibration_fixtures._context(), "primary"
        )
    assert "package is invalid" in str(captured.value)


def test_sealed_calibration_reopens_each_exact_lane_context(
    tmp_path: Path,
) -> None:
    inputs = _calibration_inputs(tmp_path, _calibration_package())
    primary, primary_manifest, primary_evidence = commission._sealed_calibration(
        inputs, _calibration_context("primary"), "primary"
    )
    reproduction, reproduction_manifest, reproduction_evidence = (
        commission._sealed_calibration(
            inputs,
            _calibration_context("reproduction"),
            "reproduction",
        )
    )

    assert primary_evidence.observations != reproduction_evidence.observations
    assert primary.context != reproduction.context
    primary_template = primary.to_dict()
    reproduction_template = reproduction.to_dict()
    del primary_template["context"]
    del reproduction_template["context"]
    assert primary_template == reproduction_template
    assert primary_manifest == derive_calibration_manifest(
        primary, primary_evidence.observations
    )
    assert reproduction_manifest == derive_calibration_manifest(
        reproduction, reproduction_evidence.observations
    )
    assert primary_manifest.raw_evidence_digest != (
        reproduction_manifest.raw_evidence_digest
    )


def test_sealed_calibration_rejects_context_rebinding(tmp_path: Path) -> None:
    inputs = _calibration_inputs(tmp_path, _calibration_package())

    with pytest.raises(
        commission.B300QualificationCommissionError,
        match="differs from the commissioned lane",
    ):
        commission._sealed_calibration(
            inputs,
            _calibration_context("reproduction"),
            "primary",
        )


def test_sealed_calibration_rejects_recycled_lane_observations(
    tmp_path: Path,
) -> None:
    package = _calibration_package()
    package["stages"]["reproduction"] = deepcopy(
        package["stages"]["primary"]
    )
    inputs = _calibration_inputs(tmp_path, package)

    with pytest.raises(
        commission.B300QualificationCommissionError,
        match="not independent",
    ):
        commission._sealed_calibration(
            inputs, calibration_fixtures._context(), "primary"
        )


def test_compose_requires_a_sealed_commission_block() -> None:
    inputs = SimpleNamespace(qualification_commission=None)
    with pytest.raises(commission.B300QualificationCommissionError) as captured:
        commission.compose_commissioned_qualifications(
            inputs, object(), object(), _capabilities()
        )
    assert "declares no qualification commission" in str(captured.value)

    with pytest.raises(commission.B300QualificationCommissionError):
        commission.compose_commissioned_qualifications(
            inputs, object(), object(), object()
        )


def test_compose_rejects_a_session_that_differs_from_the_declared_cell() -> None:
    workload = Workload(
        _h("corpus"), "seed-v1", (WorkloadCell("s8", 8192, 1024, 2, 2),)
    )
    inputs = SimpleNamespace(workload=workload, prompt_batches=(("p", "p"),) * 3)
    session = {"warmup_count": 1}
    speed = {"min_windows": 2}
    policy = SimpleNamespace(tokens_per_prompt=1024)
    commission._require_cell_conformance(inputs, policy, session, speed)

    with pytest.raises(
        commission.B300QualificationCommissionError, match="conform"
    ):
        commission._require_cell_conformance(
            inputs, SimpleNamespace(tokens_per_prompt=256), session, speed
        )
    with pytest.raises(
        commission.B300QualificationCommissionError, match="conform"
    ):
        commission._require_cell_conformance(
            inputs, policy, {"warmup_count": 2}, speed
        )
    # A floor above the cell's timed reads can never be satisfied by any run;
    # it must die at commissioning (the 2026-08-21 min_windows=12 vs 6 failure).
    with pytest.raises(
        commission.B300QualificationCommissionError, match="conform"
    ):
        commission._require_cell_conformance(
            inputs, policy, session, {"min_windows": 3}
        )

    mixed = Workload(
        _h("mixed"),
        "seed-v1",
        (
            WorkloadCell("s8", 8192, 1024, 2, 2),
            WorkloadCell("l65", 65536, 4096, 1, 3),
        ),
    )
    mixed_inputs = SimpleNamespace(
        workload=mixed,
        prompt_batches=(("a", "b"), ("c", "d"), ("e",), ("f",), ("g",), ("h",)),
        prompt_batch_cells=("s8", "s8", "s8", "l65", "l65", "l65"),
    )
    mixed_policy = SimpleNamespace(tokens_per_prompt=4096)
    with pytest.raises(commission.B300QualificationCommissionError, match="conform"):
        commission._require_cell_conformance(
            mixed_inputs, mixed_policy, {"warmup_count": 1}, {"min_windows": 5}
        )
    # The producer seals the extra prompt AND its answer before composition;
    # inserting it only in the runtime plan would shift hidden-judge identities.
    warm = SimpleNamespace(
        workload=mixed,
        prompt_batches=(mixed_inputs.prompt_batches[0], mixed_inputs.prompt_batches[3],
                        *mixed_inputs.prompt_batches[1:]),
        prompt_batch_cells=("s8", "l65", *mixed_inputs.prompt_batch_cells[1:]),
    )
    commission._require_cell_conformance(
        warm, mixed_policy, {"warmup_count": 2}, {"min_windows": 5}
    )
    assert warm.prompt_batches[2:] == mixed_inputs.prompt_batches[1:]
    assert warm.prompt_batch_cells[2:].count("s8") == 2
    assert warm.prompt_batch_cells[2:].count("l65") == 3
    with pytest.raises(commission.B300QualificationCommissionError, match="conform"):
        commission._require_cell_conformance(
            warm, mixed_policy, {"warmup_count": 3}, {"min_windows": 5}
        )


def test_commissioned_authority_materializes_the_declared_incumbent(
    tmp_path: Path,
) -> None:
    # d00e64fa regression: a real (non-empty) incumbent always materializes
    # manifest.toml, which the genesis-only reject condition treated as
    # "differs from the commissioned incumbent stack".
    import tests.test_engine_tree as engine_tree_fixtures
    from cacheon.eval import b300_screen_deployment as screen_deployment

    source = engine_tree_fixtures._copy(tmp_path)
    catalog, _, ref, _ = engine_tree_fixtures._arranged(source)
    snapshot = catalog.snapshot()
    inputs = SimpleNamespace(
        root=tmp_path / "deployment",
        runtime=SimpleNamespace(
            runtime_digest=_h("runtime"), base_engine_digest=_h("base")
        ),
    )
    manifest = SimpleNamespace(digest=_h("arena"))

    members, _, stock, stock_tree = screen_deployment._commissioned_stock_authority(
        inputs,
        manifest,
        catalog,
        snapshot,
        error=commission.B300QualificationCommissionError,
        label="pristine reference",
    )
    assert members
    assert stock.entries == {}
    assert stock_tree.runtime_manifest is None

    _, _, incumbent, incumbent_tree = screen_deployment._commissioned_stock_authority(
        inputs,
        manifest,
        catalog,
        snapshot,
        error=commission.B300QualificationCommissionError,
        label="qualification",
        entries={ref.target_id: ref},
        resolver={("proposal", ref.artifact_digest): source},
    )
    assert incumbent.entries == {ref.target_id: ref}
    assert incumbent_tree.runtime_manifest == "manifest.toml"
    assert incumbent_tree.stack_digest == incumbent.digest
    assert incumbent.digest != stock.digest


@pytest.mark.parametrize("gpu_model,tp", (("b300", 4), ("h100", 1)))
def test_full_commission_composes_both_physical_roles_without_a_gpu(tmp_path, monkeypatch, gpu_model, tp):
    from cacheon.arena_service import ArenaService
    from cacheon.chain.evaluation_coordinator import WorkerReadiness
    from cacheon.eval import b300_screen_deployment as screen
    from cacheon.eval.b300_arena_provider import B300ArenaServiceProvider
    from cacheon.eval.b300_sealed_qualification_commission import predicted_qualification_builder_digest
    from cacheon.eval.reference_quality import retained_support_policy_digest
    from tests import test_b300_screen_deployment as fixtures
    from tests.test_b300_sealed_qualification_commission import _block
    from tests.support.b300 import GLM53_REGISTERED_TARGET_IDS

    paths, gpus, ready = fixtures._case(tmp_path, gpu_model=gpu_model, host_size=2 * tp, lane=tuple(range(tp)))
    for name in ("prompt_authority", "authority_config", "measurement_config"):
        paths[name].chmod(0o600)
    prompt = json.loads(paths["prompt_authority"].read_text())
    prompt["workload_cell"]["timed_reads"] = 3
    prompt["prompt_batches"].append(["four"])
    if tp == 4:
        prompt.update(model_profile_key="GLM-5.3-NVFP4", registered_targets=list(GLM53_REGISTERED_TARGET_IDS))
        prompt["engine_config"]["engine_kwargs"].update(dp_size=4, enable_dp_attention=True)
    prompt_sha = fixtures._write(paths["prompt_authority"], prompt)
    block = _block()
    block["support_policy_digest"] = retained_support_policy_digest()
    block["policy"].update(tokens_per_prompt=1024, topk_width=0)
    block["session"]["conditioning_count"] = 1
    authority = json.loads(paths["authority_config"].read_text())
    authority.update(qualification=block, resources={"runtime": {"cpu_millis": 32000}, "prebuild": {"cpu_millis": 16000}})
    authority["prompt"]["sha256"] = prompt_sha
    authority["qualification_builder_digest"] = predicted_qualification_builder_digest(
        default_target_catalog(), registered_target_ids=tuple(prompt["registered_targets"]),
        builder_source_digest=block["builder_source_digest"], selection_store_digest=block["selection_store_digest"])
    for name in ("authority_config", "measurement_config"):
        fixtures._write(paths[name], authority)
    inputs = screen._authority_inputs(**paths, provisioner=None, provisioned_gpus=gpus)
    composition = screen._compose(inputs)
    judge = _Judge()
    judge.binding = HiddenJudgeBinding(*(inputs.prompt_identity[key] for key in
        ("hidden_corpus_commitment", "hidden_judge_digest", "hidden_task_policy_digest")))
    contexts, native_arches = [], []
    native_build = screen._native_build

    def tracked_native(*args):
        built = native_build(*args)
        native_arches.append(built.target_architecture)
        return built

    def calibration(_inputs, context, stage):
        contexts.append((stage, context))
        threshold = replace(calibration_fixtures._threshold_policy(), context=context)
        threshold = replace(threshold, speed=replace(threshold.speed, max_noise="0.02"))
        evidence = CalibrationEvidenceSet.create(threshold, calibration_fixtures._observations())
        return threshold, derive_calibration_manifest(threshold, evidence.observations), evidence

    monkeypatch.setattr(screen, "_native_build", tracked_native)
    executors = ()
    try:
        service = ArenaService(composition.manifest, B300ArenaServiceProvider(composition.manifest, composition.authorities))
        readiness = WorkerReadiness.for_service(service, ready_receipt_digest=ready["receipt_digest"], ready_epoch=1)
        commissions, executors = commission.compose_commissioned_qualifications(
            inputs, composition, readiness, _capabilities(hidden_judge=judge), calibration_loader=calibration)
        assert len(commissions) == 2 and {stage for stage, _ in contexts} == {"primary", "reproduction"}
        assert len({context.logical_hardware_digest for _, context in contexts}) == 2
        assert native_arches and set(native_arches) == {"sm103" if tp == 4 else "sm90"}
        for index, executor in enumerate(executors):
            assert tuple(g.physical_id for g in executor.device_policy.expected_gpus) == tuple(range(index * tp, (index + 1) * tp))
            assert executor.config.runtime.cpu_millis == 32000
            assert executor.config.prebuild.policy.cpu_millis == 16000
        assert all(c.construction.registered_target_ids == tuple(prompt["registered_targets"]) for c in commissions)
    finally:
        for executor in executors:
            executor.manager.close()
        composition.close()
