"""Commissioning surface for qualification in the one existing pod service.

Full composition needs the commissioned B300 host (sealed deployment root,
eight-GPU lane pair, OCI runtime); these tests pin the CPU-checkable gates:
capability typing and identity binding, the tracked deadline policy, lane
disjointness, and sealed calibration package handling.
"""

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
        _capabilities(graph_facts_builder_digest="not-a-digest")
    with pytest.raises(commission.B300QualificationCommissionError):
        _capabilities(source_resolver_digest=_h("upper").upper())
    with pytest.raises(commission.B300QualificationCommissionError):
        _capabilities(incumbent_entries=[("moe.fused_experts_reduce", object())])
    with pytest.raises(commission.B300QualificationCommissionError):
        _capabilities(incumbent_entries={"moe.fused_experts_reduce": object()})


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


def test_pristine_reference_authority_removes_seam_selection(
    tmp_path: Path,
) -> None:
    case = oci_backend_fixtures._case(tmp_path)
    incumbent_config = replace(
        case.plan.engine_config,
        seam_bindings=("collective",),
    )
    incumbent_launch = replace(
        case.launch,
        engine_config_digest=incumbent_config.digest,
    )
    incumbent_plan = replace(
        case.plan,
        launch_digest=incumbent_launch.digest,
        expected_engine_config_digest=incumbent_config.digest,
        engine_config=incumbent_config,
        expected_preflight=commission.expected_runtime_preflight(
            incumbent_launch, case.preflight
        ),
    )

    # Genesis: the declared incumbent is the empty stock stack, so the
    # pristine tree/native identities coincide with the incumbent's.
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

    assert incumbent_plan.engine_config.seam_bindings == ("collective",)
    assert pristine_plan.engine_config.seam_bindings == ()
    assert pristine_launch.digest != incumbent_launch.digest
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
    commission._require_cell_conformance(
        mixed_inputs, SimpleNamespace(tokens_per_prompt=4096),
        {"warmup_count": 1}, {"min_windows": 5}
    )
    with pytest.raises(commission.B300QualificationCommissionError, match="conform"):
        commission._require_cell_conformance(
            mixed_inputs, policy, {"warmup_count": 1}, {"min_windows": 5}
        )


def test_qualification_swap_root_is_runtime_traversable(tmp_path: Path) -> None:
    root = commission._swap_intake_root(tmp_path / "resident-intake" / "A")
    assert root.stat().st_mode & 0o777 == 0o711

    root.chmod(0o700)
    assert commission._swap_intake_root(root) == root
    assert root.stat().st_mode & 0o777 == 0o711


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


def test_sealed_incumbent_bundle_is_derived_staged_and_bounded(
    tmp_path: Path,
) -> None:
    """The v7 baseline injection identity is sealed from the stack entry.

    Digest and slot set come from the resolver-verified source and the
    registered target's members — never from a runtime swap acknowledgement —
    and the bundle bytes are staged content-addressed into the swap intake.
    Anything one swap cannot realize returns None (the two-process route),
    not an error.
    """

    import tests.test_engine_tree as engine_tree_fixtures
    from cacheon.bundle_hash import content_hash

    source = engine_tree_fixtures._copy(tmp_path)
    catalog, _, ref, _ = engine_tree_fixtures._arranged(source)
    intake = commission._swap_intake_root(tmp_path / "resident-intake" / "A")
    resolver = SimpleNamespace(resolve_proposal=lambda digest: source)

    genesis = SimpleNamespace(incumbent_entries={}, source_resolver=resolver)
    assert commission._sealed_incumbent_bundle(genesis, catalog, intake) is None

    capabilities = SimpleNamespace(
        incumbent_entries={ref.target_id: ref}, source_resolver=resolver
    )
    sealed = commission._sealed_incumbent_bundle(capabilities, catalog, intake)
    assert sealed is not None
    assert sealed.target_id == ref.target_id
    assert sealed.bundle_digest == ref.artifact_digest
    assert sealed.slots == tuple(sorted(catalog.require(ref.target_id).members))
    staged = intake / sealed.bundle_digest
    assert staged.is_dir()
    assert content_hash(staged) == sealed.bundle_digest

    multiple = SimpleNamespace(
        incumbent_entries={ref.target_id: ref, "second.target": ref},
        source_resolver=resolver,
    )
    assert commission._sealed_incumbent_bundle(multiple, catalog, intake) is None


def test_sealed_incumbent_bundle_refuses_unswappable_and_foreign_slots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tests.test_engine_tree as engine_tree_fixtures
    from cacheon.eval import resident_screen_lane

    source = engine_tree_fixtures._copy(tmp_path)
    catalog, _, ref, _ = engine_tree_fixtures._arranged(source)
    intake = commission._swap_intake_root(tmp_path / "resident-intake" / "A")
    capabilities = SimpleNamespace(
        incumbent_entries={ref.target_id: ref},
        source_resolver=SimpleNamespace(resolve_proposal=lambda digest: source),
    )

    monkeypatch.setattr(
        resident_screen_lane,
        "screen_swappability",
        lambda manifest: "native-rebuild bundles are not swappable",
    )
    assert (
        commission._sealed_incumbent_bundle(capabilities, catalog, intake) is None
    )

    import cacheon.manifest as manifest_module

    monkeypatch.setattr(resident_screen_lane, "screen_swappability", lambda m: None)
    monkeypatch.setattr(
        manifest_module,
        "load_manifest",
        lambda path: SimpleNamespace(ops=(SimpleNamespace(slot="foreign.slot"),)),
    )
    with pytest.raises(
        commission.B300QualificationCommissionError,
        match="differ from its registered target members",
    ):
        commission._sealed_incumbent_bundle(capabilities, catalog, intake)
