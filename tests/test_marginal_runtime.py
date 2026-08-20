from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, fields, replace
from pathlib import Path

import pytest

from cacheon.bundle_hash import content_hash
from cacheon.engine_tree import (
    inspect_contribution,
    materialize_engine_tree,
    reopen_materialized_engine_tree,
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
from cacheon.eval.marginal_runtime import (
    MarginalRuntimeError,
    MaterializedArmBinding,
    prepare_marginal_runtime,
)
from cacheon.eval.native_compile_profile import NativeCuTeCompileProfile
from cacheon.eval.oci_backend import (
    TrustedArenaModelMountReceipt,
    expected_runtime_preflight,
    runtime_identity_from_preflight,
)
from cacheon.eval.oci_outer_session import (
    BatchExecutionEvidence,
    SessionExecutionPlan,
)
from cacheon.eval.oci_session_protocol import (
    BatchEvidence,
    EngineSessionConfig,
    PromptEvidence,
)
from cacheon.eval.runtime_preflight import RuntimePreflightReceipt
from cacheon.stack_manifest import (
    EvaluationStackContext,
    EvaluationStackManifest,
    ProposalContributionRef,
)
from cacheon.stack_plan import MarginalArmPlan, plan_candidate_stack, plan_marginal_arm
from cacheon.target_catalog import TargetCatalog, default_target_catalog


ROOT = Path(__file__).parents[1]
FIXTURES = Path(__file__).parent / "fixtures"
SILU = ROOT / "examples" / "miner_silu_torch"
MSA = FIXTURES / "stack_msa_singleton"
FUSED = FIXTURES / "stack_fused_epilogue_atomic"


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _binding_id(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()[:32]


def _specs(catalog: TargetCatalog) -> dict[str, str]:
    targets = catalog.snapshot()["targets"]
    assert isinstance(targets, list)
    return {
        row["target_id"]: catalog.target_spec_digest(row["target_id"])
        for row in targets
    }


def _preflight() -> RuntimePreflightReceipt:
    image = _digest("image")
    return RuntimePreflightReceipt(
        schema="cacheon-runtime-preflight-v2",
        requested_image="registry.example/worker@sha256:" + image,
        image_digest=image,
        local_image_id="sha256:" + "a" * 64,
        repo_digests=("registry.example/worker@sha256:" + image,),
        oci_platform="linux/amd64",
        platform_digest=_digest("platform"),
        docker_binary="/usr/bin/docker",
        uid=max(1, os.getuid()),
        gid=max(1, os.getgid()),
        sglang_version="0.0.0.dev1+g56e290315",
        worker_distribution="cacheon-harness",
        worker_version="0.0.1",
        worker_distribution_digest=_digest("worker"),
        worker_file_count=200,
        worker_total_bytes=1_000_000,
        python_implementation="cpython",
        python_executable="/usr/local/bin/python3",
        python_version="3.12.0",
        python_abi="cpython-312-x86_64-linux-gnu",
        python_platform="linux-x86_64",
        machine="x86_64",
        package_versions=(),
        cudart_library="libcudart.so.13",
        cuda_visible_devices="",
        nvidia_visible_devices="void",
        security_argv_sha256=_digest("preflight argv"),
    )


def _ref(source: Path, catalog: TargetCatalog, *, label: str = "") -> ProposalContributionRef:
    inspected = inspect_contribution(source, catalog=catalog)
    return ProposalContributionRef(
        target_id=inspected.target_id,
        target_spec_digest=inspected.target_spec_digest,
        artifact_digest=content_hash(source),
        selected_payload_digest=inspected.selected_payload_digest,
        attribution_digest=_digest("attribution:" + inspected.target_id + label),
    )


def _resolver(*rows: tuple[ProposalContributionRef, Path]) -> dict[tuple[str, str], Path]:
    return {("proposal", ref.artifact_digest): source for ref, source in rows}


def _stack(
    context: EvaluationStackContext,
    catalog: TargetCatalog,
    *refs: ProposalContributionRef,
) -> EvaluationStackManifest:
    return EvaluationStackManifest(
        runtime_digest=context.runtime_digest,
        base_engine_digest=context.base_engine_digest,
        arena_digest=context.arena_digest,
        catalog_snapshot=catalog.snapshot(),
        catalog_digest=catalog.digest,
        entries={ref.target_id: ref for ref in refs},
    )


@dataclass
class Case:
    catalog: TargetCatalog
    context: EvaluationStackContext
    incumbent: EvaluationStackManifest
    baseline_tree: object
    arm: MarginalArmPlan
    candidate_tree: object
    preflight: RuntimePreflightReceipt
    launch: EngineLaunchSpec
    baseline_binding: MaterializedArmBinding
    candidate_binding: MaterializedArmBinding
    session: SessionExecutionPlan
    mount: TrustedArenaModelMountReceipt


def _native(tree_digest: str, preflight: RuntimePreflightReceipt) -> NativeBuildSpec:
    dependency = _digest("dependency policy")
    return NativeBuildSpec(
        tree_digest=tree_digest,
        image_digest=preflight.image_digest,
        platform_digest=preflight.platform_digest,
        worker_distribution_digest=preflight.worker_distribution_digest,
        toolchain_digest=native_toolchain_digest(
            image_digest=preflight.image_digest,
            platform_digest=preflight.platform_digest,
        ),
        patcher_digest=native_patcher_digest(
            worker_distribution_digest=preflight.worker_distribution_digest
        ),
        compiler_flags_digest=native_compiler_policy_digest(
            image_digest=preflight.image_digest,
            worker_distribution_digest=preflight.worker_distribution_digest,
            dependency_policy_digest=dependency,
            target_architecture="sm120",
        ),
        target_architecture="sm120",
        dependency_policy_digest=dependency,
    )


def _profiled_candidate_binding(case: Case) -> MaterializedArmBinding:
    hardware = case.launch.hardware
    profile = NativeCuTeCompileProfile(
        logical_architecture=hardware.architecture,
        compiler_architecture="sm_120a",
        image_digest=case.preflight.image_digest,
        platform_digest=case.preflight.platform_digest,
        worker_distribution_digest=case.preflight.worker_distribution_digest,
        logical_hardware_digest=hardware.digest,
        device_policy_digest=hardware.device_policy_digest,
        topology_digest=hardware.topology_digest,
        visible_gpu_count=hardware.visible_gpu_count,
        tp_size=hardware.tp_size,
        ep_size=hardware.ep_size,
        dp_size=hardware.dp_size,
        constants={"max_active_clusters.cluster_size_1": 1},
        measurement_digest=_digest("trusted compile-profile measurement"),
    )
    legacy = case.candidate_binding.launch_binding.native_build_spec
    native = NativeBuildSpec(
        tree_digest=case.candidate_tree.tree_digest,
        image_digest=legacy.image_digest,
        platform_digest=legacy.platform_digest,
        worker_distribution_digest=legacy.worker_distribution_digest,
        toolchain_digest=legacy.toolchain_digest,
        patcher_digest=legacy.patcher_digest,
        compiler_flags_digest=native_compiler_policy_digest(
            image_digest=legacy.image_digest,
            worker_distribution_digest=legacy.worker_distribution_digest,
            dependency_policy_digest=legacy.dependency_policy_digest,
            target_architecture=legacy.target_architecture,
            compile_profile_digest=profile.digest,
        ),
        target_architecture=legacy.target_architecture,
        dependency_policy_digest=legacy.dependency_policy_digest,
        compile_profile_digest=profile.digest,
    )
    trusted = replace(
        case.candidate_binding.launch_binding,
        native_build_spec=native,
        native_compile_profile=profile,
    )
    return MaterializedArmBinding(case.candidate_tree, trusted)


def _local_binding(
    tree: object,
    native: NativeBuildSpec,
    launch: EngineLaunchSpec,
    preflight: RuntimePreflightReceipt,
    *,
    physical_id: str = "0",
) -> MaterializedArmBinding:
    physical = PhysicalHardwareBinding(
        physical_gpu_ids=(physical_id,),
        architecture="sm120",
        topology_class="single_gpu",
        topology_digest=_digest("topology"),
        tp_size=1,
        ep_size=1,
        dp_size=1,
        device_policy_digest=_digest("device policy"),
    )
    trusted = TrustedLaunchBinding(
        materialized_tree_root=tree.root,
        controller_distribution_digest=launch.controller_distribution_digest,
        native_build_spec=native,
        runtime_preflight_receipt=preflight,
        physical_hardware=physical,
    )
    return MaterializedArmBinding(tree, trusted)


def _case(tmp_path: Path, source: Path = SILU, *, suffix: str = "") -> Case:
    catalog = default_target_catalog()
    preflight = _preflight()
    runtime = runtime_identity_from_preflight(preflight)
    context = EvaluationStackContext(
        runtime_digest=runtime.runtime_digest,
        base_engine_digest=runtime.base_engine_digest,
        arena_digest=_digest("arena"),
        catalog_snapshot=catalog.snapshot(),
        catalog_digest=catalog.digest,
        target_spec_digests=_specs(catalog),
    )
    incumbent = _stack(context, catalog)
    ref = _ref(source, catalog, label=suffix)
    candidate = plan_candidate_stack(
        incumbent, ref, catalog=catalog, expected_context=context
    )
    baseline_tree = materialize_engine_tree(
        incumbent,
        context=context,
        catalog=catalog,
        resolver={},
        destination=tmp_path / ("baseline" + suffix),
    )
    candidate_tree = materialize_engine_tree(
        candidate,
        context=context,
        catalog=catalog,
        resolver=_resolver((ref, source)),
        destination=tmp_path / ("candidate" + suffix),
    )
    arm = plan_marginal_arm(
        incumbent,
        ref,
        catalog=catalog,
        incumbent_tree_digest=baseline_tree.tree_digest,
        candidate_tree_digest=candidate_tree.tree_digest,
        expected_context=context,
    )
    native = _native(baseline_tree.tree_digest, preflight)
    config = EngineSessionConfig(
        model_path="/cacheon/input/model",
        dtype="bfloat16",
        deterministic=False,
        attention_backend="flashinfer",
        disable_cuda_graph=False,
        mem_fraction_static=0.75,
        log_level="error",
        max_running_requests=8,
        tp_size=1,
        moe_runner_backend=None,
        disable_custom_all_reduce=True,
        engine_kwargs={},
    )
    hardware = LogicalHardwareSpec(
        visible_gpu_count=1,
        architecture="sm120",
        topology_class="single_gpu",
        topology_digest=_digest("topology"),
        tp_size=1,
        ep_size=1,
        dp_size=1,
        device_policy_digest=_digest("device policy"),
    )
    launch = EngineLaunchSpec(
        runtime_digest=context.runtime_digest,
        base_engine_digest=context.base_engine_digest,
        arena_digest=context.arena_digest,
        stack_digest=incumbent.digest,
        tree_digest=baseline_tree.tree_digest,
        image_digest=preflight.image_digest,
        platform_digest=preflight.platform_digest,
        controller_distribution_digest=_digest("controller"),
        worker_distribution_digest=preflight.worker_distribution_digest,
        model_revision_digest=_digest("model revision"),
        model_manifest_digest=_digest("model manifest"),
        model_content_digest=_digest("model content"),
        validator_overlay_digest=runtime.validator_overlay_digest,
        engine_config_digest=config.digest,
        seccomp_policy_digest=_digest("seccomp"),
        resource_policy_digest=_digest("resources"),
        native_build_spec_digest=native.digest,
        hardware=hardware,
    )
    baseline_binding = _local_binding(baseline_tree, native, launch, preflight)
    candidate_native = _native(candidate_tree.tree_digest, preflight)
    candidate_binding = _local_binding(
        candidate_tree, candidate_native, launch, preflight
    )
    session = SessionExecutionPlan(
        launch_digest=launch.digest,
        expected_engine_config_digest=config.digest,
        engine_config=config,
        expected_preflight=expected_runtime_preflight(launch, preflight),
        prompt_batches=(("warmup",), ("timed one", "timed two")),
        warmup_count=1,
        conditioning_count=1,
        max_new_tokens=2,
        top_logprobs_num=2,
        temperature=0.0,
    )
    model = tmp_path / ("model" + suffix)
    model.mkdir(mode=0o700)
    mount = TrustedArenaModelMountReceipt.capture(
        model,
        arena_digest=launch.arena_digest,
        model_revision_digest=launch.model_revision_digest,
        model_manifest_digest=launch.model_manifest_digest,
        model_content_digest=launch.model_content_digest,
    )
    return Case(
        catalog,
        context,
        incumbent,
        baseline_tree,
        arm,
        candidate_tree,
        preflight,
        launch,
        baseline_binding,
        candidate_binding,
        session,
        mount,
    )


def _prepared(case: Case):
    return prepare_marginal_runtime(
        case.arm,
        catalog=case.catalog,
        expected_context=case.context,
        incumbent_launch=case.launch,
        incumbent_binding=case.baseline_binding,
        candidate_binding=case.candidate_binding,
        baseline_session_plan=case.session,
    )


def _batch_evidence(
    plan: SessionExecutionPlan, index: int, *, label: str
) -> BatchExecutionEvidence:
    token_count = len(plan.prompt_batches[index]) * plan.max_new_tokens
    prompt = PromptEvidence(
        output_ids=tuple(range(plan.max_new_tokens)),
        top_logprobs=tuple(
            tuple((float(-rank), rank) for rank in range(plan.top_logprobs_num))
            for _ in range(plan.max_new_tokens)
        ),
    )
    evidence = BatchEvidence(tuple(prompt for _ in plan.prompt_batches[index]))
    return BatchExecutionEvidence(
        index,
        _binding_id(f"request:{label}:{index}"),
        _binding_id(f"nonce:{label}:{index}"),
        float(index + 1),
        float(index + 2),
        token_count,
        evidence,
    )


def test_singleton_derives_exact_launch_and_session_plan(tmp_path: Path) -> None:
    case = _case(tmp_path)
    prepared = _prepared(case)
    candidate = prepared.candidates[0]

    launch_diff = {
        key
        for key, value in case.launch.to_dict().items()
        if candidate.launch.to_dict()[key] != value
    }
    assert launch_diff == {
        "stack_digest",
        "tree_digest",
        "native_build_spec_digest",
    }
    session_diff = {
        field.name
        for field in fields(case.session)
        if getattr(case.session, field.name) != getattr(candidate.session_plan, field.name)
    }
    assert session_diff == {"launch_digest", "expected_preflight"}
    assert prepared.source is case.arm
    assert candidate.arm is case.arm
    encoded = candidate.launch.canonical_bytes.decode()
    assert all(word not in encoded for word in ("baseline", "challenger", "target_id"))


@pytest.mark.parametrize("fixture", (MSA, FUSED), ids=("msa-singleton", "fused-atomic"))
def test_singleton_and_atomic_fixtures_bind_without_runtime_import(
    tmp_path: Path, fixture: Path
) -> None:
    case = _case(tmp_path, fixture)
    prepared = _prepared(case)
    assert prepared.candidates[0].arm.transition.target_id == _ref(
        fixture, case.catalog
    ).target_id
    assert prepared.candidates[0].launch.tree_digest == case.candidate_tree.tree_digest


def test_prepare_rejects_swaps_environment_context_and_workload(tmp_path: Path) -> None:
    case = _case(tmp_path)
    other = _case(tmp_path / "other", MSA, suffix="-other")
    with pytest.raises(MarginalRuntimeError):
        prepare_marginal_runtime(
            case.arm,
            catalog=case.catalog,
            expected_context=case.context,
            incumbent_launch=case.launch,
            incumbent_binding=case.baseline_binding,
            candidate_binding=other.candidate_binding,
            baseline_session_plan=case.session,
        )

    wrong_native = replace(
        case.candidate_binding.launch_binding.native_build_spec,
        dependency_policy_digest=_digest("other dependency"),
        compiler_flags_digest=native_compiler_policy_digest(
            image_digest=case.preflight.image_digest,
            worker_distribution_digest=case.preflight.worker_distribution_digest,
            dependency_policy_digest=_digest("other dependency"),
            target_architecture="sm120",
        ),
    )
    with pytest.raises(MarginalRuntimeError, match="native-build environment"):
        prepare_marginal_runtime(
            case.arm,
            catalog=case.catalog,
            expected_context=case.context,
            incumbent_launch=case.launch,
            incumbent_binding=case.baseline_binding,
            candidate_binding=replace(
                case.candidate_binding,
                launch_binding=replace(
                    case.candidate_binding.launch_binding,
                    native_build_spec=wrong_native,
                ),
            ),
            baseline_session_plan=case.session,
        )
    with pytest.raises(MarginalRuntimeError, match="runtime preflight, or native-build"):
        prepare_marginal_runtime(
            case.arm,
            catalog=case.catalog,
            expected_context=case.context,
            incumbent_launch=case.launch,
            incumbent_binding=case.baseline_binding,
            candidate_binding=replace(
                case.candidate_binding,
                launch_binding=replace(
                    case.candidate_binding.launch_binding,
                    physical_hardware=replace(
                        case.candidate_binding.launch_binding.physical_hardware,
                        physical_gpu_ids=("1",),
                    ),
                ),
            ),
            baseline_session_plan=case.session,
        )
    with pytest.raises(MarginalRuntimeError, match="frozen stack context"):
        _prepared(replace(case, launch=replace(case.launch, arena_digest=_digest("wrong"))))
    with pytest.raises(MarginalRuntimeError, match="baseline session"):
        _prepared(
            replace(
                case,
                session=replace(
                    case.session,
                    expected_preflight=replace(
                        case.session.expected_preflight,
                        stack_digest=_digest("wrong session stack"),
                    ),
                ),
            )
        )


def test_prepare_allows_profiled_aot_candidate_on_legacy_baseline(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path)
    profiled = _profiled_candidate_binding(case)

    prepared = prepare_marginal_runtime(
        case.arm,
        catalog=case.catalog,
        expected_context=case.context,
        incumbent_launch=case.launch,
        incumbent_binding=case.baseline_binding,
        candidate_binding=profiled,
        baseline_session_plan=case.session,
    )

    candidate = prepared.candidates[0]
    assert candidate.binding == profiled
    assert candidate.launch.native_build_spec_digest == (
        profiled.launch_binding.native_build_spec.digest
    )
    assert profiled.launch_binding.native_compile_profile is not None


def test_forged_empty_tree_claiming_candidate_stack_rejects_before_executor(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path)
    forged_root = tmp_path / "forged"
    shutil.copytree(case.baseline_tree.root, forged_root)
    metadata_path = forged_root / "metadata" / "cacheon_engine_tree.json"
    metadata_path.chmod(0o644)
    metadata = json.loads(metadata_path.read_text())
    metadata["stack_digest"] = case.arm.candidate.digest
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    metadata_path.chmod(0o444)
    forged = reopen_materialized_engine_tree(forged_root)
    forged_arm = plan_marginal_arm(
        case.incumbent,
        case.arm.transition.replacement,
        catalog=case.catalog,
        incumbent_tree_digest=case.baseline_tree.tree_digest,
        candidate_tree_digest=forged.tree_digest,
        expected_context=case.context,
    )
    forged_native = _native(forged.tree_digest, case.preflight)
    forged_binding = _local_binding(
        forged, forged_native, case.launch, case.preflight
    )
    with pytest.raises(MarginalRuntimeError, match="contribution inventory"):
        prepare_marginal_runtime(
            forged_arm,
            catalog=case.catalog,
            expected_context=case.context,
            incumbent_launch=case.launch,
            incumbent_binding=case.baseline_binding,
            candidate_binding=forged_binding,
            baseline_session_plan=case.session,
        )


def test_prepare_never_imports_candidate_top_level(tmp_path: Path) -> None:
    marker = tmp_path / "imported"
    source = tmp_path / "raising-source"
    (source / "kernels").mkdir(parents=True)
    (source / "metadata").mkdir()
    (source / "manifest.toml").write_text(
        'bundle_id="raising"\nabi_version="cacheon-op-abi-v0"\n'
        '[[ops]]\nslot="activation.silu_and_mul"\nsource="kernels/raising.py"\n'
        'entry="silu_and_mul"\ndtypes=["bfloat16"]\nmetadata="metadata/op.json"\n'
    )
    (source / "metadata/op.json").write_text(
        '{"op":"activation.silu_and_mul","dtypes":["bfloat16"]}\n'
    )
    (source / "kernels/raising.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('bad')\n"
        "raise RuntimeError('must not import')\n"
        "def silu_and_mul(x, out):\n    raise RuntimeError('unreachable')\n"
    )
    case = _case(tmp_path / "case", source)
    _prepared(case)
    assert not marker.exists()


def test_bridge_import_and_records_exclude_grading_authority() -> None:
    script = f"""
import sys
sys.path.insert(0, {str(ROOT)!r})
import cacheon.eval.marginal_runtime as module
for name in ('torch', 'sglang', 'bittensor'):
    assert name not in sys.modules
print(module.__name__)
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-c", script],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    import cacheon.eval.marginal_runtime as module

    names = {
        field.name
        for record in (
            module.PreparedMarginalRuntime,
            module.PreparedCandidateRuntime,
            module.MaterializedArmBinding,
        )
        for field in fields(record)
    }
    assert not names & {
        "score",
        "quality",
        "verdict",
        "retry",
        "crown",
        "winner",
        "hotkey",
        "weight",
    }


def test_sglang_plugin_resolves_only_materialized_namespace_after_spawn(
    tmp_path: Path,
) -> None:
    trusted = tmp_path / "trusted"
    tree = tmp_path / "engine-tree"
    module = tree / ("cacheon_c_" + "a" * 64) / "kernels"
    module.mkdir(parents=True)
    trusted.mkdir()
    target = trusted / "sglang/srt/layers"
    target.mkdir(parents=True)
    for package in (trusted / "sglang", trusted / "sglang/srt", target):
        (package / "__init__.py").write_text("")
    (target / "activation.py").write_text("""class SiluAndMul:
    def forward_cuda(self, *args): pass
    def forward_native(self, *args): pass
""")
    (tree / "torch.py").write_text("origin = 'candidate'\n")
    (module / "kernel.py").write_text("loaded = True\n")
    for path in sorted(tree.rglob("*"), reverse=True):
        path.chmod(0o755 if path.is_dir() else 0o444)
    tree.chmod(0o755)
    namespace = module.parent.name
    script = tmp_path / "spawn_probe.py"
    script.write_text(f"""import importlib, multiprocessing as mp, os, sys
sys.path[:0] = [{str(ROOT)!r}, {str(trusted)!r}]
TREE, NAMESPACE = {str(tree)!r}, {namespace!r}
def child(send, bundle):
    import cacheon.integrations.sglang_plugin as plugin
    from cacheon import seam, seams
    seam._ENGINE_TREE = TREE
    seams.SEAM_ADAPTERS = tuple(a for a in seams.SEAM_ADAPTERS if a.integration == 'sglang_silu')
    os.environ.update(CACHEON_ENGINE_WORKER='1', CACHEON_BUNDLE_PATH=bundle,
        CACHEON_ENGINE_TREE_DIGEST='1' * 64, CACHEON_STACK_DIGEST='2' * 64,
        CACHEON_ACTIVE='0')
    plugin.register()
    importlib.import_module('sglang.srt.layers.activation')
    from cacheon.integrations import sglang_silu
    found = importlib.util.find_spec(NAMESPACE)
    if bundle == TREE:
        import torch
        kernel = importlib.import_module(NAMESPACE + '.kernels.kernel')
        shadowed = os.path.realpath(torch.__file__) == os.path.join(TREE, 'torch.py')
        send.send((TREE in sys.path, shadowed, kernel.loaded, found is not None,
            sglang_silu.is_installed()))
    else:
        send.send((found is None, sglang_silu.is_installed()))
if __name__ == '__main__':
    results = []
    for bundle in (TREE, '/raw/miner/bundle'):
        receive, send = mp.get_context('spawn').Pipe(False)
        process = mp.get_context('spawn').Process(target=child, args=(send, bundle))
        process.start(); process.join(10)
        assert process.exitcode == 0
        results.append(receive.recv())
    assert results == [(False, False, True, True, True), (True, True)]
""")
    completed = subprocess.run(
        [sys.executable, "-I", str(script)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
