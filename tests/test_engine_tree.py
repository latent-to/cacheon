from __future__ import annotations

import hashlib
import importlib
import json
import os
import shutil
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType

import pytest

import cacheon.engine_tree as engine_tree
from cacheon.bundle_hash import content_hash
from cacheon.chain.publication import publish_worker_bundle
from cacheon.eval.oci_session_protocol import SlotAuditPolicy
from cacheon.engine_tree import (
    EngineTreeError,
    inspect_contribution,
    source_tree_digest,
    materialize_engine_tree,
    reopen_materialized_engine_tree,
)
from cacheon.manifest import load_manifest
from cacheon.sandbox import load_module
from cacheon.stack_manifest import (
    EvaluationStackContext,
    EvaluationStackManifest,
    ProposalContributionRef,
)
from cacheon.stack_plan import RollbackPlan, plan_candidate_stack, plan_marginal_arm
from cacheon.target_catalog import TargetCatalog, default_target_catalog


ROOT = Path(__file__).parents[1]
FIXTURES = Path(__file__).parent / "fixtures"
SINGLETON = FIXTURES / "stack_norm_singleton"
FUSED = ROOT / "examples" / "miner_dp_attention_exchange_torch"


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _audit_policy(label: str, slots: tuple[str, ...]) -> SlotAuditPolicy:
    return SlotAuditPolicy(_digest(f"audit-seed:{label}")[:32], 100_000, 32, slots, 1)


def _spec_digests(catalog: TargetCatalog) -> dict[str, str]:
    targets = catalog.snapshot()["targets"]
    assert isinstance(targets, list)
    return {
        row["target_id"]: catalog.target_spec_digest(row["target_id"])
        for row in targets
    }


def _evaluation_context(catalog: TargetCatalog) -> EvaluationStackContext:
    return EvaluationStackContext(
        runtime_digest=_digest("runtime"),
        base_engine_digest=_digest("base"),
        arena_digest=_digest("arena"),
        catalog_snapshot=catalog.snapshot(),
        catalog_digest=catalog.digest,
        target_spec_digests=_spec_digests(catalog),
    )


def _proposal_ref(source: Path, catalog: TargetCatalog) -> ProposalContributionRef:
    inspected = inspect_contribution(source, catalog=catalog)
    return ProposalContributionRef(
        target_id=inspected.target_id,
        target_spec_digest=inspected.target_spec_digest,
        artifact_digest=content_hash(source),
        selected_payload_digest=inspected.selected_payload_digest,
        attribution_digest=_digest(f"attribution:{inspected.target_id}"),
    )


def _sources(*rows: tuple[ProposalContributionRef, Path]) -> dict[tuple[str, str], Path]:
    return {("proposal", ref.artifact_digest): source for ref, source in rows}


def _evaluation_stack(
    catalog: TargetCatalog,
    context: EvaluationStackContext,
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


def _write_moe_fixture(root: Path, target: str, entry: str) -> Path:
    (root / "kernels").mkdir(parents=True)
    (root / "kernels" / "fused_epilogue.py").write_text(
        "from kernels.helper import marker\n"
        "import fused_epilogue_sm103 as native\n\n"
        "def prepare(*args):\n"
        "    return marker\n\n"
        f"def {entry}(*args):\n"
        "    return native, marker\n"
    )
    (root / "kernels" / "helper.py").write_text(f"marker = {target!r}\n")
    (root / "kernels" / "fused_epilogue_sm103.cu").write_text(
        "#include <torch/extension.h>\n"
        "PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}\n"
    )
    (root / "rebuild.json").write_text(
        '{"steps":[{"type":"repo_python","path":"build_cuda_ext.py"}]}\n'
    )
    (root / "manifest.toml").write_text(
        f'bundle_id = "fixture-{entry}"\n'
        'abi_version = "cacheon-op-abi-v0"\n\n'
        "[competition]\n"
        f'target = "{target}"\n'
        'mode = "slot"\n\n'
        "[[ops]]\n"
        f'slot = "{target}"\n'
        'source = "kernels/fused_epilogue.py"\n'
        f'entry = "{entry}"\n'
        'prepare = "prepare"\n'
        'dtypes = ["bfloat16"]\n'
        'architectures = ["sm103"]\n'
        'cuda_sources = ["kernels/fused_epilogue_sm103.cu"]\n'
    )
    return root


def _native_fixture(tmp_path: Path) -> Path:
    return _write_moe_fixture(
        tmp_path / "source", "moe.fused_experts", "fused_experts"
    )


def _materialize(stack, context, catalog, resolver, destination, **kwargs):
    return materialize_engine_tree(
        stack,
        context=context,
        catalog=catalog,
        resolver=resolver,
        destination=destination,
        **kwargs,
    )


def _arranged(source: Path):
    catalog = default_target_catalog()
    context = _evaluation_context(catalog)
    ref = _proposal_ref(source, catalog)
    return catalog, context, ref, _evaluation_stack(catalog, context, ref)


def _copy(tmp_path: Path, fixture: Path = SINGLETON, name: str = "source") -> Path:
    destination = tmp_path / name
    shutil.copytree(fixture, destination)
    return destination


def _cuda_prepend(source: Path, text: str) -> Path:
    cuda = source / "kernels" / "fused_epilogue_sm103.cu"
    cuda.write_text(text + cuda.read_text())
    return cuda


def _declare_cuda(source: Path, extra: str) -> None:
    manifest = source / "manifest.toml"
    manifest.write_text(
        manifest.read_text().replace(
            'cuda_sources = ["kernels/fused_epilogue_sm103.cu"]',
            f'cuda_sources = ["kernels/fused_epilogue_sm103.cu", "{extra}"]',
        )
    )


def test_singleton_materialization_projects_metadata_and_reopens(tmp_path: Path) -> None:
    catalog, context, ref, stack = _arranged(SINGLETON)

    result = _materialize(stack, context, catalog, _sources((ref, SINGLETON)), tmp_path / "engine")

    assert result.stack_digest == stack.digest
    assert result.runtime_manifest == "manifest.toml"
    assert reopen_materialized_engine_tree(
        result.root, expected_tree_digest=result.tree_digest
    ) == result
    manifest = load_manifest(result.root)
    assert manifest.bundle_id == "cacheon-materialized-v1"
    assert manifest.competition is None
    assert [op.slot for op in manifest.ops] == [ref.target_id]
    metadata = json.loads((result.root / manifest.ops[0].metadata).read_text())
    assert "notes" not in metadata
    assert "regime" not in metadata
    assert all(path.stat().st_mode & 0o777 == 0o444 for path in result.root.rglob("*") if path.is_file())


def test_materialization_accepts_exact_typed_worker_bundle_carrier(tmp_path: Path) -> None:
    source = tmp_path / "private-source"
    shutil.copytree(SINGLETON, source)
    for path in source.rglob("*"):
        path.chmod(0o700 if path.is_dir() else 0o600)
    source.chmod(0o700)

    catalog, context, ref, stack = _arranged(source)
    publication = publish_worker_bundle(
        source,
        tmp_path / "publications",
        ref.artifact_digest,
    )
    assert content_hash(publication.root) != ref.artifact_digest

    result = _materialize(
        stack, context, catalog, _sources((ref, publication.root)), tmp_path / "engine"
    )

    assert result.stack_digest == stack.digest
    assert reopen_materialized_engine_tree(
        result.root, expected_tree_digest=result.tree_digest
    ) == result


def test_multiple_variants_share_selected_source_without_order_authority(
    tmp_path: Path,
) -> None:
    source = _copy(tmp_path)
    wide_metadata = source / "metadata" / "blockscore_wide.json"
    wide = json.loads((source / "metadata" / "blockscore.json").read_text())
    wide["capabilities"]["block_size"] = {"exact": 256}
    wide_metadata.write_text(json.dumps(wide, indent=2) + "\n")
    with (source / "manifest.toml").open("a") as manifest:
        manifest.write(
            "\n[[ops]]\n"
            'slot = "norm.rmsnorm"\n'
            'variant = "wide"\n'
            'source = "kernels/blockscore.py"\n'
            'entry = "blockscore"\n'
            'dtypes = ["bfloat16"]\n'
            'architectures = ["sm103"]\n'
            'metadata = "metadata/blockscore_wide.json"\n'
        )
    catalog, context, ref, stack = _arranged(source)
    result = _materialize(stack, context, catalog, _sources((ref, source)), tmp_path / "engine")
    manifest = load_manifest(result.root)
    assert [op.variant for op in manifest.ops] == ["fixture", "wide"]
    assert manifest.ops[0].source == manifest.ops[1].source


def test_overlapping_variant_domains_reject_before_ref_identity(tmp_path: Path) -> None:
    source = _copy(tmp_path)
    with (source / "manifest.toml").open("a") as manifest:
        manifest.write(
            "\n[[ops]]\n"
            'slot = "norm.rmsnorm"\n'
            'variant = "overlap"\n'
            'source = "kernels/blockscore.py"\n'
            'entry = "blockscore"\n'
            'dtypes = ["bfloat16"]\n'
            'architectures = ["sm103"]\n'
            'metadata = "metadata/blockscore.json"\n'
        )
    with pytest.raises(EngineTreeError, match="overlapping capability domains"):
        inspect_contribution(source, catalog=default_target_catalog())


def test_atomic_materialization_namespaces_both_members(tmp_path: Path) -> None:
    source_hash = content_hash(FUSED)
    source_modes = {
        path.relative_to(FUSED): path.stat().st_mode
        for path in FUSED.rglob("*")
        if path.is_file()
    }
    catalog, context, ref, stack = _arranged(FUSED)

    result = _materialize(stack, context, catalog, _sources((ref, FUSED)), tmp_path / "engine")

    manifest = load_manifest(result.root)
    assert {op.slot for op in manifest.ops} == {
        "collective.all_gather_into_tensor",
        "collective.reduce_scatter_tensor",
    }
    assert all(op.source.startswith("entries/cacheon_c_") for op in manifest.ops)
    source = (result.root / manifest.ops[0].source).read_text()
    assert "from cacheon_c_" in source
    assert not (result.root / "rebuild.json").exists()
    assert content_hash(FUSED) == source_hash
    assert source_modes == {
        path.relative_to(FUSED): path.stat().st_mode
        for path in FUSED.rglob("*")
        if path.is_file()
    }


def test_stock_only_stack_has_no_runtime_bundle(tmp_path: Path) -> None:
    catalog = default_target_catalog()
    context = _evaluation_context(catalog)
    stack = _evaluation_stack(catalog, context)

    result = _materialize(stack, context, catalog, {}, tmp_path / "stock")

    assert result.runtime_manifest is None
    assert not (result.root / "manifest.toml").exists()
    assert [row.path for row in result.files] == ["metadata/cacheon_engine_tree.json"]


def test_independent_contributions_compose_without_source_name_collisions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    experts = _write_moe_fixture(
        tmp_path / "experts", "moe.fused_experts", "fused_experts"
    )
    dense = _write_moe_fixture(
        tmp_path / "dense", "linear.dense", "dense"
    )
    catalog = default_target_catalog()
    context = _evaluation_context(catalog)
    experts_ref = _proposal_ref(experts, catalog)
    dense_ref = _proposal_ref(dense, catalog)
    stack = _evaluation_stack(catalog, context, experts_ref, dense_ref)

    result = _materialize(
        stack,
        context,
        catalog,
        _sources((experts_ref, experts), (dense_ref, dense)),
        tmp_path / "engine",
    )

    manifest = load_manifest(result.root)
    assert [op.slot for op in manifest.ops] == [
        "linear.dense",
        "moe.fused_experts",
    ]
    assert manifest.ops[0].source != manifest.ops[1].source
    assert Path(manifest.ops[0].source).stem != Path(manifest.ops[1].source).stem
    assert len(set(result.root.glob("cacheon_c_*/kernels/fused_epilogue.py"))) == 2
    assert len(set(result.root.glob("cacheon_c_*/kernels/helper.py"))) == 2
    assert len(
        set(result.root.glob("cuda/cacheon_c_*/kernels/cacheon_c_*__fused_epilogue_sm103_*.cu"))
    ) == 2
    for op in manifest.ops:
        shim = (result.root / op.source).read_text()
        assert "from cacheon_c_" in shim
        assert "from kernels.helper" not in shim
        assert "import fused_epilogue_sm103" not in shim
    for emitted_path in result.root.glob("cacheon_c_*/kernels/fused_epilogue.py"):
        emitted = emitted_path.read_text()
        assert "from cacheon_c_" in emitted
        assert "from kernels.helper" not in emitted
        assert "import cacheon_c_" in emitted
        assert "import fused_epilogue_sm103" not in emitted

    monkeypatch.syspath_prepend(str(result.root))
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    before_modules = set(sys.modules)
    loaded = []
    try:
        for op in manifest.ops:
            native_name = Path(op.cuda_sources[0]).stem
            monkeypatch.setitem(sys.modules, native_name, ModuleType(native_name))
            module = load_module(result.root / op.source)
            loaded.append(module)
            _native, marker = getattr(module, op.entry)()
            assert marker == op.slot
            assert sys.modules[module.__name__] is module
        assert loaded[0].__name__ != loaded[1].__name__
        assert getattr(loaded[0], manifest.ops[0].entry).__module__ != getattr(
            loaded[1], manifest.ops[1].entry
        ).__module__
    finally:
        for name in set(sys.modules) - before_modules:
            if name.startswith(("cacheon_c_", "cacheon_kernel_cacheon_c_")):
                sys.modules.pop(name, None)


def test_plain_experts_candidate_cannot_retain_shadowing_reduce_route(
    tmp_path: Path,
) -> None:
    experts = _write_moe_fixture(
        tmp_path / "experts", "moe.fused_experts", "fused_experts"
    )
    reduce = _write_moe_fixture(
        tmp_path / "reduce", "moe.fused_experts_reduce", "fused_experts_reduce"
    )
    catalog = default_target_catalog()
    context = _evaluation_context(catalog)
    experts_ref = _proposal_ref(experts, catalog)
    reduce_ref = _proposal_ref(reduce, catalog)
    incumbent = _evaluation_stack(catalog, context, reduce_ref)

    candidate = plan_candidate_stack(
        incumbent,
        experts_ref,
        catalog=catalog,
        expected_context=context,
    )
    arm = plan_marginal_arm(
        incumbent,
        experts_ref,
        catalog=catalog,
        incumbent_tree_digest=_digest("incumbent-tree"),
        candidate_tree_digest=_digest("candidate-tree"),
        expected_context=context,
    )
    materialized = _materialize(
        candidate,
        context,
        catalog,
        _sources((experts_ref, experts), (reduce_ref, reduce)),
        tmp_path / "candidate-engine",
    )
    manifest = load_manifest(materialized.root)

    assert tuple(incumbent.entries) == ("moe.fused_experts_reduce",)
    assert tuple(candidate.entries) == ("moe.fused_experts",)
    assert tuple(ref.target_id for ref in arm.transition.displaced) == (
        "moe.fused_experts_reduce",
    )
    assert [op.slot for op in manifest.ops] == ["moe.fused_experts"]
    assert all("fused_experts_reduce" not in row.path for row in materialized.files)


def test_override_entry_shim_preserves_required_ref_and_optional_device_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    override = tmp_path / "override"
    (override / "kernels").mkdir(parents=True)
    (override / "kernels" / "epilogue.py").write_text(
        "def gemm1_epilogue_ref(gate, up):\n    return gate\n"
    )
    (override / "manifest.toml").write_text(
        'bundle_id = "override-fixture"\nabi_version = "cacheon-op-abi-v0"\n'
        '[[ops]]\nslot = "moe.fused_experts"\nsource = "kernels/epilogue.py"\n'
        'entry = "gemm1_epilogue"\nbase_kernel = "nvfp4_moe_megakernel"\n'
        'override_point = "gemm1_epilogue"\ndtypes = ["bfloat16"]\n'
    )
    catalog, context, ref, stack = _arranged(override)
    result = _materialize(stack, context, catalog, _sources((ref, override)), tmp_path / "engine")
    op = load_manifest(result.root).ops[0]
    shim = (result.root / op.source).read_text()
    assert f"import {op.entry}_ref as {op.entry}_ref" in shim
    assert "try:" in shim and f"import {op.entry} as {op.entry}" in shim

    monkeypatch.syspath_prepend(str(result.root))
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    # The SGLang/CUDA validation image installs CuTeDSL even when no GPU is
    # exposed. Make this specifically the portable-reference branch that the
    # test names, independent of ambient toolchain packages.
    monkeypatch.setitem(sys.modules, "cutlass", None)
    monkeypatch.setitem(sys.modules, "cutlass.cute", None)
    before_modules = set(sys.modules)
    try:
        module = load_module(result.root / op.source)
        assert callable(getattr(module, op.entry + "_ref"))
        assert getattr(module, op.entry, None) is None
        from cacheon_kernels.override import build_override

        entry, prepare = build_override(
            op.slot,
            op.override_point,
            op.entry,
            lambda name: getattr(module, name, None),
        )
        assert callable(entry) and callable(prepare)
    finally:
        for name in set(sys.modules) - before_modules:
            if name.startswith(("cacheon_c_", "cacheon_kernel_cacheon_c_")):
                sys.modules.pop(name, None)


def test_inert_padding_changes_artifact_not_selected_payload(tmp_path: Path) -> None:
    padded = _copy(tmp_path, name="padded")
    (padded / "README.txt").write_text("not selected by the target\n")
    catalog = default_target_catalog()

    plain = inspect_contribution(SINGLETON, catalog=catalog)
    extra = inspect_contribution(padded, catalog=catalog)

    assert content_hash(SINGLETON) != content_hash(padded)
    assert plain.selected_payload_digest == extra.selected_payload_digest
    assert plain.selected_delta_digest == extra.selected_delta_digest
    assert _proposal_ref(SINGLETON, catalog).digest != _proposal_ref(padded, catalog).digest
    context = _evaluation_context(catalog)
    ref = _proposal_ref(padded, catalog)
    result = _materialize(
        _evaluation_stack(catalog, context, ref),
        context,
        catalog,
        _sources((ref, padded)),
        tmp_path / "engine",
    )
    assert not any("README" in row.path for row in result.files)


def test_imported_local_inputs_enter_selected_identity(tmp_path: Path) -> None:
    source = _copy(tmp_path)
    catalog = default_target_catalog()
    before = inspect_contribution(source, catalog=catalog)
    kernel = source / "kernels" / "blockscore.py"
    kernel.write_text("from kernels.helper import scale\n" + kernel.read_text())
    (source / "kernels" / "helper.py").write_text("scale = 1\n")

    after = inspect_contribution(source, catalog=catalog)

    assert after.python_files == (
        "kernels/blockscore.py",
        "kernels/helper.py",
    )
    assert after.selected_payload_digest != before.selected_payload_digest


def test_native_from_import_is_rewritten(tmp_path: Path) -> None:
    source = _native_fixture(tmp_path)
    kernel = source / "kernels" / "fused_epilogue.py"
    kernel.write_text(
        "from fused_epilogue_sm103 import ar_residual_rmsnorm as native_ar\n"
        + kernel.read_text()
    )
    catalog, context, ref, stack = _arranged(source)

    result = _materialize(stack, context, catalog, _sources((ref, source)), tmp_path / "engine")

    manifest = load_manifest(result.root)
    emitted = (result.root / manifest.ops[0].source).read_text()
    assert "from cacheon_c_" in emitted
    assert "from fused_epilogue_sm103" not in emitted


@pytest.mark.parametrize(
    "source_text",
    [
        "module = __import__(\"kernels.helper\")\n",
        "import importlib\nmodule = importlib.import_module(\"kernels.helper\")\n",
        "import importlib as il\nmodule = il.import_module(\"kernels.helper\")\n",
        "from importlib import import_module\nmodule = import_module(\"kernels.helper\")\n",
        "import builtins\nmodule = builtins.__import__(\"kernels.helper\")\n",
        "from builtins import __import__ as imp\nmodule = imp(\"kernels.helper\")\n",
        "exec(\"import kernels.helper\")\n",
        "code = compile(\"import kernels.helper\", \"<miner>\", \"exec\")\n",
    ],
)
def test_dynamic_imports_fail_closed(tmp_path: Path, source_text: str) -> None:
    source = _copy(tmp_path)
    (source / "kernels" / "blockscore.py").write_text(source_text)
    with pytest.raises(EngineTreeError, match="dynamic import"):
        inspect_contribution(source, catalog=default_target_catalog())


@pytest.mark.parametrize(
    "source_text",
    [
        # The CuTe DSL compile idiom: a single import-alias binding of
        # cutlass.cute + a compile call on that module alias.
        "import cutlass.cute as cute\n\n"
        "def blockscore(q, k, out):\n"
        "    kernel = cute.compile(blockscore, q, k, out)\n"
        "    out.copy_(q @ k.transpose(-1, -2))\n",
        "from cutlass import cute\n\n"
        "def blockscore(q, k, out):\n"
        "    kernel = cute.compile(blockscore)\n"
        "    out.copy_(q @ k.transpose(-1, -2))\n",
    ],
)
def test_cute_dsl_compile_alias_is_admitted(tmp_path: Path, source_text: str) -> None:
    source = _copy(tmp_path)
    (source / "kernels" / "blockscore.py").write_text(source_text)
    inspected = inspect_contribution(source, catalog=default_target_catalog())
    assert "kernels/blockscore.py" in inspected.python_files


def test_runtime_toml_omits_only_inline_table_null_fields() -> None:
    from cacheon.engine_tree import _toml_value

    assert _toml_value({"active": 7, "inactive": None}) == '{ "active" = 7 }'
    with pytest.raises(EngineTreeError, match="unsupported TOML value"):
        _toml_value([None])


@pytest.mark.parametrize(
    "source_text",
    [
        # rebinding the alias anywhere withdraws the admission
        "import cutlass.cute as cute\n"
        "cute = cute\n\n"
        "def blockscore(q, k, out):\n"
        "    return cute.compile(blockscore)\n",
        # so does shadowing it via a parameter in any scope
        "import cutlass.cute as cute\n\n"
        "def blockscore(q, k, out, cute=None):\n"
        "    return cute.compile(blockscore)\n",
        # a second import binding the same name
        "import cutlass.cute as cute\n"
        "import types as cute\n\n"
        "def blockscore(q, k, out):\n"
        "    return cute.compile(blockscore)\n",
        # a non-allowlisted module behind the alias
        "import types as cute\n\n"
        "def blockscore(q, k, out):\n"
        "    return cute.compile(blockscore)\n",
        # .compile on anything that is not the allowlisted module alias
        "def blockscore(q, k, out):\n"
        "    gemm = object()\n"
        "    return gemm.compile(q)\n",
        # builtins reached as a module: builtins.compile is a plain attribute
        "import builtins\n\n"
        "def blockscore(q, k, out):\n"
        "    f = builtins.compile\n"
        "    return f\n",
    ],
)
def test_cute_dsl_compile_admission_fails_closed(
    tmp_path: Path, source_text: str
) -> None:
    source = _copy(tmp_path)
    (source / "kernels" / "blockscore.py").write_text(source_text)
    with pytest.raises(EngineTreeError, match="dynamic import"):
        inspect_contribution(source, catalog=default_target_catalog())


def test_cute_dsl_compile_vendored_local_module_fails_closed(tmp_path: Path) -> None:
    # A bundle-local ``cutlass/`` package must withdraw the carve-out: the
    # admitted receiver has to be the EXTERNAL pinned DSL, never bundle code.
    source = _copy(tmp_path)
    vendored = source / "cutlass"
    vendored.mkdir()
    (vendored / "__init__.py").write_text("VALUE = 1\n")
    (vendored / "cute.py").write_text("VALUE = 2\n")
    (source / "kernels" / "blockscore.py").write_text(
        "import cutlass.cute as cute\n\n"
        "def blockscore(q, k, out):\n"
        "    return cute.compile(blockscore)\n"
    )
    with pytest.raises(EngineTreeError, match="dynamic import"):
        inspect_contribution(source, catalog=default_target_catalog())


def test_package_imports_are_closed_rewritten_and_executable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _copy(tmp_path)
    package = source / "kernels" / "pkg"
    package.mkdir()
    (source / "kernels" / "__init__.py").write_text("PARENT = 3\n")
    (package / "__init__.py").write_text("PACKAGE = True\n")
    (package / "helper.py").write_text("VALUE = 7\n")
    # CPython resolves the package before this same-name module.
    (source / "kernels" / "pkg.py").write_text("VALUE = -1\n")
    (source / "kernels" / "blockscore.py").write_text(
        "if True:\n"
        "    from .pkg import helper\n"
        "    import kernels.pkg.helper\n"
        "VALUE = helper.VALUE + kernels.pkg.helper.VALUE\n\n"
        "def blockscore(q, k, out):\n"
        "    return VALUE\n"
    )
    catalog = default_target_catalog()
    inspected = inspect_contribution(source, catalog=catalog)
    assert "kernels/pkg/__init__.py" in inspected.python_files
    assert "kernels/__init__.py" in inspected.python_files
    assert "kernels/pkg/helper.py" in inspected.python_files
    assert "kernels/pkg.py" not in inspected.python_files
    catalog, context, ref, stack = _arranged(source)
    result = _materialize(stack, context, catalog, _sources((ref, source)), tmp_path / "engine")
    manifest = load_manifest(result.root)
    emitted = (result.root / manifest.ops[0].source).read_text()
    compile(emitted, manifest.ops[0].source, "exec")
    assert "from cacheon_c_" in emitted
    implementation = next(result.root.glob("cacheon_c_*/kernels/blockscore.py")).read_text()
    assert "import cacheon_c_" in implementation
    monkeypatch.syspath_prepend(str(result.root))
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    module_name = manifest.ops[0].source.removesuffix(".py").replace("/", ".")
    module = importlib.import_module(module_name)
    entry = getattr(module, manifest.ops[0].entry)
    assert entry(None, None, None) == 14
    namespace = entry.__module__.split(".", 1)[0]
    assert importlib.import_module(namespace + ".kernels").PARENT == 3
    reopen_materialized_engine_tree(
        result.root, expected_tree_digest=result.tree_digest
    )


@pytest.mark.parametrize(
    "source_text,message",
    [
        ("from .missing import value\n", "unresolved relative"),
        ("from kernels import helper, missing\n", "partially local"),
        ("import kernels.missing\n", "partially local"),
        ("from kernels.missing import value\n", "partially local"),
        ("from . import *\n", "unresolved relative"),
    ],
)
def test_unresolved_or_partial_local_imports_fail_closed(
    tmp_path: Path, source_text: str, message: str
) -> None:
    source = _copy(tmp_path)
    (source / "kernels" / "helper.py").write_text("value = 1\n")
    (source / "kernels" / "blockscore.py").write_text(source_text)
    with pytest.raises(EngineTreeError, match=message):
        inspect_contribution(source, catalog=default_target_catalog())


@pytest.mark.parametrize(
    "source_text",
    (
        "import fused_epilogue_sm103.missing\n",
        "from fused_epilogue_sm103.missing import value\n",
    ),
)
def test_partial_declared_native_import_fails_closed(
    tmp_path: Path, source_text: str,
) -> None:
    source = _native_fixture(tmp_path)
    (source / "kernels" / "fused_epilogue.py").write_text(
        source_text
        + "def prepare(*args):\n    return None\n"
        + "def fused_experts(*args):\n    return None\n"
    )
    with pytest.raises(EngineTreeError, match="partially local"):
        inspect_contribution(source, catalog=default_target_catalog())


def test_bare_namespace_and_nonidentifier_module_paths_fail_closed(tmp_path: Path) -> None:
    namespace = tmp_path / "namespace"
    shutil.copytree(SINGLETON, namespace)
    (namespace / "kernels" / "blockscore.py").write_text("import kernels\n")
    with pytest.raises(EngineTreeError, match="bare local namespace"):
        inspect_contribution(namespace, catalog=default_target_catalog())

    invalid = tmp_path / "invalid"
    shutil.copytree(SINGLETON, invalid)
    (invalid / "kernels-v2").mkdir()
    shutil.copy2(
        invalid / "kernels" / "blockscore.py",
        invalid / "kernels-v2" / "blockscore.py",
    )
    manifest = invalid / "manifest.toml"
    manifest.write_text(
        manifest.read_text().replace("kernels/blockscore.py", "kernels-v2/blockscore.py")
    )
    with pytest.raises(EngineTreeError, match="non-identifier component"):
        inspect_contribution(invalid, catalog=default_target_catalog())


def test_python_native_name_collision_fails_during_inspection(tmp_path: Path) -> None:
    source = _native_fixture(tmp_path)
    (source / "fused_epilogue_sm103.py").write_text("collision = True\n")
    with pytest.raises(EngineTreeError, match="both local Python and declared native"):
        inspect_contribution(source, catalog=default_target_catalog())


def test_ambiguous_declared_native_stems_fail_closed(tmp_path: Path) -> None:
    source = _native_fixture(tmp_path)
    duplicate = source / "other" / "fused_epilogue_sm103.cu"
    duplicate.parent.mkdir()
    shutil.copy2(source / "kernels" / "fused_epilogue_sm103.cu", duplicate)
    _declare_cuda(source, "other/fused_epilogue_sm103.cu")
    with pytest.raises(EngineTreeError, match="ambiguous native module stem"):
        inspect_contribution(source, catalog=default_target_catalog())


def test_nonregular_source_tree_entries_fail_closed(tmp_path: Path) -> None:
    source = _copy(tmp_path)
    fifo = source / "host-pipe"
    try:
        os.mkfifo(fifo)
    except (AttributeError, OSError):
        pytest.skip("FIFO creation unavailable")
    with pytest.raises(EngineTreeError, match="nonregular"):
        inspect_contribution(source, catalog=default_target_catalog())


def test_declared_cuda_headers_enter_identity_and_undeclared_headers_reject(
    tmp_path: Path,
) -> None:
    source = _native_fixture(tmp_path)
    _cuda_prepend(source, '#include "helper.cuh" /* selected */\n')
    (source / "kernels" / "helper.cuh").write_text("#define HELPER 1\n")
    with pytest.raises(EngineTreeError, match="undeclared local input"):
        inspect_contribution(source, catalog=default_target_catalog())

    _declare_cuda(source, "kernels/helper.cuh")
    inspected = inspect_contribution(source, catalog=default_target_catalog())
    assert "kernels/helper.cuh" in inspected.cuda_files


def test_dynamic_cuda_include_directives_fail_closed(tmp_path: Path) -> None:
    source = _native_fixture(tmp_path)
    _cuda_prepend(source, '#define HEADER "unbound.cuh"\n#include HEADER\n')
    with pytest.raises(EngineTreeError, match="dynamic include"):
        inspect_contribution(source, catalog=default_target_catalog())


def test_literal_cuda_includes_allow_comment_only_suffixes(tmp_path: Path) -> None:
    source = _native_fixture(tmp_path)
    cuda = source / "kernels" / "fused_epilogue_sm103.cu"
    cuda.write_text(
        cuda.read_text().replace(
            "#include <torch/extension.h>",
            "#include <torch/extension.h> // pinned toolchain header",
        )
    )
    inspect_contribution(source, catalog=default_target_catalog())


@pytest.mark.parametrize("header", ["/tmp/unbound.cuh", "../../unbound.cuh"])
def test_unsafe_system_cuda_includes_fail_closed(tmp_path: Path, header: str) -> None:
    source = _native_fixture(tmp_path)
    _cuda_prepend(source, f"#include <{header}>\n")
    with pytest.raises(EngineTreeError, match="unsafe system include"):
        inspect_contribution(source, catalog=default_target_catalog())


def test_missing_quoted_cuda_include_cannot_escape_dependency_roots(tmp_path: Path) -> None:
    source = _native_fixture(tmp_path)
    _cuda_prepend(source, '#include "../unbound.cuh"\n')
    with pytest.raises(EngineTreeError, match="unsafe dependency include"):
        inspect_contribution(source, catalog=default_target_catalog())


@pytest.mark.parametrize(
    "directive",
    [
        '#include_next "/tmp/unbound.cuh"',
        '%:include "/tmp/unbound.cuh"',
    ],
)
def test_alternate_cuda_include_directives_fail_closed(
    tmp_path: Path, directive: str
) -> None:
    source = _native_fixture(tmp_path)
    _cuda_prepend(source, directive + "\n")
    with pytest.raises(EngineTreeError, match="unsupported include"):
        inspect_contribution(source, catalog=default_target_catalog())


def test_line_spliced_cuda_include_is_still_validated(tmp_path: Path) -> None:
    source = _native_fixture(tmp_path)
    _cuda_prepend(source, '#inc\\\nlude "/tmp/unbound.cuh"\n')
    with pytest.raises(EngineTreeError, match="safe relative path"):
        inspect_contribution(source, catalog=default_target_catalog())


@pytest.mark.parametrize(
    "selected",
    [".git/blockscore.py", "kernels/._blockscore.py", "kernels/blockscore.pyc"],
)
def test_bundle_hash_excluded_paths_cannot_be_selected(
    tmp_path: Path, selected: str
) -> None:
    source = _copy(tmp_path)
    selected_path = source / selected
    selected_path.parent.mkdir(parents=True, exist_ok=True)
    selected_path.write_text((source / "kernels" / "blockscore.py").read_text())
    manifest = source / "manifest.toml"
    manifest.write_text(manifest.read_text().replace("kernels/blockscore.py", selected))
    with pytest.raises(ValueError):
        inspect_contribution(source, catalog=default_target_catalog())


def test_root_source_symlink_is_rejected(tmp_path: Path) -> None:
    alias = tmp_path / "alias"
    alias.symlink_to(SINGLETON, target_is_directory=True)
    with pytest.raises(EngineTreeError, match="must not be a symlink"):
        inspect_contribution(alias, catalog=default_target_catalog())


def test_materialization_is_location_mode_and_umask_independent(tmp_path: Path) -> None:
    left = tmp_path / "left-source"
    right = tmp_path / "right-source"
    shutil.copytree(SINGLETON, left)
    shutil.copytree(SINGLETON, right)
    os.chmod(left / "kernels" / "blockscore.py", 0o600)
    os.chmod(right / "kernels" / "blockscore.py", 0o755)
    assert source_tree_digest(left) == source_tree_digest(right)

    catalog = default_target_catalog()
    context = _evaluation_context(catalog)
    left_ref = _proposal_ref(left, catalog)
    right_ref = _proposal_ref(right, catalog)
    assert left_ref == right_ref
    stack = _evaluation_stack(catalog, context, left_ref)
    previous = os.umask(0o077)
    try:
        left_tree = _materialize(
            stack, context, catalog, _sources((left_ref, left)), tmp_path / "left-engine"
        )
    finally:
        os.umask(previous)
    previous = os.umask(0o002)
    try:
        right_tree = _materialize(
            stack, context, catalog, _sources((right_ref, right)), tmp_path / "right-engine"
        )
    finally:
        os.umask(previous)

    assert left_tree.tree_digest == right_tree.tree_digest
    assert [row.identity_data() for row in left_tree.files] == [
        row.identity_data() for row in right_tree.files
    ]


def test_semantic_aliases_and_set_order_share_selected_identity(tmp_path: Path) -> None:
    canonical = tmp_path / "canonical"
    aliases = tmp_path / "aliases"
    shutil.copytree(SINGLETON, canonical)
    shutil.copytree(SINGLETON, aliases)
    manifest = aliases / "manifest.toml"
    manifest.write_text(
        manifest.read_text()
        .replace('dtypes = ["bfloat16"]', 'dtypes = ["bfloat16", "bf16"]')
        .replace('architectures = ["sm103"]', 'architectures = ["sm_103", "sm103"]')
    )
    metadata_path = aliases / "metadata" / "blockscore.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["dtypes"] = ["bf16", "bfloat16"]
    metadata["architectures"] = ["sm_103", "sm103"]
    metadata["capabilities"] = {
        field: {"one_of": [spec["exact"], spec["exact"]]}
        for field, spec in reversed(tuple(metadata["capabilities"].items()))
    }
    metadata_path.write_text(json.dumps(metadata, indent=4) + "\n")

    left = inspect_contribution(canonical, catalog=default_target_catalog())
    right = inspect_contribution(aliases, catalog=default_target_catalog())
    assert left.selected_payload_digest == right.selected_payload_digest
    assert left.selected_delta_digest == right.selected_delta_digest


def test_packaging_order_ids_and_json_whitespace_do_not_choose_namespace(
    tmp_path: Path,
) -> None:
    canonical = tmp_path / "canonical"
    reordered = tmp_path / "reordered"
    shutil.copytree(FUSED, canonical)
    shutil.copytree(FUSED, reordered)
    manifest = reordered / "manifest.toml"
    prefix, first_op, second_op = manifest.read_text().split("[[ops]]")
    manifest.write_text(
        (prefix + "[[ops]]" + second_op + "[[ops]]" + first_op)
        .replace("fixture-dp-exchange-atomic", "ignored-packaging-id")
    )
    for metadata_path in (reordered / "metadata").glob("*.json"):
        metadata = json.loads(metadata_path.read_text())
        metadata_path.write_text(json.dumps(metadata, indent=6) + "\n")

    left = inspect_contribution(canonical, catalog=default_target_catalog())
    right = inspect_contribution(reordered, catalog=default_target_catalog())
    assert content_hash(canonical) != content_hash(reordered)
    assert left.selected_payload_digest == right.selected_payload_digest
    assert left.selected_delta_digest == right.selected_delta_digest
    assert f"cacheon_c_{left.selected_delta_digest}" == (
        f"cacheon_c_{right.selected_delta_digest}"
    )


@pytest.mark.parametrize(
    "input_class",
    ["op", "metadata", "python"],
)
def test_every_selected_executable_input_class_rotates_delta(
    tmp_path: Path, input_class: str
) -> None:
    source = _copy(tmp_path, FUSED)
    before = inspect_contribution(source, catalog=default_target_catalog())

    if input_class == "op":
        manifest = source / "manifest.toml"
        manifest.write_text(
            manifest.read_text().replace(
                'entry = "all_gather_into_tensor"',
                'entry = "all_gather_into_tensor_v2"',
                1,
            )
        )
    elif input_class == "metadata":
        path = source / "metadata" / "all_gather.json"
        metadata = json.loads(path.read_text())
        metadata["architectures"].append("sm100")
        path.write_text(json.dumps(metadata))
    else:
        path = source / "kernels" / "exchange.py"
        path.write_text(path.read_text() + "\n# selected source revision\n")

    after = inspect_contribution(source, catalog=default_target_catalog())
    assert after.selected_payload_digest != before.selected_payload_digest
    assert after.selected_delta_digest != before.selected_delta_digest


def test_source_mutation_cannot_diverge_identity_from_emitted_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _copy(tmp_path)
    catalog, context, ref, stack = _arranged(source)
    stable_read = engine_tree._stable_read
    changed = False

    def racing_read(root: Path, relative: str) -> bytes:
        nonlocal changed
        data = stable_read(root, relative)
        if root == source.resolve() and relative == "kernels/blockscore.py" and not changed:
            (source / relative).write_text("def blockscore(*args):\n    return None\n")
            changed = True
        return data

    monkeypatch.setattr(engine_tree, "_stable_read", racing_read)
    with pytest.raises(EngineTreeError, match="changed"):
        _materialize(stack, context, catalog, _sources((ref, source)), tmp_path / "engine")
    assert not (tmp_path / "engine").exists()


def test_reopen_rejects_root_mode_extra_directories_and_root_symlinks(
    tmp_path: Path,
) -> None:
    catalog, context, ref, stack = _arranged(SINGLETON)
    result = _materialize(stack, context, catalog, _sources((ref, SINGLETON)), tmp_path / "engine")
    metadata = json.loads((result.root / "metadata/cacheon_engine_tree.json").read_text())
    assert metadata["contributions"][0]["namespace"] == (
        "cacheon_c_" + ref.selected_delta_digest
    )
    with pytest.raises(EngineTreeError, match="tree digest mismatch"):
        reopen_materialized_engine_tree(
            result.root, expected_tree_digest=_digest("wrong-tree-receipt")
        )

    ghost = result.root / "ghost"
    ghost.mkdir()
    with pytest.raises(EngineTreeError, match="directory inventory"):
        reopen_materialized_engine_tree(result.root, expected_tree_digest=result.tree_digest)
    ghost.rmdir()

    os.chmod(result.root, 0o700)
    try:
        with pytest.raises(EngineTreeError, match="root directory mode"):
            reopen_materialized_engine_tree(result.root)
    finally:
        os.chmod(result.root, 0o755)

    alias = tmp_path / "engine-link"
    alias.symlink_to(result.root, target_is_directory=True)
    with pytest.raises(EngineTreeError, match="must not be a symlink"):
        reopen_materialized_engine_tree(alias)

    metadata_path = result.root / "metadata/cacheon_engine_tree.json"
    os.chmod(metadata_path, 0o644)
    metadata["contributions"][0]["namespace"] = "cacheon_c_" + "0" * 64
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    os.chmod(metadata_path, 0o444)
    with pytest.raises(EngineTreeError, match="namespace mismatch"):
        reopen_materialized_engine_tree(result.root)


def test_failed_preinstall_verification_leaves_no_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    catalog, context, ref, stack = _arranged(SINGLETON)
    reopen = engine_tree.reopen_materialized_engine_tree

    def fail_temp(root: str | Path, *, expected_tree_digest: str | None = None):
        if Path(root).name.startswith(".engine."):
            raise EngineTreeError("forced preinstall verification failure")
        return reopen(root, expected_tree_digest=expected_tree_digest)

    monkeypatch.setattr(engine_tree, "reopen_materialized_engine_tree", fail_temp)
    destination = tmp_path / "engine"
    with pytest.raises(EngineTreeError, match="forced preinstall"):
        _materialize(stack, context, catalog, _sources((ref, SINGLETON)), destination)
    assert not destination.exists()




@pytest.mark.parametrize("fixture", [SINGLETON, FUSED], ids=["norm-singleton", "atomic-dp"])
def test_fixture_materialization_binds_marginal_arm_and_exact_rollback(
    tmp_path: Path, fixture: Path,
) -> None:
    catalog = default_target_catalog()
    context = _evaluation_context(catalog)
    incumbent = _evaluation_stack(catalog, context)
    baseline = _materialize(incumbent, context, catalog, {}, tmp_path / "baseline")
    ref = _proposal_ref(fixture, catalog)
    candidate = plan_candidate_stack(
        incumbent,
        ref,
        catalog=catalog,
        expected_context=context,
    )
    challenger = _materialize(
        candidate, context, catalog, _sources((ref, fixture)), tmp_path / "challenger"
    )
    arm = plan_marginal_arm(
        incumbent,
        ref,
        catalog=catalog,
        incumbent_tree_digest=baseline.tree_digest,
        candidate_tree_digest=challenger.tree_digest,
        expected_context=context,
    )
    assert arm.candidate == candidate
    assert arm.baseline_before == arm.baseline_after
    assert arm.baseline_before is not arm.baseline_after
    rollback = RollbackPlan.from_arm(
        arm, catalog=catalog, expected_context=context
    )
    restored, restored_tree = rollback.reconstruct(
        candidate,
        tree_digest=challenger.tree_digest,
        source_arm=arm,
        catalog=catalog,
        expected_context=context,
    )
    assert restored == incumbent
    assert restored_tree == baseline.tree_digest


def test_source_and_materialized_symlinks_are_rejected(tmp_path: Path) -> None:
    source = _copy(tmp_path)
    (source / "padding-link").symlink_to(source / "manifest.toml")
    with pytest.raises(EngineTreeError, match="symlink"):
        inspect_contribution(source, catalog=default_target_catalog())

    catalog, context, ref, stack = _arranged(SINGLETON)
    result = _materialize(stack, context, catalog, _sources((ref, SINGLETON)), tmp_path / "engine")
    link = result.root / "link"
    link.symlink_to(result.root / "manifest.toml")
    with pytest.raises(EngineTreeError, match="symlink"):
        reopen_materialized_engine_tree(result.root)


def test_wrong_source_identity_and_post_write_tampering_fail_closed(tmp_path: Path) -> None:
    catalog, context, ref, stack = _arranged(SINGLETON)
    padded = _copy(tmp_path, name="padded")
    (padded / "padding.txt").write_text("changes the proposal artifact")

    with pytest.raises(EngineTreeError, match="artifact digest mismatch"):
        _materialize(stack, context, catalog, _sources((ref, padded)), tmp_path / "wrong")

    wrong_payload = replace(ref, selected_payload_digest=_digest("wrong-payload"))
    wrong_stack = _evaluation_stack(catalog, context, wrong_payload)
    with pytest.raises(EngineTreeError, match="selected payload digest mismatch"):
        _materialize(
            wrong_stack, context, catalog, _sources((wrong_payload, SINGLETON)), tmp_path / "wrong-payload"
        )

    result = _materialize(stack, context, catalog, _sources((ref, SINGLETON)), tmp_path / "engine")
    kernel = next(result.root.glob("cacheon_c_*/kernels/*.py"))
    os.chmod(kernel, 0o644)
    kernel.write_text(kernel.read_text() + "\n# tampered\n")
    os.chmod(kernel, 0o444)
    with pytest.raises(EngineTreeError, match="inventory mismatch"):
        reopen_materialized_engine_tree(result.root)


def test_destination_and_context_are_fail_closed(tmp_path: Path) -> None:
    catalog, context, ref, stack = _arranged(SINGLETON)
    destination = tmp_path / "already-there"
    destination.mkdir()
    with pytest.raises(EngineTreeError, match="already exists"):
        _materialize(stack, context, catalog, _sources((ref, SINGLETON)), destination)

    wrong_context = EvaluationStackContext(
        runtime_digest=_digest("wrong-runtime"),
        base_engine_digest=context.base_engine_digest,
        arena_digest=context.arena_digest,
        catalog_snapshot=catalog.snapshot(),
        catalog_digest=catalog.digest,
        target_spec_digests=_spec_digests(catalog),
    )
    with pytest.raises(ValueError, match="runtime digest"):
        _materialize(stack, wrong_context, catalog, _sources((ref, SINGLETON)), tmp_path / "stale")


def test_destination_cannot_mutate_a_resolved_contribution_source(tmp_path: Path) -> None:
    source = _copy(tmp_path)
    before = content_hash(source)
    catalog, context, ref, stack = _arranged(source)

    with pytest.raises(EngineTreeError, match="outside contribution source"):
        _materialize(stack, context, catalog, _sources((ref, source)), source / "emitted-engine")
    assert content_hash(source) == before
    assert not (source / "emitted-engine").exists()
