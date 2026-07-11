from __future__ import annotations

import difflib
import importlib
import json
import os
import sys
import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from optima.competition import (
    SLOT_MODE,
    SYSTEM_MODE,
    CompetitionError,
    resolve_competition,
)
from optima.compat import PINNED_SGLANG
from optima.manifest import (
    CompetitionEntry,
    ManifestError,
    SystemPatchEntry,
    all_declared_system_patches,
    load_manifest,
)
from optima.sandbox import scan_tree
from optima.system_overlay import (
    _SchedulerOverlayFinder,
    activate_scheduler_overlay,
    driver_import_is_stock,
    driver_module_is_stock,
    install_process_role_hook,
    role_for_process_target,
)
from optima.system_patch import (
    SGLANG_INFERENCE_SYSTEM_V1,
    SYSTEM_TARGETS,
    SystemPatchError,
    materialize_system_overlay,
    qualification_requirement,
    read_validated_system_overlay,
    system_patch_fingerprints,
    system_overlay_identity,
    validate_file_patch,
    validate_patch_path,
)


ARENA = "minimax-m3-b300-tp4-decode-v1"


def _diff(old: str, new: str, path: str) -> str:
    return "".join(
        difflib.unified_diff(
            old.splitlines(keepends=True),
            new.splitlines(keepends=True),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
        )
    )


def _bundle(
    tmp_path: Path,
    *,
    patch_text: str | None = None,
    competition_target: str = SGLANG_INFERENCE_SYSTEM_V1,
    competition_mode: str = SYSTEM_MODE,
    with_competition: bool = True,
) -> Path:
    bundle = tmp_path / "bundle"
    (bundle / "patches").mkdir(parents=True)
    if patch_text is None:
        patch_text = _diff(
            "VALUE = 1\n",
            "VALUE = 2\n",
            "sglang/srt/layers/activation.py",
        )
    (bundle / "patches" / "runtime.patch").write_text(patch_text)
    rows = [
        'bundle_id = "system-test"',
        'abi_version = "optima-system-patch-v1"',
        "",
    ]
    if with_competition:
        rows.extend(
            [
                "[competition]",
                f'target = "{competition_target}"',
                f'mode = "{competition_mode}"',
                "",
            ]
        )
    rows.extend(
        [
            "[system]",
            'target = "sglang"',
            'region = "inference"',
            'patches = ["patches/runtime.patch"]',
            "",
        ]
    )
    (bundle / "manifest.toml").write_text("\n".join(rows))
    return bundle


def _stock(tmp_path: Path) -> Path:
    site = tmp_path / "stock-site"
    package = site / "sglang"
    (package / "srt" / "layers").mkdir(parents=True)
    (package / "__init__.py").write_text('__version__ = "test"\n')
    (package / "srt" / "__init__.py").write_text("")
    (package / "srt" / "layers" / "__init__.py").write_text("")
    (package / "srt" / "layers" / "activation.py").write_text("VALUE = 1\n")
    (package / "untouched.py").write_text("UNCHANGED = True\n")
    (package / "jit_kernel").mkdir()
    (package / "jit_kernel" / ".clang-format").write_text("dev-only\n")
    (package / "multimodal_gen" / ".claude").mkdir(parents=True)
    (package / "multimodal_gen" / ".claude" / "notes.md").write_text("dev-only\n")
    return site


def test_system_manifest_is_a_product_without_fake_op(tmp_path):
    manifest = load_manifest(_bundle(tmp_path))
    assert manifest.ops == ()
    assert manifest.system == SystemPatchEntry(
        target="sglang",
        region="inference",
        patches=("patches/runtime.patch",),
    )

    resolved = resolve_competition(manifest, for_settlement=True)
    assert resolved.target == SGLANG_INFERENCE_SYSTEM_V1
    assert resolved.mode == SYSTEM_MODE
    assert resolved.members == ()
    assert resolved.crownable


def test_system_product_has_its_own_abi(tmp_path):
    bundle = _bundle(tmp_path)
    text = (bundle / "manifest.toml").read_text().replace(
        "optima-system-patch-v1", "optima-op-abi-v0"
    )
    (bundle / "manifest.toml").write_text(text)
    with pytest.raises(ManifestError, match="optima-system-patch-v1"):
        load_manifest(bundle)


def test_system_patch_is_explicitly_scan_allowlisted(tmp_path):
    bundle = _bundle(tmp_path)
    manifest = load_manifest(bundle)
    result = scan_tree(
        bundle,
        declared_cuda_sources=frozenset(),
        declared_dep_patches=frozenset(),
        declared_system_patches=all_declared_system_patches(bundle, manifest),
    )
    assert result.ok, result.violations

    (bundle / "patches" / "undeclared.patch").write_text(
        (bundle / "patches" / "runtime.patch").read_text()
    )
    result = scan_tree(
        bundle,
        declared_cuda_sources=frozenset(),
        declared_dep_patches=frozenset(),
        declared_system_patches=all_declared_system_patches(bundle, manifest),
    )
    assert not result.ok
    assert "system.patches" in "\n".join(result.violations)


def test_cli_scan_accepts_declared_system_patch(tmp_path, capsys):
    from argparse import Namespace

    from optima.cli import cmd_scan

    bundle = _bundle(tmp_path)
    assert cmd_scan(Namespace(bundle=str(bundle))) == 0
    output = capsys.readouterr().out
    assert "bundle: system-test" in output
    assert "VIOLATIONS" not in output


def test_eval_recursive_scan_accepts_declared_system_patch(tmp_path, capsys):
    from optima.cli import _recursive_scan_ok

    bundle = _bundle(tmp_path)
    assert _recursive_scan_ok(str(bundle), manifest=load_manifest(bundle))
    assert "FAIL" not in capsys.readouterr().out


def test_system_copy_signals_are_product_level_not_fake_slot_identity(tmp_path):
    fingerprints = system_patch_fingerprints(_bundle(tmp_path))
    assert len(fingerprints) == 2  # exact bytes + normalized unified-diff effect
    assert all(len(value) == 64 for value in fingerprints)


def test_system_and_component_products_are_mutually_exclusive(tmp_path):
    bundle = _bundle(tmp_path)
    (bundle / "kernel.py").write_text("def entry():\n    pass\n")
    with (bundle / "manifest.toml").open("a") as stream:
        stream.write(
            "\n[[ops]]\nslot = \"norm.rmsnorm\"\n"
            "source = \"kernel.py\"\nentry = \"entry\"\n"
        )
    with pytest.raises(ManifestError, match="exactly one product"):
        load_manifest(bundle)


def test_system_manifest_requires_nonempty_exact_text_patch_set(tmp_path):
    bundle = _bundle(tmp_path)
    text = (bundle / "manifest.toml").read_text().replace(
        'patches = ["patches/runtime.patch"]', "patches = []"
    )
    (bundle / "manifest.toml").write_text(text)
    with pytest.raises(ManifestError, match="non-empty list"):
        load_manifest(bundle)

    bundle = _bundle(tmp_path / "binary")
    (bundle / "patches" / "runtime.patch").write_text(
        "GIT binary patch\nliteral 4\n"
    )
    with pytest.raises(ManifestError, match="binary"):
        load_manifest(bundle)


def test_system_competition_isolation(tmp_path):
    manifest = load_manifest(_bundle(tmp_path))
    with pytest.raises(CompetitionError, match="mode 'system'"):
        resolve_competition(
            replace(
                manifest,
                competition=CompetitionEntry(
                    target=SGLANG_INFERENCE_SYSTEM_V1, mode=SLOT_MODE
                ),
            ),
            for_settlement=True,
        )

    component = tmp_path / "component"
    component.mkdir()
    (component / "k.py").write_text("def entry():\n    pass\n")
    (component / "manifest.toml").write_text(
        'bundle_id="c"\nabi_version="optima-op-abi-v0"\n'
        '[competition]\ntarget="sglang.inference.v1"\nmode="system"\n'
        '[[ops]]\nslot="norm.rmsnorm"\nsource="k.py"\nentry="entry"\n'
    )
    with pytest.raises(CompetitionError, match="whole-serving system target"):
        resolve_competition(load_manifest(component), for_settlement=True)


def test_system_target_requires_registered_arena_external_one_shot(tmp_path):
    manifest = load_manifest(_bundle(tmp_path))
    requirement = qualification_requirement(
        manifest,
        competition_target=SGLANG_INFERENCE_SYSTEM_V1,
        arena_name=ARENA,
        external_quality_gate="controller-one-shot-paired-topk-v1",
    )
    assert requirement.one_shot_external
    assert not requirement.component_receipts
    assert not requirement.component_champions

    with pytest.raises(SystemPatchError, match="not registered for arena"):
        qualification_requirement(
            manifest,
            competition_target=SGLANG_INFERENCE_SYSTEM_V1,
            arena_name="miner-chosen-arena",
            external_quality_gate="controller-one-shot-paired-topk-v1",
        )
    with pytest.raises(SystemPatchError, match="requires external one-shot"):
        qualification_requirement(
            manifest,
            competition_target=SGLANG_INFERENCE_SYSTEM_V1,
            arena_name=ARENA,
            external_quality_gate="candidate-self-report",
        )


@pytest.mark.parametrize(
    "path",
    [
        "sglang/srt/layers/activation.py",
        "sglang/srt/model_executor/model_runner.py",
        "sglang/srt/models/minimax.py",
        "sglang/srt/mem_cache/memory_pool.py",
        "sglang/srt/batch_overlap/two_batch_overlap.py",
        "sglang/srt/managers/schedule_policy.py",
        "sglang/srt/managers/scheduler.py",
        "sglang/srt/managers/scheduler_components/dp_attn.py",
    ],
)
def test_inference_region_broad_allowed_surface(path):
    validate_patch_path(SYSTEM_TARGETS[SGLANG_INFERENCE_SYSTEM_V1], path)


@pytest.mark.parametrize(
    "path",
    [
        "sglang/api.py",
        "sglang/srt/entrypoints/engine.py",
        "sglang/srt/managers/tokenizer_manager.py",
        "sglang/srt/managers/detokenizer_manager.py",
        "sglang/srt/managers/scheduler_components/logprob_result_processor.py",
        "sglang/srt/managers/scheduler_components/batch_result_processor.py",
        "sglang/srt/managers/scheduler_components/output_streamer.py",
        "sglang/srt/layers/sampler.py",
        "sglang/srt/layers/logits_processor.py",
        "sglang/srt/observability/timing.py",
        "sglang/srt/layers/kernel.so",
    ],
)
def test_policy_excludes_api_sampling_result_and_timing_surfaces(path):
    with pytest.raises(SystemPatchError):
        validate_patch_path(SYSTEM_TARGETS[SGLANG_INFERENCE_SYSTEM_V1], path)


def test_allowed_file_cannot_import_excluded_surface():
    from optima.deppatch import parse_patch_text

    text = _diff(
        "VALUE = 1\n",
        "from sglang.srt.layers.logits_processor import LogitsProcessor\nVALUE = 1\n",
        "sglang/srt/layers/activation.py",
    )
    (file_patch,) = parse_patch_text(text)
    with pytest.raises(SystemPatchError, match="excluded source surface"):
        validate_file_patch(SYSTEM_TARGETS[SGLANG_INFERENCE_SYSTEM_V1], file_patch)


def test_mixed_scheduler_file_is_narrowed_to_named_semantic_symbols():
    from optima.deppatch import parse_patch_text

    old = (
        "class Scheduler:\n"
        "    def get_next_batch_to_run(self):\n"
        "        choice = 1\n"
        "        return choice\n"
        "\n"
        "    def process_batch_result(self):\n"
        "        result = 1\n"
        "        return result\n"
    )
    allowed_new = old.replace("choice = 1", "choice = 2")
    (allowed_patch,) = parse_patch_text(
        _diff(old, allowed_new, "sglang/srt/managers/scheduler.py")
    )
    validate_file_patch(
        SYSTEM_TARGETS[SGLANG_INFERENCE_SYSTEM_V1],
        allowed_patch,
        original_source=old,
    )

    forbidden_new = old.replace("result = 1", "result = 2")
    (forbidden_patch,) = parse_patch_text(
        _diff(old, forbidden_new, "sglang/srt/managers/scheduler.py")
    )
    with pytest.raises(SystemPatchError, match="outside.*semantic regions"):
        validate_file_patch(
            SYSTEM_TARGETS[SGLANG_INFERENCE_SYSTEM_V1],
            forbidden_patch,
            original_source=old,
        )

    escaped_new = old.replace(
        "        return choice\n\n",
        "        return choice\n\n    ESCAPED_CLASS_LEVEL = True\n\n",
    )
    (escaped_patch,) = parse_patch_text(
        _diff(old, escaped_new, "sglang/srt/managers/scheduler.py")
    )
    with pytest.raises(SystemPatchError, match="resulting line.*outside"):
        validate_file_patch(
            SYSTEM_TARGETS[SGLANG_INFERENCE_SYSTEM_V1],
            escaped_patch,
            original_source=old,
        )


def test_full_content_addressed_overlay_exact_apply_and_integrity(tmp_path):
    bundle = _bundle(tmp_path)
    stock = _stock(tmp_path)
    cache = tmp_path / "cache"
    dest = materialize_system_overlay(
        bundle,
        competition_target=SGLANG_INFERENCE_SYSTEM_V1,
        arena_name=ARENA,
        cache_root=cache,
        stock_site_root=stock,
        sglang_version=PINNED_SGLANG,
    )
    assert (dest / "site/sglang/srt/layers/activation.py").read_text() == "VALUE = 2\n"
    # Full package copy, not a loose touched-file namespace package.
    assert (dest / "site/sglang/untouched.py").read_text() == "UNCHANGED = True\n"
    assert not (dest / "site/sglang/jit_kernel/.clang-format").exists()
    assert not (dest / "site/sglang/multimodal_gen/.claude").exists()

    identity, stamp, validated = read_validated_system_overlay(
        bundle,
        competition_target=SGLANG_INFERENCE_SYSTEM_V1,
        arena_name=ARENA,
        cache_root=cache,
        stock_site_root=stock,
        sglang_version=PINNED_SGLANG,
    )
    assert validated == dest
    assert stamp["cache_key"] == identity.cache_key
    assert stamp["touched_files"] == {
        "sglang/srt/layers/activation.py": stamp["touched_files"][
            "sglang/srt/layers/activation.py"
        ]
    }
    assert materialize_system_overlay(
        bundle,
        competition_target=SGLANG_INFERENCE_SYSTEM_V1,
        arena_name=ARENA,
        cache_root=cache,
        stock_site_root=stock,
        sglang_version=PINNED_SGLANG,
    ) == dest


def test_overlay_tamper_and_writable_candidate_fail_closed(tmp_path):
    bundle = _bundle(tmp_path)
    stock = _stock(tmp_path)
    cache = tmp_path / "cache"
    dest = materialize_system_overlay(
        bundle,
        competition_target=SGLANG_INFERENCE_SYSTEM_V1,
        arena_name=ARENA,
        cache_root=cache,
        stock_site_root=stock,
        sglang_version=PINNED_SGLANG,
    )
    with pytest.raises(SystemPatchError, match="read-only"):
        read_validated_system_overlay(
            bundle,
            competition_target=SGLANG_INFERENCE_SYSTEM_V1,
            arena_name=ARENA,
            cache_root=cache,
            stock_site_root=stock,
            sglang_version=PINNED_SGLANG,
            require_read_only=True,
            read_only_check=lambda _path: False,
        )

    (dest / "site/sglang/untouched.py").write_text("TAMPERED = True\n")
    with pytest.raises(SystemPatchError, match="tree differs"):
        read_validated_system_overlay(
            bundle,
            competition_target=SGLANG_INFERENCE_SYSTEM_V1,
            arena_name=ARENA,
            cache_root=cache,
            stock_site_root=stock,
            sglang_version=PINNED_SGLANG,
        )


def test_exact_context_mismatch_refuses_overlay(tmp_path):
    bundle = _bundle(
        tmp_path,
        patch_text=_diff(
            "NOT THE PINNED SOURCE\n",
            "VALUE = 2\n",
            "sglang/srt/layers/activation.py",
        ),
    )
    stock = _stock(tmp_path)
    with pytest.raises(SystemPatchError, match="context mismatch"):
        materialize_system_overlay(
            bundle,
            competition_target=SGLANG_INFERENCE_SYSTEM_V1,
            arena_name=ARENA,
            cache_root=tmp_path / "cache",
            stock_site_root=stock,
            sglang_version=PINNED_SGLANG,
        )


def test_identity_binds_stock_package_bytes(tmp_path):
    bundle = _bundle(tmp_path)
    stock = _stock(tmp_path)
    first = system_overlay_identity(
        bundle,
        competition_target=SGLANG_INFERENCE_SYSTEM_V1,
        arena_name=ARENA,
        stock_site_root=stock,
        sglang_version=PINNED_SGLANG,
    )
    (stock / "sglang/untouched.py").write_text("UNCHANGED = False\n")
    second = system_overlay_identity(
        bundle,
        competition_target=SGLANG_INFERENCE_SYSTEM_V1,
        arena_name=ARENA,
        stock_site_root=stock,
        sglang_version=PINNED_SGLANG,
    )
    assert first.cache_key != second.cache_key


def test_scheduler_role_activation_never_touches_driver_path(tmp_path):
    dest = tmp_path / "overlay"
    (dest / "site/sglang").mkdir(parents=True)
    identity = SimpleNamespace(cache_key="abc123")

    def reader(**kwargs):
        assert kwargs["require_read_only"] is True
        return identity, {}, dest

    base_env = {
        "OPTIMA_SYSTEM_OVERLAY_ARMED": "1",
        "OPTIMA_SYSTEM_BUNDLE_PATH": "/bundle",
        "OPTIMA_SYSTEM_COMPETITION_TARGET": SGLANG_INFERENCE_SYSTEM_V1,
        "OPTIMA_SYSTEM_ARENA": ARENA,
        "OPTIMA_SYSTEM_OVERLAY_ROOT": str(tmp_path / "cache"),
        "OPTIMA_SYSTEM_EXPECTED_CACHE_KEY": "abc123",
        "OPTIMA_SYSTEM_DRIVER_PID": "100",
    }
    driver_path = ["/stock/site"]
    assert activate_scheduler_overlay(
        env=dict(base_env), pid=100, parent_pid=1, modules={},
        sys_path=driver_path, reader=reader,
    ) is None
    assert driver_path == ["/stock/site"]

    scheduler_env = {
        **base_env,
        "OPTIMA_SYSTEM_PROCESS_ROLE": "scheduler",
        "OPTIMA_SYSTEM_ROLE_PARENT_PID": "100",
    }
    scheduler_path = ["/stock/site"]
    site = activate_scheduler_overlay(
        env=scheduler_env, pid=101, parent_pid=100, modules={},
        sys_path=scheduler_path, reader=reader,
    )
    assert site == (dest / "site").resolve()
    assert scheduler_path[0] == str((dest / "site").resolve())
    assert scheduler_env["OPTIMA_SYSTEM_OVERLAY_ACTIVE"] == "abc123"

    deferred_env = dict(scheduler_env)
    deferred_env.pop("OPTIMA_SYSTEM_OVERLAY_ACTIVE")
    deferred_path = ["/stock/site"]
    assert activate_scheduler_overlay(
        env=deferred_env, pid=101, parent_pid=100, modules={},
        sys_path=deferred_path, reader=reader, defer_to_import=True,
    ) == site
    assert deferred_path == ["/stock/site"]
    assert "OPTIMA_SYSTEM_OVERLAY_ACTIVE" not in deferred_env


def test_scheduler_finder_forces_overlay_after_spawn_path_reset(tmp_path, monkeypatch):
    overlay_site = tmp_path / "overlay/site"
    stock_site = tmp_path / "stock"
    (overlay_site / "sglang").mkdir(parents=True)
    (stock_site / "sglang").mkdir(parents=True)
    (overlay_site / "sglang/__init__.py").write_text("ORIGIN = 'overlay'\n")
    (stock_site / "sglang/__init__.py").write_text("ORIGIN = 'stock'\n")
    receipts = tmp_path / "receipts"
    env = {
        "OPTIMA_SYSTEM_COMPETITION_TARGET": SGLANG_INFERENCE_SYSTEM_V1,
        "OPTIMA_SYSTEM_ARENA": ARENA,
    }
    finder = _SchedulerOverlayFinder(overlay_site, "cache-key", env)
    monkeypatch.setenv("OPTIMA_SEAM_RECEIPT_DIR", str(receipts))
    monkeypatch.delenv("OPTIMA_SYSTEM_OVERLAY_FAILED", raising=False)
    monkeypatch.delitem(sys.modules, "sglang", raising=False)
    # Model multiprocessing.spawn.prepare restoring the stock parent path after
    # site startup. The meta-path finder must remain authoritative.
    monkeypatch.setattr(sys, "path", [str(stock_site)])
    monkeypatch.setattr(sys, "meta_path", [finder, *sys.meta_path])
    module = importlib.import_module("sglang")
    try:
        assert module.ORIGIN == "overlay"
        assert Path(module.__file__).resolve() == (
            overlay_site / "sglang/__init__.py"
        ).resolve()
        assert sys.path[0] == str(overlay_site.resolve())
        assert env["PYTHONPATH"].split(os.pathsep)[0] == str(overlay_site.resolve())
        assert env["OPTIMA_SYSTEM_OVERLAY_ACTIVE"] == "cache-key"
        receipt = json.loads(next(receipts.glob("system_active.*.json")).read_text())
        assert receipt["module_origin"] == str(
            (overlay_site / "sglang/__init__.py").resolve()
        )
        assert receipt["pid"] > 0
    finally:
        sys.modules.pop("sglang", None)


def test_scheduler_receipt_is_post_loader_not_pre_import(tmp_path, monkeypatch):
    overlay_site = tmp_path / "overlay/site"
    stock_site = tmp_path / "stock"
    (overlay_site / "sglang").mkdir(parents=True)
    (stock_site / "sglang").mkdir(parents=True)
    (overlay_site / "sglang/__init__.py").write_text(
        "raise RuntimeError('package init failed')\n"
    )
    (stock_site / "sglang/__init__.py").write_text("ORIGIN = 'stock'\n")
    receipts = tmp_path / "receipts"
    env = {
        "OPTIMA_SYSTEM_COMPETITION_TARGET": SGLANG_INFERENCE_SYSTEM_V1,
        "OPTIMA_SYSTEM_ARENA": ARENA,
    }
    finder = _SchedulerOverlayFinder(overlay_site, "cache-key", env)
    monkeypatch.setenv("OPTIMA_SEAM_RECEIPT_DIR", str(receipts))
    monkeypatch.delenv("OPTIMA_SYSTEM_OVERLAY_FAILED", raising=False)
    monkeypatch.delitem(sys.modules, "sglang", raising=False)
    monkeypatch.setattr(sys, "path", [str(stock_site)])
    monkeypatch.setattr(sys, "meta_path", [finder, *sys.meta_path])
    with pytest.raises(RuntimeError, match="package init failed"):
        importlib.import_module("sglang")
    sys.modules.pop("sglang", None)
    with pytest.raises(ImportError, match="activation failed closed"):
        importlib.import_module("sglang")
    assert "OPTIMA_SYSTEM_OVERLAY_ACTIVE" not in env
    assert not receipts.exists()


def test_scheduler_role_pid_and_import_order_are_fail_closed(tmp_path):
    dest = tmp_path / "overlay"
    (dest / "site/sglang").mkdir(parents=True)
    identity = SimpleNamespace(cache_key="abc")
    reader = lambda **_kwargs: (identity, {}, dest)
    env = {
        "OPTIMA_SYSTEM_OVERLAY_ARMED": "1",
        "OPTIMA_SYSTEM_PROCESS_ROLE": "scheduler",
        "OPTIMA_SYSTEM_ROLE_PARENT_PID": "10",
        "OPTIMA_SYSTEM_DRIVER_PID": "10",
        "OPTIMA_SYSTEM_BUNDLE_PATH": "/bundle",
        "OPTIMA_SYSTEM_COMPETITION_TARGET": SGLANG_INFERENCE_SYSTEM_V1,
        "OPTIMA_SYSTEM_ARENA": ARENA,
        "OPTIMA_SYSTEM_OVERLAY_ROOT": str(tmp_path),
        "OPTIMA_SYSTEM_EXPECTED_CACHE_KEY": "abc",
    }
    with pytest.raises(RuntimeError, match="timing driver"):
        activate_scheduler_overlay(
            env=dict(env), pid=10, parent_pid=10, modules={},
            sys_path=[], reader=reader,
        )
    with pytest.raises(RuntimeError, match="parent mismatch"):
        activate_scheduler_overlay(
            env=dict(env), pid=11, parent_pid=99, modules={},
            sys_path=[], reader=reader,
        )
    with pytest.raises(RuntimeError, match="after SGLang import"):
        activate_scheduler_overlay(
            env=dict(env), pid=11, parent_pid=10,
            modules={"sglang": object()}, sys_path=[], reader=reader,
        )


def test_only_exact_sglang_scheduler_target_receives_role():
    def scheduler():
        pass

    scheduler.__module__ = "sglang.srt.managers.scheduler"
    scheduler.__qualname__ = "run_scheduler_process"
    assert role_for_process_target(scheduler) == "scheduler"

    def detokenizer():
        pass

    detokenizer.__module__ = "sglang.srt.managers.detokenizer_manager"
    detokenizer.__qualname__ = "run_detokenizer_process"
    assert role_for_process_target(detokenizer) is None


def test_process_hook_marks_only_spawn_window_and_restores_environment(monkeypatch):
    import multiprocessing
    import multiprocessing.process
    import os

    observed = []

    def scheduler():
        pass

    scheduler.__module__ = "sglang.srt.managers.scheduler"
    scheduler.__qualname__ = "run_scheduler_process"

    def fake_start(_process, *_args, **_kwargs):
        observed.append(
            (
                os.environ.get("OPTIMA_SYSTEM_PROCESS_ROLE"),
                os.environ.get("OPTIMA_SYSTEM_ROLE_PARENT_PID"),
            )
        )
        return "started"

    base = multiprocessing.process.BaseProcess
    monkeypatch.setattr(base, "start", fake_start)
    monkeypatch.setenv("OPTIMA_SYSTEM_OVERLAY_ARMED", "1")
    monkeypatch.delenv("OPTIMA_SYSTEM_PROCESS_ROLE", raising=False)
    monkeypatch.delenv("OPTIMA_SYSTEM_ROLE_PARENT_PID", raising=False)
    install_process_role_hook()

    process = SimpleNamespace(_target=scheduler, _start_method="spawn")
    assert base.start(process) == "started"
    assert observed == [("scheduler", str(os.getpid()))]
    assert "OPTIMA_SYSTEM_PROCESS_ROLE" not in os.environ
    assert "OPTIMA_SYSTEM_ROLE_PARENT_PID" not in os.environ

    with pytest.raises(RuntimeError, match="require a spawn scheduler"):
        base.start(SimpleNamespace(_target=scheduler, _start_method="fork"))

    monkeypatch.setattr(multiprocessing, "get_start_method", lambda **_kwargs: "spawn")
    assert base.start(SimpleNamespace(_target=scheduler, _start_method=None)) == "started"
    monkeypatch.setattr(multiprocessing, "get_start_method", lambda **_kwargs: "fork")
    with pytest.raises(RuntimeError, match="require a spawn scheduler"):
        base.start(SimpleNamespace(_target=scheduler, _start_method=None))


def test_armed_process_starts_cannot_inherit_scheduler_role_concurrently(monkeypatch):
    import multiprocessing.process
    import os

    scheduler_entered = threading.Event()
    release_scheduler = threading.Event()
    helper_finished = threading.Event()
    observed = []

    def scheduler():
        pass

    scheduler.__module__ = "sglang.srt.managers.scheduler"
    scheduler.__qualname__ = "run_scheduler_process"

    def helper():
        pass

    def fake_start(process, *_args, **_kwargs):
        observed.append((
            process.kind,
            os.environ.get("OPTIMA_SYSTEM_PROCESS_ROLE"),
            os.environ.get("OPTIMA_SYSTEM_ROLE_PARENT_PID"),
        ))
        if process.kind == "scheduler":
            scheduler_entered.set()
            assert release_scheduler.wait(2)
        else:
            helper_finished.set()
        return process.kind

    base = multiprocessing.process.BaseProcess
    monkeypatch.setattr(base, "start", fake_start)
    monkeypatch.setenv("OPTIMA_SYSTEM_OVERLAY_ARMED", "1")
    monkeypatch.delenv("OPTIMA_SYSTEM_PROCESS_ROLE", raising=False)
    monkeypatch.delenv("OPTIMA_SYSTEM_ROLE_PARENT_PID", raising=False)
    install_process_role_hook()
    scheduler_process = SimpleNamespace(
        _target=scheduler, _start_method="spawn", kind="scheduler"
    )
    helper_process = SimpleNamespace(
        _target=helper, _start_method="spawn", kind="helper"
    )
    scheduler_thread = threading.Thread(target=base.start, args=(scheduler_process,))
    helper_thread = threading.Thread(target=base.start, args=(helper_process,))
    scheduler_thread.start()
    assert scheduler_entered.wait(2)
    helper_thread.start()
    assert not helper_finished.wait(0.05)
    release_scheduler.set()
    scheduler_thread.join(2)
    helper_thread.join(2)
    assert not scheduler_thread.is_alive()
    assert not helper_thread.is_alive()
    assert observed == [
        ("scheduler", "scheduler", str(os.getpid())),
        ("helper", None, None),
    ]


def test_scheduler_bootstrap_consumes_one_child_role_markers(tmp_path, monkeypatch):
    import optima.system_overlay as system_overlay

    site = tmp_path / "site"
    (site / "sglang").mkdir(parents=True)
    monkeypatch.setattr(system_overlay, "install_process_role_hook", lambda: None)
    monkeypatch.setattr(
        system_overlay,
        "activate_scheduler_overlay",
        lambda **kwargs: site,
    )
    monkeypatch.setattr(sys, "meta_path", [])
    monkeypatch.setenv("OPTIMA_SYSTEM_OVERLAY_ARMED", "1")
    monkeypatch.setenv("OPTIMA_SYSTEM_PROCESS_ROLE", "scheduler")
    monkeypatch.setenv("OPTIMA_SYSTEM_ROLE_PARENT_PID", "123")
    monkeypatch.setenv("OPTIMA_SYSTEM_EXPECTED_CACHE_KEY", "cache-key")
    system_overlay.install()
    assert "OPTIMA_SYSTEM_PROCESS_ROLE" not in os.environ
    assert "OPTIMA_SYSTEM_ROLE_PARENT_PID" not in os.environ
    assert len(sys.meta_path) == 1
    assert isinstance(sys.meta_path[0], _SchedulerOverlayFinder)


def test_driver_stock_path_assertion(tmp_path):
    overlay = tmp_path / "overlay"
    assert driver_import_is_stock("/stock/sglang/__init__.py", overlay)
    assert not driver_import_is_stock(
        overlay / "site/sglang/__init__.py", overlay
    )
    lazy_engine = object()
    assert driver_module_is_stock(
        SimpleNamespace(__file__="/stock/sglang/__init__.py", Engine=lazy_engine),
        overlay,
    )
    assert not driver_module_is_stock(
        SimpleNamespace(
            __file__=overlay / "site/sglang/__init__.py",
            Engine=lazy_engine,
        ),
        overlay,
    )
    assert not driver_module_is_stock(SimpleNamespace(Engine=lazy_engine), overlay)
