"""Content-addressed native-extension cache security invariants."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest


def _patcher():
    # Import is safe without OPTIMA_BUNDLE_PATH: the reviewed script's main is a no-op.
    return importlib.import_module("optima.patchers.build_cuda_ext")


def _bundle(root: Path, *, body: str = "// body\n", header: str = "#define V 1\n") -> Path:
    (root / "kernels").mkdir(parents=True)
    (root / "kernels" / "shim.py").write_text(
        "def entry(x, out):\n    out.copy_(x)\n"
    )
    (root / "kernels" / "native.cu").write_text(body)
    (root / "kernels" / "values.cuh").write_text(header)
    (root / "manifest.toml").write_text(
        'bundle_id = "miner-controlled-collision"\n'
        'abi_version = "optima-op-abi-v0"\n\n'
        '[[ops]]\n'
        'slot = "activation.silu_and_mul"\n'
        'source = "kernels/shim.py"\n'
        'entry = "entry"\n'
        'cuda_sources = ["kernels/native.cu", "kernels/values.cuh"]\n'
    )
    return root


@pytest.fixture()
def fake_cuda(monkeypatch, tmp_path):
    mod = _patcher()
    cache = tmp_path / "cuda-cache"
    monkeypatch.setenv("OPTIMA_CUDA_EXT_CACHE", str(cache))
    monkeypatch.setattr(mod.shutil, "which", lambda name: f"/fake/{name}")

    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=0: (10, 3))
    context = {
        "arch": "sm_103a",
        "nvcc": {"path": "/fake/nvcc", "sha256": "n" * 64, "version": "fake"},
        "ptxas": {"path": "/fake/ptxas", "sha256": "p" * 64, "version": "fake"},
        "torch_version": "fake",
        "cxx11_abi": 1,
        "nvcc_flags": list(mod._NVCC_FLAGS),
        "link_libraries": list(mod._LINK_LIBS),
    }
    monkeypatch.setattr(mod, "_runtime_context", lambda arch: context)

    builds: list[tuple[str, str]] = []

    def fake_compile(src, so, arch, module_name, runtime_context):
        builds.append((src.read_text(), module_name))
        so.write_bytes(f"{module_name}:{src.read_text()}".encode())

    monkeypatch.setattr(mod, "_compile", fake_compile)
    return mod, cache, context, builds


def _run(mod, monkeypatch, bundle: Path, phase: str) -> None:
    monkeypatch.setenv("OPTIMA_BUNDLE_PATH", str(bundle))
    monkeypatch.setenv("OPTIMA_REBUILD_PHASE", phase)
    monkeypatch.delenv("OPTIMA_BUNDLE_CONTENT_HASH", raising=False)
    if phase == "build":
        monkeypatch.setenv("OPTIMA_TARGET_GPU_ARCH", "sm_103a")
    mod.main()


def _stamps(cache: Path) -> list[Path]:
    return sorted(cache.rglob("artifact.json"))


def test_bundle_id_collision_and_header_change_get_distinct_artifacts(
    tmp_path, monkeypatch, fake_cuda
):
    mod, cache, _, builds = fake_cuda
    a = _bundle(tmp_path / "a", body="// same unit\n", header="#define V 1\n")
    b = _bundle(tmp_path / "b", body="// same unit\n", header="#define V 2\n")

    _run(mod, monkeypatch, a, "build")
    _run(mod, monkeypatch, b, "build")

    stamps = _stamps(cache)
    assert len(stamps) == 2
    rows = [json.loads(p.read_text()) for p in stamps]
    assert len({row["identity"]["bundle_hash"] for row in rows}) == 2
    assert len({row["artifact_id"] for row in rows}) == 2
    assert len({row["module_name"] for row in rows}) == 2
    assert all(row["module_name"].startswith("_optima_cuda_native_") for row in rows)
    assert len(builds) == 2


def test_load_rederives_identity_and_refuses_stamp_or_artifact_tampering(
    tmp_path, monkeypatch, fake_cuda
):
    mod, cache, _, _ = fake_cuda
    bundle = _bundle(tmp_path / "bundle")
    _run(mod, monkeypatch, bundle, "build")
    (stamp_path,) = _stamps(cache)
    row = json.loads(stamp_path.read_text())
    artifact = stamp_path.parent / f"{row['module_name']}.so"

    loaded = []
    monkeypatch.setattr(mod, "_load", lambda alias, name, so: loaded.append((alias, name, so)))
    _run(mod, monkeypatch, bundle, "load")
    assert loaded == [("native", row["module_name"], artifact)]

    row["identity"]["toolchain"]["arch"] = "sm_999a"
    stamp_path.write_text(json.dumps(row))
    with pytest.raises(RuntimeError, match="refusing stale/missing CUDA artifact"):
        _run(mod, monkeypatch, bundle, "load")

    # Build repairs an invalid cache entry, while load never does.
    _run(mod, monkeypatch, bundle, "build")
    artifact.write_bytes(artifact.read_bytes() + b"tampered")
    with pytest.raises(RuntimeError, match="artifact hash differs"):
        _run(mod, monkeypatch, bundle, "load")


def test_toolchain_or_flags_change_cannot_load_previous_artifact(
    tmp_path, monkeypatch, fake_cuda
):
    mod, cache, context, _ = fake_cuda
    bundle = _bundle(tmp_path / "bundle")
    _run(mod, monkeypatch, bundle, "build")
    assert len(_stamps(cache)) == 1

    changed = dict(context)
    changed["nvcc_flags"] = [*context["nvcc_flags"], "--different"]
    monkeypatch.setattr(mod, "_runtime_context", lambda arch: changed)
    with pytest.raises(RuntimeError, match="refusing stale/missing CUDA artifact"):
        _run(mod, monkeypatch, bundle, "load")
    assert len(_stamps(cache)) == 1  # load did not compile or mutate the cache


def test_duplicate_source_stems_are_rejected_before_compile(tmp_path, monkeypatch, fake_cuda):
    mod, _, _, builds = fake_cuda
    bundle = _bundle(tmp_path / "bundle")
    (bundle / "other").mkdir()
    (bundle / "other" / "native.cu").write_text("// other\n")
    manifest = bundle / "manifest.toml"
    manifest.write_text(manifest.read_text().replace(
        '"kernels/values.cuh"]', '"kernels/values.cuh", "other/native.cu"]'
    ))
    with pytest.raises(RuntimeError, match="share import alias"):
        _run(mod, monkeypatch, bundle, "build")
    assert builds == []


def test_native_import_alias_cannot_overwrite_existing_module(tmp_path):
    mod = _patcher()
    assert "json" in sys.modules
    with pytest.raises(RuntimeError, match="import alias collision"):
        mod._load("json", "_optima_cuda_unique_deadbeef", tmp_path / "missing.so")
