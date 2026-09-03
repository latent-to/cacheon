"""Runtime-cache seeding: validator-provisioned warm bytes for fresh leases.

Every OCI lease mounts an empty tmpfs cache, so each boot recompiles runtime
JIT artifacts and concurrent TP ranks race the same output path (observed
2026-08-10: a half-written minfer fmha_sm100 plan .so killed a qualification
lane). These tests pin the seeding contract: a verbatim, symlink-free mirror
of a validator-owned FMHA tree into MInfer's exact cache path, read-only modes,
and the cache byte budget.
"""

import os
from pathlib import Path

import pytest

from cacheon.eval.oci_backend import OCIBackendError, _seed_runtime_cache


def _seed_tree(root: Path) -> Path:
    seed = root / "seed"
    plan = seed / "plan"
    plan.mkdir(parents=True)
    (plan / "fmha_sm100_plan.so").write_bytes(b"x" * 64)
    variant = seed / "0_0_0_0_1_1_4"
    variant.mkdir()
    (variant / "fmha_sm100.so").write_bytes(b"y" * 8)
    return seed


def _fresh_cache(root: Path) -> Path:
    cache = root / "cache"
    cache.mkdir()
    return cache


def test_seed_mirrors_tree_with_private_modes(tmp_path):
    seed = _seed_tree(tmp_path)
    cache = _fresh_cache(tmp_path)
    _seed_runtime_cache(
        seed, cache, uid=os.geteuid(), gid=os.getegid(), max_bytes=1 << 20
    )
    plan = (
        cache / "home" / ".cache" / "minfer" / "fmha_sm100" / "plan"
        / "fmha_sm100_plan.so"
    )
    assert plan.read_bytes() == b"x" * 64
    variant = (
        cache / "home" / ".cache" / "minfer" / "fmha_sm100"
        / "0_0_0_0_1_1_4" / "fmha_sm100.so"
    )
    assert variant.read_bytes() == b"y" * 8
    assert plan.stat().st_mode & 0o777 == 0o444
    assert plan.parent.stat().st_mode & 0o777 == 0o555
    assert (cache / "home").stat().st_mode & 0o777 == 0o700


def test_seed_refuses_symlink_members(tmp_path):
    seed = _seed_tree(tmp_path)
    (seed / "link").symlink_to(seed / "plan")
    with pytest.raises(OCIBackendError, match="symlink"):
        _seed_runtime_cache(
            seed,
            _fresh_cache(tmp_path),
            uid=os.geteuid(),
            gid=os.getegid(),
            max_bytes=1 << 20,
        )


def test_seed_refuses_planless_tree_of_unknown_entries(tmp_path):
    seed = tmp_path / "seed"
    variant = seed / "0_0_0_0_1_1_4"
    variant.mkdir(parents=True)
    (variant / "fmha_sm100.so").write_bytes(b"y" * 8)
    with pytest.raises(OCIBackendError, match="unknown cache families"):
        _seed_runtime_cache(
            seed,
            _fresh_cache(tmp_path),
            uid=os.geteuid(),
            gid=os.getegid(),
            max_bytes=1 << 20,
        )


def _full_cache_seed(root: Path) -> Path:
    seed = root / "full-seed"
    (seed / "torchinductor" / "ab").mkdir(parents=True)
    (seed / "torchinductor" / "ab" / "graph.py").write_bytes(b"g" * 32)
    (seed / "triton").mkdir()
    (seed / "triton" / "kernel.cubin").write_bytes(b"k" * 16)
    (seed / "cuda").mkdir()
    (seed / "cuda" / "compute.bin").write_bytes(b"c" * 8)
    (seed / "xdg").mkdir()
    return seed


def test_full_cache_seed_installs_families_writable(tmp_path):
    seed = _full_cache_seed(tmp_path)
    cache = _fresh_cache(tmp_path)
    _seed_runtime_cache(
        seed, cache, uid=os.geteuid(), gid=os.getegid(), max_bytes=1 << 20
    )
    graph = cache / "torchinductor" / "ab" / "graph.py"
    assert graph.read_bytes() == b"g" * 32
    assert (cache / "triton" / "kernel.cubin").read_bytes() == b"k" * 16
    assert graph.stat().st_mode & 0o777 == 0o644
    assert graph.parent.stat().st_mode & 0o777 == 0o700
    assert (cache / "xdg").is_dir()


def test_full_cache_seed_refuses_unknown_family(tmp_path):
    seed = _full_cache_seed(tmp_path)
    evil = seed / "evil"
    evil.mkdir()
    (evil / "payload").write_bytes(b"z")
    with pytest.raises(OCIBackendError, match="unknown cache families.*evil"):
        _seed_runtime_cache(
            seed,
            _fresh_cache(tmp_path),
            uid=os.geteuid(),
            gid=os.getegid(),
            max_bytes=1 << 20,
        )


def test_full_cache_seed_refuses_empty_root(tmp_path):
    seed = tmp_path / "full-seed"
    seed.mkdir()
    with pytest.raises(OCIBackendError, match="no cache families"):
        _seed_runtime_cache(
            seed,
            _fresh_cache(tmp_path),
            uid=os.geteuid(),
            gid=os.getegid(),
            max_bytes=1 << 20,
        )


def test_full_cache_seed_refuses_file_free_families(tmp_path):
    seed = tmp_path / "full-seed"
    (seed / "xdg").mkdir(parents=True)
    (seed / "triton-home").mkdir()
    with pytest.raises(OCIBackendError, match="holds no files"):
        _seed_runtime_cache(
            seed,
            _fresh_cache(tmp_path),
            uid=os.geteuid(),
            gid=os.getegid(),
            max_bytes=1 << 20,
        )


def test_full_cache_seed_refuses_symlink_members(tmp_path):
    seed = _full_cache_seed(tmp_path)
    (seed / "triton" / "alias").symlink_to(seed / "triton" / "kernel.cubin")
    with pytest.raises(OCIBackendError, match="symlink"):
        _seed_runtime_cache(
            seed,
            _fresh_cache(tmp_path),
            uid=os.geteuid(),
            gid=os.getegid(),
            max_bytes=1 << 20,
        )


def test_full_cache_seed_enforces_budget(tmp_path):
    seed = _full_cache_seed(tmp_path)
    with pytest.raises(OCIBackendError, match="cache budget"):
        _seed_runtime_cache(
            seed,
            _fresh_cache(tmp_path),
            uid=os.geteuid(),
            gid=os.getegid(),
            max_bytes=16,
        )


def test_seed_refuses_file_free_tree(tmp_path):
    seed = tmp_path / "seed"
    (seed / "plan").mkdir(parents=True)
    with pytest.raises(OCIBackendError, match="holds no files"):
        _seed_runtime_cache(
            seed,
            _fresh_cache(tmp_path),
            uid=os.geteuid(),
            gid=os.getegid(),
            max_bytes=1 << 20,
        )


def test_seed_refuses_untrusted_root(tmp_path):
    with pytest.raises(OCIBackendError, match="trusted directory"):
        _seed_runtime_cache(
            tmp_path / "absent",
            _fresh_cache(tmp_path),
            uid=os.geteuid(),
            gid=os.getegid(),
            max_bytes=1 << 20,
        )


def test_seed_enforces_cache_budget(tmp_path):
    seed = _seed_tree(tmp_path)
    with pytest.raises(OCIBackendError, match="cache budget"):
        _seed_runtime_cache(
            seed,
            _fresh_cache(tmp_path),
            uid=os.geteuid(),
            gid=os.getegid(),
            max_bytes=16,
        )


def test_prebuild_config_validates_seed_root(tmp_path):
    from cacheon.eval.b300_screen_deployment import _prebuild_policy
    from cacheon.eval.oci_backend import OCIRuntimeResourcePolicy
    from cacheon.eval.oci_prebuild import OCIPrebuildConfig, OCIPrebuildError

    runtime = OCIRuntimeResourcePolicy(
        uid=max(1, os.getuid()),
        gid=max(1, os.getgid()),
        cpu_millis=8_000,
        memory_bytes=8 << 30,
        pids_limit=2_048,
        nofile_limit=32_768,
        cache_bytes=2 << 30,
        cache_inodes=10_000,
        tmpfs_bytes=1 << 30,
        shm_bytes=2 << 30,
        init_timeout_seconds=30.0,
        batch_timeout_seconds=30.0,
        container_python="/usr/local/bin/python3",
    )

    def build(seed_root):
        return OCIPrebuildConfig(
            docker_binary="/usr/bin/docker",
            recovery_root=tmp_path / "recovery",
            publication_root=tmp_path / "publications",
            seccomp_profile=tmp_path / "seccomp.json",
            executor_id="seedtest",
            policy=_prebuild_policy(runtime),
            runtime_seed_root=seed_root,
        )

    assert build(None).runtime_seed_root is None
    assert build(tmp_path / "seed").runtime_seed_root == tmp_path / "seed"
    with pytest.raises(OCIPrebuildError, match="runtime_seed_root"):
        build(Path("relative/seed"))
