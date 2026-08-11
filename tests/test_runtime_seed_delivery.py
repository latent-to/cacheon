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
