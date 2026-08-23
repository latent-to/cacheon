"""Device, runtime, and OCI policy fixtures shared by the B300 lane suites.

Five suites each carried their own copy of these. The copies were verbatim
except for two things that genuinely differ — the screen lane's resource
policies are smaller than qualification's, and the reference-device suite runs a
different GPU on purpose — so those are named presets here rather than mode
flags on one builder. Anything a caller varies per test stays an argument.
"""

from __future__ import annotations

import hashlib
import os

from cacheon.arena_service import ArenaRuntimeIdentity
from cacheon.eval.device_state import GPUConfiguration
from cacheon.eval.oci_backend import OCIRuntimeResourcePolicy
from cacheon.eval.oci_prebuild import OCIPrebuildPolicy


def sha(label: str) -> str:
    """A deterministic stand-in digest, named by what it stands for."""

    return hashlib.sha256(label.encode()).hexdigest()


def arena_runtime() -> ArenaRuntimeIdentity:
    return ArenaRuntimeIdentity(
        arena_id="production-b300-tp4",
        runtime_digest=sha("runtime"),
        base_engine_digest=sha("base-engine"),
        validator_overlay_digest=sha("validator-overlay"),
        worker_distribution_digest=sha("worker-distribution"),
        model_revision_digest=sha("model-revision"),
        model_manifest_digest=sha("model-manifest"),
        model_content_digest=sha("model-content"),
        target_architecture="sm103",
        topology_class="nvlink-domain",
        topology_digest=sha("topology"),
        gpu_count=4,
        tensor_parallel_size=4,
    )


# Qualification runs a full sealed cell; the screen lane is a short routing
# check and is provisioned for that. Keeping both as presets preserves the
# distinction a single "default" would have quietly erased.
_RUNTIME_PRESETS = {
    "qualification": dict(
        memory_bytes=32 << 30,
        pids_limit=4_096,
        nofile_limit=65_536,
        cache_bytes=4 << 30,
        cache_inodes=100_000,
        shm_bytes=8 << 30,
        init_timeout_seconds=120.0,
        batch_timeout_seconds=60.0,
    ),
    "screen": dict(
        memory_bytes=8 << 30,
        pids_limit=2_048,
        nofile_limit=32_768,
        cache_bytes=2 << 30,
        cache_inodes=10_000,
        shm_bytes=2 << 30,
        init_timeout_seconds=30.0,
        batch_timeout_seconds=30.0,
    ),
}

_PREBUILD_PRESETS = {
    "qualification": dict(
        memory_bytes=32 << 30,
        pids_limit=4_096,
        stage_bytes=16 << 30,
        stage_inodes=100_000,
        timeout_seconds=7_200.0,
        native_compile_timeout_seconds=6_000,
    ),
    "screen": dict(
        memory_bytes=8 << 30,
        pids_limit=2_048,
        stage_bytes=4 << 30,
        stage_inodes=10_000,
        timeout_seconds=300.0,
        native_compile_timeout_seconds=240,
    ),
}


def runtime_policy(preset: str = "qualification") -> OCIRuntimeResourcePolicy:
    return OCIRuntimeResourcePolicy(
        uid=max(1, os.getuid()),
        gid=max(1, os.getgid()),
        cpu_millis=8_000,
        tmpfs_bytes=1 << 30,
        container_python="/usr/local/bin/python3",
        **_RUNTIME_PRESETS[preset],
    )


def prebuild_policy(
    runtime: OCIRuntimeResourcePolicy, preset: str = "qualification"
) -> OCIPrebuildPolicy:
    return OCIPrebuildPolicy(
        uid=runtime.uid,
        gid=runtime.gid,
        cpu_millis=8_000,
        tmpfs_bytes=1 << 30,
        container_python=runtime.container_python,
        build_path=("/usr/local/cuda/bin", "/usr/local/bin", "/usr/bin", "/bin"),
        build_tmpdir="/tmp",
        pinned_build_roots=("/usr/include", "/usr/lib", "/usr/local/cuda"),
        runtime_policy_digest=runtime.digest,
        **_PREBUILD_PRESETS[preset],
    )


def gpu(index: int = 0, model: str = "b300") -> GPUConfiguration:
    """One device of ``model``. The second model is not decoration.

    A generic evaluator that resolves identity from sealed inputs has to be
    exercised on more than one device profile, or "generic" is untested. The
    reference suite runs the RTX profile for exactly that reason.
    """

    if model == "rtx6000":
        return GPUConfiguration(
            physical_id=index,
            uuid=f"GPU-00000000-{index:04x}-0000-0000-{index:012x}",
            pci_bus_id=f"00000000:{index + 1:02x}:00.0",
            name="NVIDIA RTX PRO 6000 Blackwell Server Edition",
            memory_total_mib=98_304,
            driver_version="595.71.05",
            power_limit_mw=600_000,
            compute_mode="Default",
            persistence_mode="Enabled",
            application_graphics_clock_mhz=None,
            application_memory_clock_mhz=None,
            max_graphics_clock_mhz=2_100,
            max_memory_clock_mhz=4_000,
        )
    return GPUConfiguration(
        physical_id=index,
        uuid=f"GPU-00000000-{index:04x}-0000-0000-{index:012x}",
        pci_bus_id=f"00000000:{index + 1:02x}:00.0",
        name="NVIDIA B300 SXM6 AC",
        memory_total_mib=288_000,
        driver_version="600.10.01",
        power_limit_mw=1_000_000,
        compute_mode="Default",
        persistence_mode="Enabled",
        application_graphics_clock_mhz=None,
        application_memory_clock_mhz=None,
        max_graphics_clock_mhz=2_500,
        max_memory_clock_mhz=5_000,
    )
