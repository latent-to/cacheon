"""Commission one sealed B300/TP4 screen worker from pod-owned authorities.

This is deployment composition, not a generic plugin interface.  It accepts a
fixed set of validator-owned JSON authorities, provisions exactly the selected
four-GPU lane, emits three canonical artifacts at fixed names, and later
reconstructs the same in-process service for the authenticated pod adapter.

Candidate data cannot select a Python module, class, command, argument, engine
option, host path, or output path through this module.  The only executable
authorities are the concrete Cacheon adapters assembled below.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import time
from dataclasses import dataclass, fields
from pathlib import Path, PurePosixPath
from typing import Callable, Mapping, Sequence

from cacheon.arena_service import (
    SCREEN_STAGES,
    ArenaCapacityPolicy,
    ArenaCandidateBinding,
    ArenaRuntimeIdentity,
    ArenaService,
    ArenaServiceManifest,
    NonCrownScreenPolicy,
    ScreenStagePolicy,
    ServingShape,
    WorkloadMixture,
    WorkloadRegime,
)
from cacheon.chain.evaluation_coordinator import WorkerReadiness
from cacheon.engine_tree import (
    inspect_contribution,
    materialize_engine_tree,
    reopen_materialized_engine_tree,
)
from cacheon.eval.b300_arena_provider import (
    B300ArenaServiceProvider,
    B300DeclaredQualificationAuthorities,
    B300QualificationLanePair,
    B300QualificationLanePolicy,
    B300ResidentScreenFactory,
    B300ResidentScreenLifetime,
    B300ScreenDeploymentAuthorities,
    b300_arena_provider_digest,
)
from cacheon.eval.b300_mainnet_worker import B300MainnetWorker
from cacheon.eval.b300_screen_stages import (
    B300BuildABIGraphScreenAdapter,
    B300ScreenExecutionPlan,
    B300StaticScreenAdapter,
    compose_b300_non_serving_screen_handlers,
)
from cacheon.eval.device_state import (
    DeviceStatePolicy,
    GPUConfiguration,
    provision_gpu_configurations,
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
from cacheon.eval.native_artifact import NativeArtifactLimits
from cacheon.eval.oci_backend import (
    OCIBackendConfig,
    OCIEngineExecutor,
    OCIRuntimeResourcePolicy,
    TrustedArenaModelMountReceipt,
    expected_runtime_preflight,
    runtime_identity_from_preflight,
)
from cacheon.eval.oci_outer_session import SessionExecutionPlan
from cacheon.eval.oci_prebuild import OCIPrebuildConfig, OCIPrebuildPolicy
from cacheon.eval.oci_process import OCIProcessManager
from cacheon.eval.oci_resident_session import ResidentSessionPlan
from cacheon.eval.oci_session_protocol import EngineSessionConfig, SlotAuditPolicy
from cacheon.eval.b300_screen_qualification_bridge import (
    B300ScreenQualificationBridgeError,
    derive_b300_screen_qualification,
)
from cacheon.eval.resident_queue import ScreenPolicy
from cacheon.eval.resident_screen_lane import (
    ResidentScreenLane,
    ResidentServingScreenStage,
    make_backend_lifetime_factory,
)
from cacheon.eval.runtime_preflight import (
    HOST_RECEIPT_SCHEMA,
    RuntimePreflightReceipt,
)
from cacheon.seams import SEAM_ADAPTERS
from cacheon.stack_identity import canonical_digest, canonical_json_bytes
from cacheon.stack_manifest import (
    EvaluationStackContext,
    EvaluationStackManifest,
    ProposalContributionRef,
)
from cacheon.stack_plan import plan_candidate_stack
from cacheon.target_catalog import TargetCatalog, default_target_catalog
from cacheon._strict import require_digest


DEPLOYMENT_SCHEMA = "cacheon-b300-screen-deployment-v2"
DEPLOYMENT_FILE = "screen-deployment.json"
MANIFEST_FILE = "arena-service-manifest.json"
READINESS_FILE = "worker-readiness.json"
MATERIALIZATION_SCHEMA = "cacheon-b300-screen-materialization-v1"
ARCHITECTURE = "sm103"
GPU_COUNT = 4
TP_SIZE = 4
DEFAULT_OUTPUT_ROOT = Path("/data/cacheon-b300/remote-worker/commissioned")


class B300ScreenDeploymentError(RuntimeError):
    """A commissioned authority is missing, mutable, or inconsistent."""


def _digest(value: object, field: str) -> str:
    return require_digest(value, field=field, error=B300ScreenDeploymentError)


def _canonical_bytes(value: object) -> bytes:
    try:
        return canonical_json_bytes(value) + b"\n"
    except (TypeError, ValueError) as exc:
        raise B300ScreenDeploymentError(
            f"deployment authority is not canonical JSON data: {exc}"
        ) from None


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_json(path_value: str | os.PathLike[str], field: str) -> tuple[Path, dict[str, object], str]:
    path = Path(path_value)
    if not path.is_absolute() or path.is_symlink():
        raise B300ScreenDeploymentError(f"{field} must be an absolute non-symlink file")
    try:
        before = path.stat()
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise B300ScreenDeploymentError(
                f"{field} must be a single-linked regular file"
            )
        raw = path.read_bytes()
        after = path.stat()
    except B300ScreenDeploymentError:
        raise
    except OSError as exc:
        raise B300ScreenDeploymentError(f"cannot read {field}: {exc}") from None
    stable = ("st_dev", "st_ino", "st_mode", "st_nlink", "st_size", "st_mtime_ns")
    if any(getattr(before, name) != getattr(after, name) for name in stable):
        raise B300ScreenDeploymentError(f"{field} changed while being read")
    if not raw or len(raw) > 64 << 20:
        raise B300ScreenDeploymentError(f"{field} is empty or exceeds its byte bound")
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_no_duplicate_pairs,
            parse_constant=lambda item: (_ for _ in ()).throw(
                ValueError(f"invalid number {item}")
            ),
        )
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise B300ScreenDeploymentError(f"{field} is malformed JSON: {exc}") from None
    if type(value) is not dict:
        raise B300ScreenDeploymentError(f"{field} must be a JSON object")
    return path.resolve(strict=True), value, hashlib.sha256(raw).hexdigest()


def _no_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, child in pairs:
        if key in value:
            raise ValueError(f"duplicate key {key!r}")
        value[key] = child
    return value


def _mapping(value: object, field: str) -> dict[str, object]:
    if type(value) is not dict:
        raise B300ScreenDeploymentError(f"{field} must be a JSON object")
    return value


def _text(value: object, field: str, *, maximum: int = 512) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value.strip() != value
        or len(value) > maximum
        or any(character in value for character in "\x00\r\n")
    ):
        raise B300ScreenDeploymentError(f"{field} is not canonical text")
    return value


def _integer(value: object, field: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise B300ScreenDeploymentError(f"{field} is not a bounded integer")
    return value


def _absolute_path(value: object, field: str) -> Path:
    raw = _text(value, field, maximum=4096)
    path = PurePosixPath(raw)
    if not path.is_absolute() or ".." in path.parts or "." in path.parts or str(path) != raw:
        raise B300ScreenDeploymentError(f"{field} is not a canonical absolute path")
    return Path(raw)


def _authority_ref(path: Path, digest: str) -> dict[str, str]:
    return {"path": str(path), "sha256": _digest(digest, "authority SHA-256")}


def _prepare_private_root(path: Path) -> Path:
    if not path.is_absolute() or path.is_symlink():
        raise B300ScreenDeploymentError("output root must be absolute and not a symlink")
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    root = path.resolve(strict=True)
    info = root.stat()
    if not stat.S_ISDIR(info.st_mode) or stat.S_IMODE(info.st_mode) & 0o077:
        raise B300ScreenDeploymentError("output root must be a private directory")
    return root


def _atomic_canonical(path: Path, value: object) -> None:
    raw = _canonical_bytes(value)
    if path.exists():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != raw:
            raise B300ScreenDeploymentError(
                f"refusing to replace differing commissioned artifact {path.name}"
            )
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(temporary, flags, 0o400)
        with os.fdopen(fd, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except OSError as exc:
        raise B300ScreenDeploymentError(
            f"cannot publish commissioned artifact {path.name}: {exc}"
        ) from None
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _find_preflight(value: object) -> dict[str, object]:
    """Find the sole canonical host preflight inside a sealed authority receipt."""

    found: list[dict[str, object]] = []

    def visit(item: object, depth: int) -> None:
        if depth > 16:
            raise B300ScreenDeploymentError("device authority nesting exceeds policy")
        if type(item) is dict:
            if item.get("schema") == HOST_RECEIPT_SCHEMA:
                found.append(item)
            for child in item.values():
                visit(child, depth + 1)
        elif type(item) is list:
            for child in item:
                visit(child, depth + 1)

    visit(value, 0)
    unique = {json.dumps(row, sort_keys=True, separators=(",", ":")): row for row in found}
    if len(unique) != 1:
        raise B300ScreenDeploymentError(
            "device execution authority must contain one canonical runtime preflight"
        )
    return next(iter(unique.values()))


def _runtime_preflight(row_value: object) -> RuntimePreflightReceipt:
    row = _mapping(row_value, "runtime preflight")
    worker = _mapping(row.get("worker"), "runtime preflight worker")
    python = _mapping(row.get("python"), "runtime preflight python")
    packages = _mapping(row.get("packages"), "runtime preflight packages")
    cuda = _mapping(row.get("cuda"), "runtime preflight cuda")
    try:
        receipt = RuntimePreflightReceipt(
            schema=row["schema"],
            requested_image=row["requested_image"],
            image_digest=row["image_digest"],
            local_image_id=row["local_image_id"],
            repo_digests=tuple(row["repo_digests"]),
            oci_platform=row["oci_platform"],
            platform_digest=row["platform_digest"],
            docker_binary=row["docker_binary"],
            uid=row["uid"],
            gid=row["gid"],
            sglang_version=row["sglang_version"],
            worker_distribution=worker["distribution"],
            worker_version=worker["version"],
            worker_distribution_digest=worker["digest"],
            worker_file_count=worker["file_count"],
            worker_total_bytes=worker["total_bytes"],
            python_implementation=python["implementation"],
            python_executable=python["executable"],
            python_version=python["version"],
            python_abi=python["abi"],
            python_platform=python["platform"],
            machine=python["machine"],
            package_versions=tuple(sorted(packages.items())),
            cudart_library=cuda["cudart_library"],
            cuda_visible_devices=cuda["cuda_visible_devices"],
            nvidia_visible_devices=cuda["nvidia_visible_devices"],
            security_argv_sha256=row["security_argv_sha256"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise B300ScreenDeploymentError(
            f"runtime preflight authority is malformed: {type(exc).__name__}"
        ) from None
    if receipt.schema != HOST_RECEIPT_SCHEMA:
        raise B300ScreenDeploymentError("runtime preflight schema differs")
    return receipt


def _gpu_from_dict(row_value: object) -> GPUConfiguration:
    row = _mapping(row_value, "GPU configuration")
    expected = {field.name for field in fields(GPUConfiguration)}
    if set(row) != expected:
        raise B300ScreenDeploymentError("GPU configuration fields differ")
    try:
        return GPUConfiguration(**row)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise B300ScreenDeploymentError(
            f"GPU configuration is invalid: {exc}"
        ) from None


def _device_policy(gpus: tuple[GPUConfiguration, ...]) -> DeviceStatePolicy:
    if (
        len(gpus) != GPU_COUNT
        or tuple(gpu.physical_id for gpu in gpus)
        != tuple(sorted(gpu.physical_id for gpu in gpus))
        or any("B300" not in gpu.name.upper() for gpu in gpus)
        or len({gpu.max_memory_clock_mhz for gpu in gpus}) != 1
    ):
        raise B300ScreenDeploymentError(
            "selected lane must contain exactly four ordered B300 configurations"
        )
    return DeviceStatePolicy(
        expected_gpus=gpus,
        maximum_temperature_c=90,
        maximum_gpu_utilization_percent=5,
        maximum_memory_utilization_percent=5,
        allowed_active_pstates=("P0",),
        active_maximum_graphics_clock_mhz=max(
            gpu.max_graphics_clock_mhz for gpu in gpus
        ),
        active_memory_clock_mhz=gpus[0].max_memory_clock_mhz,
        active_maximum_power_draw_mw=max(gpu.power_limit_mw for gpu in gpus),
        active_require_process_on_every_gpu=True,
        required_consecutive_idle_samples=2,
        poll_interval_s=0.05,
        ready_poll_interval_s=0.05,
        drain_timeout_s=180.0,
        maximum_samples=1024,
    )


def _runtime_policy(preflight: RuntimePreflightReceipt) -> OCIRuntimeResourcePolicy:
    return OCIRuntimeResourcePolicy(
        uid=preflight.uid,
        gid=preflight.gid,
        cpu_millis=96_000,
        memory_bytes=1 << 40,
        pids_limit=65_536,
        nofile_limit=262_144,
        cache_bytes=64 << 30,
        cache_inodes=1_000_000,
        tmpfs_bytes=16 << 30,
        shm_bytes=128 << 30,
        init_timeout_seconds=900.0,
        batch_timeout_seconds=600.0,
        container_python=preflight.python_executable,
    )


def _prebuild_policy(runtime: OCIRuntimeResourcePolicy) -> OCIPrebuildPolicy:
    return OCIPrebuildPolicy(
        uid=runtime.uid,
        gid=runtime.gid,
        cpu_millis=96_000,
        memory_bytes=512 << 30,
        pids_limit=16_384,
        tmpfs_bytes=256 << 30,
        stage_bytes=16 << 30,
        stage_inodes=500_000,
        timeout_seconds=7_200.0,
        native_compile_timeout_seconds=6_000,
        container_python=runtime.container_python,
        build_path=("/usr/local/cuda/bin", "/usr/local/bin", "/usr/bin", "/bin"),
        build_tmpdir="/tmp",
        pinned_build_roots=("/usr",),
        runtime_policy_digest=runtime.digest,
    )


def _seccomp_path() -> Path:
    from cacheon.eval import oci_backend

    path = Path(oci_backend.__file__).with_name("seccomp_moby_v0_2_1.json")
    if path.is_symlink() or not path.is_file():
        raise B300ScreenDeploymentError("fixed seccomp profile is unavailable")
    return path.resolve(strict=True)


def _backend_config(
    root: Path,
    preflight: RuntimePreflightReceipt,
    *,
    executor_id: str,
    runtime_seed_root: Path | None = None,
) -> OCIBackendConfig:
    runtime = _runtime_policy(preflight)
    return OCIBackendConfig(
        OCIPrebuildConfig(
            docker_binary=preflight.docker_binary,
            recovery_root=root / "oci" / executor_id,
            publication_root=root / "native-publications",
            seccomp_profile=_seccomp_path(),
            executor_id=executor_id,
            policy=_prebuild_policy(runtime),
            runtime_seed_root=runtime_seed_root,
        ),
        runtime,
        NativeArtifactLimits(),
    )


def _build_executor(
    root: Path,
    preflight: RuntimePreflightReceipt,
    device_policy: DeviceStatePolicy,
    *,
    executor_id: str = "b300-screen",
    runtime_seed_root: Path | None = None,
) -> OCIEngineExecutor:
    config = _backend_config(
        root,
        preflight,
        executor_id=executor_id,
        runtime_seed_root=runtime_seed_root,
    )
    manager = OCIProcessManager(
        docker_binary=config.prebuild.docker_binary,
        recovery_root=config.prebuild.recovery_root,
        executor_id=config.prebuild.executor_id,
    )
    return OCIEngineExecutor(config, device_policy, manager=manager)


def _catalog_specs(catalog: TargetCatalog) -> dict[str, str]:
    rows = catalog.snapshot().get("targets")
    if not isinstance(rows, list):  # pragma: no cover - validator table invariant
        raise B300ScreenDeploymentError("target catalog snapshot is malformed")
    return {
        str(row["target_id"]): catalog.target_spec_digest(str(row["target_id"]))
        for row in rows
        if isinstance(row, dict) and isinstance(row.get("target_id"), str)
    }


def _seam_bindings(target_members: tuple[str, ...]) -> tuple[str, ...]:
    members = set(target_members)
    return tuple(
        sorted(
            {
                adapter.binding_id
                for adapter in SEAM_ADAPTERS
                if adapter.binding_id is not None
                and members.intersection(adapter.slots)
            }
        )
    )


def _engine_config(
    target_members: tuple[str, ...],
    *,
    disable_cuda_graph: bool,
) -> EngineSessionConfig:
    bindings = _seam_bindings(target_members)
    kwargs: dict[str, object] = {
        "chunked_prefill_size": 4096,
        "context_length": 8192,
        "cuda_graph_backend_prefill": "disabled",
        "kv_cache_dtype": "auto",
        "page_size": 128,
        "quantization": "modelopt_fp4",
        "trust_remote_code": True,
    }
    if "arfusion" in bindings:
        kwargs["disable_radix_cache"] = True
        kwargs["enable_flashinfer_allreduce_fusion"] = True
    if not disable_cuda_graph:
        # SGLang's default 300s scheduler watchdog SIGKILLs ranks mid CUDA-graph
        # capture on a live resident loop (measured 2026-07-20; reproduced on
        # mainnet FIFO recommission 2026-08-04 as outer_oci_client_returncode=137).
        kwargs["watchdog_timeout"] = 1800
    return EngineSessionConfig(
        model_path="/cacheon/input/model",
        dtype="bfloat16",
        deterministic=False,
        attention_backend=None,
        disable_cuda_graph=disable_cuda_graph,
        mem_fraction_static=0.75,
        log_level="error",
        max_running_requests=32,
        tp_size=TP_SIZE,
        moe_runner_backend="flashinfer_cutlass",
        disable_custom_all_reduce=True,
        engine_kwargs=kwargs,
        seam_bindings=bindings,
    )


def _native_build(
    tree_digest: str,
    preflight: RuntimePreflightReceipt,
    policy: OCIPrebuildPolicy,
) -> NativeBuildSpec:
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
            dependency_policy_digest=policy.dependency_policy_digest,
            target_architecture=ARCHITECTURE,
        ),
        target_architecture=ARCHITECTURE,
        dependency_policy_digest=policy.dependency_policy_digest,
    )


@dataclass(frozen=True)
class _CommissionedInputs:
    root: Path
    ready: dict[str, object]
    authority: dict[str, object]
    authority_refs: dict[str, dict[str, str]]
    preflight: RuntimePreflightReceipt
    gpus: tuple[GPUConfiguration, ...]
    qualification_gpus: tuple[GPUConfiguration, ...]
    device_policy: DeviceStatePolicy
    qualification_lane_pair: B300QualificationLanePair
    runtime: ArenaRuntimeIdentity
    topology_digest: str
    controller_distribution_digest: str
    model_root: Path
    prompt_batches: tuple[tuple[str, ...], ...]
    prompt_identity: dict[str, str]
    plan_resolver_digest: str
    evidence_policy_digest: str
    resident_factory_digest: str
    resident_resource_ids: tuple[str, ...]
    declared_qualification: B300DeclaredQualificationAuthorities
    qualification_commission: dict[str, object] | None
    runtime_seed_root: Path | None = None


class _CommissionedScreenPlanResolver:
    """Concrete, validator-owned resolver for candidate OCI screen plans."""

    def __init__(
        self,
        inputs: _CommissionedInputs,
        executor: OCIEngineExecutor,
        catalog: TargetCatalog,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if type(executor) is not OCIEngineExecutor or type(catalog) is not TargetCatalog:
            raise B300ScreenDeploymentError("screen resolver inputs are not exact")
        self.inputs = inputs
        self.executor = executor
        self.catalog = catalog
        self.clock = clock

    def __call__(
        self,
        manifest: ArenaServiceManifest,
        candidate: ArenaCandidateBinding,
    ) -> B300ScreenExecutionPlan:
        if (
            type(manifest) is not ArenaServiceManifest
            or type(candidate) is not ArenaCandidateBinding
            or manifest.runtime != self.inputs.runtime
        ):
            raise B300ScreenDeploymentError("screen plan request differs from commission")
        inspected = inspect_contribution(candidate.publication.root, catalog=self.catalog)
        reservation = candidate.reservation
        target = self.catalog.require(reservation.target_id)
        if (
            inspected.target_id != reservation.target_id
            or inspected.target_spec_digest
            != self.catalog.target_spec_digest(reservation.target_id)
            or inspected.selected_delta_digest != reservation.selected_delta_digest
            or target.members != reservation.target_members
        ):
            raise B300ScreenDeploymentError(
                "candidate contribution differs from finalized reservation"
            )
        proposal = ProposalContributionRef(
            target_id=inspected.target_id,
            target_spec_digest=inspected.target_spec_digest,
            artifact_digest=candidate.publication.content_hash,
            selected_payload_digest=inspected.selected_payload_digest,
            attribution_digest=canonical_digest(
                "cacheon.eval.b300-screen-attribution.v1",
                {
                    "candidate_digest": candidate.digest,
                    "publication_digest": candidate.publication.digest,
                    "reservation_digest": reservation.reservation_digest,
                    "service_digest": manifest.digest,
                },
            ),
        )
        if proposal.selected_delta_digest != reservation.selected_delta_digest:
            raise B300ScreenDeploymentError("proposal changed selected delta identity")
        context = EvaluationStackContext(
            runtime_digest=self.inputs.runtime.runtime_digest,
            base_engine_digest=self.inputs.runtime.base_engine_digest,
            arena_digest=manifest.digest,
            catalog_snapshot=self.catalog.snapshot(),
            catalog_digest=self.catalog.digest,
            target_spec_digests=_catalog_specs(self.catalog),
        )
        incumbent = EvaluationStackManifest(
            runtime_digest=context.runtime_digest,
            base_engine_digest=context.base_engine_digest,
            arena_digest=context.arena_digest,
            catalog_snapshot=self.catalog.snapshot(),
            catalog_digest=self.catalog.digest,
            entries={},
        )
        stack = plan_candidate_stack(
            incumbent,
            proposal,
            catalog=self.catalog,
            expected_context=context,
        )
        trees_root = self.inputs.root / "engine-trees"
        trees_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        destination = trees_root / stack.digest
        if destination.exists():
            tree = reopen_materialized_engine_tree(destination)
            if tree.stack_digest != stack.digest:
                raise B300ScreenDeploymentError(
                    "existing engine tree names another candidate stack"
                )
        else:
            tree = materialize_engine_tree(
                stack,
                context=context,
                catalog=self.catalog,
                resolver={
                    ("proposal", proposal.artifact_digest): candidate.publication.root
                },
                destination=destination,
            )
        if tree.stack_digest != stack.digest:
            raise B300ScreenDeploymentError("materialized engine stack changed")

        policy = self.inputs.device_policy
        hardware = LogicalHardwareSpec(
            visible_gpu_count=GPU_COUNT,
            architecture=ARCHITECTURE,
            topology_class=self.inputs.runtime.topology_class,
            topology_digest=self.inputs.topology_digest,
            tp_size=TP_SIZE,
            ep_size=1,
            dp_size=1,
            device_policy_digest=policy.policy_sha256,
        )
        physical = PhysicalHardwareBinding(
            physical_gpu_ids=tuple(str(gpu.physical_id) for gpu in self.inputs.gpus),
            architecture=ARCHITECTURE,
            topology_class=self.inputs.runtime.topology_class,
            topology_digest=self.inputs.topology_digest,
            tp_size=TP_SIZE,
            ep_size=1,
            dp_size=1,
            device_policy_digest=policy.policy_sha256,
        )
        native = _native_build(
            tree.tree_digest,
            self.inputs.preflight,
            self.executor.config.prebuild.policy,
        )
        binding = TrustedLaunchBinding(
            materialized_tree_root=tree.root,
            controller_distribution_digest=self.inputs.controller_distribution_digest,
            native_build_spec=native,
            runtime_preflight_receipt=self.inputs.preflight,
            physical_hardware=physical,
        )
        arena_digest = manifest.digest
        mount = TrustedArenaModelMountReceipt.capture(
            self.inputs.model_root,
            arena_digest=arena_digest,
            model_revision_digest=self.inputs.runtime.model_revision_digest,
            model_manifest_digest=self.inputs.runtime.model_manifest_digest,
            model_content_digest=self.inputs.runtime.model_content_digest,
        )
        eager_config = _engine_config(target.members, disable_cuda_graph=True)
        graph_config = _engine_config(target.members, disable_cuda_graph=False)
        seccomp_digest = _file_sha256(self.executor.config.prebuild.seccomp_profile)

        def launch(config: EngineSessionConfig) -> EngineLaunchSpec:
            return EngineLaunchSpec(
                runtime_digest=self.inputs.runtime.runtime_digest,
                base_engine_digest=self.inputs.runtime.base_engine_digest,
                arena_digest=arena_digest,
                stack_digest=tree.stack_digest,
                tree_digest=tree.tree_digest,
                image_digest=self.inputs.preflight.image_digest,
                platform_digest=self.inputs.preflight.platform_digest,
                controller_distribution_digest=(
                    self.inputs.controller_distribution_digest
                ),
                worker_distribution_digest=(
                    self.inputs.preflight.worker_distribution_digest
                ),
                model_revision_digest=self.inputs.runtime.model_revision_digest,
                model_manifest_digest=self.inputs.runtime.model_manifest_digest,
                model_content_digest=self.inputs.runtime.model_content_digest,
                validator_overlay_digest=(
                    self.inputs.runtime.validator_overlay_digest
                ),
                engine_config_digest=config.digest,
                seccomp_policy_digest=seccomp_digest,
                resource_policy_digest=(
                    self.executor.config.prebuild.policy.resource_policy_digest
                ),
                native_build_spec_digest=native.digest,
                hardware=hardware,
            )

        eager_launch = launch(eager_config)
        graph_launch = launch(graph_config)
        audit = SlotAuditPolicy(
            validator_seed=canonical_digest(
                "cacheon.eval.b300-screen-audit-seed.v1",
                {
                    "candidate_digest": candidate.digest,
                    "device_execution_sha256": self.inputs.authority_refs[
                        "device_execution"
                    ]["sha256"],
                    "screen_attempt": candidate.screen_attempt,
                    "selection_policy_digest": self.inputs.prompt_identity[
                        "selection_policy_digest"
                    ],
                },
            )[:32],
            sample_rate_ppm=1_000_000,
            minimum_calls=1,
            expected_slots=target.members,
            expected_member_count=TP_SIZE,
        )
        batches = self.inputs.prompt_batches[:3]

        def session(
            engine_launch: EngineLaunchSpec,
            config: EngineSessionConfig,
        ) -> SessionExecutionPlan:
            return SessionExecutionPlan(
                launch_digest=engine_launch.digest,
                expected_engine_config_digest=config.digest,
                engine_config=config,
                expected_preflight=expected_runtime_preflight(
                    engine_launch, self.inputs.preflight
                ),
                prompt_batches=batches,
                warmup_count=1,
                conditioning_count=1,
                max_new_tokens=4,
                top_logprobs_num=0,
                temperature=0.0,
                audit_policy=audit,
            )

        deadline = float(self.clock()) + 4 * 60 * 60
        if not math.isfinite(deadline):
            raise B300ScreenDeploymentError("screen deadline clock is invalid")
        return B300ScreenExecutionPlan(
            service_digest=manifest.digest,
            candidate_digest=candidate.digest,
            screen_attempt=candidate.screen_attempt,
            selected_delta_digest=reservation.selected_delta_digest,
            eager_launch=eager_launch,
            graph_launch=graph_launch,
            binding=binding,
            model_mount=mount,
            eager_session=session(eager_launch, eager_config),
            graph_session=session(graph_launch, graph_config),
            deadline=deadline,
        )


def _resident_factory(
    inputs: _CommissionedInputs,
    executor: OCIEngineExecutor,
    catalog: TargetCatalog,
    manifest_provider: Callable[[], ArenaServiceManifest],
) -> B300ResidentScreenFactory:
    """Build one stock TP4 engine lifetime shared by queued arrivals."""

    if type(executor) is not OCIEngineExecutor or type(catalog) is not TargetCatalog:
        raise B300ScreenDeploymentError("resident factory inputs are not exact")
    if not callable(manifest_provider):
        raise B300ScreenDeploymentError("resident manifest provider is not callable")
    prompts = tuple(prompt for batch in inputs.prompt_batches[:1] for prompt in batch)

    def create() -> B300ResidentScreenLifetime:
        manifest = manifest_provider()
        if (
            type(manifest) is not ArenaServiceManifest
            or manifest.runtime != inputs.runtime
        ):
            raise B300ScreenDeploymentError(
                "resident service manifest differs from commission"
            )
        snapshot = catalog.snapshot()
        rows = snapshot.get("targets")
        if not isinstance(rows, list):
            raise B300ScreenDeploymentError("target catalog snapshot is malformed")
        target_members = tuple(
            sorted(
                {
                    member
                    for row in rows
                    if isinstance(row, dict)
                    for member in row.get("members", ())
                    if isinstance(member, str)
                }
            )
        )
        if not target_members:
            raise B300ScreenDeploymentError("resident target member set is empty")
        context = EvaluationStackContext(
            runtime_digest=inputs.runtime.runtime_digest,
            base_engine_digest=inputs.runtime.base_engine_digest,
            arena_digest=manifest.digest,
            catalog_snapshot=snapshot,
            catalog_digest=catalog.digest,
            target_spec_digests=_catalog_specs(catalog),
        )
        stock = EvaluationStackManifest(
            runtime_digest=context.runtime_digest,
            base_engine_digest=context.base_engine_digest,
            arena_digest=context.arena_digest,
            catalog_snapshot=snapshot,
            catalog_digest=catalog.digest,
            entries={},
        )
        trees_root = inputs.root / "engine-trees"
        trees_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        destination = trees_root / f"resident-stock-{stock.digest}"
        if destination.exists():
            tree = reopen_materialized_engine_tree(destination)
        else:
            tree = materialize_engine_tree(
                stock,
                context=context,
                catalog=catalog,
                resolver={},
                destination=destination,
            )
        if tree.stack_digest != stock.digest or tree.runtime_manifest is not None:
            raise B300ScreenDeploymentError(
                "resident stock tree differs from the empty commissioned stack"
            )

        policy = inputs.device_policy
        hardware = LogicalHardwareSpec(
            visible_gpu_count=GPU_COUNT,
            architecture=ARCHITECTURE,
            topology_class=inputs.runtime.topology_class,
            topology_digest=inputs.topology_digest,
            tp_size=TP_SIZE,
            ep_size=1,
            dp_size=1,
            device_policy_digest=policy.policy_sha256,
        )
        physical = PhysicalHardwareBinding(
            physical_gpu_ids=tuple(str(gpu.physical_id) for gpu in inputs.gpus),
            architecture=ARCHITECTURE,
            topology_class=inputs.runtime.topology_class,
            topology_digest=inputs.topology_digest,
            tp_size=TP_SIZE,
            ep_size=1,
            dp_size=1,
            device_policy_digest=policy.policy_sha256,
        )
        native = _native_build(
            tree.tree_digest,
            inputs.preflight,
            executor.config.prebuild.policy,
        )
        binding = TrustedLaunchBinding(
            materialized_tree_root=tree.root,
            controller_distribution_digest=inputs.controller_distribution_digest,
            native_build_spec=native,
            runtime_preflight_receipt=inputs.preflight,
            physical_hardware=physical,
        )
        config = _engine_config(target_members, disable_cuda_graph=False)
        launch = EngineLaunchSpec(
            runtime_digest=inputs.runtime.runtime_digest,
            base_engine_digest=inputs.runtime.base_engine_digest,
            arena_digest=manifest.digest,
            stack_digest=tree.stack_digest,
            tree_digest=tree.tree_digest,
            image_digest=inputs.preflight.image_digest,
            platform_digest=inputs.preflight.platform_digest,
            controller_distribution_digest=inputs.controller_distribution_digest,
            worker_distribution_digest=inputs.preflight.worker_distribution_digest,
            model_revision_digest=inputs.runtime.model_revision_digest,
            model_manifest_digest=inputs.runtime.model_manifest_digest,
            model_content_digest=inputs.runtime.model_content_digest,
            validator_overlay_digest=inputs.runtime.validator_overlay_digest,
            engine_config_digest=config.digest,
            seccomp_policy_digest=_file_sha256(
                executor.config.prebuild.seccomp_profile
            ),
            resource_policy_digest=(
                executor.config.prebuild.policy.resource_policy_digest
            ),
            native_build_spec_digest=native.digest,
            hardware=hardware,
        )
        mount = TrustedArenaModelMountReceipt.capture(
            inputs.model_root,
            arena_digest=manifest.digest,
            model_revision_digest=inputs.runtime.model_revision_digest,
            model_manifest_digest=inputs.runtime.model_manifest_digest,
            model_content_digest=inputs.runtime.model_content_digest,
        )
        plan = ResidentSessionPlan(
            launch_digest=launch.digest,
            expected_engine_config_digest=config.digest,
            engine_config=config,
            expected_preflight=expected_runtime_preflight(
                launch, inputs.preflight
            ),
            max_swaps=10_000,
            max_batches=100_000,
            max_new_tokens=4,
            top_logprobs_num=0,
            temperature=0.0,
        )
        swap_root = inputs.root / "resident-intake"
        # Owner retains write for host-side staging; other/execute lets the
        # non-root OCI --user traverse to a known digest. mode=0o700 made the
        # intake root pass ST_RDONLY preflight while digest lstat failed with
        # EACCES, surfaced as "staged swap bundle is absent or writable"
        # (mainnet FIFO 2026-08-04 after CUDA-graph capture completed).
        swap_root.mkdir(parents=True, exist_ok=True, mode=0o711)
        os.chmod(swap_root, 0o711)
        lifetime = make_backend_lifetime_factory(
            executor,
            launch,
            binding,
            mount,
            plan,
            swap_intake_root=swap_root,
            deadline_provider=lambda: float(executor.manager.clock())
            + 30 * 24 * 60 * 60,
        )
        lane = ResidentScreenLane(
            lifetime,
            prompts=prompts,
            policy=ScreenPolicy(max_candidates_per_lifetime=1_000),
            verdict_timeout_s=3600.0,
            close_timeout_s=1800.0,
        )
        stage = ResidentServingScreenStage(lane, swap_root)
        return B300ResidentScreenLifetime(stage, lane.close)

    return B300ResidentScreenFactory(
        inputs.resident_factory_digest,
        inputs.resident_resource_ids,
        create,
    )


@dataclass(frozen=True)
class _Composition:
    manifest: ArenaServiceManifest
    authorities: B300ScreenDeploymentAuthorities
    build_executor: OCIEngineExecutor
    resident_executor: OCIEngineExecutor
    pipeline: B300BuildABIGraphScreenAdapter

    def close(self) -> None:
        self.pipeline.close()
        self.build_executor.manager.close()
        self.resident_executor.manager.close()


def _compose(inputs: _CommissionedInputs) -> _Composition:
    catalog = default_target_catalog()
    build_executor = _build_executor(
        inputs.root,
        inputs.preflight,
        inputs.device_policy,
        executor_id="b300-screen-build",
        runtime_seed_root=inputs.runtime_seed_root,
    )
    resident_executor = _build_executor(
        inputs.root,
        inputs.preflight,
        inputs.device_policy,
        executor_id="b300-screen-resident",
        runtime_seed_root=inputs.runtime_seed_root,
    )
    resolver = _CommissionedScreenPlanResolver(inputs, build_executor, catalog)
    static = B300StaticScreenAdapter(catalog)
    pipeline = B300BuildABIGraphScreenAdapter(
        catalog=catalog,
        executor=build_executor,
        plan_resolver_digest=inputs.plan_resolver_digest,
        plan_resolver=resolver,
        evidence_policy_digest=inputs.evidence_policy_digest,
        evidence_root=inputs.root / "screen-evidence",
        execution_mode="resident",
    )
    handlers = compose_b300_non_serving_screen_handlers(
        static,
        pipeline,
        pipeline_resource_ids=("b300-screen-build",),
    )
    manifest_box: list[ArenaServiceManifest] = []
    authorities = B300ScreenDeploymentAuthorities(
        runtime_identity=inputs.runtime,
        screen_handlers=handlers,
        resident_screen_factory=_resident_factory(
            inputs,
            resident_executor,
            catalog,
            lambda: manifest_box[0],
        ),
        qualification=inputs.declared_qualification,
    )
    manifest = ArenaServiceManifest(
        runtime=inputs.runtime,
        workload=WorkloadMixture(
            prompt_corpus_digest=inputs.prompt_identity["sha256"],
            prompt_seed_scheme="sealed-m4l-v1",
            regimes=(
                WorkloadRegime(
                    "decode",
                    "decode",
                    500_000,
                    (ServingShape(256, 32, 32, 4),),
                ),
                WorkloadRegime(
                    "long-prefill",
                    "long_prefill",
                    500_000,
                    (ServingShape(8192, 4, 1, 4),),
                ),
            ),
        ),
        capacity=ArenaCapacityPolicy(32, 64, 1, 4, 4, 2, 2, 3),
        screens=NonCrownScreenPolicy(
            tuple(
                ScreenStagePolicy(stage, timeout)
                for stage, timeout in zip(
                    SCREEN_STAGES,
                    (60_000, 7_200_000, 1_800_000, 1_800_000, 3_600_000),
                    strict=True,
                )
            )
        ),
        qualification_policy_digest=(
            inputs.declared_qualification.qualification_policy_digest
        ),
        provider_digest=b300_arena_provider_digest(authorities),
    )
    manifest_box.append(manifest)
    return _Composition(
        manifest,
        authorities,
        build_executor,
        resident_executor,
        pipeline,
    )


def _prompt_batches(value: object) -> tuple[tuple[str, ...], ...]:
    prompt = _mapping(value, "prompt authority")
    raw = prompt.get("prompt_batches")
    if type(raw) is not list:
        raise B300ScreenDeploymentError("prompt authority has no prompt batches")
    try:
        batches = tuple(tuple(batch) for batch in raw)
    except TypeError:
        raise B300ScreenDeploymentError("prompt batches are not nested arrays") from None
    if (
        len(batches) < 3
        or any(not batch for batch in batches)
        or any(
            not isinstance(item, str)
            or not item
            or len(item) > 2_000_000
            or "\x00" in item
            for batch in batches
            for item in batch
        )
    ):
        raise B300ScreenDeploymentError(
            "prompt authority must contain at least three bounded nonempty batches"
        )
    return batches


def _prompt_identity(prompt: dict[str, object], sha256: str) -> dict[str, str]:
    return {
        "hidden_corpus_commitment": _digest(
            prompt.get("hidden_corpus_commitment"), "hidden corpus commitment"
        ),
        "hidden_judge_digest": _digest(
            prompt.get("hidden_judge_digest"), "hidden judge digest"
        ),
        "hidden_task_policy_digest": _digest(
            prompt.get("hidden_task_policy_digest"), "hidden task policy digest"
        ),
        "selection_policy_digest": _digest(
            prompt.get("selection_policy_digest"), "selection policy digest"
        ),
        "sha256": _digest(sha256, "prompt authority SHA-256"),
        "tokenizer_digest": _digest(
            prompt.get("tokenizer_digest"), "tokenizer digest"
        ),
    }


def _same_authority_identity(
    authority: dict[str, object],
    measurement: dict[str, object],
) -> None:
    for field in ("arena_id", "qualification_builder_digest"):
        if authority.get(field) != measurement.get(field):
            raise B300ScreenDeploymentError(
                f"authority and measurement differ at {field}"
            )
    for section, names in (
        ("topology", ("architecture", "gpu_count", "lane", "lane_digest", "tensor_parallel_size", "topology_class")),
        ("model", ("content_digest", "manifest_digest", "revision_digest", "root")),
        ("worker", ("base_engine_digest", "image", "local_image_id", "runtime_digest", "validator_overlay_digest", "worker_distribution_digest")),
        ("prompt", ("sha256",)),
        ("device_execution", ("sha256",)),
    ):
        left = _mapping(authority.get(section), f"authority {section}")
        right = _mapping(measurement.get(section), f"measurement {section}")
        if any(left.get(name) != right.get(name) for name in names):
            raise B300ScreenDeploymentError(
                f"authority and measurement differ at {section}"
            )


def _ready_lane(ready: dict[str, object]) -> tuple[int, ...]:
    lane = _mapping(ready.get("lane"), "READY lane")
    raw = lane.get("devices")
    if type(raw) is not list or any(type(row) is not int for row in raw):
        raise B300ScreenDeploymentError("READY lane devices are malformed")
    selected = tuple(raw)
    if (
        len(selected) != GPU_COUNT
        or selected != tuple(sorted(set(selected)))
        or lane.get("tensor_parallel_size") != TP_SIZE
    ):
        raise B300ScreenDeploymentError("READY lane must be one ordered TP4 lane")
    _digest(lane.get("lane_digest"), "READY lane digest")
    return selected


def _ready_gpu_ids(ready: dict[str, object]) -> tuple[int, ...]:
    gpu = _mapping(ready.get("gpu"), "READY GPU inventory")
    inventory = gpu.get("inventory")
    if gpu.get("count") != 8 or type(inventory) is not list or len(inventory) != 8:
        raise B300ScreenDeploymentError("READY receipt is not an eight-B300 pod")
    try:
        physical_ids = tuple(
            _integer(
                _mapping(row, "READY GPU row").get("index"),
                "READY GPU index",
            )
            for row in inventory
        )
    except B300ScreenDeploymentError:
        raise
    if physical_ids != tuple(sorted(set(physical_ids))):
        raise B300ScreenDeploymentError(
            "READY GPU inventory is not one canonical eight-device set"
        )
    return physical_ids


def _validate_ready_inventory(
    ready: dict[str, object],
    gpus: tuple[GPUConfiguration, ...],
) -> None:
    gpu = _mapping(ready.get("gpu"), "READY GPU inventory")
    inventory = gpu.get("inventory")
    if tuple(gpu.physical_id for gpu in gpus) != _ready_gpu_ids(ready):
        raise B300ScreenDeploymentError(
            "provisioned GPU set differs from the commissioned eight-device pod"
        )
    for configured in gpus:
        try:
            row = next(
                _mapping(value, "READY GPU row")
                for value in inventory  # type: ignore[union-attr]
                if _mapping(value, "READY GPU row").get("index")
                == configured.physical_id
            )
        except StopIteration:
            raise B300ScreenDeploymentError("READY GPU row is missing") from None
        if (
            row.get("index") != configured.physical_id
            or row.get("uuid") != configured.uuid
            or str(row.get("pci_bus_id", "")).lower() != configured.pci_bus_id
            or row.get("name") != configured.name
            or row.get("memory_mib") != configured.memory_total_mib
        ):
            raise B300ScreenDeploymentError(
                "provisioned GPU configuration differs from READY inventory"
            )


def _project_fmha_identity(authority: dict[str, object]) -> dict[str, object] | None:
    value = authority.get("fmha_cache_seed")
    if value is None:
        return None
    row = _mapping(value, "FMHA cache authority")
    return {
        "directory_count": _integer(row.get("directory_count"), "FMHA directory count"),
        "file_count": _integer(row.get("file_count"), "FMHA file count"),
        "plan_sha256": _digest(row.get("plan_sha256"), "FMHA plan SHA-256"),
        "total_bytes": _integer(row.get("total_bytes"), "FMHA total bytes"),
        "tree_sha256": _digest(row.get("tree_sha256"), "FMHA tree SHA-256"),
    }


def _derive_inputs(
    *,
    root: Path,
    ready: dict[str, object],
    authority: dict[str, object],
    measurement: dict[str, object],
    prompt: dict[str, object],
    authority_refs: dict[str, dict[str, str]],
    preflight: RuntimePreflightReceipt,
    gpus: tuple[GPUConfiguration, ...],
) -> _CommissionedInputs:
    _same_authority_identity(authority, measurement)
    selected = _ready_lane(ready)
    if (
        type(gpus) is not tuple
        or len(gpus) != 2 * GPU_COUNT
        or tuple(gpu.physical_id for gpu in gpus) != _ready_gpu_ids(ready)
        or any("B300" not in gpu.name.upper() for gpu in gpus)
    ):
        raise B300ScreenDeploymentError(
            "qualification identity requires the exact commissioned eight-B300 pair"
        )
    _validate_ready_inventory(ready, gpus)
    by_id = {gpu.physical_id: gpu for gpu in gpus}
    try:
        selected_gpus = tuple(by_id[physical_id] for physical_id in selected)
    except KeyError:
        raise B300ScreenDeploymentError(
            "commissioned screen lane is absent from the eight-device pair"
        ) from None
    complement_ids = tuple(
        physical_id for physical_id in _ready_gpu_ids(ready) if physical_id not in selected
    )
    if len(complement_ids) != GPU_COUNT:
        raise B300ScreenDeploymentError(
            "commissioned screen lane has no disjoint TP4 complement"
        )
    complement_gpus = tuple(by_id[physical_id] for physical_id in complement_ids)
    device_policy = _device_policy(selected_gpus)
    physical_lanes = sorted(
        (selected_gpus, complement_gpus),
        key=lambda lane: tuple(gpu.physical_id for gpu in lane),
    )
    lane_a_policy = _device_policy(physical_lanes[0])
    lane_b_policy = _device_policy(physical_lanes[1])
    qualification_lane_pair = B300QualificationLanePair(
        B300QualificationLanePolicy.from_device_policy("A", lane_a_policy),
        B300QualificationLanePolicy.from_device_policy("B", lane_b_policy),
    )

    topology = _mapping(authority.get("topology"), "topology authority")
    raw_lane = topology.get("lane")
    try:
        authority_lane = tuple(int(row) for row in raw_lane)  # type: ignore[union-attr]
    except (TypeError, ValueError):
        raise B300ScreenDeploymentError("topology authority lane is malformed") from None
    if (
        topology.get("architecture") != ARCHITECTURE
        or topology.get("gpu_count") != GPU_COUNT
        or topology.get("tensor_parallel_size") != TP_SIZE
        or authority_lane != selected
    ):
        raise B300ScreenDeploymentError(
            "sealed topology is not the commissioned sm103 TP4 lane"
        )
    authority_lane_digest = _digest(
        topology.get("lane_digest"), "topology authority lane digest"
    )
    topology_class = _text(topology.get("topology_class"), "topology class")
    # ``RuntimePreflightFacts.topology_digest`` is measured inside the OCI
    # lifetime from the visible TP lane's canonical ``nvidia-smi topo -m``
    # matrix.  The sealed authority's lane digest is that same-domain value.
    # Do not wrap it in a second deployment-identity domain: doing so makes an
    # otherwise exact live lane impossible to compare with the host policy.
    # READY's independently bound lane/inventory identity remains retained in
    # the deployment payload below and in the device execution policy.
    ready_lane = _mapping(ready.get("lane"), "READY lane")
    _digest(ready_lane.get("lane_digest"), "READY lane digest")
    topology_digest = authority_lane_digest

    worker = _mapping(authority.get("worker"), "worker authority")
    image = _text(worker.get("image"), "worker image")
    identity = runtime_identity_from_preflight(preflight)
    ready_worker_image = _text(ready.get("worker_image"), "READY worker image")
    if (
        image != ready_worker_image
        or image != preflight.requested_image
        or worker.get("local_image_id") != preflight.local_image_id
        or worker.get("runtime_digest") != identity.runtime_digest
        or worker.get("base_engine_digest") != identity.base_engine_digest
        or worker.get("validator_overlay_digest")
        != identity.validator_overlay_digest
        or worker.get("worker_distribution_digest")
        != preflight.worker_distribution_digest
    ):
        raise B300ScreenDeploymentError(
            "sealed worker authority differs from runtime preflight or READY"
        )
    model = _mapping(authority.get("model"), "model authority")
    ready_model = _mapping(ready.get("model"), "READY model")
    model_root = _absolute_path(model.get("root"), "model authority root")
    if (
        _absolute_path(ready_model.get("path"), "READY model root") != model_root
        or ready_model.get("content_digest") != model.get("content_digest")
        or ready_model.get("readonly_inventory_verified") is not True
    ):
        raise B300ScreenDeploymentError(
            "sealed model authority differs from commissioned READY model"
        )
    if model_root.is_symlink() or not model_root.is_dir():
        raise B300ScreenDeploymentError("commissioned model root is unavailable")

    source = _mapping(ready.get("source"), "READY source")
    runtime_root = _mapping(ready.get("runtime"), "READY runtime")
    # Current-pod commissions bind the exact runtime seed as runtime.path.
    # The earlier Lium bootstrap schema carried the same path in a structured
    # runtime_seed field, so accept that representation when replaying one.
    seed_value = ready.get("runtime_seed")
    if seed_value is None:
        seed_value = runtime_root.get("path")
    elif type(seed_value) is dict:
        seed_value = seed_value.get("path")
    runtime_seed_root = _absolute_path(seed_value, "READY runtime seed")
    controller_distribution_digest = _digest(
        source.get("tree_digest"), "READY source tree digest"
    )
    runtime = ArenaRuntimeIdentity(
        arena_id=_text(authority.get("arena_id"), "arena id"),
        runtime_digest=identity.runtime_digest,
        base_engine_digest=identity.base_engine_digest,
        validator_overlay_digest=identity.validator_overlay_digest,
        worker_distribution_digest=preflight.worker_distribution_digest,
        model_revision_digest=_digest(
            model.get("revision_digest"), "model revision digest"
        ),
        model_manifest_digest=_digest(
            model.get("manifest_digest"), "model manifest digest"
        ),
        model_content_digest=_digest(
            model.get("content_digest"), "model content digest"
        ),
        target_architecture=ARCHITECTURE,
        topology_class=topology_class,
        topology_digest=topology_digest,
        gpu_count=GPU_COUNT,
        tensor_parallel_size=TP_SIZE,
    )
    prompt_identity = _prompt_identity(
        prompt, authority_refs["prompt_authority"]["sha256"]
    )
    batches = _prompt_batches(prompt)
    catalog = default_target_catalog()
    # Calibration is qualification evidence bound by the sealed deployment
    # payload.  It cannot be part of the screen resolver identity: its context
    # names the arena manifest, whose provider identity names this resolver.
    policy_facts = {
        "catalog_digest": catalog.digest,
        "controller_distribution_digest": controller_distribution_digest,
        "device_execution_sha256": authority_refs["device_execution"]["sha256"],
        "device_policy_digest": device_policy.policy_sha256,
        "fmha_cache_identity": _project_fmha_identity(authority),
        "model_content_digest": runtime.model_content_digest,
        "prompt_identity": prompt_identity,
        "runtime_tree_digest": _digest(
            runtime_root.get("tree_digest"), "READY runtime tree digest"
        ),
        "topology_digest": topology_digest,
        "worker_distribution_digest": runtime.worker_distribution_digest,
    }
    plan_resolver_digest = canonical_digest(
        "cacheon.eval.b300-screen-plan-resolver.v2", policy_facts
    )
    evidence_policy_digest = canonical_digest(
        "cacheon.eval.b300-screen-evidence-policy.v1",
        {
            "audit": "all-rank-slot-audit",
            "evidence_root": "content-addressed",
            "host_regrade": True,
        },
    )
    resident_factory_digest = canonical_digest(
        "cacheon.eval.b300-resident-routing-factory.v1",
        {
            "candidate_limit_per_lifetime": 1_000,
            "engine_mode": "stock-tp4-graph-resident",
            "lifetime_deadline_seconds": 30 * 24 * 60 * 60,
            "native_rebuild_route": "qualification-waiver",
            "prompt_authority_sha256": prompt_identity["sha256"],
            "swappable_route": "all-rank-swap-recapture-read",
        },
    )

    try:
        declared, qualification_commission = derive_b300_screen_qualification(
            authority=authority,
            prompt_identity=prompt_identity,
            catalog=catalog,
            lane_pair=qualification_lane_pair,
            backend_config_factory=lambda executor_id: _backend_config(
                root, preflight, executor_id=executor_id
            ),
        )
    except B300ScreenQualificationBridgeError as exc:
        raise B300ScreenDeploymentError(str(exc)) from None
    return _CommissionedInputs(
        root=root,
        ready=ready,
        authority=authority,
        authority_refs=authority_refs,
        preflight=preflight,
        gpus=selected_gpus,
        qualification_gpus=gpus,
        device_policy=device_policy,
        qualification_lane_pair=qualification_lane_pair,
        runtime=runtime,
        topology_digest=topology_digest,
        controller_distribution_digest=controller_distribution_digest,
        model_root=model_root,
        prompt_batches=batches,
        prompt_identity=prompt_identity,
        plan_resolver_digest=plan_resolver_digest,
        evidence_policy_digest=evidence_policy_digest,
        resident_factory_digest=resident_factory_digest,
        resident_resource_ids=("b300-resident-screen-lane",),
        declared_qualification=declared,
        qualification_commission=qualification_commission,
        runtime_seed_root=runtime_seed_root,
    )


def _authority_inputs(
    *,
    ready_receipt: str | os.PathLike[str],
    authority_config: str | os.PathLike[str],
    measurement_config: str | os.PathLike[str],
    calibration_package: str | os.PathLike[str],
    calibration_projection_receipt: str | os.PathLike[str],
    prompt_authority: str | os.PathLike[str],
    output_root: str | os.PathLike[str],
    provisioner: Callable[..., tuple[GPUConfiguration, ...]] | None,
    provisioned_gpus: tuple[GPUConfiguration, ...] | None = None,
) -> _CommissionedInputs:
    root = _prepare_private_root(Path(output_root))
    ready_path, ready, ready_sha = _stable_json(ready_receipt, "READY receipt")
    authority_path, authority, authority_sha = _stable_json(
        authority_config, "authority config"
    )
    measurement_path, measurement, measurement_sha = _stable_json(
        measurement_config, "measurement config"
    )
    calibration_path, _calibration, calibration_sha = _stable_json(
        calibration_package, "calibration package"
    )
    projection_path, _projection, projection_sha = _stable_json(
        calibration_projection_receipt, "calibration projection receipt"
    )
    prompt_path, prompt, prompt_sha = _stable_json(
        prompt_authority, "prompt authority"
    )

    prompt_ref = _mapping(authority.get("prompt"), "prompt binding")
    device_ref = _mapping(authority.get("device_execution"), "device binding")
    calibration_ref = _mapping(authority.get("calibration"), "calibration binding")
    device_path_value = _absolute_path(
        device_ref.get("path"), "device execution path"
    )
    device_path, device_execution, device_sha = _stable_json(
        device_path_value, "device execution receipt"
    )
    if (
        prompt_path != _absolute_path(prompt_ref.get("path"), "prompt binding path").resolve(strict=True)
        or prompt_sha != _digest(prompt_ref.get("sha256"), "prompt binding SHA-256")
        or device_path
        != _absolute_path(device_ref.get("path"), "device binding path").resolve(strict=True)
        or device_sha
        != _digest(device_ref.get("sha256"), "device binding SHA-256")
        or calibration_path
        != _absolute_path(
            calibration_ref.get("package"), "calibration binding path"
        ).resolve(strict=True)
        or calibration_sha
        != _digest(
            calibration_ref.get("package_sha256"),
            "calibration binding SHA-256",
        )
    ):
        raise B300ScreenDeploymentError(
            "explicit sealed authority paths or SHA-256 values differ from config"
        )
    preflight = _runtime_preflight(_find_preflight(device_execution))
    selected = _ready_gpu_ids(ready)
    if (provisioner is None) == (provisioned_gpus is None):
        raise B300ScreenDeploymentError(
            "GPU configuration requires exactly one provisioner or sealed inventory"
        )
    if provisioner is not None:
        try:
            gpus = provisioner(
                selected,
                deadline=time.monotonic() + 60.0,
            )
        except Exception as exc:
            raise B300ScreenDeploymentError(
                f"fixed GPU provisioning failed: {type(exc).__name__}"
            ) from None
    else:
        assert provisioned_gpus is not None
        gpus = provisioned_gpus
    if type(gpus) is not tuple or any(type(row) is not GPUConfiguration for row in gpus):
        raise B300ScreenDeploymentError(
            "GPU provisioner did not return exact immutable configurations"
        )
    refs = {
        "authority_config": _authority_ref(authority_path, authority_sha),
        "calibration_package": _authority_ref(calibration_path, calibration_sha),
        "calibration_projection_receipt": _authority_ref(
            projection_path, projection_sha
        ),
        "device_execution": _authority_ref(device_path, device_sha),
        "measurement_config": _authority_ref(measurement_path, measurement_sha),
        "prompt_authority": _authority_ref(prompt_path, prompt_sha),
        "ready_receipt": _authority_ref(ready_path, ready_sha),
    }
    return _derive_inputs(
        root=root,
        ready=ready,
        authority=authority,
        measurement=measurement,
        prompt=prompt,
        authority_refs=refs,
        preflight=preflight,
        gpus=gpus,
    )


def _deployment_payload(inputs: _CommissionedInputs) -> dict[str, object]:
    ready_lane = _mapping(inputs.ready.get("lane"), "READY lane")
    return {
        "authorities": {
            key: dict(value) for key, value in sorted(inputs.authority_refs.items())
        },
        "controller_distribution_digest": inputs.controller_distribution_digest,
        "declared_qualification": inputs.declared_qualification.to_dict(),
        "device_configuration_digest": (
            inputs.device_policy.configuration_sha256
        ),
        "device_policy_digest": inputs.device_policy.policy_sha256,
        "evidence_policy_digest": inputs.evidence_policy_digest,
        "gpu_configurations": [
            gpu.canonical_dict() for gpu in inputs.qualification_gpus
        ],
        "plan_resolver_digest": inputs.plan_resolver_digest,
        "preflight_sha256": inputs.preflight.sha256,
        "prompt_identity": dict(inputs.prompt_identity),
        "ready": {
            "lane_devices": list(_ready_lane(inputs.ready)),
            "lane_digest": _digest(
                ready_lane.get("lane_digest"), "READY lane digest"
            ),
            "receipt_digest": _digest(
                inputs.ready.get("receipt_digest"), "READY receipt digest"
            ),
            "worker_epoch": _text(
                inputs.ready.get("worker_epoch"), "READY worker epoch"
            ),
        },
        "resident_factory_digest": inputs.resident_factory_digest,
        "resident_resource_ids": list(inputs.resident_resource_ids),
        "runtime": inputs.runtime.to_dict(),
        "schema": DEPLOYMENT_SCHEMA,
        "topology_digest": inputs.topology_digest,
    }


def materialize_b300_screen_identities(
    *,
    ready_receipt: str | os.PathLike[str],
    authority_config: str | os.PathLike[str],
    measurement_config: str | os.PathLike[str],
    calibration_package: str | os.PathLike[str],
    calibration_projection_receipt: str | os.PathLike[str],
    prompt_authority: str | os.PathLike[str],
    output_root: str | os.PathLike[str] = DEFAULT_OUTPUT_ROOT,
    gpu_provisioner: Callable[..., tuple[GPUConfiguration, ...]] = (
        provision_gpu_configurations
    ),
) -> dict[str, object]:
    """Provision and emit the fixed path-free service identities.

    This command performs the one permitted read-only GPU inventory query.  It
    does not start an engine, import candidate code, execute a candidate, or
    contact chain/network services.
    """

    inputs = _authority_inputs(
        ready_receipt=ready_receipt,
        authority_config=authority_config,
        measurement_config=measurement_config,
        calibration_package=calibration_package,
        calibration_projection_receipt=calibration_projection_receipt,
        prompt_authority=prompt_authority,
        output_root=output_root,
        provisioner=gpu_provisioner,
    )
    composition: _Composition | None = None
    try:
        composition = _compose(inputs)
        service = ArenaService(
            composition.manifest,
            B300ArenaServiceProvider(
                composition.manifest, composition.authorities
            ),
        )
        epoch = _text(inputs.ready.get("worker_epoch"), "READY worker epoch")
        if len(epoch) != 32 or any(character not in "0123456789abcdef" for character in epoch):
            raise B300ScreenDeploymentError("READY worker epoch is not 128-bit hex")
        readiness = WorkerReadiness.for_service(
            service,
            ready_receipt_digest=_digest(
                inputs.ready.get("receipt_digest"), "READY receipt digest"
            ),
            ready_epoch=int(epoch, 16),
        )
        deployment = _deployment_payload(inputs)
        _atomic_canonical(inputs.root / DEPLOYMENT_FILE, deployment)
        _atomic_canonical(inputs.root / MANIFEST_FILE, composition.manifest.to_dict())
        _atomic_canonical(inputs.root / READINESS_FILE, readiness.to_dict())
        return {
            "arena_service_manifest": str(inputs.root / MANIFEST_FILE),
            "arena_service_manifest_sha256": _file_sha256(
                inputs.root / MANIFEST_FILE
            ),
            "deployment": str(inputs.root / DEPLOYMENT_FILE),
            "deployment_sha256": _file_sha256(inputs.root / DEPLOYMENT_FILE),
            "provider_digest": composition.manifest.provider_digest,
            "schema": MATERIALIZATION_SCHEMA,
            "service_digest": composition.manifest.digest,
            "worker_readiness": str(inputs.root / READINESS_FILE),
            "worker_readiness_digest": readiness.digest,
            "worker_readiness_sha256": _file_sha256(
                inputs.root / READINESS_FILE
            ),
        }
    finally:
        if composition is not None:
            composition.close()


def _runtime_from_dict(value: object) -> ArenaRuntimeIdentity:
    row = _mapping(value, "arena runtime identity")
    try:
        return ArenaRuntimeIdentity(**row)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise B300ScreenDeploymentError(f"runtime identity is invalid: {exc}") from None


def _manifest_from_dict(value: object) -> ArenaServiceManifest:
    row = _mapping(value, "arena service manifest")
    workload_row = _mapping(row.get("workload"), "workload")
    raw_regimes = workload_row.get("regimes")
    if type(raw_regimes) is not list:
        raise B300ScreenDeploymentError("workload regimes are malformed")
    regimes: list[WorkloadRegime] = []
    for raw_regime in raw_regimes:
        regime = _mapping(raw_regime, "workload regime")
        raw_shapes = regime.get("shapes")
        if type(raw_shapes) is not list:
            raise B300ScreenDeploymentError("workload shapes are malformed")
        regimes.append(
            WorkloadRegime(
                regime["name"],  # type: ignore[arg-type]
                regime["phase"],  # type: ignore[arg-type]
                regime["weight_ppm"],  # type: ignore[arg-type]
                tuple(ServingShape(**_mapping(shape, "serving shape")) for shape in raw_shapes),
            )
        )
    screens_row = _mapping(row.get("screens"), "screen policy")
    if screens_row.get("crownable") is not False or type(screens_row.get("stages")) is not list:
        raise B300ScreenDeploymentError("screen policy is malformed")
    try:
        return ArenaServiceManifest(
            runtime=_runtime_from_dict(row["runtime"]),
            workload=WorkloadMixture(
                workload_row["prompt_corpus_digest"],  # type: ignore[arg-type]
                workload_row["prompt_seed_scheme"],  # type: ignore[arg-type]
                tuple(regimes),
            ),
            capacity=ArenaCapacityPolicy(**_mapping(row["capacity"], "capacity")),
            screens=NonCrownScreenPolicy(
                tuple(
                    ScreenStagePolicy(**_mapping(stage, "screen stage"))
                    for stage in screens_row["stages"]  # type: ignore[union-attr]
                )
            ),
            qualification_policy_digest=row["qualification_policy_digest"],  # type: ignore[arg-type]
            provider_digest=row["provider_digest"],  # type: ignore[arg-type]
            schema_version=row["schema_version"],  # type: ignore[arg-type]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise B300ScreenDeploymentError(
            f"arena service manifest is invalid: {type(exc).__name__}"
        ) from None


class _CommissionedB300ScreenWorker(B300MainnetWorker):
    """Worker that also releases composition-owned screen resources."""

    def __init__(self, composition: _Composition, readiness: WorkerReadiness) -> None:
        self._commissioned_composition = composition
        try:
            super().__init__(composition.manifest, composition.authorities, readiness)
        except BaseException:
            composition.close()
            raise

    def close(self) -> None:
        try:
            super().close()
        finally:
            self._commissioned_composition.close()


def _ref_path(refs: dict[str, object], name: str) -> Path:
    row = _mapping(refs.get(name), f"deployment authority {name}")
    path = _absolute_path(row.get("path"), f"deployment authority {name} path")
    expected = _digest(
        row.get("sha256"), f"deployment authority {name} SHA-256"
    )
    if _file_sha256(path) != expected:
        raise B300ScreenDeploymentError(
            f"deployment authority {name} changed after materialization"
        )
    return path


def _canonical_artifact(path: Path, field: str) -> dict[str, object]:
    resolved, value, _sha = _stable_json(path, field)
    if resolved != path.resolve(strict=True) or path.read_bytes() != _canonical_bytes(value):
        raise B300ScreenDeploymentError(f"{field} is not canonical JSON")
    return value


def replay_commissioned_screen_composition(
    registration: dict[str, object],
    ready_receipt: dict[str, object],
) -> tuple[_CommissionedInputs, _Composition, WorkerReadiness]:
    """Replay the fixed commissioned artifacts into one live composition.

    ``registration`` and ``ready_receipt`` come from the fixed authenticated
    transport codec.  They select no local paths: this function reopens only
    the three fixed commissioned filenames and the SHA-bound authority refs
    inside the commissioned deployment artifact.  The caller owns the returned
    composition and must close it.
    """

    if type(registration) is not dict or type(ready_receipt) is not dict:
        raise B300ScreenDeploymentError(
            "commissioned worker inputs must be exact JSON objects"
        )
    root = _prepare_private_root(DEFAULT_OUTPUT_ROOT)
    deployment_path = root / DEPLOYMENT_FILE
    manifest_path = root / MANIFEST_FILE
    readiness_path = root / READINESS_FILE
    deployment = _canonical_artifact(deployment_path, "screen deployment")
    if deployment.get("schema") != DEPLOYMENT_SCHEMA:
        raise B300ScreenDeploymentError("screen deployment schema differs")
    refs = _mapping(deployment.get("authorities"), "deployment authorities")
    required_refs = {
        "authority_config",
        "calibration_package",
        "calibration_projection_receipt",
        "device_execution",
        "measurement_config",
        "prompt_authority",
        "ready_receipt",
    }
    if set(refs) != required_refs:
        raise B300ScreenDeploymentError("deployment authority inventory differs")
    gpu_rows = deployment.get("gpu_configurations")
    if type(gpu_rows) is not list:
        raise B300ScreenDeploymentError("deployment GPU inventory is malformed")
    gpus = tuple(_gpu_from_dict(row) for row in gpu_rows)
    inputs = _authority_inputs(
        ready_receipt=_ref_path(refs, "ready_receipt"),
        authority_config=_ref_path(refs, "authority_config"),
        measurement_config=_ref_path(refs, "measurement_config"),
        calibration_package=_ref_path(refs, "calibration_package"),
        calibration_projection_receipt=_ref_path(
            refs, "calibration_projection_receipt"
        ),
        prompt_authority=_ref_path(refs, "prompt_authority"),
        output_root=root,
        provisioner=None,
        provisioned_gpus=gpus,
    )
    if inputs.ready != ready_receipt or _deployment_payload(inputs) != deployment:
        raise B300ScreenDeploymentError(
            "commissioned deployment did not replay from sealed authorities"
        )
    stored_manifest = _manifest_from_dict(
        _canonical_artifact(manifest_path, "arena service manifest")
    )
    readiness_row = _canonical_artifact(readiness_path, "worker readiness")
    try:
        readiness = WorkerReadiness(**readiness_row)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise B300ScreenDeploymentError(
            f"worker readiness is invalid: {exc}"
        ) from None
    composition: _Composition | None = None
    try:
        composition = _compose(inputs)
        if composition.manifest != stored_manifest:
            raise B300ScreenDeploymentError(
                "arena service manifest did not replay from deployment"
            )
        service = ArenaService(
            composition.manifest,
            B300ArenaServiceProvider(
                composition.manifest, composition.authorities
            ),
        )
        readiness.validate(service)
        lane_devices = registration.get("lane_devices")
        if (
            registration.get("ready_receipt_digest")
            != ready_receipt.get("receipt_digest")
            or registration.get("worker_epoch")
            != ready_receipt.get("worker_epoch")
            or registration.get("service_identity")
            != composition.manifest.service_id
            or registration.get("worker_readiness") != readiness.to_dict()
            or registration.get("worker_readiness_digest") != readiness.digest
            or type(lane_devices) is not list
            or tuple(lane_devices) != _ready_lane(ready_receipt)
        ):
            raise B300ScreenDeploymentError(
                "registration differs from commissioned service, READY, or TP4 lane"
            )
        result = (inputs, composition, readiness)
        composition = None
        return result
    finally:
        if composition is not None:
            composition.close()


def commissioned_screen_worker_from_composition(
    composition: _Composition,
    readiness: WorkerReadiness,
) -> B300MainnetWorker:
    """Adopt one replayed composition as the commissioned screen worker."""

    if type(composition) is not _Composition or type(readiness) is not WorkerReadiness:
        raise B300ScreenDeploymentError(
            "commissioned worker adoption inputs are not exact"
        )
    return _CommissionedB300ScreenWorker(composition, readiness)


def build_commissioned_b300_screen_worker(
    registration: dict[str, object],
    ready_receipt: dict[str, object],
) -> B300MainnetWorker:
    """Reopen the fixed commissioned artifacts and build one screen worker."""

    _inputs, composition, readiness = replay_commissioned_screen_composition(
        registration, ready_receipt
    )
    try:
        worker = commissioned_screen_worker_from_composition(composition, readiness)
        composition = None
        return worker
    finally:
        if composition is not None:
            composition.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m cacheon.eval.b300_screen_deployment"
    )
    subparsers = parser.add_subparsers(dest="operation", required=True)
    materialize = subparsers.add_parser("materialize")
    materialize.add_argument("--ready-receipt", required=True)
    materialize.add_argument("--authority-config", required=True)
    materialize.add_argument("--measurement-config", required=True)
    materialize.add_argument("--calibration-package", required=True)
    materialize.add_argument("--calibration-projection-receipt", required=True)
    materialize.add_argument("--prompt-authority", required=True)
    materialize.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.operation != "materialize":  # pragma: no cover - argparse is closed
        raise B300ScreenDeploymentError("unsupported deployment operation")
    result = materialize_b300_screen_identities(
        ready_receipt=args.ready_receipt,
        authority_config=args.authority_config,
        measurement_config=args.measurement_config,
        calibration_package=args.calibration_package,
        calibration_projection_receipt=args.calibration_projection_receipt,
        prompt_authority=args.prompt_authority,
        output_root=args.output_root,
    )
    print(_canonical_bytes(result).decode("utf-8"), end="")
    return 0


__all__ = [
    "B300ScreenDeploymentError",
    "DEFAULT_OUTPUT_ROOT",
    "DEPLOYMENT_FILE",
    "DEPLOYMENT_SCHEMA",
    "MANIFEST_FILE",
    "MATERIALIZATION_SCHEMA",
    "READINESS_FILE",
    "build_commissioned_b300_screen_worker",
    "commissioned_screen_worker_from_composition",
    "main",
    "materialize_b300_screen_identities",
    "replay_commissioned_screen_composition",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
