"""Parse and project a sealed arena's workload and physical allocation."""

from __future__ import annotations

from dataclasses import replace

from cacheon.arena_service import ArenaRuntimeIdentity, ArenaServiceError, Workload, WorkloadCell
from cacheon._strict import require_digest
from cacheon.eval.device_state import DeviceStatePolicy, GPUConfiguration
from cacheon.eval.engine_launch import LogicalHardwareSpec, PhysicalHardwareBinding
from cacheon.eval.oci_session_protocol import (
    CONTAINER_MODEL_PATH,
    ENGINE_CONFIG_FIELDS,
    EngineSessionConfig,
)
from cacheon.seams import SEAM_ADAPTERS
from cacheon.target_catalog import TargetCatalog

_ARENA_ENGINE_FIELDS = ENGINE_CONFIG_FIELDS - {
    "disable_cuda_graph",
    "max_running_requests",
    "model_path",
    "seam_bindings",
}
_DERIVED_ENGINE_KWARGS = {
    "context_length",
    "enable_flashinfer_allreduce_fusion",
    "watchdog_timeout",
}
_CELL_FIELDS = frozenset(WorkloadCell.__dataclass_fields__)


class B300ScreenDeploymentError(RuntimeError):
    """A commissioned authority is missing, mutable, or inconsistent."""


def _mapping(value: object, field: str) -> dict[str, object]:
    if type(value) is not dict:
        raise B300ScreenDeploymentError(f"{field} must be a JSON object")
    return value


def string_rows(value: object, label: str) -> list[str]:
    if type(value) is not list or any(type(row) is not str for row in value):
        raise B300ScreenDeploymentError(f"{label} must be a list of strings")
    return value


def _device_ids(value: object, label: str) -> tuple[int, ...]:
    if (
        type(value) is not list
        or not value
        or any(type(row) is not int or row < 0 for row in value)
        or value != sorted(set(value))
    ):
        raise B300ScreenDeploymentError(f"{label} must be ordered unique GPU indices")
    return tuple(value)


def ready_gpu_ids(ready: dict[str, object]) -> tuple[int, ...]:
    """Read the full host inventory independently of the arena allocation."""
    gpu = _mapping(ready.get("gpu"), "READY GPU inventory")
    inventory = gpu.get("inventory")
    count = gpu.get("count")
    if type(count) is not int or count < 1 or type(inventory) is not list or len(inventory) != count:
        raise B300ScreenDeploymentError("READY GPU count differs from its inventory")
    return _device_ids(
        [_mapping(row, "READY GPU row").get("index") for row in inventory],
        "READY GPU inventory",
    )


def ready_lane(ready: dict[str, object]) -> tuple[int, ...]:
    """Read the screen lane without assuming a host size or GPU model."""
    lane = _mapping(ready.get("lane"), "READY lane")
    selected = _device_ids(lane.get("devices"), "READY lane")
    tp = lane.get("tensor_parallel_size")
    if type(tp) is not int or tp != len(selected):
        raise B300ScreenDeploymentError("READY lane differs from its tensor parallel size")
    require_digest(lane.get("lane_digest"), field="READY lane digest", error=B300ScreenDeploymentError)
    return selected


def ready_lanes(ready: dict[str, object]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Resolve the commissioned pair, allowing other devices on the host.

    Existing full-host receipts have one uniquely determined complement.
    A partial-host allocation must explicitly name its baseline devices.
    """
    selected = ready_lane(ready)
    inventory = ready_gpu_ids(ready)
    lane = _mapping(ready.get("lane"), "READY lane")
    baseline = _device_ids(lane["baseline_devices"], "READY baseline lane") if "baseline_devices" in lane else tuple(
        index for index in inventory if index not in selected
    )
    if (
        len(baseline) != len(selected)
        or set(selected).intersection(baseline)
        or not set(selected + baseline).issubset(inventory)
    ):
        raise B300ScreenDeploymentError(
            "READY requires equal disjoint lanes within its inventory; "
            "partial-host allocations must name baseline_devices"
        )
    return selected, baseline


def device_policy(gpus: tuple[GPUConfiguration, ...]) -> DeviceStatePolicy:
    """Bind the lane's observed device configuration and execution envelope."""
    if (
        not gpus
        or tuple(gpu.physical_id for gpu in gpus) != tuple(sorted({gpu.physical_id for gpu in gpus}))
        or len({gpu.max_memory_clock_mhz for gpu in gpus}) != 1
    ):
        raise B300ScreenDeploymentError("selected lane must contain ordered compatible GPU configurations")
    return DeviceStatePolicy(
        expected_gpus=gpus,
        maximum_temperature_c=90,
        maximum_gpu_utilization_percent=5,
        maximum_memory_utilization_percent=5,
        allowed_active_pstates=("P0",),
        active_maximum_graphics_clock_mhz=max(gpu.max_graphics_clock_mhz for gpu in gpus),
        active_memory_clock_mhz=gpus[0].max_memory_clock_mhz,
        active_maximum_power_draw_mw=max(gpu.power_limit_mw for gpu in gpus),
        active_require_process_on_every_gpu=True,
        required_consecutive_idle_samples=2,
        poll_interval_s=0.05,
        ready_poll_interval_s=0.05,
        maximum_samples=1024,
    )


def resource_policy(policy, overrides: object):
    """Apply sealed host capacity to the existing OCI policy and its digest."""
    row = _mapping(overrides, "OCI resource capacity")
    if set(row) - {"cpu_millis", "memory_bytes", "pids_limit", "nofile_limit",
                   "cache_bytes", "cache_inodes", "tmpfs_bytes", "shm_bytes",
                   "stage_bytes", "stage_inodes"}:
        raise B300ScreenDeploymentError("unknown OCI resource capacity field")
    try:
        return replace(policy, **row)
    except (TypeError, ValueError) as exc:
        raise B300ScreenDeploymentError(f"invalid OCI resource capacity: {exc}") from None


def hardware_bindings(
    runtime: ArenaRuntimeIdentity, policy: DeviceStatePolicy, *, dp_size: int
) -> tuple[LogicalHardwareSpec, PhysicalHardwareBinding]:
    """Project both launch views from the same commissioned runtime and lane."""
    common = dict(
        architecture=runtime.target_architecture,
        topology_class=runtime.topology_class,
        topology_digest=runtime.topology_digest,
        tp_size=runtime.tensor_parallel_size,
        ep_size=1,
        dp_size=dp_size,
        device_policy_digest=policy.policy_sha256,
    )
    return (
        LogicalHardwareSpec(visible_gpu_count=runtime.gpu_count, **common),
        PhysicalHardwareBinding(physical_gpu_ids=tuple(str(gpu.physical_id) for gpu in policy.expected_gpus), **common),
    )


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


def engine_config(
    template: EngineSessionConfig,
    target_members: tuple[str, ...],
    cell: WorkloadCell | tuple[WorkloadCell, ...],
    *,
    disable_cuda_graph: bool,
) -> EngineSessionConfig:
    bindings = _seam_bindings(target_members)
    cells = (cell,) if type(cell) is WorkloadCell else tuple(cell)
    if not cells or any(type(row) is not WorkloadCell for row in cells):
        raise B300ScreenDeploymentError("engine workload cells are not exact")
    kwargs = dict(template.engine_kwargs)
    kwargs["context_length"] = max(
        row.input_tokens + row.output_tokens for row in cells
    ) + 128
    if not disable_cuda_graph:
        kwargs["watchdog_timeout"] = 1800
    return replace(
        template,
        disable_cuda_graph=disable_cuda_graph,
        max_running_requests=max(row.concurrency for row in cells),
        engine_kwargs=kwargs,
        seam_bindings=bindings,
    )


def data_parallel_size(config: EngineSessionConfig) -> int:
    value = config.engine_kwargs.get("dp_size", 1)
    if type(value) is not int or not 1 <= value <= config.tp_size:
        raise B300ScreenDeploymentError("arena engine dp_size exceeds its TP degree")
    return value


def engine_template(prompt: dict[str, object]) -> EngineSessionConfig:
    """Parse model/runtime choices; launch- and target-derived fields stay owned."""

    row = _mapping(prompt.get("engine_config"), "arena engine config")
    if set(row) != _ARENA_ENGINE_FIELDS:
        raise B300ScreenDeploymentError("arena engine config fields are not closed")
    engine_kwargs = _mapping(row.get("engine_kwargs"), "arena engine kwargs")
    forbidden = set(engine_kwargs) & _DERIVED_ENGINE_KWARGS
    if forbidden:
        raise B300ScreenDeploymentError(
            f"arena engine kwargs contain derived fields: {sorted(forbidden)!r}"
        )
    try:
        return EngineSessionConfig(
            model_path=CONTAINER_MODEL_PATH,
            disable_cuda_graph=False,
            max_running_requests=None,
            seam_bindings=(),
            **row,  # type: ignore[arg-type]
        )
    except (TypeError, ValueError) as exc:
        raise B300ScreenDeploymentError(
            f"arena engine config is invalid: {exc}"
        ) from None


def workload(
    prompt: dict[str, object],
    batches: tuple[tuple[str, ...], ...],
    sha256: str,
) -> Workload:
    singular = prompt.get("workload_cell")
    plural = prompt.get("workload_cells")
    if (singular is None) == (plural is None):
        raise B300ScreenDeploymentError(
            "prompt authority requires exactly one workload-cell form"
        )
    raw_rows = [singular] if singular is not None else plural
    if type(raw_rows) is not list or not raw_rows:
        raise B300ScreenDeploymentError("workload cells are malformed")
    rows = tuple(_mapping(row, "workload cell") for row in raw_rows)
    if any(set(row) != _CELL_FIELDS for row in rows):
        raise B300ScreenDeploymentError("workload cell fields are not closed")
    try:
        cells = tuple(WorkloadCell(**row) for row in rows)  # type: ignore[arg-type]
        parsed = Workload(
            prompt_corpus_digest=sha256,
            prompt_seed_scheme=prompt.get("prompt_seed_scheme"),  # type: ignore[arg-type]
            cells=cells,
        )
    except ArenaServiceError as exc:
        raise B300ScreenDeploymentError(
            f"workload declaration is invalid: {exc}"
        ) from None
    prompt_batch_cells(prompt, batches, parsed)
    return parsed


def prompt_batch_cells(
    prompt: dict[str, object],
    batches: tuple[tuple[str, ...], ...],
    parsed: Workload,
) -> tuple[str, ...]:
    if len(parsed.cells) == 1:
        if "prompt_batch_cells" in prompt:
            raise B300ScreenDeploymentError(
                "one-cell prompt authority cannot declare batch-cell routing"
            )
        cells = (parsed.cells[0].cell_id,) * len(batches)
    else:
        raw = prompt.get("prompt_batch_cells")
        if (
            type(raw) is not list
            or len(raw) != len(batches)
            or any(type(row) is not str for row in raw)
        ):
            raise B300ScreenDeploymentError(
                "prompt batch cells must exactly cover the sealed batches"
            )
        cells = tuple(raw)
    by_id = {row.cell_id: row for row in parsed.cells}
    if set(cells) != set(by_id) or any(
        cell_id not in by_id
        or len(batch) != by_id[cell_id].concurrency
        for batch, cell_id in zip(batches, cells, strict=True)
    ):
        raise B300ScreenDeploymentError(
            "sealed prompt batches do not match their declared cell concurrency"
        )
    return cells


def scored_cell(parsed: Workload) -> WorkloadCell:
    """Cell order is sealed; single-cell projections use its first entry."""

    return parsed.cells[0]


def target_partition(
    prompt: dict[str, object], catalog: TargetCatalog
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return the sealed active set and its catalog-complete closed complement."""

    registered = tuple(
        string_rows(prompt.get("registered_targets"), "registered targets")
    )
    if not registered or registered != tuple(sorted(set(registered))):
        raise B300ScreenDeploymentError(
            "registered targets must be a nonempty sorted unique list"
        )
    for target in registered:
        try:
            catalog.require(target)
        except (KeyError, TypeError, ValueError) as exc:
            raise B300ScreenDeploymentError(
                f"arena target is not registered: {exc}"
            ) from None
    snapshot = catalog.snapshot().get("targets")
    if type(snapshot) is not list or any(type(row) is not dict for row in snapshot):
        raise B300ScreenDeploymentError("target catalog snapshot is malformed")
    catalog_ids = tuple(row.get("target_id") for row in snapshot)
    if any(type(target) is not str for target in catalog_ids):
        raise B300ScreenDeploymentError("target catalog IDs are malformed")
    closed = tuple(target for target in catalog_ids if target not in registered)
    return registered, closed
