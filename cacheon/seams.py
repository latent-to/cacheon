"""Single source of truth for the seam ADAPTERS (the sglang chokepoints we patch).

Three places must agree on the set of seam adapters: the ``.pth`` bootstrap (which
modules to watch for import), ``seam.activate()`` (which adapters to install), and the
``compat`` canary (which chokepoints to assert survived an sglang bump). Keeping three
hand-maintained lists in lockstep is exactly the drift sglang itself fought — a clean
registry PLUS a parallel hardcoded choices list turned "add one backend" into an
N-file edit (see the project review). So all three derive from the ONE table here:
adding a seam is a single entry, and the bootstrap watch-list, the install loop, and
the canary all pick it up.

This is deliberately SEPARATE from ``slots.py``. ``SlotSpec`` is the miner-facing
contract — frozen and model-agnostic. A seam adapter is version-pinned glue to a
specific sglang internal that churns on every ``PINNED_SGLANG`` bump. Coupling them
would tie the stable contract to sglang's internals, the exact inversion of the LLVM
lesson (freeze the contract, let the adapters churn). The ``slots`` field below is a
cross-REFERENCE (which slots an adapter serves), not the source of either.

Import-light on purpose (stdlib only): the ``.pth`` bootstrap imports this at
interpreter startup, before — and without — importing torch or sglang.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping


@dataclass(frozen=True)
class SeamAdapter:
    name: str  # short label (also the integration module stem: cacheon.integrations.sglang_<name-ish>)
    target_module: str  # the sglang module whose import triggers installation
    integration: str  # the cacheon.integrations submodule exposing install(registry)
    # The patched chokepoint, for the compat canary + docs: "Class.method" for a method
    # patch, a bare "function_name" (no dot) for a module-LEVEL function rebind, or
    # "attr:Name" for a (possibly non-callable) module attribute the adapter rebinds.
    chokepoint: str
    slots: tuple[str, ...]  # the slot(s) this adapter serves (cross-ref into slots.py)
    # Optional validator protocol binding. Both fields are validator-owned table
    # metadata and must appear together. Multiple adapters may intentionally share
    # one binding/gate.
    binding_id: str | None = None
    environment_gate: str | None = None


@dataclass(frozen=True)
class SeamBinding:
    """One validator-owned activation gate for a fixed adapter set.

    Binding identifiers cross the isolated-session protocol; environment variable
    names never do. Several adapters may share one binding when they implement one
    semantic product.
    """

    binding_id: str
    environment_gate: str
    adapters: tuple[str, ...]


# THE table. Add a seam here and the bootstrap watch-list, the activate() install loop,
# and the compat canary all pick it up — no parallel list to keep in sync.
SEAM_ADAPTERS: tuple[SeamAdapter, ...] = (
    SeamAdapter("activation", "sglang.srt.layers.activation",
                "sglang_silu", "SiluAndMul.forward_cuda", ("activation.silu_and_mul",)),
    SeamAdapter("layernorm", "sglang.srt.layers.layernorm",
                "sglang_norm", "RMSNorm.forward_cuda",
                ("norm.fused_add_rmsnorm", "norm.rmsnorm")),
    SeamAdapter("dense", "sglang.srt.layers.quantization.unquant",
                "sglang_dense", "UnquantizedLinearMethod.apply", ("linear.dense",),
                binding_id="dense", environment_gate="CACHEON_DENSE_SEAM"),
    SeamAdapter("moe", "sglang.srt.layers.moe.fused_moe_triton.layer",
                "sglang_moe", "FusedMoE.forward_impl",
                ("moe.fused_experts", "moe.fused_experts_reduce",
                 "moe.fused_routed_experts"),
                binding_id="moe", environment_gate="CACHEON_MOE_SEAM"),
    SeamAdapter("moe_deferred", "sglang.srt.layers.moe.fused_moe_triton.layer",
                "sglang_moe", "FusedMoE.forward_deferred_finalize",
                ("moe.fused_routed_experts",),
                binding_id="moe", environment_gate="CACHEON_MOE_SEAM"),
    SeamAdapter("moe_deferred_finalize",
                "sglang.srt.layers.moe.moe_runner.flashinfer_trtllm",
                "sglang_moe", "finalize_flashinfer_trtllm_deferred_output",
                ("moe.fused_routed_experts",),
                binding_id="moe", environment_gate="CACHEON_MOE_SEAM"),
    SeamAdapter("collective", "sglang.srt.distributed.parallel_state",
                "sglang_allreduce", "GroupCoordinator.all_reduce", ("collective.all_reduce",),
                binding_id="collective", environment_gate="CACHEON_COLLECTIVE_SEAM"),
    SeamAdapter("collective_all_reduce_inplace",
                "sglang.srt.distributed.parallel_state", "sglang_allreduce",
                "GroupCoordinator._all_reduce_in_place", ("collective.all_reduce",),
                binding_id="collective", environment_gate="CACHEON_COLLECTIVE_SEAM"),
    SeamAdapter("collective_all_reduce_outplace",
                "sglang.srt.distributed.parallel_state", "sglang_allreduce",
                "GroupCoordinator._all_reduce_out_place", ("collective.all_reduce",),
                binding_id="collective", environment_gate="CACHEON_COLLECTIVE_SEAM"),
    SeamAdapter("collective_all_gather", "sglang.srt.distributed.parallel_state",
                "sglang_allreduce", "GroupCoordinator._all_gather_into_tensor",
                ("collective.all_gather_into_tensor",), binding_id="collective",
                environment_gate="CACHEON_COLLECTIVE_SEAM"),
    SeamAdapter("collective_reduce_scatter", "sglang.srt.distributed.parallel_state",
                "sglang_allreduce", "GroupCoordinator._reduce_scatter_tensor",
                ("collective.reduce_scatter_tensor",), binding_id="collective",
                environment_gate="CACHEON_COLLECTIVE_SEAM"),
    # Module-LEVEL function chokepoint (no dot): sglang's fused AR+residual+RMSNorm
    # epilogue waist. Callers resolve the symbol per call via a function-local import,
    # so rebinding the module attribute reroutes every call site. Only hot when the
    # arena serves --enable-flashinfer-allreduce-fusion (arena server flag).
    SeamAdapter("arfusion", "sglang.srt.layers.flashinfer_comm_fusion",
                "sglang_arfusion", "flashinfer_allreduce_residual_rmsnorm",
                ("collective.ar_residual_rmsnorm",), binding_id="arfusion",
                environment_gate="CACHEON_ARFUSION_SEAM"),
    # NOT a slot seam: the candidate-bundle load gate. sglang spawns scheduler ranks
    # AND a detokenizer (output-path!) through the same bootstrap, and the detokenizer
    # imports watched modules too — so seam.activate() never loads miner code; this
    # adapter wraps the scheduler spawn entry so the load happens only in positively-
    # identified scheduler execution processes (active receipts == tp_size exactly).
    SeamAdapter("scheduler_gate", "sglang.srt.managers.scheduler",
                "sglang_scheduler_gate", "run_scheduler_process", ()),
    # NOT a slot seam: the resident-SCREEN-tier hot-swap hook. Inert unless the
    # validator sets CACHEON_RESIDENT_SWAP (a control directory) — which only the
    # persistent screening engine does, never qualification/crown launches. BEFORE
    # hook on decode-graph (re)capture: applies a pending bundle swap in-process so
    # the recapture warmup JIT-compiles the new kernel and the recorded graphs bake
    # it in. See cacheon/integrations/sglang_resident_swap.py.
    SeamAdapter("resident_swap", "sglang.srt.model_executor.model_runner",
                "sglang_resident_swap", "ModelRunner.init_decode_cuda_graph", ()),
)


def _derive_seam_bindings(
    adapters: tuple[SeamAdapter, ...],
) -> tuple[SeamBinding, ...]:
    """Derive the closed protocol vocabulary from the adapter source of truth."""

    adapter_names: set[str] = set()
    grouped: dict[str, tuple[str, list[str]]] = {}
    gate_owners: dict[str, str] = {}
    for adapter in adapters:
        if adapter.name in adapter_names:
            raise RuntimeError(f"duplicate seam adapter name {adapter.name!r}")
        adapter_names.add(adapter.name)
        binding_id, gate = adapter.binding_id, adapter.environment_gate
        if (binding_id is None) != (gate is None):
            raise RuntimeError(
                f"seam adapter {adapter.name!r} must declare binding and gate together"
            )
        if binding_id is None:
            continue
        if (
            type(binding_id) is not str
            or not binding_id
            or binding_id != binding_id.lower()
            or not binding_id.replace("_", "").isalnum()
        ):
            raise RuntimeError(f"seam adapter {adapter.name!r} has invalid binding id")
        if (
            type(gate) is not str
            or not gate.startswith("CACHEON_")
            or not gate.endswith("_SEAM")
            or not gate.replace("_", "").isalnum()
            or gate != gate.upper()
        ):
            raise RuntimeError(f"seam adapter {adapter.name!r} has invalid fixed gate")
        owner = gate_owners.setdefault(gate, binding_id)
        if owner != binding_id:
            raise RuntimeError(
                f"seam environment gate {gate!r} belongs to multiple bindings"
            )
        existing = grouped.get(binding_id)
        if existing is None:
            grouped[binding_id] = (gate, [adapter.name])
        else:
            existing_gate, names = existing
            if existing_gate != gate:
                raise RuntimeError(
                    f"seam binding {binding_id!r} declares inconsistent gates"
                )
            names.append(adapter.name)
    return tuple(
        SeamBinding(binding_id, grouped[binding_id][0], tuple(grouped[binding_id][1]))
        for binding_id in sorted(grouped)
    )


# Closed protocol vocabulary for validator-selected live seam activation, derived
# from the same rows that own import watching/install/compat. Callers can select only
# these public IDs; process environment names and values never cross the wire.
SEAM_BINDINGS = _derive_seam_bindings(SEAM_ADAPTERS)
SEAM_BINDING_ENV_GATES: Mapping[str, str] = MappingProxyType(
    {binding.binding_id: binding.environment_gate for binding in SEAM_BINDINGS}
)


def normalize_seam_bindings(value: object) -> tuple[str, ...]:
    """Validate and freeze a canonical sequence of closed binding identifiers.

    JSON arrays arrive as lists while trusted construction normally uses tuples, so
    those are the only accepted containers.  In particular, a bare string is never
    treated as an iterable of identifiers.  Duplicates and non-canonical order fail
    rather than being repaired, keeping the same digest on every validator.
    """

    if not isinstance(value, (tuple, list)):
        raise ValueError("seam_bindings must be an array of binding identifiers")
    bindings = tuple(value)
    if any(type(binding) is not str for binding in bindings):
        raise ValueError("seam_bindings must contain only strings")
    if len(set(bindings)) != len(bindings):
        raise ValueError("seam_bindings must not contain duplicates")
    unknown = sorted(set(bindings) - set(SEAM_BINDING_ENV_GATES))
    if unknown:
        raise ValueError(f"seam_bindings contains unknown identifiers: {unknown!r}")
    if bindings != tuple(sorted(bindings)):
        raise ValueError("seam_bindings must be sorted in canonical order")
    return bindings


def seam_binding_environment(value: object) -> dict[str, str]:
    """Return the complete fixed seam environment for canonical bindings.

    Every known gate is emitted as an explicit ``"0"`` or ``"1"`` so inherited
    process state cannot activate a seam omitted by the trusted session config.
    """

    enabled = set(normalize_seam_bindings(value))
    return {
        binding.environment_gate: (
            "1" if binding.binding_id in enabled else "0"
        )
        for binding in SEAM_BINDINGS
    }

# The modules whose import should trigger seam installation (consumed by bootstrap).
TARGET_MODULES = frozenset(a.target_module for a in SEAM_ADAPTERS)
