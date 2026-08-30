"""SGLang compatibility canary — enforce the pin and integration surface.

Our harness patches sglang internals (the `SiluAndMul` / `RMSNorm` seams, the
`BaseFusedOp` base, the Engine logprob API, specific `ServerArgs` kwargs). Any
sglang upgrade can move those. This canary introspects the INSTALLED sglang —
imports + signatures only, **no GPU, no model** — and checks every seam and API we
depend on still exists.

Run `cacheon compat` after bumping sglang. If it goes red, the seams need an
adapter before that version can be used for scoring. (A green canary is necessary
but not sufficient — the runtime smoke test, "broken kernel still FAILs the gate,"
is the behavioral confirmation on the pod.)
"""

from __future__ import annotations

import dataclasses
import inspect
from dataclasses import dataclass
from typing import Optional

# The sglang version scored against. Bump DELIBERATELY and in a coordinated way —
# see docs/dev/sglang-tracking.md. All validators must run the same version (consensus).
#
# 0.5.18 (CUDA 13). This is the GLM-5.3 branch's source compatibility target.
# The exact release source and served GLM image were inspected on 2026-08-30;
# runtime seam activation, broken/faithful controls, and arena rebaseline remain
# pending. The deployed MiniMax-M3 packet binds its own source-built runtime and
# is not rewritten by this repository default.
PINNED_SGLANG = "0.5.18"


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""


def _chokepoint_present(mod, chokepoint: str) -> bool:
    """True iff the adapter's chokepoint exists on ``mod``.

    ``"Class.method"`` asserts a method on a class; a bare ``"function_name"`` (no dot)
    asserts a callable MODULE-LEVEL attribute — the rebind-style seams (e.g. arfusion's
    ``flashinfer_allreduce_residual_rmsnorm``) patch a module function, not a class.
    ``"attr:Name"`` asserts a module attribute that need not be callable (rebind
    targets like flashinfer's ``JitSpec`` / env constants).
    """
    if chokepoint.startswith("attr:"):
        return hasattr(mod, chokepoint[len("attr:"):])
    cls_name, dot, meth = chokepoint.partition(".")
    if not dot:
        return callable(getattr(mod, cls_name, None))
    cls = getattr(mod, cls_name, None)
    return cls is not None and hasattr(cls, meth)


def run_checks() -> list[Check]:
    checks: list[Check] = []

    def add(name: str, ok: bool, detail: str = "") -> None:
        checks.append(Check(name, bool(ok), str(detail)))

    try:
        import sglang
    except Exception as exc:  # noqa: BLE001
        add("import sglang", False, repr(exc))
        return checks

    ver = getattr(sglang, "__version__", "?")
    version_matches = ver == PINNED_SGLANG
    add(
        f"sglang installed (pinned {PINNED_SGLANG})",
        version_matches,
        f"found {ver}" + ("" if version_matches else "  <-- DIFFERS from pin"),
    )

    # Table-driven baseline: every adapter in the single seam table (cacheon/seams.py)
    # must have its target module import and its Class.method chokepoint present. Adding
    # a seam to that table auto-adds this canary (no separate edit here). The bespoke
    # signature checks below enrich these for the known seams.
    import importlib
    import importlib.util

    from cacheon.seams import SEAM_ADAPTERS

    def _requires_present(name: str) -> bool:
        # find_spec on a DOTTED name imports the parent packages, so a missing parent
        # RAISES (ModuleNotFoundError) instead of returning None — only top-level names
        # degrade gracefully. A dotted `requires` (e.g. the M3-only minimax_sparse_ops
        # subpackage) must mean SKIP on boxes without the parent, not a canary crash.
        try:
            return importlib.util.find_spec(name) is not None
        except ModuleNotFoundError:
            return False

    for adapter in SEAM_ADAPTERS:
        if adapter.requires is not None and not _requires_present(adapter.requires):
            # Row not assessable here (e.g. flashinfer only exists on engine boxes).
            # SKIP-as-ok so dev/intake boxes stay green; the pinned engine env — the
            # place a chokepoint break actually matters — always has the package.
            add(f"seam table: {adapter.name} ({adapter.chokepoint})", True,
                f"SKIP: {adapter.requires} not installed on this box")
            continue
        try:
            mod = importlib.import_module(adapter.target_module)
            ok = _chokepoint_present(mod, adapter.chokepoint)
            add(f"seam table: {adapter.name} ({adapter.chokepoint})", ok,
                "" if ok else f"missing {adapter.chokepoint} in {adapter.target_module}")
        except Exception as exc:  # noqa: BLE001
            add(f"seam table: {adapter.name} ({adapter.chokepoint})", False, repr(exc))

    try:
        from sglang.kernels.fused_op import BaseFusedOp
        fused_op_base = BaseFusedOp
        add("BaseFusedOp base present", True)
    except Exception as exc:  # noqa: BLE001
        fused_op_base = None
        add("BaseFusedOp base present", False, repr(exc))

    # activation seam (SiluAndMul slot)
    try:
        from sglang.srt.layers.activation import SiluAndMul

        ok = hasattr(SiluAndMul, "forward_cuda") and hasattr(SiluAndMul, "forward_native")
        if fused_op_base is not None:
            ok = ok and issubclass(SiluAndMul, fused_op_base)
        add("seam: SiluAndMul (activation)", ok, "needs forward_cuda/native on BaseFusedOp")
    except Exception as exc:  # noqa: BLE001
        add("seam: SiluAndMul (activation)", False, repr(exc))

    # norm seam (RMSNorm slot)
    try:
        from sglang.srt.layers.layernorm import RMSNorm

        params = list(inspect.signature(RMSNorm.forward_cuda).parameters)
        ok = (
            hasattr(RMSNorm, "forward_cuda")
            and {"residual", "quant_linear"} <= set(params)
        )
        if fused_op_base is not None:
            ok = ok and issubclass(RMSNorm, fused_op_base)
        add("seam: RMSNorm (layernorm)", ok, f"forward_cuda params={tuple(params)}")
    except Exception as exc:  # noqa: BLE001
        add("seam: RMSNorm (layernorm)", False, repr(exc))

    try:
        from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

        apply_params = set(inspect.signature(UnquantizedLinearMethod.apply).parameters)
        into_params = set(
            inspect.signature(UnquantizedLinearMethod.apply_into).parameters
        )
        ok = {"layer", "x", "bias"} <= apply_params and {
            "layer", "x", "output", "bias"
        } <= into_params
        add(
            "seam: UnquantizedLinearMethod (linear.dense)",
            ok,
            f"apply params={tuple(sorted(apply_params))}; "
            f"apply_into params={tuple(sorted(into_params))}",
        )
    except Exception as exc:  # noqa: BLE001
        add("seam: UnquantizedLinearMethod (linear.dense)", False, repr(exc))

    # MoE seams. Ordinary runners reach forward_impl; FlashInfer TRT-LLM skips it
    # through forward_deferred_finalize when it fuses routed finalize + shared add.
    try:
        from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE

        params = set(inspect.signature(FusedMoE.forward_impl).parameters)
        deferred = set(
            inspect.signature(FusedMoE.forward_deferred_finalize).parameters
        )
        ok = hasattr(FusedMoE, "forward_impl") and {
            "hidden_states", "topk_output", "pre_quant_input"
        } <= params and {"hidden_states", "topk_output"} <= deferred
        add(
            "seam: FusedMoE routed paths (moe.fused_experts)",
            ok,
            f"forward_impl params={tuple(sorted(params))}; "
            f"deferred params={tuple(sorted(deferred))}",
        )
    except Exception as exc:  # noqa: BLE001
        add("seam: FusedMoE.forward_impl (moe.fused_experts)", False, repr(exc))

    # collective seam (the TP-comms chokepoint: GroupCoordinator.all_reduce)
    try:
        from sglang.srt.distributed.parallel_state import GroupCoordinator

        params = set(inspect.signature(GroupCoordinator.all_reduce).parameters)
        ok = hasattr(GroupCoordinator, "all_reduce") and "input_" in params
        add("seam: GroupCoordinator.all_reduce (collective)", ok, f"all_reduce params={tuple(sorted(params))}")
    except Exception as exc:  # noqa: BLE001
        add("seam: GroupCoordinator.all_reduce (collective)", False, repr(exc))

    # Engine logprob API (we read top-k logprobs for KL)
    try:
        gp = set(inspect.signature(sglang.Engine.generate).parameters)
        need = {"prompt", "sampling_params", "return_logprob", "logprob_start_len", "top_logprobs_num"}
        add("Engine.generate logprob API", need <= gp, f"missing: {sorted(need - gp) or 'none'}")
    except Exception as exc:  # noqa: BLE001
        add("Engine.generate logprob API", False, repr(exc))

    # ServerArgs kwargs we pass to Engine(...)
    try:
        from sglang.srt.server_args import ServerArgs

        fields = {f.name for f in dataclasses.fields(ServerArgs)}
        need = {
            "model_path", "dtype", "attention_backend", "disable_cuda_graph",
            "mem_fraction_static", "enable_deterministic_inference", "random_seed", "log_level",
        }
        add("ServerArgs accepts our kwargs", need <= fields, f"missing: {sorted(need - fields) or 'none'}")
    except Exception as exc:  # noqa: BLE001
        add("ServerArgs accepts our kwargs", False, repr(exc))

    # Blessed base — the kernel-library surface a miner kernel / a composed override runs on.
    # Consensus-critical: a flashinfer/cutlass/triton skew JITs different kernels -> different
    # throughput AND numerics -> divergent weights -> Yuma penalty (the same reason sglang is
    # pinned). Record-only until the arena pins exact versions; the canary reports the surface.
    try:
        for name, ok, detail in check_blessed_base():
            add(f"blessed-base: {name}", ok, detail)
    except Exception as exc:  # noqa: BLE001
        add("blessed-base", False, repr(exc))

    return checks


# ---- blessed dependency base (consensus pin surface) ----
# The kernel-library surface the subnet scores on. A miner kernel runs against
# these libraries and a base override-kernel is composed from them, so for
# CONSENSUS they must be identical across validators: two validators on
# different flashinfer (or cutlass / triton) JIT *different* kernels ->
# different throughput AND numerics -> divergent weight vectors -> Yuma
# penalty. Only sglang is pinned (PINNED_SGLANG above); the kernel libs ride
# along implicitly, so this makes the whole import surface an explicit,
# canary-checked pin. stdlib-only (no torch import) so the canary runs
# anywhere. Per-arena: when arenas merge this becomes part of the Arena (the
# docker_image should expose this enumerated, hashed set, not an opaque blob).
# A pinned version of None = record-only: the canary reports the installed
# version (the consensus-audit surface) but does not enforce, because the
# exact arena versions aren't validated yet. Set a version to enforce it (a
# mismatch then fails the canary, like the sglang pin).


@dataclass(frozen=True)
class PinnedDep:
    dist: str  # installed distribution name (importlib.metadata.version)
    version: Optional[str]  # the pinned version, or None = record-only (not yet enforced)
    why: str


# THE blessed base. A kernel is a kernel — these are all kernel libraries (Axiom 5 excludes
# engine orchestration, not kernel libs). Versions finalized per arena; flashinfer is the
# override base (the M3 win used 0.6.12).
BLESSED_BASE: tuple[PinnedDep, ...] = (
    PinnedDep("torch", None, "tensor runtime + the dtype/layout contract at every seam"),
    PinnedDep("triton", None, "Triton kernels (the lingua franca tier)"),
    PinnedDep("flashinfer", None, "fused MoE / attention CuTe-DSL kernels (the override base; win used 0.6.12)"),
    PinnedDep("nvidia-cutlass-dsl", None, "CuTe-DSL (CUTLASS python) — the device-code substrate"),
    PinnedDep("sgl-kernel", None, "sglang's CUDA kernel package (a kernel lib; importable)"),
    PinnedDep("deepgemm", None, "DeepGEMM FP8/FP4 GEMMs (optional kernel lib)"),
)


def resolved_version(dist: str) -> Optional[str]:
    """The installed version of a distribution, or None if not installed."""
    try:
        import importlib.metadata as md

        return md.version(dist)
    except Exception:  # noqa: BLE001 - not installed / no metadata
        return None


def check_blessed_base(base: tuple[PinnedDep, ...] = BLESSED_BASE) -> list[tuple[str, bool, str]]:
    """Per-dep (name, ok, detail) for the compat canary. Record-only deps (version None) are
    always ok and just report the installed version; an enforced dep fails if absent or
    version-mismatched — the consensus break a flashinfer/cutlass skew would silently cause."""
    rows: list[tuple[str, bool, str]] = []
    for dep in base:
        inst = resolved_version(dep.dist)
        if dep.version is None:
            rows.append((dep.dist, True, f"installed={inst} (record-only; pin per arena)"))
        elif inst is None:
            rows.append((dep.dist, False, f"NOT INSTALLED (pinned {dep.version})"))
        elif inst == dep.version:
            rows.append((dep.dist, True, f"installed={inst} == pinned"))
        else:
            rows.append((dep.dist, False, f"installed={inst} pinned={dep.version}  <-- DIFFERS"))
    return rows


def format_checks(checks: list[Check]) -> str:
    lines = []
    for c in checks:
        mark = "ok  " if c.ok else "FAIL"
        lines.append(f"  [{mark}] {c.name}" + (f"  — {c.detail}" if c.detail else ""))
    n_fail = sum(1 for c in checks if not c.ok)
    lines.append("")
    lines.append(
        "PIN MATCHES; ALL SEAMS INTACT"
        if n_fail == 0
        else f"{n_fail} CHECK(S) FAILED — restore the pin or adapt seams before scoring"
    )
    return "\n".join(lines)
