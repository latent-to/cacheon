"""Check an arena's exact SGLang version and installed integration surfaces.

Imports and signatures establish compatibility without a GPU or model. The
commission's faithful/broken runtime controls establish behavioral acceptance.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Optional

# Default compatibility target for existing GLM deployments. Another arena
# supplies its own exact pin; its RuntimePreflightConfig enforces the same value.
PINNED_SGLANG = "0.5.20"


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""


def _chokepoint_present(mod, chokepoint: str) -> bool:
    """True iff the adapter's chokepoint exists on ``mod``.

    ``"Class.method"`` asserts a method on a class; a bare ``"function_name"`` (no dot)
    asserts a callable MODULE-LEVEL attribute — the rebind-style seams (e.g. the scheduler
    gate's ``run_scheduler_process``) patch a module function, not a class.
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


def run_checks(expected_sglang_version: str = PINNED_SGLANG) -> list[Check]:
    """Check the exact version commissioned for this arena and its seams."""
    checks: list[Check] = []

    def add(name: str, ok: bool, detail: str = "") -> None:
        checks.append(Check(name, bool(ok), str(detail)))

    try:
        import sglang
    except Exception as exc:  # noqa: BLE001
        add("import sglang", False, repr(exc))
        return checks

    ver = getattr(sglang, "__version__", "?")
    version_matches = ver == expected_sglang_version
    add(
        f"sglang installed (pinned {expected_sglang_version})",
        version_matches,
        f"found {ver}" + ("" if version_matches else "  <-- DIFFERS from pin"),
    )

    # Table-driven baseline: every adapter in the single seam table (cacheon/seams.py)
    # must have its target module import and its Class.method chokepoint present. Adding
    # a seam to that table auto-adds this canary (no separate edit here).
    import importlib

    from cacheon.seams import SEAM_ADAPTERS

    for adapter in SEAM_ADAPTERS:
        try:
            mod = importlib.import_module(adapter.target_module)
            ok = _chokepoint_present(mod, adapter.chokepoint)
            add(f"seam table: {adapter.name} ({adapter.chokepoint})", ok,
                "" if ok else f"missing {adapter.chokepoint} in {adapter.target_module}")
        except Exception as exc:  # noqa: BLE001
            add(f"seam table: {adapter.name} ({adapter.chokepoint})", False, repr(exc))

    # The node check's honest twin reroutes every fused op of a node onto its
    # ``forward_native``; without this base the twin is stock and measures no noise.
    try:
        from sglang.kernels.fused_op import BaseFusedOp

        ok = hasattr(BaseFusedOp, "forward_native")
        add("BaseFusedOp.forward_native present", ok)
    except Exception as exc:  # noqa: BLE001
        add("BaseFusedOp.forward_native present", False, repr(exc))

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

        fields = set(inspect.signature(ServerArgs).parameters)
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


# Kernel library versions are part of the commissioned image/runtime identity.
# None below means record-only in this canary; an explicit version is enforced.
# A matching library list does not replace the image or end-to-end acceptance.


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
