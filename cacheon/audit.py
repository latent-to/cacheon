"""In-engine slot audit — direct fidelity, replacing rollout-KL as the primary gate.

WHY (2026-07-07, measured): rollout-KL between two engine launches measures
BATCHING, not fidelity, on non-deterministic stacks — a bit-stock candidate
running at 0.545x speed scored mean_kl 0.96 purely because timing shifts batch
composition and kernels are batch-variant; and sglang's deterministic mode
refuses the arena's fa4 attention backend outright. The invariant the referee
actually needs is direct: "in the SCORED engine, the miner kernel computes the
slot's declared function." So the validator audits exactly that: on randomly
sampled dispatcher calls, run the captured STOCK baseline on pristine inputs and
grade the miner outputs against it. The node adapter does the grading and hands
this module the fraction it measured; the per-slot comparison under each slot's
declared tolerances went with the per-operation adapters that called it.
Backend-agnostic; no determinism assumptions.

Gate stack this belongs to: verify (fp32 ground truth, jittered/temporal/burst)
-> THIS audit (untimed quality launch) -> paired benchmark no-regression ->
rollout-KL demoted to advisory (still computed and reported; it is calibration
data, not a razor).

Threat notes:
  * Sampling comes from a process-private RNG seeded with os.urandom — a kernel
    cannot know ex-ante whether a call is audited, so behaving only on audited
    calls is not a strategy. (In-process miner code could in principle
    introspect this module; full isolation is the existing roadmap item that
    closes that class for good.)
  * Audits run only when CACHEON_SLOT_AUDIT is set (the eval's untimed quality
    launch sets it); timed launches never carry the overhead.
  * A failed comparison NEVER crashes the engine: violations are counted and
    receipted; the eval driver reads the receipts and fails the bundle.
  * The baseline call itself may be collective (e.g. the all-reduce
    chokepoint): safe only because every rank reaches the dispatcher for the
    same calls in lockstep AND the sampling RNG is seeded identically across
    ranks of one launch (CACHEON_SLOT_AUDIT_SEED, set by the driver) — a
    rank-divergent sample of a collective baseline would deadlock.
"""

from __future__ import annotations

import logging
import os
import random
from typing import Callable, Optional

import torch

from cacheon import receipts
from cacheon.audit_gate import gate as gate

logger = logging.getLogger("cacheon.audit")

_state: dict = {"rate": None, "rng": None}
_stats: dict[str, dict] = {}


def _rate() -> float:
    if _state["rate"] is None:
        try:
            _state["rate"] = min(1.0, max(0.0, float(os.environ.get("CACHEON_SLOT_AUDIT", "0"))))
        except ValueError:
            _state["rate"] = 0.0
        if _state["rate"] > 0.0:
            seed = os.environ.get("CACHEON_SLOT_AUDIT_SEED")
            # Collective-safe sampling REQUIRES rank-identical decisions; the driver
            # sets the seed. Without one, fall back to urandom (fine for pure-op
            # slots, unsafe for collective baselines — the driver always seeds).
            _state["rng"] = random.Random(int(seed) if seed else os.urandom(8))
    return _state["rate"]


def enabled() -> bool:
    return _rate() > 0.0


def sampled() -> bool:
    """Decide (per dispatcher call) whether this call is audited."""
    r = _rate()
    return r > 0.0 and _state["rng"].random() < r


def _slot_stats(slot: str) -> dict:
    return _stats.setdefault(slot, {
        "slot": slot, "n": 0, "violations": 0, "baseline_refused": 0,
        "compare_errors": 0, "worst_frac": 1.0, "min_ratio": None, "mode": None,
    })


def _receipt(slot: str) -> None:
    # receipts.write names the file kind.tag.pid.json -> same-call-site writes
    # OVERWRITE, giving a rolling per-rank summary; the driver reads the final state.
    receipts.write("audit", _stats[slot], tag=slot)


def baseline_refused(slot: str) -> None:
    """The stock baseline declined this call (e.g. returned (None, None)) — a
    coverage note, not a violation: there is nothing to compare against."""
    s = _slot_stats(slot)
    s["baseline_refused"] += 1
    _receipt(slot)


def record_fraction(slot: str, fraction: float, bar: float, mode: str) -> None:
    """Fold one unit the caller graded itself into the receipted stats.

    The node adapter grades against stock in the engine with a tolerance measured on
    the same calls; a flat bf16 atol of 2e-2 passed a 1.5x-wrong MoE block on 109 of
    240 calls, because early-layer outputs sit below 0.04 (H100 Qwen run, 2026-09-19).
    """
    s = _slot_stats(slot)
    s["mode"], s["min_ratio"] = mode, bar
    s["n"] += 1
    s["worst_frac"] = min(s["worst_frac"], fraction)
    if fraction < bar:
        s["violations"] += 1
        if s["violations"] <= 4:
            logger.warning("cacheon.audit VIOLATION slot=%s unit=%d frac=%.4f (bar %.4f)",
                           slot, s["n"], fraction, bar)
    _receipt(slot)


def compare_error(slot: str) -> None:
    """Count a candidate result that could not be compared with stock's."""
    _slot_stats(slot)["compare_errors"] += 1
    _receipt(slot)


def capture_reference(
    slot: str, baseline_thunk: Callable,
) -> tuple[Optional[torch.Tensor], ...] | None:
    """Retain stock outputs before a candidate can mutate inputs or shared buffers.

    MoE stock must see the original input address: upstream FP4 outputs are bound
    to it. Copy the result instead of the input (2026-09-18 audit comparison errors).
    A baseline error is a refusal, not an engine crash and not a compare_error:
    stock produced nothing to compare against, which says nothing about the
    candidate. That incident's 7,500 baseline errors, with zero violations,
    terminally failed a correct bundle. For COLLECTIVE
    baselines the thunk itself is a collective; if it errors on one rank the
    engine is already unrecoverable —
    hang-avoidance beyond that is out of scope here.
    """
    try:
        expected = baseline_thunk()
        if not isinstance(expected, (tuple, list)):
            expected = (expected,)
    except Exception:  # noqa: BLE001
        try:
            baseline_refused(slot)
        except Exception:  # noqa: BLE001
            pass
        logger.exception("cacheon.audit: baseline call failed (slot=%s)", slot)
        return None
    # Snapshot allocation failure must still abort before the candidate starts.
    return tuple(e.detach().clone() if e is not None else e for e in expected)
