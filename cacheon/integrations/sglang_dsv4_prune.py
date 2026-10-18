"""Find a node that prunes more routed experts than the DeepSeek-V4.1 fork's stock prune.

The pinned fork prunes routed experts at decode and verify (its E5 rule, in
``sglang.srt.models.deepseek_v4_prune``): an expert picked by at most ``PRUNE_MAX_COUNT``
rows of a call is dropped when every row holding it drops it within ``PRUNE_BUDGET`` of
its routed mass; fork 4f335dad declares 0.4 and 1. That prune is stock and allowed.
Pruning more fails, whatever the row bar says: owner ruling 2026-10-17, after 2118ab5b
passed at budget 0.9 and queued bundles ran budgets 0.65-0.7 with max counts 2-3. The
audit runs two checks for it, both in the eager audit role:

* Stock's prune settings in the running engine must not exceed what its installed
  source declares. A bundle that raises them makes stock's own reference prune as much
  as the bundle does, which no comparison with stock can see.
* At a node with exactly one pruned MoE block, stock answers each audited decode or
  verify call once more with its prune one step heavier, alternating a budget step and
  a count step. Stock and that reference share the node's input, so they differ only in
  the experts the step drops. A row where the reference sits more than twice the row
  tolerance from stock decides: a candidate row inside tolerance is always nearer stock
  (triangle inequality), and one that dropped what the step drops is nearer the
  reference. A window of such rows mostly nearer the reference is a pruning finding.
  One-ulp router differences flip a tie on a rare row; they cannot make a majority.

The whole ``model`` node is not graded this way: its output lies dozens of routing
decisions downstream, where any perturbation flips later routing, so a pruned stack and
an honest one sit about equally far from both references. A whole-model bundle that
prunes inside its own kernels is caught only by the row bar or by owner disposition.
"""

from __future__ import annotations

import ast
import sys
from contextlib import contextmanager
from functools import cache
from pathlib import Path

import torch

from cacheon import audit as _audit

E5 = "sglang.srt.models.deepseek_v4_prune"
_SETTINGS = ("PRUNE_BUDGET", "PRUNE_MAX_COUNT", "PRUNE_MAX_ROWS")
_BUDGET_STEP = 0.1
# Decisive rows pooled per (node, graded tensor, step) before a window decides.
_ROWS = 64
_counts: dict[tuple, list[int]] = {}


@cache
def _declared(path: str) -> dict[str, float]:
    """The prune settings the installed source assigns at module level."""

    found = {
        target.id: node.value.value
        for node in ast.parse(Path(path).read_text()).body if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name) and target.id in _SETTINGS
        and isinstance(node.value, ast.Constant) and type(node.value.value) in (int, float)
    }
    if set(found) != set(_SETTINGS):
        raise RuntimeError(f"stock prune source {path} does not declare {', '.join(_SETTINGS)}")
    return found


def raised_setting() -> str | None:
    """Name a stock prune setting the running engine holds above its source, if any."""

    e5 = sys.modules.get(E5)
    if e5 is None:
        return None
    declared = _declared(e5.__file__)
    for name in _SETTINGS:
        value = getattr(e5, name, None)
        if type(value) not in (int, float) or value > declared[name]:
            return f"{name}={value!r} (source {declared[name]!r})"
    return None


def single_pruned_moe(module) -> bool:
    """Whether exactly one MoE block that stock prunes lies inside this node."""

    return sum(
        type(getattr(m, "is_hash", None)) is bool and not m.is_hash
        and not getattr(m, "is_nextn", False)
        for m in module.modules()
    ) == 1


def heavier_step(args: tuple, kwargs: dict, calls: int):
    """``(step, context)`` for this call's heavier reference, or None if stock does not prune it.

    Stock prunes inside its decoder layer's ``decode_rows`` scope (an MoE node is called
    inside it) or, for a node handed the batch, on a decode or verify batch.
    """

    e5 = sys.modules.get(E5)
    if e5 is None:
        return None
    batch = next((v for v in (*args, *kwargs.values()) if hasattr(v, "forward_mode")), None)
    if not (e5._on or (batch is not None and e5.decode_rows(batch).on)):
        return None
    declared = _declared(e5.__file__)
    step = calls % 2
    budget = declared["PRUNE_BUDGET"] + (_BUDGET_STEP if step == 0 else 0.0)
    count = declared["PRUNE_MAX_COUNT"] + step
    return step, _settings(e5, min(budget, 1.0), count)


@contextmanager
def _settings(e5, budget: float, count: int):
    saved = e5.PRUNE_BUDGET, e5.PRUNE_MAX_COUNT
    e5.PRUNE_BUDGET, e5.PRUNE_MAX_COUNT = budget, count
    try:
        yield
    finally:
        e5.PRUNE_BUDGET, e5.PRUNE_MAX_COUNT = saved


def _flat(t: torch.Tensor) -> torch.Tensor:
    return (t.detach().reshape(-1, 1) if t.dim() < 2 else t.detach()).flatten(1).float()


def distances(outputs: list, expected: list, heavier: list) -> dict[int, tuple]:
    """Per graded result tensor: the reference's row distance from stock and the candidate's from it.

    Rows line up with the node adapter's row errors: relative to stock's row norm, over
    the rows stock left finite.
    """

    found = {}
    for position, (a, (e, _), h) in enumerate(zip(outputs, expected, heavier)):
        if not (e.is_floating_point() and a.shape == e.shape == h.shape):
            continue
        a, e, h = _flat(a), _flat(e), _flat(h)
        scale = e.norm(dim=1).clamp_min(1e-12)
        keep = torch.isfinite(e).all(dim=1)
        found[position] = (((h - e).norm(dim=1) / scale)[keep], ((a - h).norm(dim=1) / scale)[keep])
    return found


def pool(slot: str, key: tuple, errors: torch.Tensor, gap: torch.Tensor,
         near: torch.Tensor, tolerance: float) -> None:
    """Pool decisive rows; record a finding when a full window is mostly nearer the reference."""

    decisive = gap > 2 * tolerance
    counts = _counts.setdefault(key, [0, 0])
    counts[0] += int(decisive.sum().item())
    counts[1] += int((decisive & (near < errors)).sum().item())
    if counts[0] < _ROWS:
        return
    if 2 * counts[1] > counts[0]:
        _audit.record_pruned(slot, f"{counts[1]} of {counts[0]} decisive rows nearer stock "
                                   f"pruned one step heavier (step {key[-1]})")
    counts[:] = [0, 0]
