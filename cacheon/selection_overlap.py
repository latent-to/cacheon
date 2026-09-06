"""Set-overlap grading of selected-index rows, shared by verify and the live audit."""

from __future__ import annotations

from typing import Optional

import torch

NOTHING_SELECTED = "reference selected no blocks"


def selection_overlap(actual: torch.Tensor, expected: torch.Tensor, *,
                      bounds: Optional[tuple[int, int]] = None) -> tuple[float, str]:
    """Return the mean valid-row set overlap and a non-empty reason for a hard rejection.

    Positions the reference left at -1 must stay -1, every candidate index must
    lie inside the declared physical range, and a reference that selected nothing
    has nothing to grade. Both graders share this so an offline PASS and an
    in-engine violation can never disagree on what counts as a selection.
    """
    ai, ei = actual.to(torch.long), expected.to(torch.long)
    valid = ei >= 0
    if bool((ai[~valid] != -1).any()):
        return 0.0, "selection padding was not rewritten to -1"
    if bounds is not None and bool(((ai < bounds[0]) | (ai > bounds[1])).any()):
        return 0.0, f"selection index outside declared range {bounds}"
    rows = valid.any(dim=-1)
    if not bool(rows.any()):
        return 0.0, NOTHING_SELECTED
    ordered = ai.sort(dim=-1).values
    positions = torch.searchsorted(ordered, ei.contiguous()).clamp(max=ai.shape[-1] - 1)
    hit = ordered.gather(-1, positions) == ei
    overlap = (hit & valid).sum(-1).float() / valid.sum(-1).clamp(min=1)
    return float(overlap[rows].mean()), ""
