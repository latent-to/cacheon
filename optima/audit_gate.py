"""Pure trusted-host grading for in-engine slot-audit receipts.

The tensor comparison machinery lives in :mod:`optima.audit` inside the model
worker and therefore depends on torch.  Qualification controllers deliberately
do not: they only validate bounded receipt facts transported out of that worker.
Keep this module dependency-light so host regrading cannot accidentally acquire
the worker's CUDA/PyTorch runtime as an availability requirement.
"""

from __future__ import annotations

from typing import Sequence


def gate(
    audit_receipts: list[dict],
    *,
    min_calls: int,
    expected_slots: Sequence[str] | None = None,
    expected_member_count: int | None = None,
) -> tuple[bool, str]:
    """Fold per-rank rolling receipts into a fail-closed audit verdict.

    Pass iff every required slot/rank member is represented, every member meets
    the minimum call coverage, and the receipts contain neither violations nor
    comparison errors.  This function intentionally operates on plain transport
    facts and has no tensor-runtime dependency.
    """
    if type(min_calls) is not int or min_calls < 1:
        return False, "audit minimum coverage is malformed"
    if not audit_receipts:
        return False, f"no audit receipts (need >= {min_calls} audited calls)"
    if (expected_slots is None) != (expected_member_count is None):
        return False, "audit coverage authority is incomplete"
    if expected_slots is not None:
        slots = tuple(expected_slots)
        members = expected_member_count
        if (
            not slots
            or slots != tuple(sorted(set(slots)))
            or any(not isinstance(slot, str) or not slot for slot in slots)
            or type(members) is not int
            or members < 1
        ):
            return False, "audit coverage authority is malformed"
        observed: dict[tuple[str, int], dict] = {}
        rank_pids: dict[int, int] = {}
        pid_ranks: dict[int, int] = {}
        for receipt in audit_receipts:
            if type(receipt) is not dict:
                return False, "audit receipt is not an object"
            slot = receipt.get("slot")
            pid = receipt.get("pid")
            rank = receipt.get("rank")
            world_size = receipt.get("world_size")
            if (
                slot not in slots
                or type(pid) is not int
                or pid < 1
                or type(rank) is not int
                or not 0 <= rank < members
                or world_size != members
                or (rank in rank_pids and rank_pids[rank] != pid)
                or (pid in pid_ranks and pid_ranks[pid] != rank)
                or (slot, rank) in observed
            ):
                return False, "audit slot/member receipt coverage is malformed"
            rank_pids[rank] = pid
            pid_ranks[pid] = rank
            observed[(slot, rank)] = receipt
        expected = {(slot, rank) for slot in slots for rank in range(members)}
        if set(observed) != expected:
            return False, (
                "audit slot/member receipt coverage is incomplete "
                f"({len(observed)}/{len(expected)})"
            )
        under = [
            (slot, rank, row.get("n"))
            for (slot, rank), row in sorted(observed.items())
            if type(row.get("n")) is not int or row["n"] < min_calls
        ]
        if under:
            return False, (
                "audit per-slot/member coverage is insufficient; "
                f"need >= {min_calls}, under-covered={under[:8]}"
            )
    total_n = sum(r.get("n", 0) for r in audit_receipts)
    total_viol = sum(r.get("violations", 0) for r in audit_receipts)
    total_err = sum(r.get("compare_errors", 0) for r in audit_receipts)
    worst = min((r.get("worst_frac", 1.0) for r in audit_receipts), default=1.0)
    desc = (
        f"{total_n} audited calls, {total_viol} violations, "
        f"worst_frac={worst:.4f}, compare_errors={total_err}"
    )
    if total_viol > 0:
        return False, desc
    if total_err > 0:
        return False, desc + " (audit could not compare; refusing to pass unproven)"
    if total_n < min_calls:
        return False, desc + f" (insufficient coverage; need >= {min_calls})"
    return True, desc
