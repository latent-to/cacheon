"""Pure trusted-host grading for in-engine slot-audit receipts.

The tensor comparison machinery lives in :mod:`cacheon.audit` inside the model
worker and therefore depends on torch.  Qualification controllers deliberately
do not: they only validate bounded receipt facts transported out of that worker.
Keep this module dependency-light so host regrading cannot accidentally acquire
the worker's CUDA/PyTorch runtime as an availability requirement.
"""

from __future__ import annotations

from typing import Sequence

PASS, FAIL, NO_DECISION = "PASS", "FAIL", "NO_DECISION"

# A compared call this far under its slot bar is a wrong function, not rounding.
# On record (census 2026-09-18): wrong kernels scored 0.0029, 0.0177 and 0.1748,
# a validator padding defect 0.75, and every honest receipt sat within 0.004 of
# its bar (0.9862-0.9913 against 0.985).
_GROSS_MARGIN = 0.03
# Near-miss calls tolerated per slot/member, in compared calls per near miss.
# Owner ruling 2026-09-18: one near-miss call in 7,488 must not be terminal.
_CALLS_PER_NEAR_MISS = 100


def _kernel_fault(receipt: dict) -> str | None:
    """Why this member's compared calls say the kernel is wrong, else ``None``."""

    violations = receipt.get("violations", 0)
    if receipt.get("compare_errors", 0) > 0:
        return "a candidate output could not be compared"
    if violations <= 0:
        return None
    bar = receipt.get("min_ratio")
    worst = receipt.get("worst_frac", 1.0)
    # Without a recorded bar a near miss cannot be told from a gross one.
    if type(bar) not in (int, float) or worst < bar - _GROSS_MARGIN:
        return f"a compared call scored {worst:.4f}, grossly under its bar"
    if violations * _CALLS_PER_NEAR_MISS > receipt.get("n", 0):
        return "near-miss calls exceed one in " + str(_CALLS_PER_NEAR_MISS)
    return None


def gate(
    audit_receipts: list[dict],
    *,
    min_calls: int,
    expected_slots: Sequence[str] | None = None,
    expected_member_count: int | None = None,
) -> tuple[str, str]:
    """Fold per-rank rolling receipts into PASS, FAIL or NO_DECISION.

    FAIL is a claim about the kernel and needs compared calls to make it: a call
    grossly under its bar, near misses beyond the budget, or a candidate output
    that could not be compared.  NO_DECISION is a claim about the audit: malformed
    evidence, or too few compared calls.  The timed role has already proved the
    candidate executes (``engine_worker`` raises ``CandidateNeverExecutedError``
    otherwise), so an audit that compared too little is the audit role's
    shortfall.  2026-09-16: a correct, faster bundle was failed terminally on
    zero compared calls because the audit workload took a padding path the
    adapter does not bind under.  PASS is never returned on unproven coverage.
    """
    if type(min_calls) is not int or min_calls < 1:
        return NO_DECISION, "audit minimum coverage is malformed"
    if not audit_receipts:
        return NO_DECISION, f"no audit receipts (need >= {min_calls} audited calls)"
    if (expected_slots is None) != (expected_member_count is None):
        return NO_DECISION, "audit coverage authority is incomplete"
    if any(type(receipt) is not dict for receipt in audit_receipts):
        return NO_DECISION, "audit receipt is not an object"
    coverage = None
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
            return NO_DECISION, "audit coverage authority is malformed"
        observed: dict[tuple[str, int], dict] = {}
        rank_pids: dict[int, int] = {}
        pid_ranks: dict[int, int] = {}
        for receipt in audit_receipts:
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
                return NO_DECISION, "audit slot/member receipt coverage is malformed"
            rank_pids[rank] = pid
            pid_ranks[pid] = rank
            observed[(slot, rank)] = receipt
        expected = {(slot, rank) for slot in slots for rank in range(members)}
        under = [
            (slot, rank, row.get("n"))
            for (slot, rank), row in sorted(observed.items())
            if type(row.get("n")) is not int or row["n"] < min_calls
        ]
        if set(observed) != expected:
            coverage = (
                "audit slot/member receipt coverage is incomplete "
                f"({len(observed)}/{len(expected)})"
            )
        elif under:
            coverage = (
                "audit per-slot/member coverage is insufficient; "
                f"need >= {min_calls}, under-covered={under[:8]}"
            )
    total_n = sum(r.get("n", 0) for r in audit_receipts)
    total_viol = sum(r.get("violations", 0) for r in audit_receipts)
    total_err = sum(r.get("compare_errors", 0) for r in audit_receipts)
    total_refused = sum(r.get("baseline_refused", 0) for r in audit_receipts)
    worst = min((r.get("worst_frac", 1.0) for r in audit_receipts), default=1.0)
    desc = (
        f"{total_n} audited calls, {total_viol} violations, "
        f"worst_frac={worst:.4f}, compare_errors={total_err}, "
        f"baseline_refused={total_refused}"
    )
    # The kernel's own evidence is graded before coverage: a wrong kernel on a
    # thinly covered run is still wrong.
    for receipt in audit_receipts:
        fault = _kernel_fault(receipt)
        if fault is not None:
            return FAIL, f"{desc} ({fault})"
    if coverage is not None:
        return NO_DECISION, f"{coverage}; {desc}"
    if total_n < min_calls:
        return NO_DECISION, desc + f" (insufficient coverage; need >= {min_calls})"
    return PASS, desc


def grade(receipts, policy) -> tuple[str, str]:
    """Grade typed receipt facts under the sealed policy that commissioned them."""
    return gate(
        [row.to_gate_dict() for row in receipts],
        min_calls=policy.minimum_calls,
        expected_slots=policy.expected_slots,
        expected_member_count=policy.expected_member_count,
    )


def infrastructure_failure(
    audit_receipts: list[dict],
    *,
    min_calls: int,
    expected_slots: Sequence[str] | None = None,
    expected_member_count: int | None = None,
) -> str | None:
    """Identify unavailable comparisons without turning them into miner FAIL.

    Reuse the registered gate with only numerical violations suppressed. Missing
    rank coverage, insufficient calls and reference/comparator errors then keep
    their exact diagnostics; a complete numerical mismatch remains a real FAIL.
    """
    decision, detail = gate(
        [{**row, "violations": 0} if type(row) is dict else row
         for row in audit_receipts],
        min_calls=min_calls, expected_slots=expected_slots,
        expected_member_count=expected_member_count,
    )
    return None if decision == PASS else detail
