"""Validator-owned servability policy for registered contribution targets.

The target catalog answers *identity*: which registered semantic delta a
proposal claims.  This module answers a separate operational question the
catalog deliberately does not: can the live arena actually route execution
through that target's chokepoint?  A registered target can be structurally
unservable in two distinct ways:

1. **No seam adapter serves the slot.**  The seam table
   (:data:`cacheon.seams.SEAM_ADAPTERS`) is the single source of truth for
   the sglang chokepoints validator code can patch.  A target whose member
   slot appears in no adapter row can never dispatch a candidate kernel on
   any arena.  This class is derived automatically from the table, so
   adding or deleting a seam updates admission with no parallel list.

2. **An adapter exists but the arena model never calls the chokepoint.**
   Example: MiniMax-M3 instantiates ``GemmaRMSNorm`` — a sibling class, not
   a subclass, of ``RMSNorm`` — at every norm site, so the ``norm.rmsnorm``
   seam installs cleanly and then never fires.  Native-lane execution
   coverage is deterministically 0/N and every submission burns to
   ``CandidateNeverExecutedError`` after consuming an evaluation lease.
   This class cannot be derived without engine evidence; until a
   commission-time stock probe records per-slot site-reach receipts, it is
   the hand-audited deny-list below.

Intake admission consults :func:`unservable_target_reason` immediately after
fingerprinting and fails such reveals with the deterministic reason
``unservable_target:<target_id>`` instead of publishing them toward an
evaluation lease they can only lose.

Catalog identity is deliberately untouched: unservable targets stay
registered, so catalog/stack digests, historical claims, and copy
fingerprints remain stable.  Servability is admission policy, not identity.

Import-light on purpose (stdlib only), like the two modules it reads.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping

from cacheon.seams import SEAM_ADAPTERS
from cacheon.target_catalog import TargetCatalog, default_target_catalog

# Every slot that at least one seam adapter row serves.  Derived, never listed.
ADAPTER_SERVED_SLOTS: frozenset[str] = frozenset(
    slot for adapter in SEAM_ADAPTERS for slot in adapter.slots
)

# Class-2 deny-list: adapter-served targets whose chokepoint the live arena
# model structurally never reaches.  Keys must be registered catalog targets
# (a test enforces this).  Entries require an engine-evidence citation; they
# are validator policy, reviewed and shipped like any other consensus change.
KNOWN_UNSERVABLE_TARGETS: Mapping[str, str] = MappingProxyType(
    {
        "norm.rmsnorm": (
            "arena model MiniMax-M3 instantiates GemmaRMSNorm (a sibling "
            "class of RMSNorm) at every norm site, so RMSNorm.forward_cuda "
            "never executes and native-lane coverage is structurally 0/N "
            "(root-caused 2026-08-19 from reservation 495ad7fa)"
        ),
    }
)


def unservable_target_reason(
    target_id: str,
    *,
    catalog: TargetCatalog | None = None,
) -> str | None:
    """Return why ``target_id`` cannot execute on the live arena, or ``None``.

    Unknown target IDs raise :class:`cacheon.target_catalog.TargetResolutionError`
    exactly like every other catalog lookup; servability is only defined for
    registered identities.
    """

    active = catalog or default_target_catalog()
    spec = active.require(target_id)
    known = KNOWN_UNSERVABLE_TARGETS.get(target_id)
    if known is not None:
        return known
    unserved = tuple(
        member for member in spec.members if member not in ADAPTER_SERVED_SLOTS
    )
    if unserved:
        return f"no seam adapter serves member slot(s) {unserved!r}"
    for member in spec.members:
        member_known = KNOWN_UNSERVABLE_TARGETS.get(member)
        if member_known is not None:
            return f"member {member!r} is unservable: {member_known}"
    return None


def unservable_targets(
    *, catalog: TargetCatalog | None = None
) -> dict[str, str]:
    """Map every unservable registered target to its reason (empty if none)."""

    active = catalog or default_target_catalog()
    result: dict[str, str] = {}
    for row in active.snapshot()["targets"]:
        target_id = row["target_id"]
        reason = unservable_target_reason(target_id, catalog=active)
        if reason is not None:
            result[target_id] = reason
    return result
