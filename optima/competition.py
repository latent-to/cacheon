"""Validator-owned competition target resolution.

A manifest's op list answers "what code should be loaded?".  It does not by
itself answer the distinct settlement question "what incumbent may this score
replace?".  In particular, treating ``ops[0]`` as the latter loses the identity
of an atomic multi-op improvement.

The registry below is deliberately small and exact.  Miners may request a
target, but only validator code defines its canonical member set.  Legacy
multi-op bundles can still parse, scan, and verify; they are non-crownable until
their exact semantic competition has been registered.  The already-proven deep
fused-epilogue pair retains a visible legacy *semantic* mapping, but its
miner-controlled scheduler Python is no longer grandfathered across the component
trust boundary; it migrates to whole-serving system qualification.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Mapping

from optima.manifest import Manifest


SLOT_MODE = "slot"
ATOMIC_MODE = "atomic"
SYSTEM_MODE = "system"

# Target IDs and member order are validator-owned.  Member order is semantic:
# callers must use the resolved order for receipts/serialization rather than
# preserving miner-controlled manifest order.
_ATOMIC_TARGETS = {
    "collective.moe_epilogue.v1": (
        "collective.ar_residual_rmsnorm",
        "collective.moe_finalize_ar_rmsnorm",
    ),
}
ATOMIC_TARGETS: Mapping[str, tuple[str, ...]] = MappingProxyType(_ATOMIC_TARGETS)


class CompetitionError(ValueError):
    """A manifest cannot compete for its requested settlement target."""


class CompetitionLegacyWarning(UserWarning):
    """A legacy manifest relies on validator compatibility inference."""


@dataclass(frozen=True)
class ResolvedCompetition:
    """Settlement-facing identity resolved independently of manifest op order."""

    target: str | None
    mode: str | None
    members: tuple[str, ...]
    crownable: bool
    legacy: bool = False
    reason: str | None = None

    def require_crownable(self) -> "ResolvedCompetition":
        """Return ``self`` or fail closed with the resolver's concrete reason."""
        if not self.crownable:
            raise CompetitionError(self.reason or "competition target is not crownable")
        return self


def _member_set(manifest: Manifest) -> frozenset[str]:
    return frozenset(op.slot for op in manifest.ops)


def _declared_members(manifest: Manifest) -> tuple[str, ...]:
    """Semantic slots in first-declaration order, independent of variant count."""
    return tuple(dict.fromkeys(op.slot for op in manifest.ops))


def _registered_target_for_members(members: frozenset[str]) -> str | None:
    matches = [
        target
        for target, canonical in ATOMIC_TARGETS.items()
        if frozenset(canonical) == members
    ]
    if len(matches) > 1:  # validator programming error, never miner ambiguity
        raise RuntimeError(
            "multiple atomic competition targets register the same member set: "
            f"{sorted(matches)!r}"
        )
    return matches[0] if matches else None


def _finish(
    resolved: ResolvedCompetition,
    *,
    for_settlement: bool,
) -> ResolvedCompetition:
    return resolved.require_crownable() if for_settlement else resolved


def _finish_component(
    manifest: Manifest,
    resolved: ResolvedCompetition,
    *,
    for_settlement: bool,
) -> ResolvedCompetition:
    """Apply the host-execution trust boundary after semantic resolution.

    Slots and atomic targets describe what is graded, not what code is safe to
    execute as a component.  A miner Python wrapper in the scheduler remains
    arbitrary host control even when ``setup`` is absent.  Only registered
    validator-device ABIs may therefore emit a component settlement identity.
    """

    if resolved.crownable:
        from optima.device_component import component_crown_rejection

        reason = component_crown_rejection(manifest)
        if reason is not None:
            resolved = replace(resolved, crownable=False, reason=reason)
    return _finish(resolved, for_settlement=for_settlement)


def resolve_competition(
    manifest: Manifest,
    *,
    for_settlement: bool = False,
    warn_legacy: bool = True,
) -> ResolvedCompetition:
    """Resolve a manifest to a validator-owned competition identity.

    ``for_settlement=False`` is useful during intake: an undeclared, unknown
    multi-op bundle receives a structured non-crownable result but remains free
    to scan and verify.  Settlement callers pass ``for_settlement=True`` and get
    a fail-closed :class:`CompetitionError` instead.

    ``warn_legacy`` makes the grandfathered deep bundle warning explicitly
    suppressible for machine callers while retaining a dedicated warning class
    for normal Python warning filters and tests.
    """
    declared = manifest.competition
    members = _member_set(manifest)
    declared_members = _declared_members(manifest)

    # A system patch is one arena-scoped serving product, never a collection of
    # component slots.  The manifest requests an identity; validator policy owns
    # the registered target, source dependency, semantic region, and admissible
    # arenas.  Keep ``members`` empty so downstream component receipt/reward code
    # cannot accidentally manufacture slot champions from a system qualification.
    if manifest.system is not None:
        if members:
            raise CompetitionError(
                "system competition may not contain component op identities"
            )
        if declared is None:
            return _finish(
                ResolvedCompetition(
                    target=None,
                    mode=SYSTEM_MODE,
                    members=(),
                    crownable=False,
                    reason=(
                        "system submissions require an explicit validator-owned "
                        "[competition] target with mode='system'"
                    ),
                ),
                for_settlement=for_settlement,
            )
        if declared.mode != SYSTEM_MODE:
            raise CompetitionError(
                "a [system] product must request competition mode 'system', not "
                f"{declared.mode!r}"
            )
        from optima.system_patch import SYSTEM_TARGETS

        policy = SYSTEM_TARGETS.get(declared.target)
        if policy is None:
            raise CompetitionError(
                f"unknown system competition target {declared.target!r}; target IDs "
                "and arena admission are validator-owned"
            )
        if (
            manifest.system.target != policy.source_target
            or manifest.system.region != policy.region
        ):
            raise CompetitionError(
                f"system competition target {declared.target!r} requires "
                f"target/region {(policy.source_target, policy.region)!r}; manifest "
                f"declares {(manifest.system.target, manifest.system.region)!r}"
            )
        return ResolvedCompetition(
            target=declared.target,
            mode=SYSTEM_MODE,
            members=(),
            crownable=True,
        )

    if declared is not None and declared.mode == SYSTEM_MODE:
        # Miner-controlled scheduler Python is not a component merely because it
        # was packaged as [[ops]].  Preserve that useful, inspectable bundle format
        # while settling the entire candidate as one externally-qualified serving
        # product.  It receives no component members/receipts/champions.
        from optima.device_component import (
            UNTRUSTED_HOST_SYSTEM_TARGET,
            untrusted_host_system_rejection,
        )

        if declared.target != UNTRUSTED_HOST_SYSTEM_TARGET:
            raise CompetitionError(
                f"untrusted-host op bundles may request only the validator-owned "
                f"whole-serving system target {UNTRUSTED_HOST_SYSTEM_TARGET!r}, not "
                f"{declared.target!r}"
            )
        reason = untrusted_host_system_rejection(manifest)
        if reason is not None:
            raise CompetitionError(reason)
        return ResolvedCompetition(
            target=UNTRUSTED_HOST_SYSTEM_TARGET,
            mode=SYSTEM_MODE,
            members=(),
            crownable=True,
        )

    # A singleton's historical settlement identity was already unambiguous.
    if declared is None and len(members) == 1:
        slot = declared_members[0]
        return _finish_component(
            manifest,
            ResolvedCompetition(
                target=slot,
                mode=SLOT_MODE,
                members=(slot,),
                crownable=True,
                legacy=True,
            ),
            for_settlement=for_settlement,
        )

    if declared is None:
        registered = _registered_target_for_members(members)
        if registered is not None:
            canonical = ATOMIC_TARGETS[registered]
            if warn_legacy:
                warnings.warn(
                    f"legacy multi-op bundle {manifest.bundle_id!r} implicitly resolves "
                    f"to atomic competition target {registered!r}; add "
                    "[competition] target/mode to manifest.toml",
                    CompetitionLegacyWarning,
                    stacklevel=2,
                )
            return _finish_component(
                manifest,
                ResolvedCompetition(
                    target=registered,
                    mode=ATOMIC_MODE,
                    members=canonical,
                    crownable=True,
                    legacy=True,
                ),
                for_settlement=for_settlement,
            )

        reason = (
            f"legacy multi-op bundle {manifest.bundle_id!r} has no registered exact "
            f"competition target for members {tuple(sorted(members))!r}; it may verify "
            "but cannot settle"
        )
        return _finish_component(
            manifest,
            ResolvedCompetition(
                target=None,
                mode=None,
                members=tuple(sorted(members)),
                crownable=False,
                legacy=True,
                reason=reason,
            ),
            for_settlement=for_settlement,
        )

    if declared.mode == SLOT_MODE:
        if len(members) != 1:
            raise CompetitionError(
                f"slot competition target {declared.target!r} requires exactly one op "
                f"identity (one semantic slot; variants are allowed); manifest declares "
                f"{declared_members!r}"
            )
        slot = declared_members[0]
        if declared.target != slot:
            raise CompetitionError(
                f"slot competition target {declared.target!r} does not match manifest "
                f"slot {slot!r}"
            )
        return _finish_component(
            manifest,
            ResolvedCompetition(
                target=slot,
                mode=SLOT_MODE,
                members=(slot,),
                crownable=True,
            ),
            for_settlement=for_settlement,
        )

    # Manifest parsing admits only slot/atomic modes, but keep this module safe
    # for programmatically constructed Manifest objects as well.
    if declared.mode != ATOMIC_MODE:
        raise CompetitionError(f"unknown competition mode {declared.mode!r}")

    canonical = ATOMIC_TARGETS.get(declared.target)
    if canonical is None:
        raise CompetitionError(
            f"unknown atomic competition target {declared.target!r}; target IDs are "
            "validator-owned"
        )
    if members != frozenset(canonical):
        raise CompetitionError(
            f"atomic competition target {declared.target!r} requires exact members "
            f"{canonical!r}; manifest declares {declared_members!r}"
        )
    return _finish_component(
        manifest,
        ResolvedCompetition(
            target=declared.target,
            mode=ATOMIC_MODE,
            members=canonical,
            crownable=True,
        ),
        for_settlement=for_settlement,
    )
