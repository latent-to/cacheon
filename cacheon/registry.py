"""Validator-owned kernel registry + eligibility + active toggle.

The dispatcher (see ``cacheon/dispatch.py``) consults a single process-global
registry to decide, per call, whether to use a miner kernel or fall back to the
baseline. Validator-supplied loader code constructs the registry from the parsed
manifest after source scanning, inside the candidate worker. This defines routing
semantics; it is not a tamper boundary against code already running in that
untrusted engine.

The ``active`` flag lets the end-to-end eval flip between the *reference* run
(miner disabled -> baseline kernels) and the *candidate* run (miner enabled) in
a single engine process with identical weights, so KL and speedup isolate the
op's effect exactly.
"""

from __future__ import annotations

import re
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

from cacheon.capabilities import (
    CallDescriptor,
    CapabilityDomain,
    CapabilityMatch,
    CapabilityMismatch,
    NUMERIC_FIELDS,
    SUPPORTED_FIELDS,
    capability_domain_from_metadata,
    canonical_value,
)
from cacheon import receipts as _receipts
from cacheon.manifest import DEFAULT_VARIANT


@dataclass(frozen=True)
class Eligibility:
    """Declarative gate parsed from a bundle op's metadata.

    A miner kernel is only used when the live tensors match what it claims to
    support. Anything outside the claim falls back to baseline rather than
    risking a wrong or unsafe launch.
    """

    dtypes: frozenset[str] = frozenset()
    architectures: frozenset[str] = frozenset()
    max_last_dim: Optional[int] = None  # cap on x.shape[-1]
    # The quantization formats this kernel's (prepare, forward) handle, e.g.
    # ``{"nvfp4"}`` or ``{"fp8"}``. EMPTY (default) means the kernel takes DENSE
    # (unquantized) expert weights only. The MoE dispatcher pairs a kernel to a layer
    # by format: a dense layer runs only an empty-``quant`` kernel; a quantized layer
    # runs only a kernel that declares its exact format (else fall back to the trusted
    # baseline). This is the gate that lets an NVFP4 expert kernel reach the seam without
    # feeding a dense kernel packed FP4 bytes + separate scales it would mis-read.
    quant: frozenset[str] = frozenset()
    # Cap on the token/row count (x.shape[0]) the kernel claims. A kernel with a
    # MEASURED dispatch window (e.g. the fused AR+norm epilogue wins at decode
    # T<=1024 and must fall through for prefill-sized T) declares it here, so the
    # seam routes oversized calls to the trusted baseline instead of trusting the
    # kernel to decline. None -> no cap.
    max_num_tokens: Optional[int] = None
    # Floor on the token/row count. The measured counterpart of max_num_tokens for
    # kernels that LOSE at tiny T (the deep fused epilogue wins at decode T>=48 but
    # loses at long-ctx T=4 — the vendor tuner's finalize-fused tactics win there);
    # below the floor the seam serves the stock path (or a sibling shallow kernel).
    # For the deep slot the EXPORT side applies the same gate, so an ineligible T
    # never has its finalize skipped in the first place. None -> no floor.
    min_num_tokens: Optional[int] = None

    # Normative, named specialization predicates.  Legacy fields above remain
    # supported while arena bindings migrate from the positional lookup API to
    # ``KernelRegistry.select(CallDescriptor(...))``.
    capabilities: CapabilityDomain = field(default_factory=CapabilityDomain)

    def __post_init__(self) -> None:
        try:
            object.__setattr__(
                self,
                "dtypes",
                frozenset(str(canonical_value("dtype", value)) for value in self.dtypes),
            )
            object.__setattr__(
                self,
                "architectures",
                frozenset(
                    str(canonical_value("architecture", value))
                    for value in self.architectures
                ),
            )
            object.__setattr__(
                self,
                "quant",
                frozenset(str(canonical_value("quant", value)) for value in self.quant),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid legacy eligibility domain: {exc}") from exc
        for name in ("max_last_dim", "min_num_tokens", "max_num_tokens"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError(f"{name} must be a non-negative integer or None")
        if (
            self.min_num_tokens is not None
            and self.max_num_tokens is not None
            and self.min_num_tokens > self.max_num_tokens
        ):
            raise ValueError("min_num_tokens must not exceed max_num_tokens")

    def match(self, descriptor: CallDescriptor) -> CapabilityMatch:
        """Match a complete canonical descriptor, including legacy constraints.

        Unlike ``accepts`` this API is fail-closed when a constrained legacy
        field is missing.  New integrations should call ``KernelRegistry.select``
        with every dimension/context value they own and inspect its decision.
        """
        mismatches: list[CapabilityMismatch] = []

        def _missing(field_name: str, expected: str) -> None:
            mismatches.append(CapabilityMismatch(field_name, "missing", expected))

        def _outside(field_name: str, expected: str, actual: Any) -> None:
            mismatches.append(
                CapabilityMismatch(field_name, "outside_domain", expected, actual)
            )

        if self.dtypes:
            if "dtype" not in descriptor:
                _missing("dtype", f"one of {sorted(self.dtypes)!r}")
            elif descriptor["dtype"] not in self.dtypes:
                _outside("dtype", f"one of {sorted(self.dtypes)!r}", descriptor["dtype"])
        if self.architectures:
            if "architecture" not in descriptor:
                _missing("architecture", f"one of {sorted(self.architectures)!r}")
            elif descriptor["architecture"] not in self.architectures:
                _outside(
                    "architecture",
                    f"one of {sorted(self.architectures)!r}",
                    descriptor["architecture"],
                )
        if self.max_last_dim is not None:
            if "last_dim" not in descriptor:
                _missing("last_dim", f"at most {self.max_last_dim}")
            elif descriptor["last_dim"] > self.max_last_dim:
                _outside("last_dim", f"at most {self.max_last_dim}", descriptor["last_dim"])
        if self.max_num_tokens is not None or self.min_num_tokens is not None:
            lo = "-inf" if self.min_num_tokens is None else str(self.min_num_tokens)
            hi = "+inf" if self.max_num_tokens is None else str(self.max_num_tokens)
            expected = f"in [{lo}, {hi}]"
            if "num_tokens" not in descriptor:
                _missing("num_tokens", expected)
            else:
                value = descriptor["num_tokens"]
                if ((self.min_num_tokens is not None and value < self.min_num_tokens)
                        or (self.max_num_tokens is not None and value > self.max_num_tokens)):
                    _outside("num_tokens", expected, value)
        if self.quant:
            if "quant" not in descriptor:
                _missing("quant", f"one of {sorted(self.quant)!r}")
            elif descriptor["quant"] not in self.quant:
                _outside("quant", f"one of {sorted(self.quant)!r}", descriptor["quant"])
        elif descriptor.get("quant") not in (None, "dense"):
            _outside("quant", "dense", descriptor["quant"])
        mismatches.extend(self.capabilities.match(descriptor).mismatches)
        # A field may be constrained by both legacy and new metadata.  Preserve
        # the intersection semantics but avoid duplicate identical diagnostics.
        deduped: list[CapabilityMismatch] = []
        for mismatch in mismatches:
            if mismatch not in deduped:
                deduped.append(mismatch)
        return CapabilityMatch(tuple(deduped))


@dataclass
class _FieldConstraint:
    """Internal intersection model used only for registration-time overlap checks."""

    allowed: set[Any] | None = None
    minimum: int | None = None
    maximum: int | None = None
    excluded: set[Any] = field(default_factory=set)

    def allow(self, values: set[Any]) -> None:
        self.allowed = values if self.allowed is None else self.allowed & values

    def range(self, minimum: int | None, maximum: int | None) -> None:
        if minimum is not None:
            self.minimum = minimum if self.minimum is None else max(self.minimum, minimum)
        if maximum is not None:
            self.maximum = maximum if self.maximum is None else min(self.maximum, maximum)

    def is_empty(self, *, numeric: bool) -> bool:
        minimum = self.minimum
        maximum = self.maximum
        if numeric:
            minimum = 0 if minimum is None else max(0, minimum)
            if maximum is not None and minimum > maximum:
                return True
        if self.allowed is not None:
            for value in self.allowed:
                if value in self.excluded:
                    continue
                if isinstance(value, bool) or not isinstance(value, int):
                    if self.minimum is None and self.maximum is None:
                        return False
                    continue
                if minimum is not None and value < minimum:
                    continue
                if maximum is not None and value > maximum:
                    continue
                return False
            return True
        # Text domains are infinite, so a finite exclusion set cannot empty one.
        if not numeric or maximum is None:
            return False
        assert minimum is not None
        # Numeric descriptor values are non-negative integers.  A finite interval
        # is empty only when every value is explicitly excluded.
        width = maximum - minimum + 1
        if width > len(self.excluded):
            return False
        return all(value in self.excluded for value in range(minimum, maximum + 1))


def _add_eligibility_constraints(
    constraints: dict[str, _FieldConstraint], eligibility: Eligibility
) -> bool:
    """Fold one Eligibility into constraints; return whether analysis is complete."""

    def _field(name: str) -> _FieldConstraint:
        return constraints.setdefault(name, _FieldConstraint())

    if eligibility.dtypes:
        _field("dtype").allow(set(eligibility.dtypes))
    if eligibility.architectures:
        _field("architecture").allow(set(eligibility.architectures))
    if eligibility.max_last_dim is not None:
        _field("last_dim").range(None, eligibility.max_last_dim)
    if eligibility.min_num_tokens is not None or eligibility.max_num_tokens is not None:
        _field("num_tokens").range(
            eligibility.min_num_tokens, eligibility.max_num_tokens
        )
    _field("quant").allow(set(eligibility.quant) or {"dense"})

    complete = True
    for predicate in eligibility.capabilities.predicates:
        # CapabilityDomain deliberately permits programmatic future/private fields
        # even though public metadata cannot declare them.  Their matching semantics
        # may evolve, so defer those intersections to selection-time.
        if predicate.field not in SUPPORTED_FIELDS:
            complete = False
        constraint = _field(predicate.field)
        if predicate.allowed:
            constraint.allow(set(predicate.allowed))
        else:
            constraint.range(predicate.minimum, predicate.maximum)
    return complete


def eligibility_domains_overlap(
    left: Eligibility, right: Eligibility
) -> bool | None:
    """Return whether two domains overlap, or ``None`` when it cannot be proven.

    Public capability metadata and all legacy eligibility fields are fully
    analyzable.  Programmatically constructed future/private predicates are
    intentionally deferred; ``KernelRegistry.select`` remains the final
    fail-closed ambiguity guard for those.
    """

    constraints: dict[str, _FieldConstraint] = {}
    complete = _add_eligibility_constraints(constraints, left)
    complete = _add_eligibility_constraints(constraints, right) and complete
    for name, constraint in constraints.items():
        if constraint.is_empty(numeric=name in NUMERIC_FIELDS):
            return False
    return True if complete else None


def eligibility_domain_is_empty(eligibility: Eligibility) -> bool | None:
    """Return whether one domain is contradictory, or ``None`` if not provable."""

    constraints: dict[str, _FieldConstraint] = {}
    complete = _add_eligibility_constraints(constraints, eligibility)
    for name, constraint in constraints.items():
        if constraint.is_empty(numeric=name in NUMERIC_FIELDS):
            return True
    return False if complete else None


class VariantRegistrationError(ValueError):
    """A slot variant cannot be registered without making routing unambiguous."""


_VARIANT_RE = re.compile(r"^[0-9A-Za-z._\-]+$")


@dataclass
class KernelImpl:
    slot: str
    bundle_id: str
    entry: Callable[..., Any]  # called as entry(*inputs, out)
    # Optional 2nd callable for (prepare, forward) slots (e.g. moe.fused_experts): the
    # validator-owned dispatcher runs it ONCE on the layer's raw weights at the first
    # call and memoizes the result, then passes that `prepared` to `entry` each step.
    # None for plain forward-only slots (silu / rmsnorm / attention).
    prepare: Optional[Callable[..., Any]] = None
    eligibility: Eligibility = field(default_factory=Eligibility)
    variant: str = DEFAULT_VARIANT


# Public architecture spelling; KernelImpl remains the compatibility name used
# throughout existing dispatchers and tests.
KernelVariant = KernelImpl


class SelectionOutcome(str, Enum):
    """Validator-owned routing result; every non-selected outcome means stock."""

    SELECTED = "selected"
    REGISTRY_INACTIVE = "registry_inactive"
    SLOT_UNREGISTERED = "slot_unregistered"
    OUT_OF_DOMAIN = "out_of_domain"
    AMBIGUOUS = "ambiguous"


@dataclass(frozen=True)
class VariantCapabilityMatch:
    """Diagnostic match result for one registered variant."""

    variant: str
    match: CapabilityMatch


@dataclass(frozen=True)
class SelectionDecision:
    slot: str
    descriptor: CallDescriptor
    outcome: SelectionOutcome
    candidate: KernelImpl | None = None
    capability_match: CapabilityMatch | None = None
    variant_matches: tuple[VariantCapabilityMatch, ...] = ()

    @property
    def impl(self) -> KernelImpl | None:
        return self.candidate if self.outcome is SelectionOutcome.SELECTED else None

    @property
    def use_candidate(self) -> bool:
        return self.impl is not None


class KernelRegistry:
    """Process-global registry. One active bundle at a time (MVP)."""

    def __init__(self) -> None:
        # Registration order is retained for reproducible diagnostics and load
        # behavior, but never acts as routing priority: exactly one variant must
        # match a call or stock is served.
        self._by_slot: dict[str, list[KernelImpl]] = {}
        self._active: bool = False
        self._lock = threading.Lock()

    # ---- registration (validator-side) ----

    def register(self, impl: KernelImpl) -> None:
        with self._lock:
            if (
                not isinstance(impl.variant, str)
                or not impl.variant
                or not _VARIANT_RE.fullmatch(impl.variant)
            ):
                raise VariantRegistrationError(
                    "kernel variant must be a non-empty simple identifier"
                )
            if eligibility_domain_is_empty(impl.eligibility) is True:
                raise VariantRegistrationError(
                    f"variant {impl.variant!r} for slot {impl.slot!r} has an empty "
                    "or contradictory capability domain"
                )
            variants = self._by_slot.setdefault(impl.slot, [])
            if any(existing.variant == impl.variant for existing in variants):
                raise VariantRegistrationError(
                    f"duplicate variant {impl.variant!r} for slot {impl.slot!r}"
                )
            for existing in variants:
                overlap = eligibility_domains_overlap(
                    existing.eligibility, impl.eligibility
                )
                if overlap is True:
                    raise VariantRegistrationError(
                        f"overlapping capability domains for slot {impl.slot!r}: "
                        f"variants {existing.variant!r} and {impl.variant!r}"
                    )
            variants.append(impl)

    def clear(self) -> None:
        with self._lock:
            self._by_slot.clear()
            self._active = False

    # ---- active toggle (used by the eval to swap reference<->candidate) ----

    def enable(self) -> None:
        self._active = True

    def disable(self) -> None:
        self._active = False

    @property
    def active(self) -> bool:
        return self._active

    # ---- lookup (dispatcher-side, hot path) ----

    def select(
        self,
        slot: str,
        descriptor: CallDescriptor,
    ) -> SelectionDecision:
        """Choose candidate or baseline for a canonical live call.

        The decision is validator-owned and diagnostic: out-of-domain calls
        carry structured mismatches rather than being indistinguishable from an
        unregistered slot.  No miner code runs here.  Existing dispatchers can
        continue using ``lookup`` until their arena bindings populate complete
        descriptors.
        """
        if not self._active:
            return SelectionDecision(
                slot, descriptor, SelectionOutcome.REGISTRY_INACTIVE
            )
        variants = self._by_slot.get(slot)
        if not variants:
            return SelectionDecision(
                slot, descriptor, SelectionOutcome.SLOT_UNREGISTERED
            )
        variant_matches = tuple(
            VariantCapabilityMatch(impl.variant, impl.eligibility.match(descriptor))
            for impl in variants
        )
        accepted = [
            (impl, result.match)
            for impl, result in zip(variants, variant_matches)
            if result.match.accepted
        ]
        if not accepted:
            only_impl = variants[0] if len(variants) == 1 else None
            only_match = variant_matches[0].match if len(variants) == 1 else None
            _receipts.not_selected(
                slot,
                SelectionOutcome.OUT_OF_DOMAIN.value,
                # Every variant's reasons, deduplicated by the receipt: with two
                # variants the useful answer is why BOTH declined, not why one did.
                {m for row in variant_matches for m in row.match.mismatches},
            )
            return SelectionDecision(
                slot,
                descriptor,
                SelectionOutcome.OUT_OF_DOMAIN,
                candidate=only_impl,
                capability_match=only_match,
                variant_matches=variant_matches,
            )
        if len(accepted) != 1:
            # Registration catches every overlap expressible in today's public
            # capability vocabulary.  This remains the authoritative guard for
            # future/private predicates whose intersection was not provable then.
            _receipts.not_selected(
                slot, SelectionOutcome.AMBIGUOUS.value, ()
            )
            return SelectionDecision(
                slot,
                descriptor,
                SelectionOutcome.AMBIGUOUS,
                variant_matches=variant_matches,
            )
        impl, match = accepted[0]
        return SelectionDecision(
            slot,
            descriptor,
            SelectionOutcome.SELECTED,
            candidate=impl,
            capability_match=match,
            variant_matches=variant_matches,
        )

    def variants(self, slot: str) -> tuple[KernelImpl, ...]:
        """Return registered variants for ``slot`` in deterministic load order."""
        return tuple(self._by_slot.get(slot, ()))

    def slots(self) -> list[str]:
        return sorted(self._by_slot)


# A single process-global registry. The dispatcher and the eval both reach this.
REGISTRY = KernelRegistry()


def eligibility_from_metadata(
    meta: dict | None,
    manifest_dtypes: tuple[str, ...],
    manifest_architectures: tuple[str, ...] = (),
) -> Eligibility:
    """Build an Eligibility from a bundle op's metadata json (+ manifest dtypes)."""
    if meta is None:
        meta = {}
    if not isinstance(meta, Mapping):
        raise ValueError("eligibility metadata must be a JSON object")

    def _string_values(name: str) -> set[str]:
        values = meta.get(name, ())
        if isinstance(values, str) or not isinstance(
            values, (list, tuple, set, frozenset)
        ):
            raise ValueError(f"metadata {name!r} must be a list of strings")
        if any(not isinstance(value, str) or not value.strip() for value in values):
            raise ValueError(f"metadata {name!r} must contain non-empty strings")
        return {value for value in values}

    def _canonical_declared(
        field_name: str, values: set[str] | tuple[str, ...]
    ) -> set[str]:
        try:
            return {
                str(canonical_value(field_name, value))
                for value in values
            }
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"invalid declared {field_name} eligibility: {exc}"
            ) from exc

    def _intersection(
        field_name: str, manifest_values: tuple[str, ...], metadata_name: str
    ) -> set[str]:
        manifest_set = _canonical_declared(field_name, manifest_values)
        metadata_set = _canonical_declared(
            field_name, _string_values(metadata_name)
        )
        if manifest_set and metadata_set:
            combined = manifest_set & metadata_set
            if not combined:
                raise ValueError(
                    f"manifest and metadata {metadata_name!r} eligibility are disjoint"
                )
            return combined
        return manifest_set or metadata_set

    dtypes = _intersection("dtype", manifest_dtypes, "dtypes")
    archs = _intersection(
        "architecture", manifest_architectures, "architectures"
    )
    def _optional_bound(name: str) -> int | None:
        value = meta.get(name)
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"metadata {name!r} must be a non-negative integer")
        return value

    return Eligibility(
        dtypes=frozenset(dtypes),
        architectures=frozenset(archs),
        max_last_dim=_optional_bound("max_last_dim"),
        quant=frozenset(_string_values("quant")),
        max_num_tokens=_optional_bound("max_num_tokens"),
        min_num_tokens=_optional_bound("min_num_tokens"),
        capabilities=capability_domain_from_metadata(meta),
    )
