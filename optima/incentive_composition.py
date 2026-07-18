"""Pure reviewed-discovery and registered-CROWN incentive composition.

The registered-CROWN arithmetic remains in :mod:`optima.finite_debt`.  This
module adds one separately versioned, finite reviewed-discovery class and the
deterministic epoch composition between the two classes.  It owns no database,
chain client, wallet, settlement router, publication journal, or selected
parameter defaults.

One discovery review chooses exactly one economic path: promotion into a
registered target (which issues no discovery debt), or a bounded bounty.  The
two payout classes are allocated independently by remaining principal and only
then aggregated by hotkey::

    P_d = min(discovery_cap, live_discovery_debt)
    P_c = min(1_000_000 - reserve_floor - P_d, live_crown_debt)
    reserve = 1_000_000 - P_d - P_c

All arithmetic is integer and every remainder tie is broken by claim digest.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Mapping

from optima._strict import require_digest, require_exact_fields, require_int
from optima.finite_debt import (
    PPM,
    REFERENCE_EPOCH_UNITS,
    DebtClaimState,
    DebtHotkeyWeight,
    FiniteDebtError,
    FiniteDebtPolicyManifest,
    pay_claim_balance,
)
from optima.stack_identity import canonical_digest


COMPOSITION_POLICY_SCHEMA_VERSION = 1
COMPOSITION_POLICY_VERSION = "optima.incentive-composition.v1"
DISCOVERY_BOUNTY_ONLY = "bounty_only"
DISCOVERY_REGISTERED_PROMOTION = "registered_promotion"
DISCOVERY_DECISIONS = frozenset(
    {DISCOVERY_BOUNTY_ONLY, DISCOVERY_REGISTERED_PROMOTION}
)
DISCOVERY_BALANCE_STATUSES = frozenset({"open", "paid", "expired", "cancelled"})
_HOTKEY = re.compile(r"[^\s]{1,256}\Z")


class IncentiveCompositionError(ValueError):
    """A composition policy, review, claim, balance, or projection is invalid."""


def _integer(
    value: object,
    field: str,
    *,
    minimum: int | None = 0,
    maximum: int | None = None,
) -> int:
    return require_int(
        value,
        field=field,
        error=IncentiveCompositionError,
        minimum=minimum,
        maximum=maximum,
    )


def _digest(value: object, field: str, *, optional: bool = False) -> str:
    if optional and value == "":
        return ""
    return require_digest(value, field=field, error=IncentiveCompositionError)


def _hotkey(value: object, field: str = "hotkey") -> str:
    if not isinstance(value, str) or _HOTKEY.fullmatch(value) is None:
        raise IncentiveCompositionError(f"{field} is malformed")
    return value


def _strict(value: object, fields: set[str], label: str) -> dict[str, object]:
    return dict(
        require_exact_fields(
            value,
            fields=frozenset(fields),
            label=label,
            error=IncentiveCompositionError,
            exact_dict=True,
        )
    )


def _reason(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 2_048
        or any(char in value for char in "\x00\r\n")
    ):
        raise IncentiveCompositionError(f"{field} is malformed")
    return value


@dataclass(frozen=True)
class IncentiveCompositionPolicyManifest:
    """Complete parameters for one reviewed-discovery composition policy."""

    innovation_policy_digest: str
    selection_report_digest: str
    reserve_ppm: int
    epoch_blocks: int
    discovery_cap_units: int
    per_award_principal_cap_epochs: int
    discovery_lifetime_blocks: int
    schema_version: int = COMPOSITION_POLICY_SCHEMA_VERSION
    policy_version: str = COMPOSITION_POLICY_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "innovation_policy_digest",
            _digest(self.innovation_policy_digest, "innovation_policy_digest"),
        )
        object.__setattr__(
            self,
            "selection_report_digest",
            _digest(self.selection_report_digest, "selection_report_digest"),
        )
        reserve = _integer(self.reserve_ppm, "reserve_ppm", maximum=PPM - 1)
        _integer(self.epoch_blocks, "epoch_blocks", minimum=1)
        _integer(
            self.discovery_cap_units,
            "discovery_cap_units",
            minimum=1,
            maximum=PPM - reserve,
        )
        _integer(
            self.per_award_principal_cap_epochs,
            "per_award_principal_cap_epochs",
            minimum=1,
        )
        _integer(
            self.discovery_lifetime_blocks,
            "discovery_lifetime_blocks",
            minimum=1,
        )
        if self.schema_version != COMPOSITION_POLICY_SCHEMA_VERSION:
            raise IncentiveCompositionError("composition policy schema is unsupported")
        if self.policy_version != COMPOSITION_POLICY_VERSION:
            raise IncentiveCompositionError("composition policy version is unsupported")

    @property
    def reserve_floor_units(self) -> int:
        return self.reserve_ppm

    @property
    def discovery_principal_cap_units(self) -> int:
        return self.discovery_cap_units * self.per_award_principal_cap_epochs

    def validate_innovation_policy(
        self, policy: FiniteDebtPolicyManifest
    ) -> None:
        if type(policy) is not FiniteDebtPolicyManifest:
            raise IncentiveCompositionError("innovation policy is not exactly typed")
        if (
            policy.digest != self.innovation_policy_digest
            or policy.reserve_ppm != self.reserve_ppm
            or policy.epoch_blocks != self.epoch_blocks
        ):
            raise IncentiveCompositionError(
                "composition policy differs from its innovation policy"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            field: getattr(self, field) for field in self.__dataclass_fields__
        }

    @classmethod
    def from_dict(cls, value: object) -> "IncentiveCompositionPolicyManifest":
        row = _strict(value, set(cls.__dataclass_fields__), "composition policy")
        return cls(**row)  # type: ignore[arg-type]

    @property
    def digest(self) -> str:
        return canonical_digest(
            "optima.incentive-composition.policy", self.to_dict()
        )


@dataclass(frozen=True)
class ReviewedDiscoveryDisposition:
    """One immutable, mutually exclusive reviewed discovery decision."""

    policy_digest: str
    win_digest: str
    proposal_digest: str
    retained_evidence_digest: str
    review_digest: str
    hotkey: str
    win_block: int
    authority_block: int
    decision: str
    requested_principal_epochs: int
    promoted_target_digest: str

    def __post_init__(self) -> None:
        for field in (
            "policy_digest",
            "win_digest",
            "proposal_digest",
            "retained_evidence_digest",
            "review_digest",
        ):
            object.__setattr__(self, field, _digest(getattr(self, field), field))
        object.__setattr__(self, "hotkey", _hotkey(self.hotkey))
        win_block = _integer(self.win_block, "win_block")
        authority_block = _integer(self.authority_block, "authority_block")
        if authority_block < win_block:
            raise IncentiveCompositionError(
                "discovery review authority predates its retained win"
            )
        if self.decision not in DISCOVERY_DECISIONS:
            raise IncentiveCompositionError("discovery decision is unsupported")
        if self.decision == DISCOVERY_BOUNTY_ONLY:
            _integer(
                self.requested_principal_epochs,
                "requested_principal_epochs",
                minimum=1,
            )
            if self.promoted_target_digest != "":
                raise IncentiveCompositionError(
                    "bounty-only disposition cannot name a promoted target"
                )
        else:
            if self.requested_principal_epochs != 0:
                raise IncentiveCompositionError(
                    "registered promotion cannot request discovery principal"
                )
            object.__setattr__(
                self,
                "promoted_target_digest",
                _digest(self.promoted_target_digest, "promoted_target_digest"),
            )

    def validate_policy(self, policy: IncentiveCompositionPolicyManifest) -> None:
        if type(policy) is not IncentiveCompositionPolicyManifest:
            raise IncentiveCompositionError("composition policy is not exactly typed")
        if self.policy_digest != policy.digest:
            raise IncentiveCompositionError("disposition policy digest differs")

    def to_dict(self) -> dict[str, object]:
        return {
            field: getattr(self, field) for field in self.__dataclass_fields__
        }

    @classmethod
    def from_dict(cls, value: object) -> "ReviewedDiscoveryDisposition":
        row = _strict(
            value,
            set(cls.__dataclass_fields__),
            "reviewed discovery disposition",
        )
        return cls(**row)  # type: ignore[arg-type]

    @property
    def digest(self) -> str:
        return canonical_digest(
            "optima.incentive-composition.discovery-disposition", self.to_dict()
        )


def review_discovery_disposition(
    policy: IncentiveCompositionPolicyManifest,
    *,
    win_digest: str,
    proposal_digest: str,
    retained_evidence_digest: str,
    review_digest: str,
    hotkey: str,
    win_block: int,
    authority_block: int,
    decision: str,
    requested_principal_epochs: int = 0,
    promoted_target_digest: str = "",
) -> ReviewedDiscoveryDisposition:
    """Construct one policy-bound review; it does not itself issue debt."""

    if type(policy) is not IncentiveCompositionPolicyManifest:
        raise IncentiveCompositionError("composition policy is not exactly typed")
    result = ReviewedDiscoveryDisposition(
        policy.digest,
        win_digest,
        proposal_digest,
        retained_evidence_digest,
        review_digest,
        hotkey,
        win_block,
        authority_block,
        decision,
        requested_principal_epochs,
        promoted_target_digest,
    )
    result.validate_policy(policy)
    return result


@dataclass(frozen=True)
class DiscoveryDebtClaim:
    """One non-renewable finite discovery bounty claim."""

    policy_digest: str
    disposition_digest: str
    proposal_digest: str
    retained_evidence_digest: str
    review_digest: str
    hotkey: str
    awarded_block: int
    expires_block: int
    requested_principal_epochs: int
    capped_principal_epochs: int
    principal_units: int

    def __post_init__(self) -> None:
        for field in (
            "policy_digest",
            "disposition_digest",
            "proposal_digest",
            "retained_evidence_digest",
            "review_digest",
        ):
            object.__setattr__(self, field, _digest(getattr(self, field), field))
        object.__setattr__(self, "hotkey", _hotkey(self.hotkey))
        awarded = _integer(self.awarded_block, "awarded_block")
        expires = _integer(self.expires_block, "expires_block", minimum=1)
        requested = _integer(
            self.requested_principal_epochs,
            "requested_principal_epochs",
            minimum=1,
        )
        capped = _integer(
            self.capped_principal_epochs,
            "capped_principal_epochs",
            minimum=1,
        )
        _integer(self.principal_units, "principal_units", minimum=1)
        if expires <= awarded or capped > requested:
            raise IncentiveCompositionError("discovery claim lifetime or cap is invalid")

    @property
    def settlement_block(self) -> int:
        return self.awarded_block

    def validate_policy(
        self,
        policy: IncentiveCompositionPolicyManifest,
        disposition: ReviewedDiscoveryDisposition | None = None,
    ) -> None:
        if type(policy) is not IncentiveCompositionPolicyManifest:
            raise IncentiveCompositionError("composition policy is not exactly typed")
        if (
            self.policy_digest != policy.digest
            or self.capped_principal_epochs
            != min(
                self.requested_principal_epochs,
                policy.per_award_principal_cap_epochs,
            )
            or self.principal_units
            != self.capped_principal_epochs * policy.discovery_cap_units
            or self.expires_block
            != self.awarded_block + policy.discovery_lifetime_blocks
        ):
            raise IncentiveCompositionError("discovery claim differs from its policy")
        if disposition is not None:
            if type(disposition) is not ReviewedDiscoveryDisposition:
                raise IncentiveCompositionError("disposition is not exactly typed")
            disposition.validate_policy(policy)
            if (
                disposition.decision != DISCOVERY_BOUNTY_ONLY
                or self.disposition_digest != disposition.digest
                or self.proposal_digest != disposition.proposal_digest
                or self.retained_evidence_digest
                != disposition.retained_evidence_digest
                or self.review_digest != disposition.review_digest
                or self.hotkey != disposition.hotkey
                or self.awarded_block != disposition.win_block
                or disposition.authority_block >= self.expires_block
                or self.requested_principal_epochs
                != disposition.requested_principal_epochs
            ):
                raise IncentiveCompositionError(
                    "discovery claim differs from its reviewed disposition"
                )

    def to_dict(self) -> dict[str, object]:
        return {
            field: getattr(self, field) for field in self.__dataclass_fields__
        }

    @classmethod
    def from_dict(cls, value: object) -> "DiscoveryDebtClaim":
        row = _strict(value, set(cls.__dataclass_fields__), "discovery debt claim")
        return cls(**row)  # type: ignore[arg-type]

    @property
    def digest(self) -> str:
        return canonical_digest(
            "optima.incentive-composition.discovery-claim", self.to_dict()
        )


def issue_discovery_claim(
    policy: IncentiveCompositionPolicyManifest,
    disposition: ReviewedDiscoveryDisposition,
) -> DiscoveryDebtClaim | None:
    """Issue only the bounded-bounty path; promotion returns no claim."""

    if (
        type(policy) is not IncentiveCompositionPolicyManifest
        or type(disposition) is not ReviewedDiscoveryDisposition
    ):
        raise IncentiveCompositionError("discovery issuance authority is not typed")
    disposition.validate_policy(policy)
    if disposition.decision == DISCOVERY_REGISTERED_PROMOTION:
        return None
    expires_block = disposition.win_block + policy.discovery_lifetime_blocks
    if disposition.authority_block >= expires_block:
        raise IncentiveCompositionError(
            "discovery bounty review is at or after the retained win expiry"
        )
    capped = min(
        disposition.requested_principal_epochs,
        policy.per_award_principal_cap_epochs,
    )
    result = DiscoveryDebtClaim(
        policy.digest,
        disposition.digest,
        disposition.proposal_digest,
        disposition.retained_evidence_digest,
        disposition.review_digest,
        disposition.hotkey,
        disposition.win_block,
        expires_block,
        disposition.requested_principal_epochs,
        capped,
        policy.discovery_cap_units * capped,
    )
    result.validate_policy(policy, disposition)
    return result


@dataclass(frozen=True)
class DiscoveryClaimBalance:
    """Immutable cumulative accounting for one discovery claim."""

    claim_digest: str
    principal_units: int
    paid_units: int
    forfeited_units: int
    remaining_units: int
    status: str
    terminal_block: int | None
    terminal_reason: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "claim_digest", _digest(self.claim_digest, "claim_digest")
        )
        principal = _integer(self.principal_units, "principal_units", minimum=1)
        paid = _integer(self.paid_units, "paid_units")
        forfeited = _integer(self.forfeited_units, "forfeited_units")
        remaining = _integer(self.remaining_units, "remaining_units")
        if paid + forfeited + remaining != principal:
            raise IncentiveCompositionError(
                "discovery balance does not conserve principal"
            )
        if self.status not in DISCOVERY_BALANCE_STATUSES:
            raise IncentiveCompositionError("discovery balance status is unsupported")
        if self.status == "open":
            if (
                remaining <= 0
                or self.terminal_block is not None
                or self.terminal_reason != ""
            ):
                raise IncentiveCompositionError(
                    "open discovery balance has terminal state"
                )
        else:
            if remaining != 0 or self.terminal_block is None:
                raise IncentiveCompositionError(
                    "terminal discovery balance is incomplete"
                )
            _integer(self.terminal_block, "terminal_block")
            _reason(self.terminal_reason, "terminal_reason")
        if self.status == "paid" and paid != principal:
            raise IncentiveCompositionError(
                "paid discovery claim did not consume principal"
            )
        if self.status in {"expired", "cancelled"} and forfeited <= 0:
            raise IncentiveCompositionError(
                "forfeited discovery balance has no forfeited principal"
            )

    @classmethod
    def open(cls, claim: DiscoveryDebtClaim) -> "DiscoveryClaimBalance":
        if type(claim) is not DiscoveryDebtClaim:
            raise IncentiveCompositionError("discovery claim is not exactly typed")
        return cls(
            claim.digest,
            claim.principal_units,
            0,
            0,
            claim.principal_units,
            "open",
            None,
            "",
        )

    def to_dict(self) -> dict[str, object]:
        return {
            field: getattr(self, field) for field in self.__dataclass_fields__
        }

    @classmethod
    def from_dict(cls, value: object) -> "DiscoveryClaimBalance":
        row = _strict(value, set(cls.__dataclass_fields__), "discovery balance")
        return cls(**row)  # type: ignore[arg-type]

    @property
    def digest(self) -> str:
        return canonical_digest(
            "optima.incentive-composition.discovery-balance", self.to_dict()
        )


def _validate_claim_balance(
    claim: DiscoveryDebtClaim, balance: DiscoveryClaimBalance
) -> None:
    if (
        type(claim) is not DiscoveryDebtClaim
        or type(balance) is not DiscoveryClaimBalance
        or balance.claim_digest != claim.digest
        or balance.principal_units != claim.principal_units
    ):
        raise IncentiveCompositionError("discovery claim and balance differ")


def pay_discovery_balance(
    claim: DiscoveryDebtClaim,
    balance: DiscoveryClaimBalance,
    amount: int,
    *,
    at_block: int,
) -> DiscoveryClaimBalance:
    _validate_claim_balance(claim, balance)
    paid = _integer(amount, "payment amount", minimum=1)
    block = _integer(at_block, "payment block")
    if (
        balance.status != "open"
        or block < claim.awarded_block
        or block >= claim.expires_block
        or paid > balance.remaining_units
    ):
        raise IncentiveCompositionError(
            "discovery payment is outside its live remaining principal"
        )
    remaining = balance.remaining_units - paid
    return DiscoveryClaimBalance(
        balance.claim_digest,
        balance.principal_units,
        balance.paid_units + paid,
        balance.forfeited_units,
        remaining,
        "open" if remaining else "paid",
        None if remaining else block,
        "" if remaining else "principal_paid",
    )


def expire_discovery_balance(
    claim: DiscoveryDebtClaim,
    balance: DiscoveryClaimBalance,
    *,
    at_block: int,
) -> DiscoveryClaimBalance:
    _validate_claim_balance(claim, balance)
    block = _integer(at_block, "expiration block")
    if balance.status != "open" or block < claim.expires_block:
        return balance
    return DiscoveryClaimBalance(
        balance.claim_digest,
        balance.principal_units,
        balance.paid_units,
        balance.forfeited_units + balance.remaining_units,
        0,
        "expired",
        block,
        "claim_lifetime_expired",
    )


def cancel_discovery_balance(
    claim: DiscoveryDebtClaim,
    balance: DiscoveryClaimBalance,
    *,
    at_block: int,
    reason: str,
) -> DiscoveryClaimBalance:
    _validate_claim_balance(claim, balance)
    block = _integer(at_block, "cancellation block")
    why = _reason(reason, "cancellation reason")
    if balance.status != "open" or block < claim.awarded_block:
        raise IncentiveCompositionError(
            "discovery cancellation is not open or predates its award"
        )
    return DiscoveryClaimBalance(
        balance.claim_digest,
        balance.principal_units,
        balance.paid_units,
        balance.forfeited_units + balance.remaining_units,
        0,
        "cancelled",
        block,
        why,
    )


@dataclass(frozen=True)
class DiscoveryClaimState:
    claim: DiscoveryDebtClaim
    balance: DiscoveryClaimBalance

    def __post_init__(self) -> None:
        _validate_claim_balance(self.claim, self.balance)
        terminal = self.balance.terminal_block
        if terminal is not None and terminal < self.claim.awarded_block:
            raise IncentiveCompositionError(
                "discovery terminal block predates its award"
            )
        if (
            self.balance.status == "expired"
            and terminal is not None
            and terminal < self.claim.expires_block
        ):
            raise IncentiveCompositionError(
                "discovery claim expired before its lifetime ended"
            )

    def to_dict(self) -> dict[str, object]:
        return {"balance": self.balance.to_dict(), "claim": self.claim.to_dict()}

    @classmethod
    def from_dict(cls, value: object) -> "DiscoveryClaimState":
        row = _strict(value, {"balance", "claim"}, "discovery claim state")
        return cls(
            DiscoveryDebtClaim.from_dict(row["claim"]),
            DiscoveryClaimBalance.from_dict(row["balance"]),
        )

    @property
    def digest(self) -> str:
        return canonical_digest(
            "optima.incentive-composition.discovery-state", self.to_dict()
        )


@dataclass(frozen=True)
class CompositionClassAllocation:
    claim_digest: str
    hotkey: str
    units: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "claim_digest", _digest(self.claim_digest, "claim_digest")
        )
        object.__setattr__(self, "hotkey", _hotkey(self.hotkey))
        _integer(self.units, "allocation units", minimum=1, maximum=PPM)

    def to_dict(self) -> dict[str, object]:
        return {
            "claim_digest": self.claim_digest,
            "hotkey": self.hotkey,
            "units": self.units,
        }

    @classmethod
    def from_dict(cls, value: object) -> "CompositionClassAllocation":
        row = _strict(value, set(cls.__dataclass_fields__), "class allocation")
        return cls(**row)  # type: ignore[arg-type]


@dataclass(frozen=True)
class ComposedEpochProjection:
    composition_policy_digest: str
    innovation_policy_digest: str
    effective_block: int
    reference_epoch_units: int
    reserve_floor_units: int
    discovery_capacity_units: int
    discovery_total_remaining_units: int
    discovery_payout_units: int
    innovation_capacity_units: int
    innovation_total_remaining_units: int
    innovation_payout_units: int
    reserve_hotkey: str
    reserve_units: int
    discovery_input_state_digests: tuple[str, ...]
    innovation_input_state_digests: tuple[str, ...]
    discovery_allocations: tuple[CompositionClassAllocation, ...]
    innovation_allocations: tuple[CompositionClassAllocation, ...]
    weights: tuple[DebtHotkeyWeight, ...]

    def __post_init__(self) -> None:
        for field in ("composition_policy_digest", "innovation_policy_digest"):
            object.__setattr__(self, field, _digest(getattr(self, field), field))
        _integer(self.effective_block, "effective_block")
        if self.reference_epoch_units != REFERENCE_EPOCH_UNITS:
            raise IncentiveCompositionError("reference_epoch_units is unsupported")
        reserve_floor = _integer(
            self.reserve_floor_units,
            "reserve_floor_units",
            maximum=PPM - 1,
        )
        discovery_capacity = _integer(
            self.discovery_capacity_units,
            "discovery_capacity_units",
            minimum=1,
            maximum=PPM - reserve_floor,
        )
        discovery_remaining = _integer(
            self.discovery_total_remaining_units,
            "discovery_total_remaining_units",
        )
        discovery_payout = _integer(
            self.discovery_payout_units,
            "discovery_payout_units",
            maximum=discovery_capacity,
        )
        innovation_capacity = _integer(
            self.innovation_capacity_units,
            "innovation_capacity_units",
            maximum=PPM - reserve_floor,
        )
        innovation_remaining = _integer(
            self.innovation_total_remaining_units,
            "innovation_total_remaining_units",
        )
        innovation_payout = _integer(
            self.innovation_payout_units,
            "innovation_payout_units",
            maximum=innovation_capacity,
        )
        reserve = _integer(
            self.reserve_units, "reserve_units", minimum=reserve_floor, maximum=PPM
        )
        object.__setattr__(
            self, "reserve_hotkey", _hotkey(self.reserve_hotkey, "reserve_hotkey")
        )
        for field in (
            "discovery_input_state_digests",
            "innovation_input_state_digests",
        ):
            values = tuple(getattr(self, field))
            if values != tuple(sorted(set(values))):
                raise IncentiveCompositionError(
                    f"{field} is not uniquely digest-sorted"
                )
            for value in values:
                _digest(value, field)
            object.__setattr__(self, field, values)
        discovery_allocations = tuple(self.discovery_allocations)
        innovation_allocations = tuple(self.innovation_allocations)
        weights = tuple(self.weights)
        for label, allocations in (
            ("discovery", discovery_allocations),
            ("innovation", innovation_allocations),
        ):
            if (
                any(type(row) is not CompositionClassAllocation for row in allocations)
                or tuple(row.claim_digest for row in allocations)
                != tuple(sorted({row.claim_digest for row in allocations}))
            ):
                raise IncentiveCompositionError(
                    f"{label} allocations are not canonical"
                )
        if (
            any(type(row) is not DebtHotkeyWeight for row in weights)
            or tuple(row.hotkey for row in weights)
            != tuple(sorted({row.hotkey for row in weights}))
        ):
            raise IncentiveCompositionError("composed weights are not canonical")
        if (
            discovery_payout != min(discovery_capacity, discovery_remaining)
            or innovation_capacity != PPM - reserve_floor - discovery_payout
            or innovation_payout != min(innovation_capacity, innovation_remaining)
            or sum(row.units for row in discovery_allocations) != discovery_payout
            or sum(row.units for row in innovation_allocations) != innovation_payout
            or reserve != PPM - discovery_payout - innovation_payout
            or sum(row.units for row in weights) != PPM
        ):
            raise IncentiveCompositionError(
                "composed projection does not conserve its ordered capacities"
            )
        by_hotkey: dict[str, int] = {}
        for allocation in (*discovery_allocations, *innovation_allocations):
            by_hotkey[allocation.hotkey] = (
                by_hotkey.get(allocation.hotkey, 0) + allocation.units
            )
        by_hotkey[self.reserve_hotkey] = (
            by_hotkey.get(self.reserve_hotkey, 0) + reserve
        )
        if {row.hotkey: row.units for row in weights} != {
            hotkey: units for hotkey, units in by_hotkey.items() if units > 0
        }:
            raise IncentiveCompositionError(
                "composed weights differ from class allocations and reserve"
            )
        object.__setattr__(self, "discovery_allocations", discovery_allocations)
        object.__setattr__(self, "innovation_allocations", innovation_allocations)
        object.__setattr__(self, "weights", weights)

    @property
    def policy_digest(self) -> str:
        return self.composition_policy_digest

    @property
    def payout_units(self) -> int:
        return self.discovery_payout_units + self.innovation_payout_units

    @property
    def weights_by_hotkey(self) -> Mapping[str, int]:
        return {row.hotkey: row.units for row in self.weights}

    def to_dict(self) -> dict[str, object]:
        return {
            "composition_policy_digest": self.composition_policy_digest,
            "discovery_allocations": [
                row.to_dict() for row in self.discovery_allocations
            ],
            "discovery_capacity_units": self.discovery_capacity_units,
            "discovery_input_state_digests": list(
                self.discovery_input_state_digests
            ),
            "discovery_payout_units": self.discovery_payout_units,
            "discovery_total_remaining_units": self.discovery_total_remaining_units,
            "effective_block": self.effective_block,
            "innovation_allocations": [
                row.to_dict() for row in self.innovation_allocations
            ],
            "innovation_capacity_units": self.innovation_capacity_units,
            "innovation_input_state_digests": list(
                self.innovation_input_state_digests
            ),
            "innovation_payout_units": self.innovation_payout_units,
            "innovation_policy_digest": self.innovation_policy_digest,
            "innovation_total_remaining_units": self.innovation_total_remaining_units,
            "reference_epoch_units": self.reference_epoch_units,
            "reserve_floor_units": self.reserve_floor_units,
            "reserve_hotkey": self.reserve_hotkey,
            "reserve_units": self.reserve_units,
            "weights": [row.to_dict() for row in self.weights],
        }

    @classmethod
    def from_dict(cls, value: object) -> "ComposedEpochProjection":
        row = _strict(value, set(cls.__dataclass_fields__), "composed projection")
        for field in (
            "discovery_allocations",
            "discovery_input_state_digests",
            "innovation_allocations",
            "innovation_input_state_digests",
            "weights",
        ):
            if type(row[field]) is not list:
                raise IncentiveCompositionError(f"projection {field} must be an array")
        row["discovery_allocations"] = tuple(
            CompositionClassAllocation.from_dict(item)
            for item in row["discovery_allocations"]  # type: ignore[union-attr]
        )
        row["innovation_allocations"] = tuple(
            CompositionClassAllocation.from_dict(item)
            for item in row["innovation_allocations"]  # type: ignore[union-attr]
        )
        row["discovery_input_state_digests"] = tuple(
            row["discovery_input_state_digests"]  # type: ignore[arg-type]
        )
        row["innovation_input_state_digests"] = tuple(
            row["innovation_input_state_digests"]  # type: ignore[arg-type]
        )
        row["weights"] = tuple(
            DebtHotkeyWeight.from_dict(item)
            for item in row["weights"]  # type: ignore[union-attr]
        )
        return cls(**row)  # type: ignore[arg-type]

    @property
    def digest(self) -> str:
        return canonical_digest(
            "optima.incentive-composition.epoch-projection", self.to_dict()
        )


def _pro_rata(
    rows: Iterable[tuple[str, str, int]], capacity: int
) -> tuple[CompositionClassAllocation, ...]:
    values = tuple(sorted(rows, key=lambda row: row[0]))
    if len({row[0] for row in values}) != len(values):
        raise IncentiveCompositionError("class allocation contains duplicate claims")
    total = sum(row[2] for row in values)
    payout = min(capacity, total)
    if payout == 0:
        return ()
    amounts: dict[str, int] = {}
    if total <= capacity:
        amounts = {digest: remaining for digest, _hotkey_value, remaining in values}
    else:
        remainders: list[tuple[int, str]] = []
        for digest, _hotkey_value, remaining in values:
            quotient, remainder = divmod(remaining * payout, total)
            amounts[digest] = quotient
            remainders.append((remainder, digest))
        missing = payout - sum(amounts.values())
        for _remainder, digest in sorted(
            remainders, key=lambda item: (-item[0], item[1])
        )[:missing]:
            amounts[digest] += 1
    return tuple(
        CompositionClassAllocation(digest, hotkey, amounts[digest])
        for digest, hotkey, _remaining in values
        if amounts.get(digest, 0) > 0
    )


def _canonical_innovation_states(
    policy: FiniteDebtPolicyManifest,
    states: Iterable[DebtClaimState],
    block: int,
) -> tuple[DebtClaimState, ...]:
    rows = tuple(states)
    if any(type(row) is not DebtClaimState for row in rows):
        raise IncentiveCompositionError("innovation states are not exactly typed")
    rows = tuple(sorted(rows, key=lambda row: row.claim.digest))
    if len({row.claim.digest for row in rows}) != len(rows):
        raise IncentiveCompositionError("innovation states contain duplicate claims")
    for row in rows:
        try:
            row.claim.validate_policy(policy)
        except FiniteDebtError as exc:
            raise IncentiveCompositionError(
                f"innovation claim differs from policy: {exc}"
            ) from None
        if row.claim.hotkey == policy.reserve_hotkey and row.balance.status == "open":
            raise IncentiveCompositionError(
                "reserve hotkey cannot own open innovation debt"
            )
        if row.balance.status == "open" and (
            block < row.claim.settlement_block or block >= row.claim.expires_block
        ):
            raise IncentiveCompositionError(
                "open innovation claim is outside its payout window"
            )
    return rows


def _canonical_discovery_states(
    policy: IncentiveCompositionPolicyManifest,
    states: Iterable[DiscoveryClaimState],
    block: int,
    reserve_hotkey: str,
) -> tuple[DiscoveryClaimState, ...]:
    rows = tuple(states)
    if any(type(row) is not DiscoveryClaimState for row in rows):
        raise IncentiveCompositionError("discovery states are not exactly typed")
    rows = tuple(sorted(rows, key=lambda row: row.claim.digest))
    if len({row.claim.digest for row in rows}) != len(rows):
        raise IncentiveCompositionError("discovery states contain duplicate claims")
    for row in rows:
        row.claim.validate_policy(policy)
        if row.claim.hotkey == reserve_hotkey and row.balance.status == "open":
            raise IncentiveCompositionError(
                "reserve hotkey cannot own open discovery debt"
            )
        if row.balance.status == "open" and (
            block < row.claim.awarded_block or block >= row.claim.expires_block
        ):
            raise IncentiveCompositionError(
                "open discovery claim is outside its payout window"
            )
    return rows


def project_composed_epoch(
    innovation_policy: FiniteDebtPolicyManifest,
    composition_policy: IncentiveCompositionPolicyManifest,
    *,
    effective_block: int,
    innovation_states: Iterable[DebtClaimState],
    discovery_states: Iterable[DiscoveryClaimState],
) -> ComposedEpochProjection:
    """Build one read-only ordered two-class epoch projection."""

    if type(innovation_policy) is not FiniteDebtPolicyManifest:
        raise IncentiveCompositionError("innovation policy is not exactly typed")
    if type(composition_policy) is not IncentiveCompositionPolicyManifest:
        raise IncentiveCompositionError("composition policy is not exactly typed")
    composition_policy.validate_innovation_policy(innovation_policy)
    block = _integer(effective_block, "effective_block")
    innovation_rows = _canonical_innovation_states(
        innovation_policy, innovation_states, block
    )
    discovery_rows = _canonical_discovery_states(
        composition_policy,
        discovery_states,
        block,
        innovation_policy.reserve_hotkey,
    )
    open_discovery = tuple(
        row for row in discovery_rows if row.balance.status == "open"
    )
    open_innovation = tuple(
        row for row in innovation_rows if row.balance.status == "open"
    )
    discovery_remaining = sum(
        row.balance.remaining_units for row in open_discovery
    )
    discovery_payout = min(
        composition_policy.discovery_cap_units, discovery_remaining
    )
    discovery_allocations = _pro_rata(
        (
            (row.claim.digest, row.claim.hotkey, row.balance.remaining_units)
            for row in open_discovery
        ),
        composition_policy.discovery_cap_units,
    )
    innovation_capacity = (
        PPM - composition_policy.reserve_floor_units - discovery_payout
    )
    innovation_remaining = sum(
        row.balance.remaining_units for row in open_innovation
    )
    innovation_payout = min(innovation_capacity, innovation_remaining)
    innovation_allocations = _pro_rata(
        (
            (row.claim.digest, row.claim.hotkey, row.balance.remaining_units)
            for row in open_innovation
        ),
        innovation_capacity,
    )
    reserve = PPM - discovery_payout - innovation_payout
    by_hotkey: dict[str, int] = {}
    for allocation in (*discovery_allocations, *innovation_allocations):
        by_hotkey[allocation.hotkey] = (
            by_hotkey.get(allocation.hotkey, 0) + allocation.units
        )
    by_hotkey[innovation_policy.reserve_hotkey] = (
        by_hotkey.get(innovation_policy.reserve_hotkey, 0) + reserve
    )
    weights = tuple(
        DebtHotkeyWeight(hotkey, by_hotkey[hotkey]) for hotkey in sorted(by_hotkey)
    )
    return ComposedEpochProjection(
        composition_policy.digest,
        innovation_policy.digest,
        block,
        REFERENCE_EPOCH_UNITS,
        composition_policy.reserve_floor_units,
        composition_policy.discovery_cap_units,
        discovery_remaining,
        discovery_payout,
        innovation_capacity,
        innovation_remaining,
        innovation_payout,
        innovation_policy.reserve_hotkey,
        reserve,
        tuple(sorted(row.digest for row in discovery_rows)),
        tuple(sorted(row.digest for row in innovation_rows)),
        discovery_allocations,
        innovation_allocations,
        weights,
    )


def apply_composed_epoch(
    innovation_states: Iterable[DebtClaimState],
    discovery_states: Iterable[DiscoveryClaimState],
    projection: ComposedEpochProjection,
) -> tuple[tuple[DebtClaimState, ...], tuple[DiscoveryClaimState, ...]]:
    """Apply one retained composed projection exactly once to matching states."""

    if type(projection) is not ComposedEpochProjection:
        raise IncentiveCompositionError("composed projection is not exactly typed")
    innovation_values = tuple(innovation_states)
    discovery_values = tuple(discovery_states)
    if (
        any(type(row) is not DebtClaimState for row in innovation_values)
        or any(type(row) is not DiscoveryClaimState for row in discovery_values)
    ):
        raise IncentiveCompositionError(
            "composed projection inputs are not exactly typed"
        )
    innovation_rows = tuple(
        sorted(innovation_values, key=lambda row: row.claim.digest)
    )
    discovery_rows = tuple(
        sorted(discovery_values, key=lambda row: row.claim.digest)
    )
    if (
        tuple(sorted(row.digest for row in innovation_rows))
        != projection.innovation_input_state_digests
        or tuple(sorted(row.digest for row in discovery_rows))
        != projection.discovery_input_state_digests
    ):
        raise IncentiveCompositionError(
            "composed projection input balances changed before application"
        )
    innovation_amounts = {
        row.claim_digest: row.units for row in projection.innovation_allocations
    }
    discovery_amounts = {
        row.claim_digest: row.units for row in projection.discovery_allocations
    }
    innovation_by_digest = {
        row.claim.digest: row for row in innovation_rows
    }
    discovery_by_digest = {
        row.claim.digest: row for row in discovery_rows
    }
    if any(
        allocation.claim_digest not in innovation_by_digest
        or innovation_by_digest[allocation.claim_digest].claim.hotkey
        != allocation.hotkey
        for allocation in projection.innovation_allocations
    ) or any(
        allocation.claim_digest not in discovery_by_digest
        or discovery_by_digest[allocation.claim_digest].claim.hotkey
        != allocation.hotkey
        for allocation in projection.discovery_allocations
    ):
        raise IncentiveCompositionError(
            "composed projection allocation differs from its claim owner"
        )
    updated_innovation: list[DebtClaimState] = []
    for row in innovation_rows:
        amount = innovation_amounts.get(row.claim.digest, 0)
        try:
            balance = (
                row.balance
                if amount == 0
                else pay_claim_balance(
                    row.claim,
                    row.balance,
                    amount,
                    at_block=projection.effective_block,
                )
            )
        except FiniteDebtError as exc:
            raise IncentiveCompositionError(
                f"innovation projection cannot apply: {exc}"
            ) from None
        updated_innovation.append(DebtClaimState(row.claim, balance))
    updated_discovery: list[DiscoveryClaimState] = []
    for row in discovery_rows:
        amount = discovery_amounts.get(row.claim.digest, 0)
        balance = (
            row.balance
            if amount == 0
            else pay_discovery_balance(
                row.claim,
                row.balance,
                amount,
                at_block=projection.effective_block,
            )
        )
        updated_discovery.append(DiscoveryClaimState(row.claim, balance))
    innovation_paid = sum(
        after.balance.paid_units - before.balance.paid_units
        for before, after in zip(
            innovation_rows, updated_innovation, strict=True
        )
    )
    discovery_paid = sum(
        after.balance.paid_units - before.balance.paid_units
        for before, after in zip(discovery_rows, updated_discovery, strict=True)
    )
    if (
        innovation_paid != projection.innovation_payout_units
        or discovery_paid != projection.discovery_payout_units
    ):
        raise IncentiveCompositionError(
            "composed projection application did not consume both payouts"
        )
    return tuple(updated_innovation), tuple(updated_discovery)


__all__ = [
    "COMPOSITION_POLICY_SCHEMA_VERSION",
    "COMPOSITION_POLICY_VERSION",
    "DISCOVERY_BOUNTY_ONLY",
    "DISCOVERY_REGISTERED_PROMOTION",
    "ComposedEpochProjection",
    "CompositionClassAllocation",
    "DiscoveryClaimBalance",
    "DiscoveryClaimState",
    "DiscoveryDebtClaim",
    "IncentiveCompositionError",
    "IncentiveCompositionPolicyManifest",
    "ReviewedDiscoveryDisposition",
    "apply_composed_epoch",
    "cancel_discovery_balance",
    "expire_discovery_balance",
    "issue_discovery_claim",
    "pay_discovery_balance",
    "project_composed_epoch",
    "review_discovery_disposition",
]
