"""Pure, fixed-point economics for finite relative-improvement debt.

This module is deliberately separate from :mod:`optima.economics`.  The latter
is the deployed v1 standing-claim policy; changing it would rewrite retained
projection authority.  This module owns only immutable policy/claim/balance
values and deterministic claim issuance and epoch-allocation math.  It owns no
database, chain client, wallet, settlement transition, or policy defaults.

Accounting is denominated in ``weight-ppm epochs``.  One reference epoch has
``1_000_000`` units and ``epoch_blocks`` binds that economic unit to chain
time.  The policy reserve is removed before claim principal is created, so a
one-log-unit, one-epoch claim pays off in one uncontended post-reserve
claim-pool epoch rather than being lengthened by the reserve.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_FLOOR, localcontext
from typing import Iterable, Mapping

from optima._strict import require_digest, require_exact_fields, require_int
from optima.stack_identity import canonical_digest


PPM = 1_000_000
REFERENCE_EPOCH_UNITS = PPM
LOG_UNIT_STEP = Decimal("1.01")
DECIMAL_PRECISION = 80
POLICY_SCHEMA_VERSION = 2
POLICY_VERSION = "optima.finite-debt.v2"
MAX_ACTIVE_CAMPAIGNS = 2
IMPROVEMENT_GROSS = "gross"
IMPROVEMENT_EXCESS = "excess_over_threshold"
IMPROVEMENT_BASES = frozenset({IMPROVEMENT_GROSS, IMPROVEMENT_EXCESS})
BALANCE_STATUSES = frozenset({"open", "paid", "expired", "cancelled"})
_HOTKEY = re.compile(r"[^\s]{1,256}\Z")


class FiniteDebtError(ValueError):
    """A finite-debt policy, claim, balance, or projection is invalid."""


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
        error=FiniteDebtError,
        minimum=minimum,
        maximum=maximum,
    )


def _digest(value: object, field: str) -> str:
    return require_digest(value, field=field, error=FiniteDebtError)


def _hotkey(value: object, field: str = "hotkey") -> str:
    if not isinstance(value, str) or _HOTKEY.fullmatch(value) is None:
        raise FiniteDebtError(f"{field} is malformed")
    return value


def _strict(value: object, fields: set[str], label: str) -> dict[str, object]:
    return dict(
        require_exact_fields(
            value,
            fields=frozenset(fields),
            label=label,
            error=FiniteDebtError,
            exact_dict=True,
        )
    )


def _canonical_decimal(
    value: object,
    field: str,
    *,
    minimum: Decimal,
    strict_minimum: bool,
) -> tuple[str, Decimal]:
    if not isinstance(value, str):
        raise FiniteDebtError(f"{field} must be a canonical decimal string")
    try:
        parsed = Decimal(value)
    except InvalidOperation:
        raise FiniteDebtError(f"{field} is not decimal") from None
    if not parsed.is_finite() or (
        parsed <= minimum if strict_minimum else parsed < minimum
    ):
        relation = "greater than" if strict_minimum else "at least"
        raise FiniteDebtError(f"{field} must be finite and {relation} {minimum}")
    canonical = format(parsed.normalize(), "f")
    if "." in canonical:
        canonical = canonical.rstrip("0").rstrip(".")
    if value != canonical:
        raise FiniteDebtError(
            f"{field} must use canonical decimal spelling {canonical!r}"
        )
    return canonical, parsed


@dataclass(frozen=True)
class CampaignBudgetShare:
    """One model campaign's claim-sizing share of reference capacity."""

    campaign_id: str
    share_ppm: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "campaign_id", _digest(self.campaign_id, "campaign_id")
        )
        _integer(self.share_ppm, "campaign share_ppm", minimum=1, maximum=PPM)

    def to_dict(self) -> dict[str, object]:
        return {"campaign_id": self.campaign_id, "share_ppm": self.share_ppm}

    @classmethod
    def from_dict(cls, value: object) -> "CampaignBudgetShare":
        row = _strict(value, set(cls.__dataclass_fields__), "campaign budget share")
        return cls(**row)  # type: ignore[arg-type]


@dataclass(frozen=True)
class RewardFamilyCampaign:
    """Validator-owned mapping from one technical frontier to one campaign."""

    family_id: str
    campaign_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "family_id", _digest(self.family_id, "family_id"))
        object.__setattr__(
            self, "campaign_id", _digest(self.campaign_id, "campaign_id")
        )

    def to_dict(self) -> dict[str, object]:
        return {"campaign_id": self.campaign_id, "family_id": self.family_id}

    @classmethod
    def from_dict(cls, value: object) -> "RewardFamilyCampaign":
        row = _strict(value, set(cls.__dataclass_fields__), "reward-family campaign")
        return cls(**row)  # type: ignore[arg-type]


def equal_campaign_budget_shares(
    campaign_ids: Iterable[str],
) -> tuple[CampaignBudgetShare, ...]:
    """Return the only supported launch roster: 100%, or two equal 50% shares.

    Model campaigns are the funded economic unit.  Reward families remain
    independent technical frontiers and clocks, but adding a family to a model
    campaign must not dilute the principal earned by another family.
    """

    try:
        campaigns = tuple(
            sorted(_digest(value, "campaign_id") for value in campaign_ids)
        )
    except TypeError:
        raise FiniteDebtError("campaign_ids must be iterable") from None
    if not campaigns or len(set(campaigns)) != len(campaigns):
        raise FiniteDebtError("campaign catalog must be nonempty and unique")
    if len(campaigns) > MAX_ACTIVE_CAMPAIGNS:
        raise FiniteDebtError("policy supports at most two active campaigns")
    share = PPM // len(campaigns)
    return tuple(
        CampaignBudgetShare(campaign, share) for campaign in campaigns
    )


@dataclass(frozen=True)
class FiniteDebtPolicyManifest:
    """Complete, content-addressed consensus parameters for finite debt.

    Every economic parameter is required.  There are intentionally no selected
    mechanism defaults in this module.
    """

    campaign_budget_shares: tuple[CampaignBudgetShare, ...]
    reward_family_campaigns: tuple[RewardFamilyCampaign, ...]
    selection_report_digest: str
    reserve_hotkey: str
    reserve_ppm: int
    epoch_blocks: int
    beta_ppm: int
    tau_blocks: int
    lifetime_blocks: int
    k_ppm: int
    improvement_basis: str
    clock_reset_threshold_log_units_ppm: int
    schema_version: int = POLICY_SCHEMA_VERSION
    policy_version: str = POLICY_VERSION

    def __post_init__(self) -> None:
        shares = tuple(self.campaign_budget_shares)
        if any(type(row) is not CampaignBudgetShare for row in shares):
            raise FiniteDebtError("campaign budget shares are not exactly typed")
        shares = tuple(sorted(shares, key=lambda row: row.campaign_id))
        if (
            not shares
            or len({row.campaign_id for row in shares}) != len(shares)
            or shares
            != equal_campaign_budget_shares(row.campaign_id for row in shares)
        ):
            raise FiniteDebtError(
                "campaign budget must be one 100% campaign or two 50% campaigns"
            )
        object.__setattr__(self, "campaign_budget_shares", shares)
        mappings = tuple(self.reward_family_campaigns)
        if any(type(row) is not RewardFamilyCampaign for row in mappings):
            raise FiniteDebtError("reward-family campaigns are not exactly typed")
        mappings = tuple(sorted(mappings, key=lambda row: row.family_id))
        campaign_ids = {row.campaign_id for row in shares}
        if (
            not mappings
            or len({row.family_id for row in mappings}) != len(mappings)
            or {row.campaign_id for row in mappings} != campaign_ids
        ):
            raise FiniteDebtError(
                "every reward family must map once and every campaign must have a family"
            )
        object.__setattr__(self, "reward_family_campaigns", mappings)
        object.__setattr__(
            self,
            "selection_report_digest",
            _digest(self.selection_report_digest, "selection_report_digest"),
        )
        object.__setattr__(self, "reserve_hotkey", _hotkey(self.reserve_hotkey, "reserve_hotkey"))
        _integer(self.reserve_ppm, "reserve_ppm", maximum=PPM - 1)
        reference_pool = (PPM - self.reserve_ppm) * REFERENCE_EPOCH_UNITS // PPM
        if any(reference_pool * row.share_ppm // PPM <= 0 for row in shares):
            raise FiniteDebtError(
                "a campaign reference claim pool rounds to zero"
            )
        _integer(self.epoch_blocks, "epoch_blocks", minimum=1)
        _integer(self.beta_ppm, "beta_ppm", maximum=PPM)
        _integer(self.tau_blocks, "tau_blocks", minimum=1)
        _integer(self.lifetime_blocks, "lifetime_blocks", minimum=1)
        _integer(self.k_ppm, "k_ppm", minimum=1)
        if self.improvement_basis not in IMPROVEMENT_BASES:
            raise FiniteDebtError("improvement_basis is unsupported")
        _integer(
            self.clock_reset_threshold_log_units_ppm,
            "clock_reset_threshold_log_units_ppm",
            minimum=1,
        )
        if self.schema_version != POLICY_SCHEMA_VERSION:
            raise FiniteDebtError("finite-debt policy schema_version is unsupported")
        if self.policy_version != POLICY_VERSION:
            raise FiniteDebtError("finite-debt policy_version is unsupported")

    @property
    def campaign_shares(self) -> Mapping[str, int]:
        return {row.campaign_id: row.share_ppm for row in self.campaign_budget_shares}

    @property
    def family_ids(self) -> tuple[str, ...]:
        return tuple(row.family_id for row in self.reward_family_campaigns)

    def campaign_for_family(self, family_id: str) -> str:
        family = _digest(family_id, "family_id")
        for row in self.reward_family_campaigns:
            if row.family_id == family:
                return row.campaign_id
        raise FiniteDebtError("claim family is absent from the policy campaign map")

    def campaign_share_ppm_for_family(self, family_id: str) -> int:
        return self.campaign_shares[self.campaign_for_family(family_id)]

    @property
    def reference_claim_pool_units(self) -> int:
        return (PPM - self.reserve_ppm) * REFERENCE_EPOCH_UNITS // PPM

    def to_dict(self) -> dict[str, object]:
        return {
            "beta_ppm": self.beta_ppm,
            "campaign_budget_shares": [
                row.to_dict() for row in self.campaign_budget_shares
            ],
            "clock_reset_threshold_log_units_ppm": self.clock_reset_threshold_log_units_ppm,
            "epoch_blocks": self.epoch_blocks,
            "improvement_basis": self.improvement_basis,
            "k_ppm": self.k_ppm,
            "lifetime_blocks": self.lifetime_blocks,
            "policy_version": self.policy_version,
            "reserve_hotkey": self.reserve_hotkey,
            "reserve_ppm": self.reserve_ppm,
            "reward_family_campaigns": [
                row.to_dict() for row in self.reward_family_campaigns
            ],
            "schema_version": self.schema_version,
            "selection_report_digest": self.selection_report_digest,
            "tau_blocks": self.tau_blocks,
        }

    @classmethod
    def from_dict(cls, value: object) -> "FiniteDebtPolicyManifest":
        row = _strict(value, set(cls.__dataclass_fields__), "finite-debt policy")
        shares = row["campaign_budget_shares"]
        if type(shares) is not list:
            raise FiniteDebtError("campaign_budget_shares must be an array")
        mappings = row["reward_family_campaigns"]
        if type(mappings) is not list:
            raise FiniteDebtError("reward_family_campaigns must be an array")
        row["campaign_budget_shares"] = tuple(
            CampaignBudgetShare.from_dict(item) for item in shares
        )
        row["reward_family_campaigns"] = tuple(
            RewardFamilyCampaign.from_dict(item) for item in mappings
        )
        return cls(**row)  # type: ignore[arg-type]

    @property
    def digest(self) -> str:
        return canonical_digest("optima.finite-debt.policy", self.to_dict())


def log_improvement_units_ppm(
    settled_speedup: str,
    *,
    basis: str,
    threshold_speedup: str = "1",
) -> int:
    """Convert one canonical relative speedup to deterministic 1%-log units.

    ``gross`` measures ``ln(settled_speedup)``. ``excess_over_threshold``
    measures ``ln(settled_speedup / threshold_speedup)``.  One full log unit is
    a multiplicative 1.01 improvement and is represented by ``1_000_000``.

    Production deliberately floors at ppm precision.  The frozen V2 selection
    simulator used half-even rounding at nano-unit precision; changing either
    convention is a policy-version change.  A conformance vector in the focused
    tests bounds the resulting selected-curve principal difference.
    """

    if basis not in IMPROVEMENT_BASES:
        raise FiniteDebtError("improvement basis is unsupported")
    _speed_text, speed = _canonical_decimal(
        settled_speedup,
        "settled_speedup",
        minimum=Decimal(1),
        strict_minimum=True,
    )
    threshold_text, threshold = _canonical_decimal(
        threshold_speedup,
        "threshold_speedup",
        minimum=Decimal(1),
        strict_minimum=False,
    )
    if basis == IMPROVEMENT_GROSS:
        if threshold_text != "1":
            raise FiniteDebtError("gross improvement requires threshold_speedup='1'")
        effective = speed
    else:
        if speed <= threshold:
            raise FiniteDebtError(
                "excess-over-threshold speedup must exceed its threshold"
            )
        with localcontext() as context:
            context.prec = DECIMAL_PRECISION
            context.rounding = ROUND_FLOOR
            effective = speed / threshold
    with localcontext() as context:
        context.prec = DECIMAL_PRECISION
        context.rounding = ROUND_FLOOR
        units = (
            context.ln(effective)
            / context.ln(LOG_UNIT_STEP)
            * Decimal(PPM)
        ).to_integral_value(rounding=ROUND_FLOOR)
    result = int(units)
    if result <= 0:
        raise FiniteDebtError("improvement rounds to zero 1%-log-unit ppm")
    return result


def rational_time_multiplier_ppm(
    policy: FiniteDebtPolicyManifest,
    *,
    accepted_crown_block: int,
    prior_accepted_crown_block: int | None,
) -> int:
    """Return ``1 + beta*delta/(delta+tau)`` in ppm.

    A family's first accepted crown has no prior family clock and therefore has
    multiplier exactly one, independent of chain age.
    """

    if type(policy) is not FiniteDebtPolicyManifest:
        raise FiniteDebtError("policy is not exactly typed")
    current = _integer(accepted_crown_block, "accepted_crown_block")
    if prior_accepted_crown_block is None:
        return PPM
    prior = _integer(prior_accepted_crown_block, "prior_accepted_crown_block")
    if prior > current:
        raise FiniteDebtError("prior family crown is newer than the accepted crown")
    delta = current - prior
    bonus = policy.beta_ppm * delta // (delta + policy.tau_blocks)
    result = PPM + bonus
    if result > PPM + policy.beta_ppm:
        raise FiniteDebtError("rational time multiplier exceeded its cap")
    return result


def resets_family_clock(
    policy: FiniteDebtPolicyManifest, log_units_ppm: int
) -> bool:
    """Whether an accepted improvement is material enough to reset its clock."""

    if type(policy) is not FiniteDebtPolicyManifest:
        raise FiniteDebtError("policy is not exactly typed")
    units = _integer(log_units_ppm, "log_units_ppm")
    return (
        units > 0
        and units >= policy.clock_reset_threshold_log_units_ppm
    )


def _claim_principal_units(
    policy: FiniteDebtPolicyManifest,
    *,
    family_id: str,
    log_units_ppm: int,
    time_multiplier_ppm: int,
) -> tuple[str, int, int, int]:
    campaign_id = policy.campaign_for_family(family_id)
    campaign_share = policy.campaign_share_ppm_for_family(family_id)
    campaign_pool = policy.reference_claim_pool_units * campaign_share // PPM
    principal = (
        campaign_pool
        * policy.k_ppm
        * log_units_ppm
        * time_multiplier_ppm
        // (PPM * PPM * PPM)
    )
    if campaign_pool <= 0 or principal <= 0:
        raise FiniteDebtError("claim principal rounds to zero")
    return campaign_id, campaign_share, campaign_pool, principal


@dataclass(frozen=True)
class InnovationDebtClaim:
    """Immutable principal issued by one accepted and reproduced crown."""

    policy_digest: str
    family_id: str
    campaign_id: str
    candidate_digest: str
    retained_evidence_digest: str
    hotkey: str
    settled_speedup: str
    threshold_speedup: str
    accepted_crown_block: int
    prior_accepted_crown_block: int | None
    settlement_block: int
    expires_block: int
    campaign_budget_ppm: int
    reference_campaign_pool_units: int
    log_units_ppm: int
    time_multiplier_ppm: int
    principal_units: int
    resets_clock: bool

    def __post_init__(self) -> None:
        for field in (
            "policy_digest",
            "family_id",
            "campaign_id",
            "candidate_digest",
            "retained_evidence_digest",
        ):
            object.__setattr__(self, field, _digest(getattr(self, field), field))
        object.__setattr__(self, "hotkey", _hotkey(self.hotkey))
        speed_text, _speed = _canonical_decimal(
            self.settled_speedup,
            "settled_speedup",
            minimum=Decimal(1),
            strict_minimum=True,
        )
        threshold_text, _threshold = _canonical_decimal(
            self.threshold_speedup,
            "threshold_speedup",
            minimum=Decimal(1),
            strict_minimum=False,
        )
        object.__setattr__(self, "settled_speedup", speed_text)
        object.__setattr__(self, "threshold_speedup", threshold_text)
        accepted = _integer(self.accepted_crown_block, "accepted_crown_block")
        settlement = _integer(self.settlement_block, "settlement_block")
        expiry = _integer(self.expires_block, "expires_block")
        if accepted > settlement or expiry <= settlement:
            raise FiniteDebtError("claim reveal, settlement, and expiry chronology is invalid")
        if self.prior_accepted_crown_block is not None:
            prior = _integer(
                self.prior_accepted_crown_block, "prior_accepted_crown_block"
            )
            if prior > accepted:
                raise FiniteDebtError("claim prior family crown is newer than this crown")
        _integer(
            self.campaign_budget_ppm,
            "campaign_budget_ppm",
            minimum=1,
            maximum=PPM,
        )
        _integer(
            self.reference_campaign_pool_units,
            "reference_campaign_pool_units",
            minimum=1,
        )
        _integer(self.log_units_ppm, "log_units_ppm", minimum=1)
        _integer(self.time_multiplier_ppm, "time_multiplier_ppm", minimum=PPM)
        _integer(self.principal_units, "principal_units", minimum=1)
        if type(self.resets_clock) is not bool:
            raise FiniteDebtError("resets_clock must be boolean")

    def validate_policy(self, policy: FiniteDebtPolicyManifest) -> None:
        """Recompute every policy-derived claim term."""

        if type(policy) is not FiniteDebtPolicyManifest or self.policy_digest != policy.digest:
            raise FiniteDebtError("claim differs from its finite-debt policy")
        units = log_improvement_units_ppm(
            self.settled_speedup,
            basis=policy.improvement_basis,
            threshold_speedup=self.threshold_speedup,
        )
        multiplier = rational_time_multiplier_ppm(
            policy,
            accepted_crown_block=self.accepted_crown_block,
            prior_accepted_crown_block=self.prior_accepted_crown_block,
        )
        campaign, campaign_share, campaign_pool, principal = _claim_principal_units(
            policy,
            family_id=self.family_id,
            log_units_ppm=units,
            time_multiplier_ppm=multiplier,
        )
        if (
            self.expires_block != self.settlement_block + policy.lifetime_blocks
            or self.campaign_id != campaign
            or self.campaign_budget_ppm != campaign_share
            or self.reference_campaign_pool_units != campaign_pool
            or self.log_units_ppm != units
            or self.time_multiplier_ppm != multiplier
            or self.principal_units != principal
            or self.resets_clock != resets_family_clock(policy, units)
        ):
            raise FiniteDebtError("claim derived terms differ from policy arithmetic")

    def to_dict(self) -> dict[str, object]:
        return {
            field: getattr(self, field) for field in self.__dataclass_fields__
        }

    @classmethod
    def from_dict(cls, value: object) -> "InnovationDebtClaim":
        row = _strict(value, set(cls.__dataclass_fields__), "innovation debt claim")
        return cls(**row)  # type: ignore[arg-type]

    @property
    def digest(self) -> str:
        return canonical_digest("optima.finite-debt.claim", self.to_dict())


def issue_innovation_claim(
    policy: FiniteDebtPolicyManifest,
    *,
    family_id: str,
    candidate_digest: str,
    retained_evidence_digest: str,
    hotkey: str,
    settled_speedup: str,
    threshold_speedup: str,
    accepted_crown_block: int,
    prior_accepted_crown_block: int | None,
    settlement_block: int,
) -> InnovationDebtClaim:
    """Issue one finite principal from immutable accepted-crown authority."""

    if type(policy) is not FiniteDebtPolicyManifest:
        raise FiniteDebtError("policy is not exactly typed")
    family = _digest(family_id, "family_id")
    accepted = _integer(accepted_crown_block, "accepted_crown_block")
    settled = _integer(settlement_block, "settlement_block")
    if accepted > settled:
        raise FiniteDebtError("accepted crown cannot be newer than settlement")
    units = log_improvement_units_ppm(
        settled_speedup,
        basis=policy.improvement_basis,
        threshold_speedup=threshold_speedup,
    )
    multiplier = rational_time_multiplier_ppm(
        policy,
        accepted_crown_block=accepted,
        prior_accepted_crown_block=prior_accepted_crown_block,
    )
    campaign, campaign_share, campaign_pool, principal = _claim_principal_units(
        policy,
        family_id=family,
        log_units_ppm=units,
        time_multiplier_ppm=multiplier,
    )
    claim = InnovationDebtClaim(
        policy_digest=policy.digest,
        family_id=family,
        campaign_id=campaign,
        candidate_digest=candidate_digest,
        retained_evidence_digest=retained_evidence_digest,
        hotkey=hotkey,
        settled_speedup=settled_speedup,
        threshold_speedup=threshold_speedup,
        accepted_crown_block=accepted,
        prior_accepted_crown_block=prior_accepted_crown_block,
        settlement_block=settled,
        expires_block=settled + policy.lifetime_blocks,
        campaign_budget_ppm=campaign_share,
        reference_campaign_pool_units=campaign_pool,
        log_units_ppm=units,
        time_multiplier_ppm=multiplier,
        principal_units=principal,
        resets_clock=resets_family_clock(policy, units),
    )
    claim.validate_policy(policy)
    return claim


@dataclass(frozen=True)
class DebtClaimBalance:
    """Immutable cumulative accounting state for one immutable claim."""

    claim_digest: str
    principal_units: int
    paid_units: int
    forfeited_units: int
    remaining_units: int
    status: str
    terminal_block: int | None
    terminal_reason: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "claim_digest", _digest(self.claim_digest, "claim_digest"))
        principal = _integer(self.principal_units, "principal_units", minimum=1)
        paid = _integer(self.paid_units, "paid_units")
        forfeited = _integer(self.forfeited_units, "forfeited_units")
        remaining = _integer(self.remaining_units, "remaining_units")
        if paid + forfeited + remaining != principal:
            raise FiniteDebtError("claim balance does not conserve principal")
        if self.status not in BALANCE_STATUSES:
            raise FiniteDebtError("claim balance status is unsupported")
        terminal = self.status != "open"
        if terminal:
            if remaining != 0 or self.terminal_block is None or not self.terminal_reason:
                raise FiniteDebtError("terminal claim balance is incomplete")
            _integer(self.terminal_block, "terminal_block")
        elif (
            remaining <= 0
            or self.terminal_block is not None
            or self.terminal_reason
        ):
            raise FiniteDebtError("open claim balance has terminal state")
        if (
            not isinstance(self.terminal_reason, str)
            or len(self.terminal_reason) > 2_048
            or any(char in self.terminal_reason for char in "\x00\r\n")
        ):
            raise FiniteDebtError("terminal_reason is malformed")
        if self.status == "paid" and paid != principal:
            raise FiniteDebtError("paid claim did not pay its full principal")
        if self.status in {"expired", "cancelled"} and forfeited <= 0:
            raise FiniteDebtError("forfeited claim has no forfeited principal")

    @classmethod
    def open(cls, claim: InnovationDebtClaim) -> "DebtClaimBalance":
        if type(claim) is not InnovationDebtClaim:
            raise FiniteDebtError("claim is not exactly typed")
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
        return {field: getattr(self, field) for field in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, value: object) -> "DebtClaimBalance":
        row = _strict(value, set(cls.__dataclass_fields__), "debt claim balance")
        return cls(**row)  # type: ignore[arg-type]

    @property
    def digest(self) -> str:
        return canonical_digest("optima.finite-debt.balance", self.to_dict())


def pay_claim_balance(
    claim: InnovationDebtClaim,
    balance: DebtClaimBalance,
    amount: int,
    *,
    at_block: int,
) -> DebtClaimBalance:
    if type(claim) is not InnovationDebtClaim or type(balance) is not DebtClaimBalance:
        raise FiniteDebtError("claim payment authority is not exactly typed")
    if balance.claim_digest != claim.digest or balance.principal_units != claim.principal_units:
        raise FiniteDebtError("claim and balance differ")
    if balance.status != "open":
        raise FiniteDebtError("only an open claim balance may be paid")
    paid = _integer(amount, "payment amount", minimum=1)
    block = _integer(at_block, "payment block")
    if block < claim.settlement_block or block >= claim.expires_block:
        raise FiniteDebtError("claim payment is outside its live settlement window")
    if paid > balance.remaining_units:
        raise FiniteDebtError("claim payment exceeds remaining principal")
    remaining = balance.remaining_units - paid
    return DebtClaimBalance(
        balance.claim_digest,
        balance.principal_units,
        balance.paid_units + paid,
        balance.forfeited_units,
        remaining,
        "open" if remaining else "paid",
        None if remaining else block,
        "" if remaining else "principal_paid",
    )


def expire_claim_balance(
    claim: InnovationDebtClaim,
    balance: DebtClaimBalance,
    *,
    at_block: int,
) -> DebtClaimBalance:
    if type(claim) is not InnovationDebtClaim or type(balance) is not DebtClaimBalance:
        raise FiniteDebtError("claim expiration authority is not exactly typed")
    if balance.claim_digest != claim.digest or balance.principal_units != claim.principal_units:
        raise FiniteDebtError("claim and balance differ")
    block = _integer(at_block, "expiration block")
    if balance.status != "open" or block < claim.expires_block:
        return balance
    return DebtClaimBalance(
        balance.claim_digest,
        balance.principal_units,
        balance.paid_units,
        balance.forfeited_units + balance.remaining_units,
        0,
        "expired",
        block,
        "claim_lifetime_expired",
    )


def cancel_claim_balance(
    claim: InnovationDebtClaim,
    balance: DebtClaimBalance,
    *,
    at_block: int,
    reason: str,
) -> DebtClaimBalance:
    if type(claim) is not InnovationDebtClaim or type(balance) is not DebtClaimBalance:
        raise FiniteDebtError("claim cancellation authority is not exactly typed")
    if balance.claim_digest != claim.digest or balance.principal_units != claim.principal_units:
        raise FiniteDebtError("claim and balance differ")
    block = _integer(at_block, "cancellation block")
    if (
        balance.status != "open"
        or block < claim.settlement_block
        or not isinstance(reason, str)
        or not reason
        or len(reason) > 2_048
        or any(char in reason for char in "\x00\r\n")
    ):
        raise FiniteDebtError("claim cancellation is malformed or not open")
    return DebtClaimBalance(
        balance.claim_digest,
        balance.principal_units,
        balance.paid_units,
        balance.forfeited_units + balance.remaining_units,
        0,
        "cancelled",
        block,
        reason,
    )


@dataclass(frozen=True)
class DebtClaimState:
    claim: InnovationDebtClaim
    balance: DebtClaimBalance

    def __post_init__(self) -> None:
        if (
            type(self.claim) is not InnovationDebtClaim
            or type(self.balance) is not DebtClaimBalance
        ):
            raise FiniteDebtError("debt claim state is not exactly typed")
        if (
            self.balance.claim_digest != self.claim.digest
            or self.balance.principal_units != self.claim.principal_units
        ):
            raise FiniteDebtError("debt claim and balance differ")
        terminal = self.balance.terminal_block
        if terminal is not None and terminal < self.claim.settlement_block:
            raise FiniteDebtError("claim balance terminal block predates settlement")
        if (
            self.balance.status == "expired"
            and terminal is not None
            and terminal < self.claim.expires_block
        ):
            raise FiniteDebtError("claim balance expired before its lifetime ended")

    def to_dict(self) -> dict[str, object]:
        return {"balance": self.balance.to_dict(), "claim": self.claim.to_dict()}

    @classmethod
    def from_dict(cls, value: object) -> "DebtClaimState":
        row = _strict(value, {"balance", "claim"}, "debt claim state")
        return cls(
            InnovationDebtClaim.from_dict(row["claim"]),
            DebtClaimBalance.from_dict(row["balance"]),
        )

    @property
    def digest(self) -> str:
        return canonical_digest("optima.finite-debt.claim-state", self.to_dict())


@dataclass(frozen=True)
class DebtEpochAllocation:
    claim_digest: str
    hotkey: str
    units: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "claim_digest", _digest(self.claim_digest, "claim_digest"))
        object.__setattr__(self, "hotkey", _hotkey(self.hotkey))
        _integer(self.units, "allocation units", minimum=1)

    def to_dict(self) -> dict[str, object]:
        return {"claim_digest": self.claim_digest, "hotkey": self.hotkey, "units": self.units}

    @classmethod
    def from_dict(cls, value: object) -> "DebtEpochAllocation":
        row = _strict(value, set(cls.__dataclass_fields__), "debt epoch allocation")
        return cls(**row)  # type: ignore[arg-type]


@dataclass(frozen=True)
class DebtHotkeyWeight:
    hotkey: str
    units: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "hotkey", _hotkey(self.hotkey))
        _integer(self.units, "hotkey weight units", minimum=1, maximum=REFERENCE_EPOCH_UNITS)

    def to_dict(self) -> dict[str, object]:
        return {"hotkey": self.hotkey, "units": self.units}

    @classmethod
    def from_dict(cls, value: object) -> "DebtHotkeyWeight":
        row = _strict(value, set(cls.__dataclass_fields__), "debt hotkey weight")
        return cls(**row)  # type: ignore[arg-type]


@dataclass(frozen=True)
class DebtEpochProjection:
    policy_digest: str
    effective_block: int
    reference_epoch_units: int
    claim_pool_capacity_units: int
    total_remaining_units: int
    payout_units: int
    reserve_hotkey: str
    reserve_units: int
    input_state_digests: tuple[str, ...]
    allocations: tuple[DebtEpochAllocation, ...]
    weights: tuple[DebtHotkeyWeight, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy_digest", _digest(self.policy_digest, "policy_digest"))
        _integer(self.effective_block, "effective_block")
        if self.reference_epoch_units != REFERENCE_EPOCH_UNITS:
            raise FiniteDebtError("reference_epoch_units is unsupported")
        capacity = _integer(
            self.claim_pool_capacity_units,
            "claim_pool_capacity_units",
            maximum=REFERENCE_EPOCH_UNITS,
        )
        remaining = _integer(self.total_remaining_units, "total_remaining_units")
        payout = _integer(self.payout_units, "payout_units", maximum=capacity)
        reserve = _integer(
            self.reserve_units,
            "reserve_units",
            maximum=REFERENCE_EPOCH_UNITS,
        )
        object.__setattr__(self, "reserve_hotkey", _hotkey(self.reserve_hotkey, "reserve_hotkey"))
        inputs = tuple(self.input_state_digests)
        if inputs != tuple(sorted(set(inputs))):
            raise FiniteDebtError("projection input states are not canonical")
        for digest in inputs:
            _digest(digest, "input state digest")
        object.__setattr__(self, "input_state_digests", inputs)
        allocations = tuple(self.allocations)
        weights = tuple(self.weights)
        if (
            any(type(row) is not DebtEpochAllocation for row in allocations)
            or tuple(row.claim_digest for row in allocations)
            != tuple(sorted({row.claim_digest for row in allocations}))
            or any(type(row) is not DebtHotkeyWeight for row in weights)
            or tuple(row.hotkey for row in weights)
            != tuple(sorted({row.hotkey for row in weights}))
        ):
            raise FiniteDebtError("projection allocations or weights are not canonical")
        if (
            payout != min(capacity, remaining)
            or sum(row.units for row in allocations) != payout
            or reserve != REFERENCE_EPOCH_UNITS - payout
            or sum(row.units for row in weights) != REFERENCE_EPOCH_UNITS
        ):
            raise FiniteDebtError("debt epoch projection does not conserve capacity")
        by_hotkey: dict[str, int] = {}
        for allocation in allocations:
            by_hotkey[allocation.hotkey] = by_hotkey.get(allocation.hotkey, 0) + allocation.units
        by_hotkey[self.reserve_hotkey] = by_hotkey.get(self.reserve_hotkey, 0) + reserve
        if {row.hotkey: row.units for row in weights} != {
            hotkey: units for hotkey, units in by_hotkey.items() if units > 0
        }:
            raise FiniteDebtError("projection hotkey weights differ from claim allocations")
        object.__setattr__(self, "allocations", allocations)
        object.__setattr__(self, "weights", weights)

    def to_dict(self) -> dict[str, object]:
        return {
            "allocations": [row.to_dict() for row in self.allocations],
            "claim_pool_capacity_units": self.claim_pool_capacity_units,
            "effective_block": self.effective_block,
            "input_state_digests": list(self.input_state_digests),
            "payout_units": self.payout_units,
            "policy_digest": self.policy_digest,
            "reference_epoch_units": self.reference_epoch_units,
            "reserve_hotkey": self.reserve_hotkey,
            "reserve_units": self.reserve_units,
            "total_remaining_units": self.total_remaining_units,
            "weights": [row.to_dict() for row in self.weights],
        }

    @classmethod
    def from_dict(cls, value: object) -> "DebtEpochProjection":
        row = _strict(value, set(cls.__dataclass_fields__), "debt epoch projection")
        for field in ("allocations", "input_state_digests", "weights"):
            if type(row[field]) is not list:
                raise FiniteDebtError(f"projection {field} must be an array")
        row["allocations"] = tuple(
            DebtEpochAllocation.from_dict(item)
            for item in row["allocations"]  # type: ignore[union-attr]
        )
        row["input_state_digests"] = tuple(row["input_state_digests"])  # type: ignore[arg-type]
        row["weights"] = tuple(
            DebtHotkeyWeight.from_dict(item) for item in row["weights"]  # type: ignore[union-attr]
        )
        return cls(**row)  # type: ignore[arg-type]

    @property
    def weights_by_hotkey(self) -> Mapping[str, int]:
        return {row.hotkey: row.units for row in self.weights}

    @property
    def digest(self) -> str:
        return canonical_digest("optima.finite-debt.epoch-projection", self.to_dict())


def project_debt_epoch(
    policy: FiniteDebtPolicyManifest,
    *,
    effective_block: int,
    states: Iterable[DebtClaimState],
) -> DebtEpochProjection:
    """Allocate one finite claim-pool epoch, with all residue sent to reserve."""

    if type(policy) is not FiniteDebtPolicyManifest:
        raise FiniteDebtError("policy is not exactly typed")
    block = _integer(effective_block, "effective_block")
    rows = tuple(states)
    if any(type(row) is not DebtClaimState for row in rows):
        raise FiniteDebtError("debt claim states are not exactly typed")
    if len({row.claim.digest for row in rows}) != len(rows):
        raise FiniteDebtError("debt claim states contain duplicate claims")
    rows = tuple(sorted(rows, key=lambda row: row.claim.digest))
    for row in rows:
        row.claim.validate_policy(policy)
        if row.claim.hotkey == policy.reserve_hotkey and row.balance.status == "open":
            raise FiniteDebtError("reserve hotkey cannot own an open debt claim")
        if row.balance.status == "open" and block < row.claim.settlement_block:
            raise FiniteDebtError("claim cannot receive payout before settlement")
        if row.balance.status == "open" and block >= row.claim.expires_block:
            raise FiniteDebtError("open claim is expired; expire its balance before projection")
    open_rows = tuple(row for row in rows if row.balance.status == "open")
    total_remaining = sum(row.balance.remaining_units for row in open_rows)
    capacity = policy.reference_claim_pool_units
    payout = min(capacity, total_remaining)
    amounts: dict[str, int] = {}
    if payout and total_remaining <= capacity:
        amounts = {
            row.claim.digest: row.balance.remaining_units for row in open_rows
        }
    elif payout:
        remainders = []
        for row in open_rows:
            quotient, remainder = divmod(
                row.balance.remaining_units * payout, total_remaining
            )
            amounts[row.claim.digest] = quotient
            remainders.append((remainder, row.claim.digest))
        missing = payout - sum(amounts.values())
        for _remainder, claim_digest in sorted(
            remainders, key=lambda item: (-item[0], item[1])
        )[:missing]:
            amounts[claim_digest] += 1
    allocations = tuple(
        DebtEpochAllocation(row.claim.digest, row.claim.hotkey, amounts[row.claim.digest])
        for row in open_rows
        if amounts.get(row.claim.digest, 0) > 0
    )
    by_hotkey: dict[str, int] = {}
    for allocation in allocations:
        by_hotkey[allocation.hotkey] = by_hotkey.get(allocation.hotkey, 0) + allocation.units
    reserve = REFERENCE_EPOCH_UNITS - payout
    if reserve:
        by_hotkey[policy.reserve_hotkey] = reserve
    weights = tuple(
        DebtHotkeyWeight(hotkey, by_hotkey[hotkey]) for hotkey in sorted(by_hotkey)
    )
    result = DebtEpochProjection(
        policy.digest,
        block,
        REFERENCE_EPOCH_UNITS,
        capacity,
        total_remaining,
        payout,
        policy.reserve_hotkey,
        reserve,
        tuple(sorted(row.digest for row in rows)),
        allocations,
        weights,
    )
    remaining_by_claim = {
        row.claim.digest: row.balance.remaining_units for row in open_rows
    }
    if any(
        allocation.units > remaining_by_claim[allocation.claim_digest]
        for allocation in result.allocations
    ):
        raise FiniteDebtError("projection allocation exceeds remaining principal")
    return result


def apply_debt_epoch_projection(
    states: Iterable[DebtClaimState], projection: DebtEpochProjection
) -> tuple[DebtClaimState, ...]:
    """Apply a retained projection exactly once to matching immutable balances."""

    if type(projection) is not DebtEpochProjection:
        raise FiniteDebtError("projection is not exactly typed")
    rows = tuple(states)
    if any(type(row) is not DebtClaimState for row in rows):
        raise FiniteDebtError("debt claim states are not exactly typed")
    rows = tuple(sorted(rows, key=lambda row: row.claim.digest))
    if tuple(sorted(row.digest for row in rows)) != projection.input_state_digests:
        raise FiniteDebtError("projection input balances changed before application")
    state_by_claim = {row.claim.digest: row for row in rows}
    for allocation in projection.allocations:
        state = state_by_claim.get(allocation.claim_digest)
        if (
            state is None
            or state.balance.status != "open"
            or state.claim.hotkey != allocation.hotkey
            or allocation.units > state.balance.remaining_units
        ):
            raise FiniteDebtError("projection allocation differs from its claim state")
    amounts = {row.claim_digest: row.units for row in projection.allocations}
    updated = []
    for row in rows:
        amount = amounts.get(row.claim.digest, 0)
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
        updated.append(DebtClaimState(row.claim, balance))
    if sum(
        row.balance.paid_units - state_by_claim[row.claim.digest].balance.paid_units
        for row in updated
    ) != projection.payout_units:
        raise FiniteDebtError("projection application did not consume its full payout")
    return tuple(updated)


__all__ = [
    "BALANCE_STATUSES",
    "DECIMAL_PRECISION",
    "DebtClaimBalance",
    "DebtClaimState",
    "DebtEpochAllocation",
    "DebtEpochProjection",
    "DebtHotkeyWeight",
    "CampaignBudgetShare",
    "FiniteDebtError",
    "FiniteDebtPolicyManifest",
    "IMPROVEMENT_BASES",
    "IMPROVEMENT_EXCESS",
    "IMPROVEMENT_GROSS",
    "InnovationDebtClaim",
    "LOG_UNIT_STEP",
    "MAX_ACTIVE_CAMPAIGNS",
    "POLICY_SCHEMA_VERSION",
    "POLICY_VERSION",
    "PPM",
    "REFERENCE_EPOCH_UNITS",
    "RewardFamilyCampaign",
    "apply_debt_epoch_projection",
    "cancel_claim_balance",
    "expire_claim_balance",
    "equal_campaign_budget_shares",
    "issue_innovation_claim",
    "log_improvement_units_ppm",
    "pay_claim_balance",
    "project_debt_epoch",
    "rational_time_multiplier_ppm",
    "resets_family_clock",
]
