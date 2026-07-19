from __future__ import annotations

from dataclasses import replace
from decimal import Decimal, ROUND_HALF_EVEN, localcontext

import pytest

from optima.finite_debt import (
    CampaignBudgetShare,
    IMPROVEMENT_EXCESS,
    IMPROVEMENT_GROSS,
    PPM,
    DebtClaimBalance,
    DebtClaimState,
    DebtEpochProjection,
    FiniteDebtError,
    FiniteDebtPolicyManifest,
    RewardFamilyCampaign,
    apply_debt_epoch_projection,
    cancel_claim_balance,
    equal_campaign_budget_shares,
    expire_claim_balance,
    issue_innovation_claim,
    log_improvement_units_ppm,
    pay_claim_balance,
    project_debt_epoch,
    rational_time_multiplier_ppm,
    resets_family_clock,
)
from optima.stack_identity import sha256_hex


FAMILY_A = sha256_hex(b"family-a")
FAMILY_B = sha256_hex(b"family-b")
CAMPAIGN_A = sha256_hex(b"campaign-a")
CAMPAIGN_B = sha256_hex(b"campaign-b")
SELECTION_REPORT = sha256_hex(b"d015-selection-report")


def _d(label: str) -> str:
    return sha256_hex(label.encode())


def _policy(
    *,
    campaign_shares: tuple[CampaignBudgetShare, ...] | None = None,
    family_campaigns: tuple[RewardFamilyCampaign, ...] | None = None,
    reserve_ppm: int = 100_000,
    epoch_blocks: int = 7_200,
    beta_ppm: int = 500_000,
    tau_blocks: int = 90,
    lifetime_blocks: int = 900,
    k_ppm: int = PPM,
    basis: str = IMPROVEMENT_GROSS,
    reset_threshold: int = 1,
) -> FiniteDebtPolicyManifest:
    return FiniteDebtPolicyManifest(
        campaign_budget_shares=campaign_shares
        or (CampaignBudgetShare(CAMPAIGN_A, PPM),),
        reward_family_campaigns=family_campaigns
        or (RewardFamilyCampaign(FAMILY_A, CAMPAIGN_A),),
        selection_report_digest=SELECTION_REPORT,
        reserve_hotkey="reserve",
        reserve_ppm=reserve_ppm,
        epoch_blocks=epoch_blocks,
        beta_ppm=beta_ppm,
        tau_blocks=tau_blocks,
        lifetime_blocks=lifetime_blocks,
        k_ppm=k_ppm,
        improvement_basis=basis,
        clock_reset_threshold_log_units_ppm=reset_threshold,
    )


def _claim(
    policy: FiniteDebtPolicyManifest,
    *,
    label: str = "one",
    family: str = FAMILY_A,
    speedup: str = "1.01",
    threshold: str = "1",
    accepted: int = 100,
    prior: int | None = None,
    settlement: int = 120,
    hotkey: str = "miner",
):
    return issue_innovation_claim(
        policy,
        family_id=family,
        candidate_digest=_d(f"candidate:{label}"),
        retained_evidence_digest=_d(f"evidence:{label}"),
        hotkey=hotkey,
        settled_speedup=speedup,
        threshold_speedup=threshold,
        accepted_crown_block=accepted,
        prior_accepted_crown_block=prior,
        settlement_block=settlement,
    )


def _state(claim) -> DebtClaimState:
    return DebtClaimState(claim, DebtClaimBalance.open(claim))


def test_policy_is_strict_content_addressed_and_campaign_map_is_canonical() -> None:
    shares = (
        CampaignBudgetShare(CAMPAIGN_B, 500_000),
        CampaignBudgetShare(CAMPAIGN_A, 500_000),
    )
    mappings = (
        RewardFamilyCampaign(FAMILY_B, CAMPAIGN_B),
        RewardFamilyCampaign(FAMILY_A, CAMPAIGN_A),
    )
    policy = _policy(campaign_shares=shares, family_campaigns=mappings)
    assert tuple(row.campaign_id for row in policy.campaign_budget_shares) == tuple(
        sorted((CAMPAIGN_A, CAMPAIGN_B))
    )
    assert policy.family_ids == tuple(sorted((FAMILY_A, FAMILY_B)))
    assert FiniteDebtPolicyManifest.from_dict(policy.to_dict()) == policy
    assert FiniteDebtPolicyManifest.from_dict(policy.to_dict()).digest == policy.digest
    with pytest.raises(FiniteDebtError, match="one 100% campaign or two 50%"):
        _policy(
            campaign_shares=(CampaignBudgetShare(CAMPAIGN_A, 999_999),)
        )
    with pytest.raises(FiniteDebtError, match="every reward family"):
        _policy(
            campaign_shares=shares,
            family_campaigns=(RewardFamilyCampaign(FAMILY_A, CAMPAIGN_A),),
        )
    with pytest.raises(FiniteDebtError, match="every reward family"):
        _policy(
            family_campaigns=(
                RewardFamilyCampaign(FAMILY_A, CAMPAIGN_A),
                RewardFamilyCampaign(FAMILY_A, CAMPAIGN_A),
            )
        )
    with pytest.raises(FiniteDebtError, match="fields"):
        FiniteDebtPolicyManifest.from_dict({**policy.to_dict(), "extra": 1})
    with pytest.raises(FiniteDebtError, match="reserve_ppm"):
        _policy(reserve_ppm=PPM)
    with pytest.raises(FiniteDebtError, match="epoch_blocks"):
        _policy(epoch_blocks=0)
    with pytest.raises(FiniteDebtError, match="campaign reference claim pool"):
        _policy(
            campaign_shares=shares,
            family_campaigns=mappings,
            reserve_ppm=999_999,
        )


def test_campaign_roster_supports_only_one_full_or_two_equal_campaigns() -> None:
    one = equal_campaign_budget_shares((CAMPAIGN_A,))
    assert one == (CampaignBudgetShare(CAMPAIGN_A, PPM),)
    two = equal_campaign_budget_shares((CAMPAIGN_B, CAMPAIGN_A))
    assert tuple(row.campaign_id for row in two) == tuple(
        sorted((CAMPAIGN_A, CAMPAIGN_B))
    )
    assert tuple(row.share_ppm for row in two) == (500_000, 500_000)
    with pytest.raises(FiniteDebtError, match="nonempty"):
        equal_campaign_budget_shares(())
    with pytest.raises(FiniteDebtError, match="unique"):
        equal_campaign_budget_shares((CAMPAIGN_A, CAMPAIGN_A))
    with pytest.raises(FiniteDebtError, match="at most two"):
        equal_campaign_budget_shares(
            (
                CAMPAIGN_A,
                CAMPAIGN_B,
                _d("campaign-c"),
            ),
        )


def test_selected_curve_policy_vector_binds_epoch_cadence_and_claim_terms() -> None:
    policy = _policy(
        epoch_blocks=7_200,
        beta_ppm=100_000,
        tau_blocks=648_000,
        lifetime_blocks=648_000,
        k_ppm=PPM,
        reserve_ppm=100_000,
        basis=IMPROVEMENT_GROSS,
        reset_threshold=1,
    )
    assert policy.to_dict() == {
        "beta_ppm": 100_000,
        "campaign_budget_shares": [
            {"campaign_id": CAMPAIGN_A, "share_ppm": PPM}
        ],
        "clock_reset_threshold_log_units_ppm": 1,
        "epoch_blocks": 7_200,
        "improvement_basis": IMPROVEMENT_GROSS,
        "k_ppm": PPM,
        "lifetime_blocks": 648_000,
        "policy_version": "optima.finite-debt.v2",
        "reserve_hotkey": "reserve",
        "reserve_ppm": 100_000,
        "reward_family_campaigns": [
            {"campaign_id": CAMPAIGN_A, "family_id": FAMILY_A}
        ],
        "schema_version": 2,
        "selection_report_digest": SELECTION_REPORT,
        "tau_blocks": 648_000,
    }
    assert FiniteDebtPolicyManifest.from_dict(policy.to_dict()) == policy
    assert replace(policy, epoch_blocks=7_201).digest != policy.digest

    claim = _claim(policy, speedup="1.044", prior=None)
    assert claim.log_units_ppm == 4_327_442
    assert claim.time_multiplier_ppm == PPM
    assert claim.principal_units == 3_894_697
    assert claim.expires_block == claim.settlement_block + 648_000
    five_percent = _claim(policy, label="five-percent", speedup="1.05", prior=None)
    assert five_percent.principal_units == 4_413_033


def test_production_floor_conforms_to_frozen_v2_half_even_within_one_unit() -> None:
    """The selected 1.044x vector preserves the intentional rounding boundary."""

    policy = _policy(
        epoch_blocks=7_200,
        beta_ppm=100_000,
        tau_blocks=648_000,
        lifetime_blocks=648_000,
        k_ppm=PPM,
        reserve_ppm=100_000,
        basis=IMPROVEMENT_GROSS,
        reset_threshold=1,
    )
    production = _claim(policy, speedup="1.044", prior=None)

    # Frozen V2 selected curves converted integer-ppm speedup to 1%-log
    # nano-units with ROUND_HALF_EVEN before flooring claim principal.
    with localcontext() as context:
        context.prec = 50
        v2_log_units_nano = int(
            (
                (Decimal(1_044_000) / Decimal(PPM)).ln()
                / (Decimal(1_010_000) / Decimal(PPM)).ln()
                * Decimal(1_000_000_000)
            ).to_integral_value(rounding=ROUND_HALF_EVEN)
        )
    v2_principal = (
        policy.reference_claim_pool_units
        * policy.k_ppm
        * v2_log_units_nano
        * PPM  # first-crown time multiplier
        * policy.campaign_share_ppm_for_family(FAMILY_A)
        // (1_000_000_000 * PPM * PPM * PPM)
    )

    assert production.log_units_ppm == 4_327_442
    assert v2_log_units_nano == 4_327_442_986
    assert (production.principal_units, v2_principal) == (3_894_697, 3_894_698)
    assert abs(production.principal_units - v2_principal) <= 1


def test_one_log_unit_one_epoch_principal_uses_post_reserve_campaign_pool() -> None:
    policy = _policy(reserve_ppm=100_000, k_ppm=PPM)
    claim = _claim(policy, prior=None)
    assert claim.log_units_ppm == PPM
    assert claim.time_multiplier_ppm == PPM
    assert claim.campaign_id == CAMPAIGN_A
    assert claim.reference_campaign_pool_units == 900_000
    assert claim.principal_units == 900_000
    assert claim.expires_block == claim.settlement_block + policy.lifetime_blocks

    state = _state(claim)
    projection = project_debt_epoch(
        policy, effective_block=121, states=(state,)
    )
    assert projection.payout_units == claim.principal_units
    assert projection.weights_by_hotkey == {"miner": 900_000, "reserve": 100_000}
    paid = apply_debt_epoch_projection((state,), projection)[0].balance
    assert paid.status == "paid" and paid.paid_units == claim.principal_units


def test_family_count_does_not_dilute_campaign_principal() -> None:
    one_campaign = _policy(
        family_campaigns=(
            RewardFamilyCampaign(FAMILY_A, CAMPAIGN_A),
            RewardFamilyCampaign(FAMILY_B, CAMPAIGN_A),
        ),
    )
    left = _claim(one_campaign, family=FAMILY_A, label="left")
    right = _claim(one_campaign, family=FAMILY_B, label="right")
    assert (left.principal_units, right.principal_units) == (900_000, 900_000)

    hundred_families = (FAMILY_A,) + tuple(
        _d(f"unused-family:{index}") for index in range(99)
    )
    large_catalog = _policy(
        family_campaigns=tuple(
            RewardFamilyCampaign(family, CAMPAIGN_A)
            for family in hundred_families
        )
    )
    large_claim = _claim(large_catalog, family=FAMILY_A, label="large-catalog")
    assert large_claim.principal_units == left.principal_units
    assert large_claim.reference_campaign_pool_units == 900_000

    two_campaigns = _policy(
        campaign_shares=(
            CampaignBudgetShare(CAMPAIGN_A, 500_000),
            CampaignBudgetShare(CAMPAIGN_B, 500_000),
        ),
        family_campaigns=(
            RewardFamilyCampaign(FAMILY_A, CAMPAIGN_A),
            RewardFamilyCampaign(FAMILY_B, CAMPAIGN_B),
        ),
    )
    left = _claim(two_campaigns, family=FAMILY_A, label="half-left")
    right = _claim(two_campaigns, family=FAMILY_B, label="half-right")
    assert left.reference_campaign_pool_units == 450_000
    assert right.reference_campaign_pool_units == 450_000
    assert (left.principal_units, right.principal_units) == (450_000, 450_000)
    material = _claim(
        two_campaigns,
        family=FAMILY_A,
        label="half-material",
        speedup="1.044",
    )
    assert material.principal_units == 1_947_348


def test_one_percent_log_units_are_path_independent_on_exact_power_vectors() -> None:
    one = log_improvement_units_ppm(
        "1.01", basis=IMPROVEMENT_GROSS, threshold_speedup="1"
    )
    two = log_improvement_units_ppm(
        "1.0201", basis=IMPROVEMENT_GROSS, threshold_speedup="1"
    )
    three = log_improvement_units_ppm(
        "1.030301", basis=IMPROVEMENT_GROSS, threshold_speedup="1"
    )
    assert (one, two, three) == (PPM, 2 * PPM, 3 * PPM)
    assert three == one + two


def test_excess_over_threshold_uses_log_ratio_not_subtracted_percentages() -> None:
    assert log_improvement_units_ppm(
        "1.0201",
        basis=IMPROVEMENT_EXCESS,
        threshold_speedup="1.01",
    ) == PPM
    policy = _policy(basis=IMPROVEMENT_EXCESS)
    claim = _claim(policy, speedup="1.0201", threshold="1.01")
    assert claim.log_units_ppm == PPM
    with pytest.raises(FiniteDebtError, match="must exceed"):
        log_improvement_units_ppm(
            "1.01", basis=IMPROVEMENT_EXCESS, threshold_speedup="1.01"
        )
    with pytest.raises(FiniteDebtError, match="canonical"):
        log_improvement_units_ppm(
            "1.010", basis=IMPROVEMENT_GROSS, threshold_speedup="1"
        )


def test_first_family_crown_has_no_chain_age_bonus_and_rational_curve_is_bounded() -> None:
    policy = _policy(beta_ppm=500_000, tau_blocks=90)
    assert rational_time_multiplier_ppm(
        policy, accepted_crown_block=10_000_000, prior_accepted_crown_block=None
    ) == PPM
    assert rational_time_multiplier_ppm(
        policy, accepted_crown_block=190, prior_accepted_crown_block=100
    ) == 1_250_000
    late = rational_time_multiplier_ppm(
        policy, accepted_crown_block=1_000_000, prior_accepted_crown_block=100
    )
    assert 1_250_000 < late < 1_500_000
    claim = _claim(policy, accepted=10_000_000, settlement=10_000_001, prior=None)
    assert claim.time_multiplier_ppm == PPM


def test_tiny_accepted_improvement_can_earn_without_resetting_family_clock() -> None:
    policy = _policy(reset_threshold=101)
    assert not resets_family_clock(policy, 100)
    assert resets_family_clock(policy, 101)
    tiny = _claim(policy, speedup="1.000001")
    assert tiny.log_units_ppm == 100
    assert tiny.principal_units == 90
    assert not tiny.resets_clock


def test_no_debt_sends_the_entire_epoch_to_the_policy_reserve() -> None:
    policy = _policy(reserve_ppm=100_000)
    projection = project_debt_epoch(policy, effective_block=500, states=())
    assert projection.claim_pool_capacity_units == 900_000
    assert projection.payout_units == 0
    assert projection.reserve_units == PPM
    assert projection.weights_by_hotkey == {"reserve": PPM}
    assert DebtEpochProjection.from_dict(projection.to_dict()) == projection
    assert DebtEpochProjection.from_dict(projection.to_dict()).digest == projection.digest


def test_pro_rata_epoch_is_bounded_by_each_remaining_principal() -> None:
    policy = _policy()
    small = _state(_claim(policy, label="small", speedup="1.01", hotkey="alice"))
    large = _state(_claim(policy, label="large", speedup="1.0201", hotkey="bob"))
    projection = project_debt_epoch(
        policy, effective_block=121, states=(large, small)
    )
    by_claim = {row.claim_digest: row.units for row in projection.allocations}
    assert by_claim[small.claim.digest] == 300_000
    assert by_claim[large.claim.digest] == 600_000
    assert all(
        by_claim[row.claim.digest] <= row.balance.remaining_units
        for row in (small, large)
    )
    updated = apply_debt_epoch_projection((small, large), projection)
    assert sum(row.balance.paid_units for row in updated) == projection.payout_units
    assert all(
        row.balance.paid_units
        + row.balance.forfeited_units
        + row.balance.remaining_units
        == row.balance.principal_units
        for row in updated
    )


def test_largest_remainder_tie_breaks_by_claim_digest_not_input_order() -> None:
    policy = _policy(reserve_ppm=999_999)
    left = _state(_claim(policy, label="left", hotkey="alice"))
    right = _state(_claim(policy, label="right", hotkey="bob"))
    first = project_debt_epoch(
        policy, effective_block=121, states=(left, right)
    )
    reordered = project_debt_epoch(
        policy, effective_block=121, states=(right, left)
    )
    assert first.digest == reordered.digest
    assert len(first.allocations) == 1
    assert first.allocations[0].claim_digest == min(
        left.claim.digest, right.claim.digest
    )
    assert first.allocations[0].units == 1


def test_repeated_epochs_never_overspend_a_claim() -> None:
    policy = _policy(reserve_ppm=999_999)
    states = (
        _state(_claim(policy, label="left", hotkey="alice")),
        _state(_claim(policy, label="right", hotkey="bob")),
    )
    first = project_debt_epoch(policy, effective_block=121, states=states)
    states = apply_debt_epoch_projection(states, first)
    second = project_debt_epoch(policy, effective_block=122, states=states)
    states = apply_debt_epoch_projection(states, second)
    assert all(row.balance.status == "paid" for row in states)
    assert sum(row.balance.paid_units for row in states) == 2
    third = project_debt_epoch(policy, effective_block=123, states=states)
    assert third.payout_units == 0 and third.reserve_units == PPM
    with pytest.raises(FiniteDebtError, match="changed"):
        apply_debt_epoch_projection(states, first)


def test_expiration_and_cancellation_forfeit_only_unpaid_principal() -> None:
    policy = _policy(lifetime_blocks=10)
    expiring = _claim(policy, label="expiring", settlement=120)
    partly_paid = pay_claim_balance(
        expiring, DebtClaimBalance.open(expiring), 100, at_block=121
    )
    before_expiry = expire_claim_balance(
        expiring, partly_paid, at_block=129
    )
    assert before_expiry == partly_paid
    expired = expire_claim_balance(expiring, partly_paid, at_block=130)
    assert expired.status == "expired"
    assert expired.paid_units == 100
    assert expired.forfeited_units == expiring.principal_units - 100

    cancelled_claim = _claim(policy, label="cancelled")
    cancelled = cancel_claim_balance(
        cancelled_claim,
        DebtClaimBalance.open(cancelled_claim),
        at_block=121,
        reason="hotkey_departed",
    )
    assert cancelled.status == "cancelled"
    assert cancelled.forfeited_units == cancelled_claim.principal_units

    projection = project_debt_epoch(
        policy,
        effective_block=131,
        states=(
            DebtClaimState(expiring, expired),
            DebtClaimState(cancelled_claim, cancelled),
        ),
    )
    assert projection.weights_by_hotkey == {"reserve": PPM}


def test_projection_refuses_unclosed_expiry_and_reserve_owned_claim() -> None:
    policy = _policy(lifetime_blocks=10)
    claim = _claim(policy, settlement=120)
    with pytest.raises(FiniteDebtError, match="expire"):
        project_debt_epoch(policy, effective_block=130, states=(_state(claim),))
    reserve_claim = _state(_claim(policy, label="reserve", hotkey="reserve"))
    with pytest.raises(FiniteDebtError, match="reserve hotkey"):
        project_debt_epoch(policy, effective_block=121, states=(reserve_claim,))


def test_claim_balance_and_projection_identities_reject_inconsistent_terms() -> None:
    policy = _policy()
    claim = _claim(policy)
    assert type(claim).from_dict(claim.to_dict()) == claim
    balance = DebtClaimBalance.open(claim)
    assert DebtClaimBalance.from_dict(balance.to_dict()) == balance
    with pytest.raises(FiniteDebtError, match="conserve"):
        replace(balance, remaining_units=balance.remaining_units - 1)
    with pytest.raises(FiniteDebtError, match="derived terms"):
        replace(claim, principal_units=claim.principal_units + 1).validate_policy(
            policy
        )
    with pytest.raises(FiniteDebtError, match="derived terms"):
        replace(claim, campaign_id=CAMPAIGN_B).validate_policy(policy)
    with pytest.raises(FiniteDebtError, match="derived terms"):
        replace(claim, campaign_budget_ppm=500_000).validate_policy(policy)
