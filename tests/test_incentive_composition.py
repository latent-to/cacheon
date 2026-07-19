from __future__ import annotations

from dataclasses import replace
from typing import Callable

import pytest

from optima.finite_debt import (
    CampaignBudgetShare,
    IMPROVEMENT_GROSS,
    POLICY_SCHEMA_VERSION,
    POLICY_VERSION,
    PPM,
    DebtClaimBalance,
    DebtClaimState,
    FiniteDebtPolicyManifest,
    RewardFamilyCampaign,
    issue_innovation_claim,
    pay_claim_balance,
)
from optima.incentive_composition import (
    COMPOSITION_POLICY_SCHEMA_VERSION,
    COMPOSITION_POLICY_VERSION,
    DISCOVERY_BOUNTY_ONLY,
    DISCOVERY_REGISTERED_PROMOTION,
    ComposedEpochProjection,
    CompositionClassAllocation,
    DiscoveryClaimBalance,
    DiscoveryClaimState,
    DiscoveryDebtClaim,
    IncentiveCompositionError,
    IncentiveCompositionPolicyManifest,
    ReviewedDiscoveryDisposition,
    apply_composed_epoch,
    cancel_discovery_balance,
    expire_discovery_balance,
    issue_discovery_claim,
    pay_discovery_balance,
    project_composed_epoch,
    review_discovery_disposition,
)
from optima.stack_identity import sha256_hex


FAMILY = sha256_hex(b"composition-family")
CAMPAIGN = sha256_hex(b"minimax-m3-campaign")
CORE_SELECTION_REPORT_DIGEST = (
    "7975a10b2924330cd527e29b0dfe6f2d9dcb40039f9d8f695b558ec6c6f46590"
)
SELECTION_REPORT_DIGEST = (
    "7369c6890dcc880b5f7295a94d07f915d59241e23d95b2c9328295780c99fb38"
)


def _d(label: str) -> str:
    return sha256_hex(label.encode())


def _innovation_policy(
    *,
    reserve_ppm: int = 100_000,
    epoch_blocks: int = 7_200,
) -> FiniteDebtPolicyManifest:
    """The selected D-015 registered-CROWN policy, with every term explicit."""

    return FiniteDebtPolicyManifest(
        campaign_budget_shares=(CampaignBudgetShare(CAMPAIGN, PPM),),
        reward_family_campaigns=(RewardFamilyCampaign(FAMILY, CAMPAIGN),),
        selection_report_digest=CORE_SELECTION_REPORT_DIGEST,
        reserve_hotkey="reserve",
        reserve_ppm=reserve_ppm,
        epoch_blocks=epoch_blocks,
        beta_ppm=100_000,
        tau_blocks=648_000,
        lifetime_blocks=648_000,
        k_ppm=PPM,
        improvement_basis=IMPROVEMENT_GROSS,
        clock_reset_threshold_log_units_ppm=1,
        schema_version=POLICY_SCHEMA_VERSION,
        policy_version=POLICY_VERSION,
    )


def _composition_policy(
    innovation_policy: FiniteDebtPolicyManifest | None = None,
    *,
    reserve_ppm: int = 100_000,
    epoch_blocks: int = 7_200,
    discovery_cap_units: int = 50_000,
    per_award_principal_cap_epochs: int = 1,
    discovery_lifetime_blocks: int = 648_000,
) -> IncentiveCompositionPolicyManifest:
    """The selected D-013 composition policy, with no economic defaults."""

    core = innovation_policy or _innovation_policy()
    return IncentiveCompositionPolicyManifest(
        innovation_policy_digest=core.digest,
        selection_report_digest=SELECTION_REPORT_DIGEST,
        reserve_ppm=reserve_ppm,
        epoch_blocks=epoch_blocks,
        discovery_cap_units=discovery_cap_units,
        per_award_principal_cap_epochs=per_award_principal_cap_epochs,
        discovery_lifetime_blocks=discovery_lifetime_blocks,
        schema_version=COMPOSITION_POLICY_SCHEMA_VERSION,
        policy_version=COMPOSITION_POLICY_VERSION,
    )


def _disposition(
    policy: IncentiveCompositionPolicyManifest,
    *,
    label: str = "one",
    hotkey: str = "discovery-miner",
    block: int = 0,
    win_block: int | None = None,
    decision: str = DISCOVERY_BOUNTY_ONLY,
    requested_epochs: int = 1,
    promoted_target_digest: str = "",
) -> ReviewedDiscoveryDisposition:
    return review_discovery_disposition(
        policy,
        win_digest=_d(f"win:{label}"),
        proposal_digest=_d(f"proposal:{label}"),
        retained_evidence_digest=_d(f"discovery-evidence:{label}"),
        review_digest=_d(f"review:{label}"),
        hotkey=hotkey,
        win_block=block if win_block is None else win_block,
        authority_block=block,
        decision=decision,
        requested_principal_epochs=requested_epochs,
        promoted_target_digest=promoted_target_digest,
    )


def _discovery_state(
    policy: IncentiveCompositionPolicyManifest,
    *,
    label: str = "one",
    hotkey: str = "discovery-miner",
    block: int = 0,
    requested_epochs: int = 1,
) -> DiscoveryClaimState:
    claim = issue_discovery_claim(
        policy,
        _disposition(
            policy,
            label=label,
            hotkey=hotkey,
            block=block,
            requested_epochs=requested_epochs,
        ),
    )
    assert claim is not None
    return DiscoveryClaimState(claim, DiscoveryClaimBalance.open(claim))


def _innovation_state(
    policy: FiniteDebtPolicyManifest,
    *,
    label: str = "one",
    hotkey: str = "innovation-miner",
    block: int = 0,
) -> DebtClaimState:
    claim = issue_innovation_claim(
        policy,
        family_id=FAMILY,
        candidate_digest=_d(f"candidate:{label}"),
        retained_evidence_digest=_d(f"innovation-evidence:{label}"),
        hotkey=hotkey,
        settled_speedup="1.01",
        threshold_speedup="1",
        accepted_crown_block=block,
        prior_accepted_crown_block=None,
        settlement_block=block,
    )
    return DebtClaimState(claim, DebtClaimBalance.open(claim))


def _projection(
    innovation_policy: FiniteDebtPolicyManifest,
    composition_policy: IncentiveCompositionPolicyManifest,
    innovation_states: tuple[DebtClaimState, ...],
    discovery_states: tuple[DiscoveryClaimState, ...],
    *,
    block: int = 0,
) -> ComposedEpochProjection:
    return project_composed_epoch(
        innovation_policy,
        composition_policy,
        effective_block=block,
        innovation_states=innovation_states,
        discovery_states=discovery_states,
    )


def test_selected_policy_and_discovery_values_have_golden_strict_serialization() -> None:
    innovation_policy = _innovation_policy()
    policy = _composition_policy(innovation_policy)
    disposition = _disposition(policy, label="golden", requested_epochs=9)
    claim = issue_discovery_claim(policy, disposition)
    assert claim is not None
    balance = DiscoveryClaimBalance.open(claim)
    state = DiscoveryClaimState(claim, balance)

    assert policy.to_dict() == {
        "innovation_policy_digest": innovation_policy.digest,
        "selection_report_digest": SELECTION_REPORT_DIGEST,
        "reserve_ppm": 100_000,
        "epoch_blocks": 7_200,
        "discovery_cap_units": 50_000,
        "per_award_principal_cap_epochs": 1,
        "discovery_lifetime_blocks": 648_000,
        "schema_version": 1,
        "policy_version": "optima.incentive-composition.v1",
    }
    assert disposition.to_dict() == {
        "policy_digest": policy.digest,
        "win_digest": _d("win:golden"),
        "proposal_digest": _d("proposal:golden"),
        "retained_evidence_digest": _d("discovery-evidence:golden"),
        "review_digest": _d("review:golden"),
        "hotkey": "discovery-miner",
        "win_block": 0,
        "authority_block": 0,
        "decision": "bounty_only",
        "requested_principal_epochs": 9,
        "promoted_target_digest": "",
    }
    assert claim.to_dict() == {
        "policy_digest": policy.digest,
        "disposition_digest": disposition.digest,
        "proposal_digest": disposition.proposal_digest,
        "retained_evidence_digest": disposition.retained_evidence_digest,
        "review_digest": disposition.review_digest,
        "hotkey": "discovery-miner",
        "awarded_block": 0,
        "expires_block": 648_000,
        "requested_principal_epochs": 9,
        "capped_principal_epochs": 1,
        "principal_units": 50_000,
    }
    assert DiscoveryClaimState.from_dict(state.to_dict()) == state
    assert IncentiveCompositionPolicyManifest.from_dict(policy.to_dict()) == policy
    assert ReviewedDiscoveryDisposition.from_dict(disposition.to_dict()) == disposition
    assert DiscoveryDebtClaim.from_dict(claim.to_dict()) == claim
    assert DiscoveryClaimBalance.from_dict(balance.to_dict()) == balance

    # These literals make any consensus-serialization drift an explicit change.
    assert policy.digest == "47da366bcc2abf80153fda3d206bef7fc06e31bebceaa58b09630e1c6df7b013"
    assert disposition.digest == "a7d5ebf011c0765c3212f586fd0728f76d9752fcb07446a6ec61d1c21d6da234"
    assert claim.digest == "fa34d13594e125eaccf60d8b31a4472f005a6b004eb1a7065d676ca17d6909ae"
    assert balance.digest == "a40049ed2b1fbd6d4ccd57361c23560b078c06ae8887bf32be4c10c5a6dd376b"
    assert state.digest == "66ed679cb6fae27f0dc34f150a30ab2ed3fcb03ab7bca4a6e0bf0da23c4c679e"


def test_every_public_payload_parser_rejects_missing_extra_and_non_plain_dicts() -> None:
    innovation_policy = _innovation_policy()
    policy = _composition_policy(innovation_policy)
    disposition = _disposition(policy)
    claim = issue_discovery_claim(policy, disposition)
    assert claim is not None
    balance = DiscoveryClaimBalance.open(claim)
    state = DiscoveryClaimState(claim, balance)
    projection = _projection(
        innovation_policy,
        policy,
        (_innovation_state(innovation_policy),),
        (state,),
    )
    allocation = projection.discovery_allocations[0]

    parsers_and_rows: tuple[
        tuple[Callable[[object], object], dict[str, object]], ...
    ] = (
        (IncentiveCompositionPolicyManifest.from_dict, policy.to_dict()),
        (ReviewedDiscoveryDisposition.from_dict, disposition.to_dict()),
        (DiscoveryDebtClaim.from_dict, claim.to_dict()),
        (DiscoveryClaimBalance.from_dict, balance.to_dict()),
        (DiscoveryClaimState.from_dict, state.to_dict()),
        (CompositionClassAllocation.from_dict, allocation.to_dict()),
        (ComposedEpochProjection.from_dict, projection.to_dict()),
    )

    class DictSubclass(dict[str, object]):
        pass

    for parser, row in parsers_and_rows:
        assert parser(row).to_dict() == row  # type: ignore[union-attr]
        missing = dict(row)
        missing.pop(next(iter(missing)))
        with pytest.raises(IncentiveCompositionError, match="fields mismatch"):
            parser(missing)
        with pytest.raises(IncentiveCompositionError, match="fields mismatch"):
            parser({**row, "extra": 1})
        with pytest.raises(IncentiveCompositionError, match="JSON object"):
            parser(DictSubclass(row))

    malformed_projection = projection.to_dict()
    malformed_projection["weights"] = tuple(malformed_projection["weights"])  # type: ignore[arg-type]
    with pytest.raises(IncentiveCompositionError, match="must be an array"):
        ComposedEpochProjection.from_dict(malformed_projection)


def test_review_paths_are_mutually_exclusive_and_promotion_issues_no_bounty() -> None:
    policy = _composition_policy()
    target = _d("promoted-target")
    promoted = _disposition(
        policy,
        label="promotion",
        decision=DISCOVERY_REGISTERED_PROMOTION,
        requested_epochs=0,
        promoted_target_digest=target,
    )
    assert promoted.promoted_target_digest == target
    assert issue_discovery_claim(policy, promoted) is None

    with pytest.raises(IncentiveCompositionError, match="cannot request"):
        _disposition(
            policy,
            decision=DISCOVERY_REGISTERED_PROMOTION,
            requested_epochs=1,
            promoted_target_digest=target,
        )
    with pytest.raises(IncentiveCompositionError, match="promoted_target_digest"):
        _disposition(
            policy,
            decision=DISCOVERY_REGISTERED_PROMOTION,
            requested_epochs=0,
        )
    with pytest.raises(IncentiveCompositionError, match="cannot name"):
        _disposition(
            policy,
            decision=DISCOVERY_BOUNTY_ONLY,
            requested_epochs=1,
            promoted_target_digest=target,
        )
    with pytest.raises(IncentiveCompositionError, match="unsupported"):
        _disposition(policy, decision="both", requested_epochs=1)


def test_bounty_is_unique_review_bound_cap_one_principal_and_has_no_crown_terms() -> None:
    policy = _composition_policy()
    disposition = _disposition(policy, label="identity", requested_epochs=999)
    replayed_disposition = _disposition(
        policy, label="identity", requested_epochs=999
    )
    claim = issue_discovery_claim(policy, disposition)
    replayed_claim = issue_discovery_claim(policy, replayed_disposition)
    assert claim is not None and replayed_claim is not None
    assert disposition.digest == replayed_disposition.digest
    assert claim.digest == replayed_claim.digest
    assert claim.disposition_digest == disposition.digest
    assert claim.requested_principal_epochs == 999
    assert claim.capped_principal_epochs == 1
    assert claim.principal_units == 50_000

    changed_review = _disposition(
        policy, label="identity-changed", requested_epochs=999
    )
    changed_claim = issue_discovery_claim(policy, changed_review)
    assert changed_claim is not None
    assert changed_review.digest != disposition.digest
    assert changed_claim.digest != claim.digest

    forbidden_fragments = (
        "family",
        "prior_crown",
        "time_multiplier",
        "log_unit",
        "renew",
        "clock_reset",
    )
    keys = set(disposition.to_dict()) | set(claim.to_dict())
    assert not any(fragment in key for key in keys for fragment in forbidden_fragments)
    assert claim.expires_block == claim.awarded_block + 648_000


def test_bounty_lifetime_is_anchored_to_retained_win_not_later_review() -> None:
    policy = _composition_policy(discovery_lifetime_blocks=10)
    disposition = _disposition(
        policy,
        label="late-valid-review",
        win_block=100,
        block=109,
    )
    claim = issue_discovery_claim(policy, disposition)
    assert claim is not None
    assert claim.awarded_block == 100
    assert claim.expires_block == 110

    with pytest.raises(IncentiveCompositionError, match="at or after.*expiry"):
        issue_discovery_claim(
            policy,
            _disposition(
                policy,
                label="expired-review",
                win_block=100,
                block=110,
            ),
        )

    promoted = _disposition(
        policy,
        label="late-promotion",
        win_block=100,
        block=1_000,
        decision=DISCOVERY_REGISTERED_PROMOTION,
        requested_epochs=0,
        promoted_target_digest=_d("late-promotion-target"),
    )
    assert issue_discovery_claim(policy, promoted) is None

    with pytest.raises(IncentiveCompositionError, match="predates"):
        _disposition(
            policy,
            label="pre-win-review",
            win_block=100,
            block=99,
        )


def test_composition_policy_is_exactly_bound_to_core_digest_reserve_and_epoch() -> None:
    core = _innovation_policy()
    policy = _composition_policy(core)
    policy.validate_innovation_policy(core)

    other_core = _innovation_policy(reserve_ppm=99_999)
    with pytest.raises(IncentiveCompositionError, match="differs"):
        policy.validate_innovation_policy(other_core)
    with pytest.raises(IncentiveCompositionError, match="differs"):
        replace(policy, innovation_policy_digest=_d("other-policy")).validate_innovation_policy(
            core
        )
    with pytest.raises(IncentiveCompositionError, match="differs"):
        replace(policy, reserve_ppm=99_999).validate_innovation_policy(core)
    with pytest.raises(IncentiveCompositionError, match="differs"):
        replace(policy, epoch_blocks=7_201).validate_innovation_policy(core)
    with pytest.raises(IncentiveCompositionError, match="discovery_cap_units"):
        replace(policy, discovery_cap_units=900_001)


def test_discovery_lifecycle_is_finite_nonrenewing_and_departure_forfeits_remainder() -> None:
    policy = _composition_policy()
    state = _discovery_state(
        policy, label="lifecycle", block=100, requested_epochs=8
    )
    claim = state.claim
    original_expiry = claim.expires_block
    assert original_expiry == 648_100

    partially_paid = pay_discovery_balance(
        claim, state.balance, 12_345, at_block=100
    )
    assert partially_paid.paid_units == 12_345
    assert partially_paid.remaining_units == 37_655
    assert expire_discovery_balance(
        claim, partially_paid, at_block=original_expiry - 1
    ) is partially_paid
    with pytest.raises(IncentiveCompositionError, match="outside"):
        pay_discovery_balance(
            claim, partially_paid, 1, at_block=original_expiry
        )

    expired = expire_discovery_balance(
        claim, partially_paid, at_block=original_expiry
    )
    assert expired.status == "expired"
    assert expired.terminal_block == original_expiry
    assert expired.terminal_reason == "claim_lifetime_expired"
    assert expired.paid_units == 12_345
    assert expired.forfeited_units == 37_655
    assert expired.paid_units + expired.forfeited_units == claim.principal_units
    assert claim.expires_block == original_expiry

    departed_state = _discovery_state(policy, label="departed", block=100)
    departed_partial = pay_discovery_balance(
        departed_state.claim, departed_state.balance, 10_000, at_block=120
    )
    departed = cancel_discovery_balance(
        departed_state.claim,
        departed_partial,
        at_block=121,
        reason="hotkey_departed",
    )
    assert departed.status == "cancelled"
    assert departed.terminal_reason == "hotkey_departed"
    assert departed.paid_units == 10_000
    assert departed.forfeited_units == 40_000
    assert departed.remaining_units == 0
    with pytest.raises(IncentiveCompositionError, match="not open"):
        cancel_discovery_balance(
            departed_state.claim,
            departed,
            at_block=122,
            reason="hotkey_departed",
        )


def test_projection_allocates_discovery_then_core_in_separate_pro_rata_classes() -> None:
    core = _innovation_policy()
    policy = _composition_policy(core)
    discovery_states = tuple(
        _discovery_state(policy, label=f"d{index}", hotkey=f"dminer{index}")
        for index in range(3)
    )
    innovation_states = tuple(
        _innovation_state(core, label=f"c{index}", hotkey=f"cminer{index}")
        for index in range(3)
    )
    projection = _projection(core, policy, innovation_states, discovery_states)

    assert projection.discovery_total_remaining_units == 150_000
    assert projection.discovery_capacity_units == 50_000
    assert projection.discovery_payout_units == 50_000
    assert projection.innovation_total_remaining_units == 2_700_000
    assert projection.innovation_capacity_units == 850_000
    assert projection.innovation_payout_units == 850_000
    assert projection.reserve_units == 100_000

    discovery_by_digest = {
        row.claim_digest: row.units for row in projection.discovery_allocations
    }
    discovery_digests = sorted(row.claim.digest for row in discovery_states)
    assert [discovery_by_digest[digest] for digest in discovery_digests] == [
        16_667,
        16_667,
        16_666,
    ]
    innovation_by_digest = {
        row.claim_digest: row.units for row in projection.innovation_allocations
    }
    innovation_digests = sorted(row.claim.digest for row in innovation_states)
    assert [innovation_by_digest[digest] for digest in innovation_digests] == [
        283_334,
        283_333,
        283_333,
    ]
    assert tuple(row.claim_digest for row in projection.discovery_allocations) == tuple(
        discovery_digests
    )
    assert tuple(row.claim_digest for row in projection.innovation_allocations) == tuple(
        innovation_digests
    )


def test_projection_is_input_order_invariant_and_roundtrips_to_one_golden_digest() -> None:
    core = _innovation_policy()
    policy = _composition_policy(core)
    discovery_states = (
        _discovery_state(policy, label="order-a", hotkey="same-miner"),
        _discovery_state(policy, label="order-b", hotkey="discovery-b"),
    )
    innovation_states = (
        _innovation_state(core, label="order-a", hotkey="same-miner"),
        _innovation_state(core, label="order-b", hotkey="innovation-b"),
    )
    forward = _projection(core, policy, innovation_states, discovery_states)
    reversed_inputs = _projection(
        core,
        policy,
        tuple(reversed(innovation_states)),
        tuple(reversed(discovery_states)),
    )
    assert forward.to_dict() == reversed_inputs.to_dict()
    assert forward.digest == reversed_inputs.digest
    assert ComposedEpochProjection.from_dict(forward.to_dict()) == forward
    assert forward.digest == "03931bbea5d60050ff34f6bf6be6dcc3689c5d04960b06e79aa68d1ddb07b29f"


def test_weights_aggregate_both_classes_by_hotkey_and_conserve_reserve() -> None:
    core = _innovation_policy()
    policy = _composition_policy(core)
    discovery = _discovery_state(policy, label="aggregate", hotkey="same-miner")
    innovation = _innovation_state(core, label="aggregate", hotkey="same-miner")
    projection = _projection(core, policy, (innovation,), (discovery,))

    assert projection.discovery_payout_units == 50_000
    assert projection.innovation_payout_units == 850_000
    assert projection.weights_by_hotkey == {
        "reserve": 100_000,
        "same-miner": 900_000,
    }
    assert sum(projection.weights_by_hotkey.values()) == PPM
    assert projection.reserve_units >= projection.reserve_floor_units == 100_000
    assert projection.payout_units + projection.reserve_units == PPM


def test_unused_class_capacity_flows_to_reserve_without_inventing_debt() -> None:
    core = _innovation_policy()
    policy = _composition_policy(core)
    discovery = _discovery_state(policy, label="small-d", hotkey="dminer")
    innovation = _innovation_state(core, label="small-c", hotkey="cminer")
    discovery_balance = pay_discovery_balance(
        discovery.claim, discovery.balance, 40_000, at_block=0
    )
    innovation_balance = pay_claim_balance(
        innovation.claim,
        innovation.balance,
        innovation.claim.principal_units - 20_000,
        at_block=0,
    )
    discovery = DiscoveryClaimState(discovery.claim, discovery_balance)
    innovation = DebtClaimState(innovation.claim, innovation_balance)
    projection = _projection(core, policy, (innovation,), (discovery,))

    assert projection.discovery_payout_units == 10_000
    assert projection.innovation_capacity_units == 890_000
    assert projection.innovation_payout_units == 20_000
    assert projection.reserve_units == 970_000
    assert projection.weights_by_hotkey == {
        "cminer": 20_000,
        "dminer": 10_000,
        "reserve": 970_000,
    }


@pytest.mark.parametrize("mutate_class", ["discovery", "innovation"])
def test_application_rejects_any_balance_mutation_after_projection(
    mutate_class: str,
) -> None:
    core = _innovation_policy()
    policy = _composition_policy(core)
    discovery = _discovery_state(policy, label="mutation")
    innovation = _innovation_state(core, label="mutation")
    projection = _projection(core, policy, (innovation,), (discovery,))

    if mutate_class == "discovery":
        changed_discovery = DiscoveryClaimState(
            discovery.claim,
            pay_discovery_balance(
                discovery.claim, discovery.balance, 1, at_block=0
            ),
        )
        innovation_inputs = (innovation,)
        discovery_inputs = (changed_discovery,)
    else:
        changed_innovation = DebtClaimState(
            innovation.claim,
            pay_claim_balance(
                innovation.claim, innovation.balance, 1, at_block=0
            ),
        )
        innovation_inputs = (changed_innovation,)
        discovery_inputs = (discovery,)

    with pytest.raises(IncentiveCompositionError, match="balances changed"):
        apply_composed_epoch(
            innovation_inputs, discovery_inputs, projection
        )


def test_application_is_canonical_consumes_each_class_and_is_apply_once() -> None:
    core = _innovation_policy()
    policy = _composition_policy(core)
    discovery_states = (
        _discovery_state(policy, label="apply-b", hotkey="d-b"),
        _discovery_state(policy, label="apply-a", hotkey="d-a"),
    )
    innovation_states = (
        _innovation_state(core, label="apply-b", hotkey="c-b"),
        _innovation_state(core, label="apply-a", hotkey="c-a"),
    )
    projection = _projection(core, policy, innovation_states, discovery_states)
    updated_innovation, updated_discovery = apply_composed_epoch(
        tuple(reversed(innovation_states)),
        tuple(reversed(discovery_states)),
        projection,
    )

    assert tuple(row.claim.digest for row in updated_innovation) == tuple(
        sorted(row.claim.digest for row in innovation_states)
    )
    assert tuple(row.claim.digest for row in updated_discovery) == tuple(
        sorted(row.claim.digest for row in discovery_states)
    )
    innovation_paid = {
        row.claim.digest: row.balance.paid_units for row in updated_innovation
    }
    discovery_paid = {
        row.claim.digest: row.balance.paid_units for row in updated_discovery
    }
    assert innovation_paid == {
        row.claim_digest: row.units for row in projection.innovation_allocations
    }
    assert discovery_paid == {
        row.claim_digest: row.units for row in projection.discovery_allocations
    }
    assert sum(innovation_paid.values()) == projection.innovation_payout_units
    assert sum(discovery_paid.values()) == projection.discovery_payout_units

    with pytest.raises(IncentiveCompositionError, match="balances changed"):
        apply_composed_epoch(
            updated_innovation, updated_discovery, projection
        )


def test_projection_rejects_duplicate_claims_reserve_owners_and_out_of_window_open_debt() -> None:
    core = _innovation_policy()
    policy = _composition_policy(core)
    discovery = _discovery_state(policy, label="guards")
    innovation = _innovation_state(core, label="guards")
    with pytest.raises(IncentiveCompositionError, match="duplicate"):
        _projection(core, policy, (innovation,), (discovery, discovery))
    with pytest.raises(IncentiveCompositionError, match="duplicate"):
        _projection(core, policy, (innovation, innovation), (discovery,))

    reserve_discovery = _discovery_state(
        policy, label="reserve-d", hotkey="reserve"
    )
    with pytest.raises(IncentiveCompositionError, match="reserve hotkey"):
        _projection(core, policy, (), (reserve_discovery,))
    reserve_innovation = _innovation_state(
        core, label="reserve-c", hotkey="reserve"
    )
    with pytest.raises(IncentiveCompositionError, match="reserve hotkey"):
        _projection(core, policy, (reserve_innovation,), ())

    future_discovery = _discovery_state(policy, label="future-d", block=10)
    with pytest.raises(IncentiveCompositionError, match="payout window"):
        _projection(core, policy, (), (future_discovery,), block=9)
    future_innovation = _innovation_state(core, label="future-c", block=10)
    with pytest.raises(IncentiveCompositionError, match="payout window"):
        _projection(core, policy, (future_innovation,), (), block=9)

    with pytest.raises(IncentiveCompositionError, match="payout window"):
        _projection(core, policy, (), (discovery,), block=discovery.claim.expires_block)
    with pytest.raises(IncentiveCompositionError, match="payout window"):
        _projection(core, policy, (innovation,), (), block=innovation.claim.expires_block)
