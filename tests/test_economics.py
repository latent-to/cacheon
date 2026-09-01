from __future__ import annotations

from dataclasses import replace

import pytest

from cacheon.economics import (
    AcceptedRewardClaim,
    ArenaRewardAuthority,
    CREDIT_SCALE,
    DiscoveryBountyClaim,
    EconomicsError,
    EmissionsPolicyManifest,
    GlobalRewardProjectionContext,
    MetagraphMember,
    StandingRewardClaim,
    WEIGHT_PPM,
    decay_factor,
    improvement_units,
    project_global_rewards,
    time_multiplier,
)
from cacheon.stack_identity import canonical_digest
from cacheon.stack_manifest import EvaluationStackManifest, ProposalContributionRef
from cacheon.target_catalog import (
    CorrectnessContractRef,
    FEATURE_ENTRY,
    TargetCatalog,
    TargetContractRef,
    TargetKind,
    TargetSpec,
    ToleranceContractRef,
)


from decimal import ROUND_FLOOR, Decimal


def _d(char: str) -> str:
    return char * 64


def _slot(target_id: str) -> TargetSpec:
    return TargetSpec(
        target_id=target_id,
        kind=TargetKind.SLOT,
        members=(target_id,),
        contract_ref=TargetContractRef(
            schema_version=1,
            slot_id=target_id,
            kind="op",
            entry="entry",
            prepare=None,
            graph_dynamic_inputs=("x",),
            input_abi_id=f"{target_id}.input.v1",
            output_abi_id=f"{target_id}.output.v1",
            reference_id=f"{target_id}.reference.v1",
            verification_profile_id=f"{target_id}.verify.v1",
            binding_family_id=f"{target_id}.binding.v1",
            correctness=CorrectnessContractRef(),
            tolerances=(ToleranceContractRef("float32", "0.001", "0.001"),),
        ),
    )


def _catalog() -> TargetCatalog:
    return TargetCatalog(
        (
            _slot("slot.a"),
            _slot("slot.b"),
            TargetSpec(
                target_id="atomic.ab",
                kind=TargetKind.ATOMIC,
                members=("slot.a", "slot.b"),
                displaces=frozenset({"slot.a", "slot.b"}),
                allowed_features=frozenset({FEATURE_ENTRY}),
                atomic_semantics_id="atomic.ab.v1",
            ),
        )
    )


def _contribution(catalog: TargetCatalog, target: str, char: str):
    return ProposalContributionRef(
        target_id=target,
        target_spec_digest=catalog.target_spec_digest(target),
        artifact_digest=_d(char),
        selected_payload_digest=canonical_digest("test.selected", {"target": target, "char": char}),
        attribution_digest=canonical_digest("test.attribution", {"char": char}),
    )


def _stack(catalog: TargetCatalog, targets=("slot.a", "slot.b"), arena="c"):
    entries = {
        target: _contribution(catalog, target, "123"[index])
        for index, target in enumerate(targets)
    }
    return EvaluationStackManifest(
        runtime_digest=_d("a"),
        base_engine_digest=_d("b"),
        arena_digest=_d(arena),
        catalog_snapshot=catalog.snapshot(),
        catalog_digest=catalog.digest,
        entries=entries,
    )


def _global_context(block: int = 200, members=None):
    return GlobalRewardProjectionContext(
        chain_scope_digest=_d("d"),
        validator_hotkey="validator",
        current_block=block,
        current_block_hash="0x" + "e" * 64,
        metagraph_members=tuple(
            members
            or (
                MetagraphMember(2, "bob"),
                MetagraphMember(0, "validator"),
                MetagraphMember(4, "dave"),
                MetagraphMember(1, "alice"),
                MetagraphMember(3, "carol"),
            )
        ),
    )


def _project(policy, catalog, stack, context, standing, discovery=(), generation=7, earning=None):
    standing_t = tuple(standing)
    if earning is None:
        earning = tuple(AcceptedRewardClaim.from_standing(row) for row in standing_t)
    return project_global_rewards(
        policy,
        context,
        (ArenaRewardAuthority(catalog, stack, generation, standing_t),),
        earning,
        discovery,
    )


def _policy(**kwargs):
    values = dict(
        half_life_blocks=100,
        discovery_lifetime_blocks=50,
        discovery_pool_ppm=200_000,
        time_multiplier_scale_blocks=1_800,
    )
    values.update(kwargs)
    return EmissionsPolicyManifest(**values)


def _floor(value: Decimal) -> int:
    return int(value.to_integral_value(rounding=ROUND_FLOOR))


def _allocate(credits: dict[str, int], pool: int) -> dict[str, int]:
    positive = {hotkey: credit for hotkey, credit in credits.items() if credit > 0}
    total = sum(positive.values())
    result: dict[str, int] = {}
    remainders = []
    for hotkey in sorted(positive):
        quotient, remainder = divmod(positive[hotkey] * pool, total)
        result[hotkey] = quotient
        remainders.append((remainder, hotkey))
    missing = pool - sum(result.values())
    for _remainder, hotkey in sorted(remainders, key=lambda row: (-row[0], row[1]))[
        :missing
    ]:
        result[hotkey] += 1
    return result


def _accepted(stack, target, hotkey, speedup_ppm, crowned_block=100, evidence="4", predecessor_block=None):
    return AcceptedRewardClaim.from_standing(
        _claim(stack, target, hotkey, speedup_ppm, crowned_block, evidence),
        predecessor_block,
    )


def _claim(stack, target, hotkey, speedup_ppm, crowned_block=100, evidence="4"):
    ref = stack.entries[target]
    return StandingRewardClaim(
        stack.arena_digest,
        target,
        ref.target_spec_digest,
        ref.digest,
        hotkey,
        speedup_ppm,
        crowned_block,
        _d(evidence),
    )


def _expected_standing_weights(policy, context, claims, pool=WEIGHT_PPM):
    credits: dict[str, int] = {}
    for row in claims:
        claim = (
            row
            if type(row) is AcceptedRewardClaim
            else AcceptedRewardClaim.from_standing(row)
        )
        if policy.excludes_accepted(claim):
            continue
        credits[claim.hotkey] = credits.get(claim.hotkey, 0) + claim.credit_at(
            context.current_block, policy
        )
    return _allocate(credits, pool)


def _discovery(hotkey="carol", units=1, block=180, proposal="5", evidence="6"):
    return DiscoveryBountyClaim(_d(proposal), _d(evidence), hotkey, units, block)


def test_policy_is_exact_versioned_content_addressed_data() -> None:
    policy = _policy()
    assert EmissionsPolicyManifest.from_dict(policy.to_dict()) == policy
    assert policy.digest == EmissionsPolicyManifest.from_dict(policy.to_dict()).digest
    assert replace(policy, half_life_blocks=101).digest != policy.digest
    for value in (
        {**policy.to_dict(), "unknown": 1},
        {**policy.to_dict(), "half_life_blocks": True},
        {**policy.to_dict(), "policy_version": "future"},
    ):
        with pytest.raises(EconomicsError):
            EmissionsPolicyManifest.from_dict(value)
    with pytest.raises(EconomicsError, match="standing reward"):
        _policy(discovery_pool_ppm=WEIGHT_PPM)
    assert replace(policy, time_multiplier_scale_blocks=1_801).digest != policy.digest
    assert replace(policy, excluded_hotkeys=("alice",)).digest != policy.digest
    assert replace(policy, excluded_claim_digests=(_d("1"),)).digest != policy.digest
    assert policy.policy_version == "cacheon.emissions.v1.3"


def test_log_relative_units_are_path_independent() -> None:
    compounded = improvement_units(1_210_000)
    sequential = improvement_units(1_100_000) + improvement_units(1_100_000)
    assert abs(compounded - sequential) * Decimal(CREDIT_SCALE) < Decimal(1)
    assert _floor(compounded * Decimal(CREDIT_SCALE)) == _floor(
        sequential * Decimal(CREDIT_SCALE)
    )


def test_first_crown_time_multiplier_is_one() -> None:
    catalog = _catalog()
    stack = _stack(catalog, ("slot.a",))
    claim = _accepted(stack, "slot.a", "alice", 1_100_000, predecessor_block=100)
    policy = _policy()
    expected = _floor(improvement_units(1_100_000) * Decimal(CREDIT_SCALE))
    assert claim.elapsed_blocks() == 0
    assert time_multiplier(0, policy.time_multiplier_scale_blocks) == Decimal(1)
    assert claim.credit_at(100, policy) == expected


def test_stall_multiplier_follows_square_root() -> None:
    scale = 1_800
    assert time_multiplier(0, scale) == Decimal(1)
    assert time_multiplier(scale, scale) == Decimal(2)
    assert time_multiplier(4 * scale, scale) == Decimal(3)
    catalog = _catalog()
    stack = _stack(catalog, ("slot.a",))
    claim = _accepted(
        stack, "slot.a", "alice", 1_100_000, crowned_block=1_900, predecessor_block=100
    )
    policy = _policy()
    expected = _floor(
        improvement_units(1_100_000) * Decimal(2) * Decimal(CREDIT_SCALE)
    )
    assert claim.credit_at(1_900, policy) == expected


def test_exponential_decay_uses_exact_half_life() -> None:
    catalog = _catalog()
    stack = _stack(catalog, ("slot.a",))
    claim = _accepted(stack, "slot.a", "alice", 1_100_001)
    policy = _policy()
    start = claim.credit_at(100, policy)
    half = claim.credit_at(200, policy)
    assert start == _floor(improvement_units(1_100_001) * Decimal(CREDIT_SCALE))
    assert half == _floor(
        improvement_units(1_100_001) * decay_factor(100, 100) * Decimal(CREDIT_SCALE)
    )
    assert decay_factor(100_800, 100_800) == Decimal("0.5")
    with pytest.raises(EconomicsError, match="newer"):
        claim.credit_at(99, policy)


def test_standing_projection_is_relative_grouped_and_exactly_normalized() -> None:
    catalog = _catalog()
    stack = _stack(catalog)
    policy = _policy()
    context = _global_context()
    standing = (
        _claim(stack, "slot.a", "alice", 1_100_000),
        _claim(stack, "slot.b", "bob", 1_200_000, evidence="7"),
    )
    result = _project(policy, catalog, stack, context, standing)
    assert result.weights_by_hotkey == _expected_standing_weights(policy, context, standing)
    assert result.weights_by_hotkey["alice"] < result.weights_by_hotkey["bob"]
    assert sum(result.weights_by_hotkey.values()) == WEIGHT_PPM
    assert len(result.standing) == 2
    assert result.discovery == ()


def test_multiple_families_for_one_hotkey_are_summed_before_normalization() -> None:
    catalog = _catalog()
    stack = _stack(catalog)
    result = _project(
        _policy(),
        catalog,
        stack,
        _global_context(),
        (
            _claim(stack, "slot.a", "alice", 1_100_000),
            _claim(stack, "slot.b", "alice", 1_200_000, evidence="7"),
        ),
    )
    assert result.weights_by_hotkey == {"alice": WEIGHT_PPM}


def test_atomic_target_is_one_family_and_suppresses_singletons() -> None:
    catalog = _catalog()
    atomic = _stack(catalog, ("atomic.ab",))
    claim = _claim(atomic, "atomic.ab", "alice", 1_250_000)
    result = _project(_policy(), catalog, atomic, _global_context(), (claim,))
    assert len(result.standing) == 1
    assert result.standing[0].target_id == "atomic.ab"
    assert result.weights_by_hotkey == {"alice": WEIGHT_PPM}

    overlap = _stack(catalog, ("atomic.ab", "slot.a"))
    with pytest.raises(EconomicsError, match="overlap"):
        _project(
            _policy(),
            catalog,
            overlap,
            _global_context(),
            (
                _claim(overlap, "atomic.ab", "alice", 1_250_000),
                _claim(overlap, "slot.a", "bob", 1_100_000, evidence="7"),
            ),
        )


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "contribution", "spec"])
def test_projection_holds_if_any_active_family_is_not_exact(mutation: str) -> None:
    catalog = _catalog()
    stack = _stack(catalog)
    left = _claim(stack, "slot.a", "alice", 1_100_000)
    right = _claim(stack, "slot.b", "bob", 1_200_000, evidence="7")
    claims = (left,) if mutation == "missing" else (left, right)
    if mutation == "duplicate":
        claims = (left, left)
    elif mutation == "contribution":
        claims = (left, replace(right, contribution_digest=_d("8")))
    elif mutation == "spec":
        claims = (left, replace(right, target_spec_digest=_d("8")))
    with pytest.raises(EconomicsError):
        _project(_policy(), catalog, stack, _global_context(), claims)


def test_missing_live_hotkey_burns_its_share_to_the_validator() -> None:
    catalog = _catalog()
    stack = _stack(catalog, ("slot.a",))
    standing = _claim(stack, "slot.a", "ghost", 1_100_000)
    result = _project(_policy(), catalog, stack, _global_context(), (standing,))
    assert result.standing[0].hotkey == "ghost"
    assert result.weights_by_hotkey == {"validator": WEIGHT_PPM}

    standing = replace(standing, hotkey="alice")
    expired = _discovery(hotkey="ghost", block=100)
    result = _project(
        _policy(), catalog, stack, _global_context(), (standing,), (expired,)
    )
    assert result.expired_discovery_claims == (expired.digest,)
    assert result.weights_by_hotkey == {"alice": WEIGHT_PPM}


def test_present_families_keep_their_ppm_when_an_absent_share_is_burned() -> None:
    catalog = _catalog()
    stack = _stack(catalog)
    policy = _policy()
    context = _global_context()
    alice = _claim(stack, "slot.a", "alice", 1_100_000)
    bob = _claim(stack, "slot.b", "bob", 1_200_000, evidence="7")
    present = _project(policy, catalog, stack, context, (alice, bob))
    burned = _project(
        policy,
        catalog,
        stack,
        context,
        (alice, _claim(stack, "slot.b", "ghost", 1_200_000, evidence="7")),
    )
    expected = _expected_standing_weights(policy, context, (alice, bob))
    assert present.weights_by_hotkey == expected
    assert burned.weights_by_hotkey == {
        "alice": expected["alice"],
        "validator": expected["bob"],
    }
    assert burned.standing[1].hotkey == "ghost"


def test_missing_live_discovery_hotkey_burns_only_its_bounty_share() -> None:
    catalog = _catalog()
    stack = _stack(catalog)
    policy = _policy()
    context = _global_context()
    standing = (
        _claim(stack, "slot.a", "alice", 1_100_000),
        _claim(stack, "slot.b", "bob", 1_200_000, evidence="7"),
    )
    discoveries = (
        _discovery("carol", 1, proposal="5", evidence="6"),
        _discovery("ghost", 3, proposal="8", evidence="9"),
    )
    result = _project(policy, catalog, stack, context, standing, discoveries)
    standing_weights = _expected_standing_weights(
        policy, context, standing, WEIGHT_PPM - policy.discovery_pool_ppm
    )
    assert result.weights_by_hotkey == {
        **standing_weights,
        "carol": 50_000,
        "validator": 150_000,
    }
    assert result.discovery[1].hotkey == "ghost"


def test_live_discovery_claims_share_only_the_bounded_pool() -> None:
    catalog = _catalog()
    stack = _stack(catalog)
    policy = _policy()
    context = _global_context()
    standing = (
        _claim(stack, "slot.a", "alice", 1_100_000),
        _claim(stack, "slot.b", "bob", 1_200_000, evidence="7"),
    )
    discoveries = (
        _discovery("carol", 1, proposal="5", evidence="6"),
        _discovery("dave", 3, proposal="8", evidence="9"),
    )
    result = _project(policy, catalog, stack, context, standing, discoveries)
    standing_weights = _expected_standing_weights(
        policy, context, standing, WEIGHT_PPM - policy.discovery_pool_ppm
    )
    assert result.weights_by_hotkey == {
        **standing_weights,
        "carol": 50_000,
        "dave": 150_000,
    }


def test_discovery_expiry_is_exact_and_claims_cannot_be_renewed() -> None:
    catalog = _catalog()
    stack = _stack(catalog, ("slot.a",))
    standing = (_claim(stack, "slot.a", "alice", 1_100_000),)
    claim = _discovery(block=150)
    assert _project(
        _policy(), catalog, stack, _global_context(199), standing, (claim,)
    ).discovery
    assert _project(
        _policy(), catalog, stack, _global_context(200), standing, (claim,)
    ).discovery == ()

    renewed = replace(claim, retained_evidence_digest=_d("7"))
    with pytest.raises(EconomicsError, match="renewed"):
        _project(
            _policy(), catalog, stack, _global_context(199), standing, (claim, renewed)
        )
    reused_evidence = replace(claim, proposal_digest=_d("8"))
    with pytest.raises(EconomicsError, match="duplicated"):
        _project(
            _policy(), catalog, stack, _global_context(199), standing, (claim, reused_evidence)
        )


def test_disabled_discovery_pool_fails_on_a_live_claim() -> None:
    catalog = _catalog()
    stack = _stack(catalog, ("slot.a",))
    with pytest.raises(EconomicsError, match="disabled"):
        _project(
            _policy(discovery_pool_ppm=0),
            catalog,
            stack,
            _global_context(),
            (_claim(stack, "slot.a", "alice", 1_100_000),),
            (_discovery(),),
        )


def test_projection_identity_is_order_stable_and_binds_every_authority() -> None:
    catalog = _catalog()
    stack = _stack(catalog)
    claims = (
        _claim(stack, "slot.a", "alice", 1_100_000),
        _claim(stack, "slot.b", "bob", 1_200_000, evidence="7"),
    )
    context = _global_context()
    left = _project(_policy(), catalog, stack, context, claims)
    reordered = GlobalRewardProjectionContext(
        context.chain_scope_digest,
        context.validator_hotkey,
        context.current_block,
        context.current_block_hash,
        tuple(reversed(context.metagraph_members)),
    )
    right = _project(_policy(), catalog, stack, reordered, reversed(claims))
    assert left.digest == right.digest
    changed = _project(_policy(), catalog, stack, context, claims, generation=8)
    assert changed.digest != left.digest
    assert context.metagraph_digest in context.to_dict().values()


def test_empty_stack_zero_credit_and_future_bounty_fail_closed() -> None:
    catalog = _catalog()
    empty = _stack(catalog, ())
    with pytest.raises(EconomicsError, match="active crown"):
        _project(_policy(), catalog, empty, _global_context(), ())

    stack = _stack(catalog, ("slot.a",))
    ancient = _claim(stack, "slot.a", "alice", 1_000_001, crowned_block=0)
    with pytest.raises(EconomicsError, match="decayed"):
        _project(_policy(), catalog, stack, _global_context(1_000_000), (ancient,))
    future = _discovery(block=201)
    with pytest.raises(EconomicsError, match="newer"):
        _project(
            _policy(),
            catalog,
            stack,
            _global_context(200),
            (_claim(stack, "slot.a", "alice", 1_100_000),),
            (future,),
        )


def test_claim_round_trip_and_zero_evidence_are_strict() -> None:
    catalog = _catalog()
    stack = _stack(catalog, ("slot.a",))
    standing = _claim(stack, "slot.a", "alice", 1_100_000)
    discovery = _discovery()
    assert StandingRewardClaim.from_dict(standing.to_dict()) == standing
    assert DiscoveryBountyClaim.from_dict(discovery.to_dict()) == discovery
    accepted = AcceptedRewardClaim.from_standing(standing)
    assert AcceptedRewardClaim.from_dict(accepted.to_dict()) == accepted
    with pytest.raises(EconomicsError, match="all-zero"):
        replace(standing, retained_evidence_digest="0" * 64)
    with pytest.raises(EconomicsError, match="fields"):
        StandingRewardClaim.from_dict({**standing.to_dict(), "extra": 1})
    with pytest.raises(EconomicsError, match="fields"):
        AcceptedRewardClaim.from_dict({**accepted.to_dict(), "extra": 1})


def _genesis(authorities):
    return tuple(
        AcceptedRewardClaim.from_standing(claim)
        for authority in authorities
        for claim in authority.standing_claims
    )


def test_global_projection_pools_families_before_one_normalization() -> None:
    catalog = _catalog()
    first = _stack(catalog, ("slot.a",), arena="c")
    second = _stack(catalog, ("slot.a", "slot.b"), arena="f")
    authorities = (
        ArenaRewardAuthority(
            catalog,
            first,
            3,
            (_claim(first, "slot.a", "alice", 1_100_000),),
        ),
        ArenaRewardAuthority(
            catalog,
            second,
            9,
            (
                _claim(second, "slot.a", "bob", 1_200_000, 200, evidence="7"),
                _claim(second, "slot.b", "carol", 1_200_000, evidence="8"),
            ),
        ),
    )
    policy = _policy()
    context = _global_context()
    earning = _genesis(authorities)
    result = project_global_rewards(
        policy, context, reversed(authorities), earning
    )
    expected = _expected_standing_weights(policy, context, earning)
    assert result.weights_by_hotkey == expected
    assert len(result.arena_authority_digests) == 2
    assert len({row.family_id for row in result.standing}) == 3


def test_any_invalid_arena_holds_the_complete_global_vector() -> None:
    catalog = _catalog()
    first = _stack(catalog, ("slot.a",), arena="c")
    second = _stack(catalog, ("slot.a", "slot.b"), arena="f")
    valid = ArenaRewardAuthority(
        catalog,
        first,
        3,
        (_claim(first, "slot.a", "alice", 1_100_000),),
    )
    incomplete = ArenaRewardAuthority(
        catalog,
        second,
        9,
        (_claim(second, "slot.a", "bob", 1_200_000),),
    )
    with pytest.raises(EconomicsError, match="every active target"):
        project_global_rewards(
            _policy(),
            _global_context(),
            (valid, incomplete),
            _genesis((valid, incomplete)),
        )


def test_same_target_in_two_arenas_has_distinct_reward_family_identity() -> None:
    catalog = _catalog()
    first = _stack(catalog, ("slot.a",), arena="c")
    second = _stack(catalog, ("slot.a",), arena="f")
    left = _claim(first, "slot.a", "alice", 1_100_000)
    right = _claim(second, "slot.a", "bob", 1_100_000, 200, evidence="7")
    assert left.family_id != right.family_id
    policy = _policy()
    context = _global_context()
    authorities = (
        ArenaRewardAuthority(catalog, first, 1, (left,)),
        ArenaRewardAuthority(catalog, second, 2, (right,)),
    )
    earning = _genesis(authorities)
    result = project_global_rewards(policy, context, authorities, earning)
    expected = _expected_standing_weights(policy, context, earning)
    assert result.weights_by_hotkey == expected
    assert expected["alice"] != 200_000
    assert expected["bob"] != 800_000


def test_historical_accepted_claims_keep_decaying_tails_without_champion_floor() -> None:
    catalog = _catalog()
    stack = _stack(catalog, ("slot.a",))
    incumbent = _claim(stack, "slot.a", "bob", 1_150_000, 200, evidence="7")
    prior = AcceptedRewardClaim(
        stack.arena_digest,
        "slot.a",
        incumbent.target_spec_digest,
        _d("9"),
        "alice",
        1_100_000,
        100,
        100,
        _d("4"),
    )
    current = AcceptedRewardClaim.from_standing(incumbent, predecessor_block=100)
    policy = _policy()
    context = _global_context()
    result = _project(
        policy,
        catalog,
        stack,
        context,
        (incumbent,),
        earning=(prior, current),
    )
    expected = _expected_standing_weights(policy, context, (prior, current))
    assert result.weights_by_hotkey == expected
    by_hotkey = {row.hotkey: row.credit for row in result.standing}
    total = sum(by_hotkey.values())
    assert by_hotkey["bob"] * 5 < total * 4
    assert len(result.standing) == 2


def test_arena_stall_clock_resets_from_the_previous_accepted_crown() -> None:
    catalog = _catalog()
    stack = _stack(catalog)
    first = _accepted(
        stack, "slot.a", "alice", 1_100_000, crowned_block=1_900, predecessor_block=1_900
    )
    second = _accepted(
        stack,
        "slot.b",
        "bob",
        1_100_000,
        crowned_block=1_900,
        evidence="7",
        predecessor_block=100,
    )
    policy = _policy()
    assert first.elapsed_blocks() == 0
    assert second.elapsed_blocks() == 1_800
    standing = (
        _claim(stack, "slot.a", "alice", 1_100_000, 1_900),
        _claim(stack, "slot.b", "bob", 1_100_000, 1_900, evidence="7"),
    )
    result = _project(
        policy,
        catalog,
        stack,
        _global_context(1_900),
        standing,
        earning=(first, second),
    )
    units = improvement_units(1_100_000)
    by_target = {row.target_id: row.credit for row in result.standing}
    assert by_target["slot.a"] == _floor(units * Decimal(CREDIT_SCALE))
    assert by_target["slot.b"] == _floor(units * Decimal(2) * Decimal(CREDIT_SCALE))
    assert sum(result.weights_by_hotkey.values()) == WEIGHT_PPM


def test_excluded_hotkey_is_omitted_and_remaining_credit_renormalizes() -> None:
    catalog = _catalog()
    stack = _stack(catalog)
    standing = (
        _claim(stack, "slot.a", "alice", 1_100_000),
        _claim(stack, "slot.b", "bob", 1_200_000, evidence="7"),
    )
    alice = AcceptedRewardClaim.from_standing(standing[0])
    bob = AcceptedRewardClaim.from_standing(standing[1])
    policy = _policy(excluded_hotkeys=("alice",))
    context = _global_context()
    result = _project(policy, catalog, stack, context, standing, earning=(alice, bob))
    assert result.weights_by_hotkey == {"bob": WEIGHT_PPM}
    assert {row.hotkey for row in result.standing} == {"bob"}
    assert sum(result.weights_by_hotkey.values()) == WEIGHT_PPM


def test_excluded_claim_digest_omits_one_crown_without_dropping_the_miner() -> None:
    catalog = _catalog()
    stack = _stack(catalog, ("slot.a",))
    incumbent = _claim(stack, "slot.a", "bob", 1_150_000, 200, evidence="7")
    prior = AcceptedRewardClaim(
        stack.arena_digest,
        "slot.a",
        incumbent.target_spec_digest,
        _d("9"),
        "alice",
        1_100_000,
        100,
        100,
        _d("4"),
    )
    current = AcceptedRewardClaim.from_standing(incumbent, predecessor_block=100)
    policy = _policy(excluded_claim_digests=(prior.digest,))
    context = _global_context()
    result = _project(
        policy,
        catalog,
        stack,
        context,
        (incumbent,),
        earning=(prior, current),
    )
    assert result.weights_by_hotkey == {"bob": WEIGHT_PPM}
    assert [row.claim_digest for row in result.standing] == [current.digest]


def test_unknown_excluded_claim_digest_fails_closed() -> None:
    catalog = _catalog()
    stack = _stack(catalog, ("slot.a",))
    standing = (_claim(stack, "slot.a", "alice", 1_100_000),)
    with pytest.raises(EconomicsError, match="excluded claim digest"):
        _project(
            _policy(excluded_claim_digests=(_d("f"),)),
            catalog,
            stack,
            _global_context(),
            standing,
        )


def test_excluding_every_earning_claim_fails_closed() -> None:
    catalog = _catalog()
    stack = _stack(catalog, ("slot.a",))
    standing = (_claim(stack, "slot.a", "alice", 1_100_000),)
    with pytest.raises(EconomicsError, match="been excluded"):
        _project(
            _policy(excluded_hotkeys=("alice",)),
            catalog,
            stack,
            _global_context(),
            standing,
        )


def test_excluded_discovery_hotkey_does_not_consume_the_discovery_pool() -> None:
    catalog = _catalog()
    stack = _stack(catalog)
    policy = _policy(excluded_hotkeys=("carol",))
    context = _global_context()
    standing = (
        _claim(stack, "slot.a", "alice", 1_100_000),
        _claim(stack, "slot.b", "bob", 1_200_000, evidence="7"),
    )
    discoveries = (
        _discovery("carol", 1, proposal="5", evidence="6"),
        _discovery("dave", 3, proposal="8", evidence="9"),
    )
    result = _project(policy, catalog, stack, context, standing, discoveries)
    standing_weights = _expected_standing_weights(
        policy, context, standing, WEIGHT_PPM - policy.discovery_pool_ppm
    )
    assert result.weights_by_hotkey == {
        **standing_weights,
        "dave": policy.discovery_pool_ppm,
    }
    assert [row.hotkey for row in result.discovery] == ["dave"]

