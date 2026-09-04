from __future__ import annotations

from dataclasses import replace

import pytest

from cacheon.economics import (
    ArenaRewardAuthority,
    DiscoveryBountyClaim,
    EconomicsError,
    EmissionsPolicyManifest,
    GlobalRewardProjectionContext,
    MetagraphMember,
    StandingRewardClaim,
    WEIGHT_PPM,
    project_global_rewards,
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


def _project(policy, catalog, stack, context, standing, discovery=(), generation=7):
    standing = tuple(standing)
    return project_global_rewards(
        policy,
        context,
        (ArenaRewardAuthority(catalog, stack, generation, tuple(standing)),),
        standing,
        discovery,
    )


def _policy(**kwargs):
    values = dict(
        half_life_blocks=100,
        discovery_lifetime_blocks=50,
        discovery_pool_ppm=200_000,
    )
    values.update(kwargs)
    return EmissionsPolicyManifest(**values)


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


def test_pass_credit_uses_log_speedup_stall_time_and_half_life() -> None:
    catalog = _catalog()
    stack = _stack(catalog, ("slot.a",))
    claim = _claim(stack, "slot.a", "alice", 1_100_001)
    passed = claim
    initial = passed.credit_at(100, _policy())
    assert passed.credit_at(200, _policy()) == initial // 2
    assert passed.credit_at(201, _policy()) < initial // 2
    assert claim.credit_at(100, _policy(), predecessor_block=0) > initial
    with pytest.raises(EconomicsError, match="credit block"):
        passed.credit_at(99, _policy())


def test_standing_projection_is_relative_grouped_and_exactly_normalized() -> None:
    catalog = _catalog()
    stack = _stack(catalog)
    result = _project(
        _policy(),
        catalog,
        stack,
        _global_context(),
        (
            _claim(stack, "slot.a", "alice", 1_100_000),
            _claim(stack, "slot.b", "bob", 1_200_000, evidence="7"),
        ),
    )
    assert result.weights_by_hotkey == {"alice": 343_297, "bob": 656_703}
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
    alice = _claim(stack, "slot.a", "alice", 1_100_000)
    present = _project(
        _policy(),
        catalog,
        stack,
        _global_context(),
        (alice, _claim(stack, "slot.b", "bob", 1_200_000, evidence="7")),
    )
    burned = _project(
        _policy(),
        catalog,
        stack,
        _global_context(),
        (alice, _claim(stack, "slot.b", "ghost", 1_200_000, evidence="7")),
    )
    assert present.weights_by_hotkey == {"alice": 343_297, "bob": 656_703}
    assert burned.weights_by_hotkey == {"alice": 343_297, "validator": 656_703}
    assert burned.standing[1].hotkey == "ghost"


def test_missing_live_discovery_hotkey_burns_only_its_bounty_share() -> None:
    catalog = _catalog()
    stack = _stack(catalog)
    standing = (
        _claim(stack, "slot.a", "alice", 1_100_000),
        _claim(stack, "slot.b", "bob", 1_200_000, evidence="7"),
    )
    discoveries = (
        _discovery("carol", 1, proposal="5", evidence="6"),
        _discovery("ghost", 3, proposal="8", evidence="9"),
    )
    result = _project(
        _policy(), catalog, stack, _global_context(), standing, discoveries
    )
    assert result.weights_by_hotkey == {
        "alice": 274_638,
        "bob": 525_362,
        "carol": 50_000,
        "validator": 150_000,
    }
    assert result.discovery[1].hotkey == "ghost"


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


def test_sub_ppm_credit_is_retained_but_not_published_as_zero() -> None:
    catalog = _catalog()
    stack = _stack(catalog)
    claims = (
        _claim(stack, "slot.a", "alice", 1_000_001, crowned_block=100),
        _claim(stack, "slot.b", "bob", 1_200_000, crowned_block=1_600, evidence="7"),
    )
    result = _project(_policy(), catalog, stack, _global_context(1_600), claims)
    assert len(result.standing) == 2
    assert result.weights_by_hotkey == {"bob": WEIGHT_PPM}


def test_claim_round_trip_and_zero_evidence_are_strict() -> None:
    catalog = _catalog()
    stack = _stack(catalog, ("slot.a",))
    standing = _claim(stack, "slot.a", "alice", 1_100_000)
    discovery = _discovery()
    assert StandingRewardClaim.from_dict(standing.to_dict()) == standing
    assert DiscoveryBountyClaim.from_dict(discovery.to_dict()) == discovery
    with pytest.raises(EconomicsError, match="all-zero"):
        replace(standing, retained_evidence_digest="0" * 64)
    with pytest.raises(EconomicsError, match="fields"):
        StandingRewardClaim.from_dict({**standing.to_dict(), "extra": 1})


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
    result = project_global_rewards(
        _policy(),
        _global_context(),
        reversed(authorities),
        tuple(claim for row in authorities for claim in row.standing_claims),
    )
    assert result.weights_by_hotkey["bob"] > result.weights_by_hotkey["carol"]
    assert result.weights_by_hotkey["carol"] > result.weights_by_hotkey["alice"]
    assert sum(result.weights_by_hotkey.values()) == WEIGHT_PPM
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
            tuple(
                claim
                for row in (valid, incomplete)
                for claim in row.standing_claims
            ),
        )



def test_non_crowned_pass_keeps_earning_after_another_pass_becomes_active() -> None:
    catalog = _catalog()
    stack = _stack(catalog, ("slot.a",))
    active = _claim(stack, "slot.a", "bob", 1_150_000, 200, evidence="7")
    held = replace(
        active,
        arena_digest=_d("f"),
        contribution_digest=_d("9"),
        hotkey="alice",
        speedup_ppm=1_100_000,
        crowned_block=100,
        retained_evidence_digest=_d("8"),
    )
    result = project_global_rewards(
        _policy(),
        _global_context(),
        (ArenaRewardAuthority(catalog, stack, 2, (active,)),),
        (held, active),
    )
    assert {row.hotkey for row in result.standing} == {"alice", "bob"}
    assert all(result.weights_by_hotkey[hotkey] > 0 for hotkey in ("alice", "bob"))
