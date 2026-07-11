"""Per-slot championships (report misalignment M4): pay specialists, not only the
single best end-to-end bundle. One champion per slot; emission split across slots."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from optima.arenas import (
    MINIMAX_M3_B300_TP4_DECODE_V1,
    MINIMAX_M3_B300_TP4_LONGPREFILL_V1,
    derive_prompt_seed,
)
from optima.commit_reveal import (
    EvalRecord,
    Ledger,
    LedgerAttestationError,
    make_chain_scope,
    make_commitment,
)


TEST_CHAIN_SCOPE = make_chain_scope(genesis_hash="0xtest", netuid=1)
HOST_ATTESTATION_SHA256 = "sha256:" + "3" * 64
VALIDATOR_HOTKEY = "validator"
EVALUATION_ID = "4" * 64
QUALIFICATION_EVIDENCE_SHA256 = "sha256:" + "5" * 64


def _verify_host_attestation(reference, context):
    valid = (
        reference == HOST_ATTESTATION_SHA256
        and context["arena_fingerprint"] in {
            MINIMAX_M3_B300_TP4_DECODE_V1.fingerprint,
            MINIMAX_M3_B300_TP4_LONGPREFILL_V1.fingerprint,
        }
        and context["chain_scope"] == TEST_CHAIN_SCOPE
        and context["validator_hotkey"] == VALIDATOR_HOTKEY
        and context["evaluation_id"] == EVALUATION_ID
        and context["miner_hotkey"]
        in {"decode-miner", "prefill-miner", "tree-miner", "host-miner",
            "split-host", "dynamic", "coordinated", "bound", "coherent"}
        and context["settlement_round_id"] == 0
        and context["evaluation_block"] == 0
        and context["passed_quality"] is True
        and context["passed_timed_quality"] is True
        and context["passed_warmup_quality"] is True
        and context["passed_speedup"] is True
        and context["confident"] is True
        and context["crownable"] is True
        and context["qualification_evidence_sha256"]
        == QUALIFICATION_EVIDENCE_SHA256
    )
    if not valid:
        return None
    return SimpleNamespace(
        qualification_evidence_sha256=QUALIFICATION_EVIDENCE_SHA256
    )


def _score(led, hotkey, ch, slot, score, *, rnd=0, pin="0.5.12.post1", passed=True):
    led.commit(hotkey, make_commitment(ch, hotkey, "s"), rnd)
    led.reveal(hotkey, ch, "s", rnd, fingerprint=ch)
    led.record_score(hotkey, ch, rnd, score, kl_mean=0.0, passed=passed, sglang_version=pin, slot=slot)


def test_two_specialists_split_emission_across_slots():
    led = Ledger()
    _score(led, "alice", "H_A", "moe.fused_experts", 1.20)
    _score(led, "bob", "H_B", "collective.all_reduce", 1.15)
    res = led.settle_per_slot(0, margin=0.02, current_sglang_version="0.5.12.post1")
    assert res.champions["moe.fused_experts"].hotkey == "alice"
    assert res.champions["collective.all_reduce"].hotkey == "bob"
    # Each owns one of two slots -> 50/50 split (winner-take-all would have given alice 100%).
    assert res.weights == {"alice": 0.5, "bob": 0.5}


def test_one_hotkey_owning_two_slots_gets_full_share():
    led = Ledger()
    _score(led, "alice", "H_A", "moe.fused_experts", 1.20)
    _score(led, "alice", "H_A2", "norm.rmsnorm", 1.10)
    res = led.settle_per_slot(0, margin=0.02, current_sglang_version="0.5.12.post1")
    assert res.weights == {"alice": 1.0}


def test_per_slot_king_of_the_hill_within_slot():
    led = Ledger()
    _score(led, "alice", "H_A", "moe.fused_experts", 1.30, rnd=0)
    led.settle_per_slot(0, margin=0.05, current_sglang_version="0.5.12.post1")
    # A weak challenger in the SAME slot doesn't clear the margin.
    _score(led, "bob", "H_B", "moe.fused_experts", 1.33, rnd=1)
    res = led.settle_per_slot(1, margin=0.05, current_sglang_version="0.5.12.post1")
    assert res.champions["moe.fused_experts"].hotkey == "alice"
    assert not res.title_changes.get("moe.fused_experts")


def test_per_slot_copy_excluded():
    led = Ledger()
    ch = "shared"
    led.commit("alice", make_commitment(ch, "alice", "a"), 0)
    led.commit("bob", make_commitment(ch, "bob", "b"), 0)
    led.reveal("alice", ch, "a", 0, fingerprint=ch)
    led.reveal("bob", ch, "b", 0, fingerprint=ch)  # demoted
    led.record_score("alice", ch, 0, 1.40, 0.0, True, slot="moe.fused_experts")
    led.record_score("bob", ch, 0, 1.40, 0.0, True, slot="moe.fused_experts")
    res = led.settle_per_slot(0, margin=0.02)
    assert res.champions["moe.fused_experts"].hotkey == "alice"
    assert "bob" in res.rejected_copies


def test_per_slot_persists_across_load(tmp_path):
    led = Ledger()
    _score(led, "alice", "a" * 64, "moe.fused_experts", 1.20)
    led.settle_per_slot(0, current_sglang_version="0.5.12.post1")
    p = tmp_path / "l.json"
    led.save(p)
    led2 = Ledger.load(p)
    assert led2.champions["moe.fused_experts"].hotkey == "alice"


def test_stale_champion_flagged_even_when_its_slot_gets_no_submissions():
    # A pin bump makes alice's frozen score incomparable; her slot receives no
    # challengers this round (only ANOTHER slot does), but she still holds emission —
    # the slot must be flagged stale for re-baseline anyway.
    led = Ledger()
    _score(led, "alice", "H_A", "moe.fused_experts", 1.20, pin="0.5.11")
    led.settle_per_slot(0, current_sglang_version="0.5.11")
    _score(led, "bob", "H_B", "norm.rmsnorm", 1.10, rnd=1, pin="0.5.12")
    res = led.settle_per_slot(1, current_sglang_version="0.5.12")
    assert "moe.fused_experts" in res.stale_slots
    # And with NO submissions anywhere the flag still raises.
    res2 = led.settle_per_slot(2, current_sglang_version="0.5.12")
    assert "moe.fused_experts" in res2.stale_slots


def test_atomic_bundle_gets_one_target_champion_and_no_member_champions():
    led = Ledger()
    hotkey = "deep-miner"
    content = "deep-hash"
    members = (
        "collective.ar_residual_rmsnorm",
        "collective.moe_finalize_ar_rmsnorm",
    )
    target = "collective.moe_epilogue.v1"
    led.commit(hotkey, make_commitment(content, hotkey, "s"), 0)
    led.reveal(hotkey, content, "s", 0, fingerprint=content)
    led.record_score(
        hotkey,
        content,
        0,
        1.08,
        kl_mean=0.0,
        passed=True,
        target=target,
        mode="atomic",
        member_slots=members,
    )

    result = led.settle_per_target(0, margin=0.02)

    assert set(result.champions) == {target}
    assert result.champions[target].hotkey == hotkey
    assert not (set(members) & set(result.champions))
    assert result.weights == {hotkey: 1.0}

    # The historical API alias delegates to target settlement; it never expands
    # the same atomic score into member-slot titles.
    alias_result = led.settle_per_slot(0, margin=0.02)
    assert set(alias_result.champions) == {target}


def _arena_score(led, *, arena, hotkey, content, score, target="norm.rmsnorm"):
    if not led.chain_scope:
        led.bind_chain_scope(TEST_CHAIN_SCOPE)
    led.bind_validator_hotkey(VALIDATOR_HOTKEY)
    led.commit(hotkey, make_commitment(content, hotkey, "arena"), 0)
    led.reveal(hotkey, content, "arena", 0, fingerprint=content)
    seed_block_hash = "0x" + "1" * 64
    prompt_seed = derive_prompt_seed(
        arena,
        bundle_hash=content,
        round_id=0,
        block_hash=seed_block_hash,
    )
    led.record_score(
        hotkey,
        content,
        0,
        score,
        kl_mean=0.0,
        passed=True,
        sglang_version=arena.sglang_version,
        target=target,
        mode="slot",
        member_slots=(target,),
        arena=arena,
        prompt_seed=prompt_seed,
        prompt_engine_version=arena.workload.prompt_engine_version,
        prompt_seed_scheme=arena.workload.prompt_seed_scheme,
        seed_round_id=0,
        seed_block=0,
        seed_block_hash=seed_block_hash,
        host_attestation_sha256=HOST_ATTESTATION_SHA256,
        quality_evidence="controller paired top-k evidence: pass",
        validator_hotkey=VALIDATOR_HOTKEY,
        evaluation_id=EVALUATION_ID,
        miner_hotkey=hotkey,
        settlement_round_id=0,
        evaluation_block=0,
        passed_quality=True,
        passed_timed_quality=True,
        passed_warmup_quality=True,
        passed_speedup=True,
        confident=True,
        crownable=True,
        qualification_evidence_sha256=QUALIFICATION_EVIDENCE_SHA256,
    )
    recorded = led.scores[-1]
    led.record_eval(EvalRecord(
        hotkey=hotkey,
        bundle_hash=content,
        slot=recorded.slot,
        round_id=0,
        score=score,
        passed=True,
        target=recorded.target,
        mode=recorded.mode,
        member_slots=recorded.member_slots,
        arena_name=recorded.arena_name,
        arena_fingerprint=recorded.arena_fingerprint,
        arena_bracket=recorded.arena_bracket,
        regime=recorded.regime,
        sglang_version=recorded.sglang_version,
        validator_image=recorded.validator_image,
        referee_source_digest=recorded.referee_source_digest,
        referee_tree_digest=recorded.referee_tree_digest,
        model_revision=recorded.model_revision,
        model_manifest_digest=recorded.model_manifest_digest,
        model_content_digest=recorded.model_content_digest,
        host_attestation_sha256=recorded.host_attestation_sha256,
        prompt_seed=recorded.prompt_seed,
        prompt_engine_version=recorded.prompt_engine_version,
        prompt_seed_scheme=recorded.prompt_seed_scheme,
        seed_round_id=recorded.seed_round_id,
        seed_block=recorded.seed_block,
        seed_block_hash=recorded.seed_block_hash,
        quality_evidence=recorded.quality_evidence,
        chain_scope=recorded.chain_scope,
        validator_hotkey=recorded.validator_hotkey,
        evaluation_id=recorded.evaluation_id,
        miner_hotkey=recorded.miner_hotkey,
        settlement_round_id=recorded.settlement_round_id,
        evaluation_block=recorded.evaluation_block,
        passed_quality=recorded.passed_quality,
        passed_timed_quality=recorded.passed_timed_quality,
        passed_warmup_quality=recorded.passed_warmup_quality,
        passed_speedup=recorded.passed_speedup,
        confident=recorded.confident,
        crownable=recorded.crownable,
        qualification_evidence_sha256=(
            recorded.qualification_evidence_sha256
        ),
    ))


def test_arena_brackets_never_mix_scores_champions_or_weights(tmp_path):
    decode = MINIMAX_M3_B300_TP4_DECODE_V1
    prefill = MINIMAX_M3_B300_TP4_LONGPREFILL_V1
    led = Ledger()
    _arena_score(
        led, arena=decode, hotkey="decode-miner", content="d" * 64, score=1.05
    )
    _arena_score(
        led, arena=prefill, hotkey="prefill-miner", content="e" * 64, score=1.20
    )

    decode_result = led.settle_per_target(
        0, margin=0.02, current_sglang_version=decode.sglang_version, arena=decode,
        host_attestation_verifier=_verify_host_attestation,
        validator_hotkey=VALIDATOR_HOTKEY,
    )
    prefill_result = led.settle_per_target(
        0, margin=0.02, current_sglang_version=prefill.sglang_version, arena=prefill,
        host_attestation_verifier=_verify_host_attestation,
        validator_hotkey=VALIDATOR_HOTKEY,
    )

    assert decode_result.weights == {"decode-miner": 1.0}
    assert prefill_result.weights == {"prefill-miner": 1.0}
    assert led.current_weights(
        arena=decode, host_attestation_verifier=_verify_host_attestation,
        validator_hotkey=VALIDATOR_HOTKEY,
    ) == {"decode-miner": 1.0}
    assert led.current_weights(
        arena=prefill, host_attestation_verifier=_verify_host_attestation,
        validator_hotkey=VALIDATOR_HOTKEY,
    ) == {"prefill-miner": 1.0}
    assert led.current_weights_across_arenas(
        (decode, prefill),
        host_attestation_verifier=_verify_host_attestation,
        validator_hotkey=VALIDATOR_HOTKEY,
    ) == {"decode-miner": 0.5, "prefill-miner": 0.5}
    assert not led.champions  # registered titles never leak into the legacy namespace

    path = tmp_path / "scoped.json"
    led.save(path)
    loaded = Ledger.load(path)
    assert loaded.current_weights(
        arena=decode, host_attestation_verifier=_verify_host_attestation,
        validator_hotkey=VALIDATOR_HOTKEY,
    ) == {"decode-miner": 1.0}
    assert loaded.current_weights(
        arena=prefill, host_attestation_verifier=_verify_host_attestation,
        validator_hotkey=VALIDATOR_HOTKEY,
    ) == {"prefill-miner": 1.0}
    assert loaded.current_weights_across_arenas(
        (decode, prefill),
        host_attestation_verifier=_verify_host_attestation,
        validator_hotkey=VALIDATOR_HOTKEY,
    ) == {"decode-miner": 0.5, "prefill-miner": 0.5}


def test_global_weights_never_redistribute_an_invalid_arena_title():
    decode = MINIMAX_M3_B300_TP4_DECODE_V1
    prefill = MINIMAX_M3_B300_TP4_LONGPREFILL_V1
    ledger = Ledger()
    _arena_score(
        ledger,
        arena=decode,
        hotkey="decode-miner",
        content="d" * 64,
        score=1.05,
    )
    _arena_score(
        ledger,
        arena=prefill,
        hotkey="prefill-miner",
        content="e" * 64,
        score=1.20,
    )
    for arena in (decode, prefill):
        ledger.settle_per_target(
            0,
            margin=arena.settlement.dethrone_margin,
            current_sglang_version=arena.sglang_version,
            arena=arena,
            host_attestation_verifier=_verify_host_attestation,
            validator_hotkey=VALIDATOR_HOTKEY,
        )
    ledger.arena_champions[decode.bracket]["norm.rmsnorm"].evaluation_id = (
        "f" * 64
    )

    with pytest.raises(LedgerAttestationError, match="refusing to redistribute"):
        ledger.current_weights_across_arenas(
            (decode, prefill),
            host_attestation_verifier=_verify_host_attestation,
            validator_hotkey=VALIDATOR_HOTKEY,
        )


def test_legacy_rows_cannot_be_adopted_into_a_registered_chain_scope():
    led = Ledger()
    _score(led, "legacy", "legacy-hash", "norm.rmsnorm", 9.0)
    legacy = led.settle_per_target(0, margin=0.02)
    assert legacy.weights == {"legacy": 1.0}
    with pytest.raises(LedgerAttestationError, match="legacy ledger contains"):
        led.bind_chain_scope(TEST_CHAIN_SCOPE)


def test_referee_tree_mismatch_invalidates_persisted_arena_champion():
    arena = MINIMAX_M3_B300_TP4_DECODE_V1
    led = Ledger()
    _arena_score(
        led, arena=arena, hotkey="tree-miner", content="f" * 64, score=1.05
    )
    settled = led.settle_per_target(
        0,
        margin=arena.settlement.dethrone_margin,
        current_sglang_version=arena.sglang_version,
        arena=arena,
        host_attestation_verifier=_verify_host_attestation,
        validator_hotkey=VALIDATOR_HOTKEY,
    )
    assert settled.weights == {"tree-miner": 1.0}

    champion = next(iter(led.arena_champions[arena.bracket].values()))
    champion.referee_tree_digest = "sha256:" + "f" * 64

    assert led.current_weights(
        arena=arena, host_attestation_verifier=_verify_host_attestation,
        validator_hotkey=VALIDATOR_HOTKEY,
    ) == {}


@pytest.mark.parametrize("row_kind", ["score", "eval", "champion"])
def test_host_attestation_must_match_every_crown_authority_row(row_kind):
    arena = MINIMAX_M3_B300_TP4_DECODE_V1
    led = Ledger()
    _arena_score(
        led, arena=arena, hotkey="host-miner", content="9" * 64, score=1.05
    )
    settled = led.settle_per_target(
        0,
        margin=arena.settlement.dethrone_margin,
        current_sglang_version=arena.sglang_version,
        arena=arena,
        host_attestation_verifier=_verify_host_attestation,
        validator_hotkey=VALIDATOR_HOTKEY,
    )
    assert settled.weights == {"host-miner": 1.0}

    if row_kind == "score":
        led.scores[-1].host_attestation_sha256 = "sha256:" + "4" * 64
    elif row_kind == "eval":
        # EvalRecord is frozen; replace the canonical retained row with a copy
        # whose sidecar reference no longer agrees with score/champion.
        key, record = next(iter(led.evals.items()))
        led.evals[key] = replace(
            record, host_attestation_sha256="sha256:" + "4" * 64
        )
    else:
        champion = next(iter(led.arena_champions[arena.bracket].values()))
        champion.host_attestation_sha256 = "sha256:" + "4" * 64

    assert led.current_weights(
        arena=arena, host_attestation_verifier=_verify_host_attestation,
        validator_hotkey=VALIDATOR_HOTKEY,
    ) == {}


def test_first_settlement_never_emits_mismatched_host_attestation_rows():
    arena = MINIMAX_M3_B300_TP4_DECODE_V1
    led = Ledger()
    _arena_score(
        led, arena=arena, hotkey="split-host", content="7" * 64, score=1.05
    )
    key, record = next(iter(led.evals.items()))
    led.evals[key] = replace(
        record, host_attestation_sha256="sha256:" + "4" * 64
    )

    settled = led.settle_per_target(
        0,
        margin=arena.settlement.dethrone_margin,
        current_sglang_version=arena.sglang_version,
        arena=arena,
        host_attestation_verifier=_verify_host_attestation,
        validator_hotkey=VALIDATOR_HOTKEY,
    )
    assert settled.weights == {}
    assert led.arena_champions[arena.bracket] == {}


@pytest.mark.parametrize("row_kind", ["score", "eval"])
@pytest.mark.parametrize(
    "field,value",
    [
        ("sglang_version", "different"),
        ("prompt_engine_version", "different"),
        ("prompt_seed_scheme", "different"),
        ("seed_round_id", 1),
        ("seed_block", 1),
        ("seed_block_hash", "0x" + "2" * 64),
        ("quality_evidence", "different controller evidence"),
        ("validator_hotkey", "different-validator"),
        ("evaluation_id", "6" * 64),
        ("miner_hotkey", "different-miner"),
        ("settlement_round_id", 1),
        ("evaluation_block", 100),
        ("passed_quality", False),
        ("passed_timed_quality", False),
        ("passed_warmup_quality", False),
        ("passed_speedup", False),
        ("confident", False),
        ("crownable", False),
        ("qualification_evidence_sha256", "sha256:" + "6" * 64),
    ],
)
def test_all_dynamic_crown_evidence_must_match_score_eval_and_champion(
    row_kind, field, value
):
    arena = MINIMAX_M3_B300_TP4_DECODE_V1
    led = Ledger()
    _arena_score(
        led, arena=arena, hotkey="dynamic", content="6" * 64, score=1.05
    )
    settled = led.settle_per_target(
        0,
        margin=arena.settlement.dethrone_margin,
        current_sglang_version=arena.sglang_version,
        arena=arena,
        host_attestation_verifier=_verify_host_attestation,
        validator_hotkey=VALIDATOR_HOTKEY,
    )
    assert settled.weights == {"dynamic": 1.0}

    if row_kind == "score":
        setattr(led.scores[-1], field, value)
    else:
        key, record = next(iter(led.evals.items()))
        led.evals[key] = replace(record, **{field: value})
    assert led.current_weights(
        arena=arena, host_attestation_verifier=_verify_host_attestation,
        validator_hotkey=VALIDATOR_HOTKEY,
    ) == {}


def test_registered_arena_score_refuses_missing_host_attestation():
    arena = MINIMAX_M3_B300_TP4_DECODE_V1
    led = Ledger()
    led.bind_chain_scope(TEST_CHAIN_SCOPE)
    content = "8" * 64
    led.commit("hostless", make_commitment(content, "hostless", "arena"), 0)
    led.reveal("hostless", content, "arena", 0, fingerprint=content)
    block_hash = "0x" + "1" * 64
    prompt_seed = derive_prompt_seed(
        arena, bundle_hash=content, round_id=0, block_hash=block_hash
    )

    with pytest.raises(ValueError, match="trusted-host attestation"):
        led.record_score(
            "hostless",
            content,
            0,
            1.05,
            kl_mean=0.0,
            passed=True,
            sglang_version=arena.sglang_version,
            target="norm.rmsnorm",
            mode="slot",
            member_slots=("norm.rmsnorm",),
            arena=arena,
            prompt_seed=prompt_seed,
            prompt_engine_version=arena.workload.prompt_engine_version,
            prompt_seed_scheme=arena.workload.prompt_seed_scheme,
            seed_round_id=0,
            seed_block=0,
            seed_block_hash=block_hash,
            quality_evidence="controller evidence",
        )


def test_coordinated_fake_digest_cannot_replace_retained_sidecar_authority():
    arena = MINIMAX_M3_B300_TP4_DECODE_V1
    led = Ledger()
    _arena_score(
        led, arena=arena, hotkey="coordinated", content="5" * 64, score=1.05
    )
    led.settle_per_target(
        0,
        margin=arena.settlement.dethrone_margin,
        current_sglang_version=arena.sglang_version,
        arena=arena,
        host_attestation_verifier=_verify_host_attestation,
        validator_hotkey=VALIDATOR_HOTKEY,
    )
    fake = "sha256:" + "4" * 64
    led.scores[-1].host_attestation_sha256 = fake
    key, record = next(iter(led.evals.items()))
    led.evals[key] = replace(record, host_attestation_sha256=fake)
    champion = next(iter(led.arena_champions[arena.bracket].values()))
    champion.host_attestation_sha256 = fake

    with pytest.raises(LedgerAttestationError, match="no authoritative evidence"):
        led.current_weights(
            arena=arena, host_attestation_verifier=_verify_host_attestation,
            validator_hotkey=VALIDATOR_HOTKEY,
        )


def test_copied_ledger_and_sidecar_cannot_emit_for_another_active_validator(
    tmp_path,
):
    arena = MINIMAX_M3_B300_TP4_DECODE_V1
    led = Ledger()
    _arena_score(
        led, arena=arena, hotkey="bound", content="4" * 64, score=1.05
    )
    led.settle_per_target(
        0,
        margin=arena.settlement.dethrone_margin,
        current_sglang_version=arena.sglang_version,
        arena=arena,
        host_attestation_verifier=_verify_host_attestation,
        validator_hotkey=VALIDATOR_HOTKEY,
    )
    copied_path = tmp_path / "copied-ledger.json"
    led.save(copied_path)
    copied = Ledger.load(copied_path)

    with pytest.raises(
        LedgerAttestationError,
        match="independently-known active validator",
    ):
        copied.current_weights(
            arena=arena,
            host_attestation_verifier=_verify_host_attestation,
            validator_hotkey="different-validator",
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("validator_hotkey", "different-validator"),
        ("evaluation_id", "7" * 64),
        ("qualification_evidence_sha256", "sha256:" + "7" * 64),
    ],
)
def test_coordinated_authority_field_edits_still_fail_closed(field, value):
    arena = MINIMAX_M3_B300_TP4_DECODE_V1
    led = Ledger()
    _arena_score(
        led, arena=arena, hotkey="coherent", content="3" * 64, score=1.05
    )
    led.settle_per_target(
        0,
        margin=arena.settlement.dethrone_margin,
        current_sglang_version=arena.sglang_version,
        arena=arena,
        host_attestation_verifier=_verify_host_attestation,
        validator_hotkey=VALIDATOR_HOTKEY,
    )

    setattr(led.scores[-1], field, value)
    key, record = next(iter(led.evals.items()))
    led.evals[key] = replace(record, **{field: value})
    champion = next(iter(led.arena_champions[arena.bracket].values()))
    setattr(champion, field, value)

    if field == "validator_hotkey":
        assert led.current_weights(
            arena=arena,
            host_attestation_verifier=_verify_host_attestation,
            validator_hotkey=VALIDATOR_HOTKEY,
        ) == {}
    else:
        with pytest.raises(
            LedgerAttestationError,
            match="no authoritative evidence",
        ):
            led.current_weights(
                arena=arena,
                host_attestation_verifier=_verify_host_attestation,
                validator_hotkey=VALIDATOR_HOTKEY,
            )
