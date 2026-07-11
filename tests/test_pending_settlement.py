"""Crash-recovery tests for authoritative settlement dispositions.

These tests are intentionally self-contained: they exercise the public chain pass
and ledger recovery APIs without depending on the large validator-loop fixture
module or invoking an evaluator more than once.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from optima.arenas import (
    MINIMAX_M3_B300_TP4_DECODE_V1 as ARENA,
    derive_prompt_seed,
)
from optima.chain.fetch import package_bundle
from optima.chain.payload import encode_payload_for_testing as encode_payload
from optima.chain.validator_loop import (
    EvalOutcome,
    WeightSafetyError,
    _atomic_write_weights_state,
    _context_identity,
    _recover_pending_settlements,
    run_pass,
)
from optima.commit_reveal import (
    EvalRecord,
    Ledger,
    LedgerAttestationError,
    PendingSettlementError,
    make_chain_scope,
    make_commitment,
)


HOST_ATTESTATION = "sha256:" + "7" * 64
QUALIFICATION_EVIDENCE = "sha256:" + "8" * 64
VALIDATOR_HOTKEY = "validator"
WALLET = SimpleNamespace(hotkey=SimpleNamespace(ss58_address="validator"))


@pytest.fixture(autouse=True)
def _allow_legacy_test_component_lane(monkeypatch):
    monkeypatch.setattr(
        "optima.device_component.component_crown_rejection",
        lambda _manifest: None,
    )


class _Metagraph:
    def __init__(self, hotkeys, weights, last_update):
        self.uids = list(range(len(hotkeys)))
        self.hotkeys = list(hotkeys)
        self.last_update = list(last_update)
        self.validator_permit = [True] * len(hotkeys)
        self.W = weights


class _Subtensor:
    def __init__(self, *, revealed, block=400):
        self.revealed = dict(revealed)
        self.block = block
        self.hotkeys = ["validator", "miner"]
        self._weights = [[0.0, 0.0], [0.0, 0.0]]
        self._last_update = [0, 0]
        self.set_weights_calls = []

    def get_current_block(self):
        return self.block

    def get_finalized_block_number(self):
        return self.block

    def get_block_hash(self, block):
        digest = hashlib.sha256(f"block:{int(block)}".encode()).hexdigest()
        return "0x" + digest

    def get_all_revealed_commitments(self, netuid=None, block=None):
        return {
            hotkey: tuple(
                entry for entry in history
                if block is None or entry[0] <= block
            )[-10:]
            for hotkey, history in self.revealed.items()
        }

    def metagraph(self, netuid=None):
        return _Metagraph(self.hotkeys, self._weights, self._last_update)

    def weights(self, netuid=None):
        rows = []
        for source_uid, dense_row in enumerate(self._weights):
            targets = [
                (target_uid, round(float(weight) * 65_535))
                for target_uid, weight in enumerate(dense_row)
                if float(weight) > 0
            ]
            if targets:
                rows.append((source_uid, targets))
        return rows

    def set_weights(
        self,
        *,
        wallet,
        netuid,
        uids,
        weights,
        version_key,
        wait_for_inclusion,
        wait_for_finalization,
    ):
        self.set_weights_calls.append((list(uids), list(weights)))
        self._weights[0] = [0.0, 0.0]
        for uid, weight in zip(uids, weights):
            self._weights[0][int(uid)] = float(weight)
        self._last_update[0] = self.block
        return True


class _DeferredWeightSubtensor(_Subtensor):
    """Accept a CR commit without changing the live sparse row until revealed."""

    def set_weights(
        self,
        *,
        wallet,
        netuid,
        uids,
        weights,
        version_key,
        wait_for_inclusion,
        wait_for_finalization,
    ):
        self.set_weights_calls.append((list(uids), list(weights)))
        return SimpleNamespace(
            success=True,
            message="commit included; reveal pending",
            data={"reveal_round": 12_345},
        )

    def reveal_weights(self, *, update_block: int) -> None:
        uids, weights = self.set_weights_calls[-1]
        self._weights[0] = [0.0, 0.0]
        for uid, weight in zip(uids, weights):
            self._weights[0][int(uid)] = float(weight)
        self._last_update[0] = int(update_block)


def _bundle_submission(tmp_path: Path) -> tuple[str, str]:
    bundle = tmp_path / "bundle"
    (bundle / "kernels").mkdir(parents=True)
    (bundle / "manifest.toml").write_text(
        'bundle_id = "pending-test"\n'
        'abi_version = "optima-op-abi-v0"\n\n'
        "[[ops]]\n"
        'slot = "activation.silu_and_mul"\n'
        'source = "kernels/k.cu"\n'
        'entry = "k"\n'
        'execution_class = "validator_device"\n'
        'device_abi = "activation.silu_and_mul.cuda.v1"\n'
        'dtypes = ["bfloat16"]\n'
    )
    (bundle / "kernels" / "k.cu").write_text(
        'extern "C" __global__ void k() {}\n'
    )
    archive, content_hash = package_bundle(
        bundle, tmp_path / "hosted" / "bundle.tar.gz"
    )
    return content_hash, archive.as_uri()


def _context_scope(context) -> dict[str, object]:
    """Use the production controller projection; no ABI compatibility shim."""

    return _context_identity(
        context,
        qualification_evidence_sha256=QUALIFICATION_EVIDENCE,
    )


def _owned_evaluator(fn, verifier=None):
    fn.validator_owned_oci = True
    if verifier is None:
        def verifier(reference, context):
            if (
                reference != HOST_ATTESTATION
                or context["validator_hotkey"] != VALIDATOR_HOTKEY
                or context["qualification_evidence_sha256"]
                != QUALIFICATION_EVIDENCE
            ):
                return None
            return SimpleNamespace(
                qualification_evidence_sha256=QUALIFICATION_EVIDENCE
            )
    fn.host_attestation_verifier = verifier
    return fn


def _retained_verifier(reference, context):
    if (
        reference != HOST_ATTESTATION
        or context["validator_hotkey"] != VALIDATOR_HOTKEY
    ):
        return None
    return SimpleNamespace(
        qualification_evidence_sha256=(
            context["qualification_evidence_sha256"]
        )
    )


def _passing_outcome(context) -> EvalOutcome:
    return EvalOutcome(
        passed=True,
        score=1.08,
        kl_mean=0.001,
        target="activation.silu_and_mul",
        mode="slot",
        member_slots=("activation.silu_and_mul",),
        crownable=True,
        passed_timed_quality=True,
        passed_warmup_quality=True,
        passed_speedup=True,
        host_attestation_sha256=HOST_ATTESTATION,
        quality_evidence="trusted external fidelity evidence",
        **_context_scope(context),
    )


def test_round_rollover_recovers_pending_without_gpu_replay_after_two_crashes(
    tmp_path, monkeypatch
):
    content_hash, url = _bundle_submission(tmp_path)
    subtensor = _Subtensor(
        revealed={"miner": ((5, encode_payload(content_hash, url)),)},
    )
    ledger_path = str(tmp_path / "ledger.json")
    calls = {"evaluator": 0}

    @_owned_evaluator
    def evaluator(bundle_dir, context):
        calls["evaluator"] += 1
        return _passing_outcome(context)

    real_settle = Ledger.settle_per_target

    def fail_first_settlement(self, *args, **kwargs):
        if kwargs.get("candidate_evidence_sha256") is not None:
            raise RuntimeError("forced settlement controller failure")
        return real_settle(self, *args, **kwargs)

    monkeypatch.setattr(Ledger, "settle_per_target", fail_first_settlement)
    with pytest.raises(RuntimeError, match="forced settlement"):
        run_pass(
            subtensor,
            WALLET,
            1,
            ledger_path=ledger_path,
            bundles_dir=str(tmp_path / "cache"),
            evaluator=evaluator,
            arena=ARENA,
            validator_hotkey="validator",
            test_only_allow_local_file_urls=True,
        )

    first_round = 400 // ARENA.settlement.round_blocks
    durable = Ledger.load(ledger_path)
    assert calls["evaluator"] == 1
    assert len(durable.scores) == 1 and len(durable.evals) == 1
    assert len(durable.pending_settlements) == 1
    assert ARENA.bracket not in durable.arena_champions

    # Move into a later settlement window. Recovery must happen before the known
    # submission loop, so this evaluator is deliberately not callable.
    subtensor.block += ARENA.settlement.round_blocks + 1
    monkeypatch.setattr(Ledger, "settle_per_target", real_settle)

    def must_not_evaluate(bundle_dir, context):  # pragma: no cover - assertion path
        raise AssertionError("authoritative GPU evaluation was replayed")

    must_not_evaluate = _owned_evaluator(must_not_evaluate)

    # Simulate death after settlement mutated the in-memory champion but before
    # the first post-settlement save. The old on-disk pending marker must survive.
    real_save = Ledger.save
    crash_armed = {"value": True}

    def crash_before_champion_save(self, path):
        if (
            crash_armed["value"]
            and self.pending_settlements
            and self.arena_champions.get(ARENA.bracket)
        ):
            crash_armed["value"] = False
            raise RuntimeError("simulated death between settle and save")
        return real_save(self, path)

    monkeypatch.setattr(Ledger, "save", crash_before_champion_save)
    with pytest.raises(RuntimeError, match="between settle and save"):
        run_pass(
            subtensor,
            WALLET,
            1,
            ledger_path=ledger_path,
            bundles_dir=str(tmp_path / "cache"),
            evaluator=must_not_evaluate,
            arena=ARENA,
            validator_hotkey="validator",
            test_only_allow_local_file_urls=True,
        )
    after_crash = Ledger.load(ledger_path)
    assert len(after_crash.pending_settlements) == 1
    assert not after_crash.arena_champions.get(ARENA.bracket)

    monkeypatch.setattr(Ledger, "save", real_save)
    recovered = run_pass(
        subtensor,
        WALLET,
        1,
        ledger_path=ledger_path,
        bundles_dir=str(tmp_path / "cache"),
        evaluator=must_not_evaluate,
        arena=ARENA,
        validator_hotkey="validator",
        test_only_allow_local_file_urls=True,
    )

    final = Ledger.load(ledger_path)
    champion = final.arena_champions[ARENA.bracket][
        "activation.silu_and_mul"
    ]
    assert champion.hotkey == "miner"
    assert champion.round_id == first_round
    assert recovered.round_id > first_round
    assert recovered.weights == {"miner": 1.0}
    assert recovered.weights_pushed
    assert not final.pending_settlements
    assert calls["evaluator"] == 1


def test_weight_commit_is_pending_until_authoritative_chain_readback(tmp_path):
    content_hash, url = _bundle_submission(tmp_path)
    subtensor = _DeferredWeightSubtensor(
        revealed={"miner": ((5, encode_payload(content_hash, url)),)},
    )
    ledger_path = str(tmp_path / "ledger.json")
    calls = {"evaluator": 0}

    @_owned_evaluator
    def evaluator(bundle_dir, context):
        calls["evaluator"] += 1
        return _passing_outcome(context)

    first = run_pass(
        subtensor,
        WALLET,
        1,
        ledger_path=ledger_path,
        bundles_dir=str(tmp_path / "cache"),
        evaluator=evaluator,
        arena=ARENA,
        validator_hotkey=VALIDATOR_HOTKEY,
        test_only_allow_local_file_urls=True,
    )
    assert first.weights == {"miner": 1.0}
    assert first.weights_submitted is True
    assert first.weights_pending is True
    assert first.weights_confirmed is False
    assert first.weights_pushed is False
    assert len(subtensor.set_weights_calls) == 1
    state_path = next(tmp_path.glob("ledger.json.weights_state.*.global.json"))
    state = json.loads(state_path.read_text())
    assert state["status"] == "pending"
    assert state["submit_block"] == 400
    assert state["reveal_round"] == 12_345
    assert state["expected_weights"] == {"miner": 1.0}

    subtensor.block += 1
    restarted = run_pass(
        subtensor,
        WALLET,
        1,
        ledger_path=ledger_path,
        bundles_dir=str(tmp_path / "cache"),
        evaluator=evaluator,
        arena=ARENA,
        validator_hotkey=VALIDATOR_HOTKEY,
        test_only_allow_local_file_urls=True,
    )
    assert calls["evaluator"] == 1
    assert len(subtensor.set_weights_calls) == 1
    assert restarted.weights_submitted is False
    assert restarted.weights_pending is True
    assert restarted.weights_confirmed is False

    subtensor.reveal_weights(update_block=subtensor.block)
    subtensor.block += 1
    confirmed = run_pass(
        subtensor,
        WALLET,
        1,
        ledger_path=ledger_path,
        bundles_dir=str(tmp_path / "cache"),
        evaluator=evaluator,
        arena=ARENA,
        validator_hotkey=VALIDATOR_HOTKEY,
        test_only_allow_local_file_urls=True,
    )
    assert len(subtensor.set_weights_calls) == 1
    assert confirmed.weights_submitted is False
    assert confirmed.weights_pending is False
    assert confirmed.weights_confirmed is True
    assert confirmed.weights_pushed is False
    state = json.loads(state_path.read_text())
    assert state["status"] == "confirmed"
    assert state["confirmed_last_update"] == 401


def test_weight_intent_is_durable_before_sdk_call_and_suppresses_crash_replay(
    tmp_path
):
    content_hash, url = _bundle_submission(tmp_path)
    subtensor = _DeferredWeightSubtensor(
        revealed={"miner": ((5, encode_payload(content_hash, url)),)},
    )
    ledger_path = str(tmp_path / "ledger.json")
    sdk_calls = {"count": 0}

    class SimulatedProcessDeath(BaseException):
        pass

    def die_inside_sdk(**kwargs):
        sdk_calls["count"] += 1
        state_path = next(tmp_path.glob("ledger.json.weights_state.*.global.json"))
        state = json.loads(state_path.read_text())
        assert state["status"] == "intent"
        assert state["expected_weights"] == {"miner": 1.0}
        raise SimulatedProcessDeath("death after durable intent")

    subtensor.set_weights = die_inside_sdk
    with pytest.raises(SimulatedProcessDeath, match="durable intent"):
        run_pass(
            subtensor,
            WALLET,
            1,
            ledger_path=ledger_path,
            bundles_dir=str(tmp_path / "cache"),
            evaluator=_owned_evaluator(
                lambda _bundle, context: _passing_outcome(context)
            ),
            arena=ARENA,
            validator_hotkey=VALIDATOR_HOTKEY,
            test_only_allow_local_file_urls=True,
        )
    assert sdk_calls["count"] == 1

    def forbidden_replay(**kwargs):  # pragma: no cover - assertion path
        raise AssertionError("ambiguous weight submission was replayed")

    subtensor.set_weights = forbidden_replay
    subtensor.block += 1
    restarted = run_pass(
        subtensor,
        WALLET,
        1,
        ledger_path=ledger_path,
        bundles_dir=str(tmp_path / "cache"),
        evaluator=_owned_evaluator(
            lambda *_args: (_ for _ in ()).throw(
                AssertionError("terminal GPU evaluation replayed")
            )
        ),
        arena=ARENA,
        validator_hotkey=VALIDATOR_HOTKEY,
        test_only_allow_local_file_urls=True,
    )
    assert restarted.weights_pending is True
    assert restarted.weights_submitted is False
    assert sdk_calls["count"] == 1

    state_path = next(tmp_path.glob("ledger.json.weights_state.*.global.json"))
    state = json.loads(state_path.read_text())
    subtensor.block = state["retry_after_block"]
    held = run_pass(
        subtensor,
        WALLET,
        1,
        ledger_path=ledger_path,
        bundles_dir=str(tmp_path / "cache"),
        evaluator=_owned_evaluator(
            lambda *_args: (_ for _ in ()).throw(
                AssertionError("terminal GPU evaluation replayed")
            )
        ),
        arena=ARENA,
        validator_hotkey=VALIDATOR_HOTKEY,
        test_only_allow_local_file_urls=True,
    )
    assert held.weights_pending is True
    assert held.weights_held is True
    assert sdk_calls["count"] == 1
    assert json.loads(state_path.read_text())["status"] == "held"


@pytest.mark.parametrize(
    "mutation",
    [
        {"chain_scope": "genesis-netuid-v1:sha256:" + "f" * 64},
        {"arena_set_sha256": "sha256:" + "f" * 64},
        {"emission_policy": "tampered-emission-policy"},
        {"unexpected": "field"},
    ],
)
def test_v1_weight_journal_context_or_extra_field_tamper_fails_closed(
    tmp_path, mutation
):
    content_hash, url = _bundle_submission(tmp_path)
    subtensor = _DeferredWeightSubtensor(
        revealed={"miner": ((5, encode_payload(content_hash, url)),)},
    )
    ledger_path = str(tmp_path / "ledger.json")
    first = run_pass(
        subtensor,
        WALLET,
        1,
        ledger_path=ledger_path,
        bundles_dir=str(tmp_path / "cache"),
        evaluator=_owned_evaluator(
            lambda _bundle, context: _passing_outcome(context)
        ),
        arena=ARENA,
        validator_hotkey=VALIDATOR_HOTKEY,
        test_only_allow_local_file_urls=True,
    )
    assert first.weights_pending
    state_path = next(tmp_path.glob("ledger.json.weights_state.*.global.json"))
    state = json.loads(state_path.read_text())
    state.update(mutation)
    _atomic_write_weights_state(state_path, state)

    subtensor.block += 1
    with pytest.raises(
        WeightSafetyError,
        match="context differs|fields differ",
    ):
        run_pass(
            subtensor,
            WALLET,
            1,
            ledger_path=ledger_path,
            bundles_dir=str(tmp_path / "cache"),
            evaluator=_owned_evaluator(
                lambda *_args: (_ for _ in ()).throw(
                    AssertionError("terminal GPU evaluation replayed")
                )
            ),
            arena=ARENA,
            validator_hotkey=VALIDATOR_HOTKEY,
            test_only_allow_local_file_urls=True,
        )
    assert len(subtensor.set_weights_calls) == 1


def test_unchanged_refresh_commit_deduplicates_until_last_update_advances(
    tmp_path
):
    content_hash, url = _bundle_submission(tmp_path)
    subtensor = _Subtensor(
        revealed={"miner": ((5, encode_payload(content_hash, url)),)},
    )
    ledger_path = str(tmp_path / "ledger.json")
    first = run_pass(
        subtensor,
        WALLET,
        1,
        ledger_path=ledger_path,
        bundles_dir=str(tmp_path / "cache"),
        evaluator=_owned_evaluator(lambda _bundle, context: _passing_outcome(context)),
        arena=ARENA,
        validator_hotkey=VALIDATOR_HOTKEY,
        test_only_allow_local_file_urls=True,
    )
    assert first.weights_pushed and first.weights_confirmed
    assert len(subtensor.set_weights_calls) == 1

    # Make the existing live vector stale, then accept a refresh without applying
    # it yet. The old vector remains confirmed while the refresh is pending.
    subtensor.block += ARENA.settlement.weights_refresh_blocks + 1
    original_set_weights = subtensor.set_weights

    must_not_evaluate = _owned_evaluator(
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("terminal evaluation replayed")
        )
    )

    def accept_without_reveal(**kwargs):
        subtensor.set_weights_calls.append(
            (list(kwargs["uids"]), list(kwargs["weights"]))
        )
        return SimpleNamespace(success=True, message="pending", data={})

    subtensor.set_weights = accept_without_reveal
    refresh = run_pass(
        subtensor,
        WALLET,
        1,
        ledger_path=ledger_path,
        bundles_dir=str(tmp_path / "cache"),
        evaluator=must_not_evaluate,
        arena=ARENA,
        validator_hotkey=VALIDATOR_HOTKEY,
        test_only_allow_local_file_urls=True,
    )
    assert refresh.weights_submitted is True
    assert refresh.weights_pending is True
    assert refresh.weights_confirmed is True
    assert refresh.weights_pushed is False
    assert len(subtensor.set_weights_calls) == 2

    subtensor.block += 1
    restart = run_pass(
        subtensor,
        WALLET,
        1,
        ledger_path=ledger_path,
        bundles_dir=str(tmp_path / "cache"),
        evaluator=must_not_evaluate,
        arena=ARENA,
        validator_hotkey=VALIDATOR_HOTKEY,
        test_only_allow_local_file_urls=True,
    )
    assert restart.weights_pending is True
    assert len(subtensor.set_weights_calls) == 2

    # The same vector is not enough to confirm a refresh; its authoritative
    # last-update block must advance past the durable submit block.
    subtensor._last_update[0] = subtensor.block - 1
    subtensor.set_weights = original_set_weights
    confirmed = run_pass(
        subtensor,
        WALLET,
        1,
        ledger_path=ledger_path,
        bundles_dir=str(tmp_path / "cache"),
        evaluator=must_not_evaluate,
        arena=ARENA,
        validator_hotkey=VALIDATOR_HOTKEY,
        test_only_allow_local_file_urls=True,
    )
    assert confirmed.weights_confirmed is True
    assert confirmed.weights_pending is False
    assert len(subtensor.set_weights_calls) == 2


def _record_pending(
    ledger: Ledger,
    *,
    hotkey: str,
    content_hash: str,
    settlement_round: int,
    score_value: float,
) -> None:
    ledger.bind_validator_hotkey(VALIDATOR_HOTKEY)
    salt = f"salt-{hotkey}"
    ledger.commit(
        hotkey,
        make_commitment(content_hash, hotkey, salt),
        settlement_round,
    )
    ledger.reveal(hotkey, content_hash, salt, settlement_round)
    seed_block = 5
    seed_hash = "0x" + hashlib.sha256(b"seed-block").hexdigest()
    seed_round = seed_block // ARENA.settlement.round_blocks
    prompt_seed = derive_prompt_seed(
        ARENA,
        bundle_hash=content_hash,
        round_id=seed_round,
        block_hash=seed_hash,
    )
    evaluation_id = hashlib.sha256(
        f"evaluation:{hotkey}:{content_hash}:{settlement_round}".encode()
    ).hexdigest()
    qualification_evidence = "sha256:" + hashlib.sha256(
        f"qualification:{evaluation_id}".encode()
    ).hexdigest()
    evaluation_block = settlement_round * ARENA.settlement.round_blocks
    decisions = dict(
        miner_hotkey=hotkey,
        settlement_round_id=settlement_round,
        evaluation_block=evaluation_block,
        passed_quality=True,
        passed_timed_quality=True,
        passed_warmup_quality=True,
        passed_speedup=True,
        confident=True,
        crownable=True,
    )
    recorded = ledger.record_score(
        hotkey,
        content_hash,
        settlement_round,
        score_value,
        0.001,
        True,
        sglang_version=ARENA.sglang_version,
        target="activation.silu_and_mul",
        mode="slot",
        member_slots=("activation.silu_and_mul",),
        arena=ARENA,
        prompt_seed=prompt_seed,
        prompt_engine_version=ARENA.workload.prompt_engine_version,
        prompt_seed_scheme=ARENA.workload.prompt_seed_scheme,
        seed_round_id=seed_round,
        seed_block=seed_block,
        seed_block_hash=seed_hash,
        quality_evidence="trusted evidence",
        host_attestation_sha256=HOST_ATTESTATION,
        validator_hotkey=VALIDATOR_HOTKEY,
        evaluation_id=evaluation_id,
        **decisions,
        qualification_evidence_sha256=qualification_evidence,
    )
    ledger.record_eval(EvalRecord(
        hotkey=hotkey,
        bundle_hash=content_hash,
        slot="activation.silu_and_mul",
        target="activation.silu_and_mul",
        mode="slot",
        member_slots=("activation.silu_and_mul",),
        round_id=settlement_round,
        score=score_value,
        passed=True,
        mean_kl=0.001,
        arena_name=ARENA.name,
        arena_fingerprint=ARENA.fingerprint,
        arena_bracket=ARENA.bracket,
        regime=ARENA.workload.regime,
        sglang_version=ARENA.sglang_version,
        validator_image=ARENA.validator_image,
        referee_source_digest=ARENA.referee_source_digest,
        referee_tree_digest=ARENA.referee_tree_digest,
        model_revision=ARENA.model_revision,
        model_manifest_digest=ARENA.model_manifest_digest,
        model_content_digest=ARENA.model_content_digest,
        host_attestation_sha256=HOST_ATTESTATION,
        prompt_seed=prompt_seed,
        prompt_engine_version=ARENA.workload.prompt_engine_version,
        prompt_seed_scheme=ARENA.workload.prompt_seed_scheme,
        seed_round_id=seed_round,
        seed_block=seed_block,
        seed_block_hash=seed_hash,
        quality_evidence="trusted evidence",
        chain_scope=ledger.chain_scope,
        validator_hotkey=VALIDATOR_HOTKEY,
        evaluation_id=evaluation_id,
        **decisions,
        qualification_evidence_sha256=qualification_evidence,
        development_only=False,
    ))
    ledger.mark_pending_settlement(recorded)


def test_pending_rows_settle_one_reveal_at_a_time_then_rounds_in_order(
    tmp_path, monkeypatch
):
    ledger = Ledger()
    scope = make_chain_scope(genesis_hash="0x" + "1" * 64, netuid=1)
    ledger.bind_chain_scope(scope)
    target_hashes = ["a" * 64, "b" * 64, "c" * 64]
    _record_pending(
        ledger,
        hotkey="z-chain-first",
        content_hash=target_hashes[0],
        settlement_round=1,
        score_value=1.05,
    )
    _record_pending(
        ledger,
        hotkey="best-in-first-round",
        content_hash=target_hashes[1],
        settlement_round=1,
        score_value=1.10,
    )
    _record_pending(
        ledger,
        hotkey="later-improver",
        content_hash=target_hashes[2],
        settlement_round=2,
        score_value=1.13,
    )
    ledger_path = str(tmp_path / "ledger.json")
    ledger.save(ledger_path)

    settled_rounds = []
    real_settle = Ledger.settle_per_target

    def observed_settle(self, round_id, *args, **kwargs):
        settled_rounds.append(round_id)
        return real_settle(self, round_id, *args, **kwargs)

    monkeypatch.setattr(Ledger, "settle_per_target", observed_settle)
    _recover_pending_settlements(
        ledger,
        ledger_path=ledger_path,
        arena=ARENA,
        margin=ARENA.settlement.dethrone_margin,
        host_attestation_verifier=_retained_verifier,
        validator_hotkey=VALIDATOR_HOTKEY,
    )

    reloaded = Ledger.load(ledger_path)
    champion = reloaded.arena_champions[ARENA.bracket][
        "activation.silu_and_mul"
    ]
    assert settled_rounds == [1, 1, 2]
    assert champion.hotkey == "later-improver"
    assert champion.score == 1.13
    assert not reloaded.pending_settlements


def test_same_round_settlement_is_independent_of_pass_batching(tmp_path):
    """A dethrone margin must see the same incumbent in one pass or two.

    With a 2% margin, 1.06 cannot dethrone 1.05. The former batch behavior
    crowned 1.06 when both rows happened to be pending at once, but retained
    1.05 when the exact same reveals arrived across separate passes.
    """

    scope = make_chain_scope(genesis_hash="0x" + "5" * 64, netuid=1)

    together = Ledger()
    together.bind_chain_scope(scope)
    _record_pending(
        together,
        hotkey="z-chain-first",
        content_hash="1" * 64,
        settlement_round=7,
        score_value=1.05,
    )
    _record_pending(
        together,
        hotkey="a-chain-second",
        content_hash="2" * 64,
        settlement_round=7,
        score_value=1.06,
    )
    together_path = str(tmp_path / "together.json")
    together.save(together_path)
    _recover_pending_settlements(
        together,
        ledger_path=together_path,
        arena=ARENA,
        margin=ARENA.settlement.dethrone_margin,
        host_attestation_verifier=_retained_verifier,
        validator_hotkey=VALIDATOR_HOTKEY,
    )

    separate = Ledger()
    separate.bind_chain_scope(scope)
    _record_pending(
        separate,
        hotkey="z-chain-first",
        content_hash="1" * 64,
        settlement_round=7,
        score_value=1.05,
    )
    separate_path = str(tmp_path / "separate.json")
    separate.save(separate_path)
    _recover_pending_settlements(
        separate,
        ledger_path=separate_path,
        arena=ARENA,
        margin=ARENA.settlement.dethrone_margin,
        host_attestation_verifier=_retained_verifier,
        validator_hotkey=VALIDATOR_HOTKEY,
    )
    _record_pending(
        separate,
        hotkey="a-chain-second",
        content_hash="2" * 64,
        settlement_round=7,
        score_value=1.06,
    )
    separate.save(separate_path)
    _recover_pending_settlements(
        separate,
        ledger_path=separate_path,
        arena=ARENA,
        margin=ARENA.settlement.dethrone_margin,
        host_attestation_verifier=_retained_verifier,
        validator_hotkey=VALIDATOR_HOTKEY,
    )

    together_champion = together.arena_champions[ARENA.bracket][
        "activation.silu_and_mul"
    ]
    separate_champion = separate.arena_champions[ARENA.bracket][
        "activation.silu_and_mul"
    ]
    assert together_champion == separate_champion
    # Lexical hotkey order is deliberately opposite chain/commit order.
    assert together_champion.hotkey == "z-chain-first"
    assert together_champion.score == 1.05


def test_standalone_verifier_fault_leaves_pending_durable(tmp_path):
    ledger = Ledger()
    scope = make_chain_scope(genesis_hash="0x" + "2" * 64, netuid=1)
    ledger.bind_chain_scope(scope)
    _record_pending(
        ledger,
        hotkey="miner",
        content_hash="d" * 64,
        settlement_round=3,
        score_value=1.08,
    )
    ledger_path = str(tmp_path / "ledger.json")
    ledger.save(ledger_path)

    class VerifierFault(RuntimeError):
        validator_fault = True

    def unavailable(reference, context):
        raise VerifierFault("retained host store unavailable")

    with pytest.raises(VerifierFault, match="store unavailable"):
        _recover_pending_settlements(
            ledger,
            ledger_path=ledger_path,
            arena=ARENA,
            margin=ARENA.settlement.dethrone_margin,
            host_attestation_verifier=unavailable,
            validator_hotkey=VALIDATOR_HOTKEY,
        )

    durable = Ledger.load(ledger_path)
    assert len(durable.pending_settlements) == 1
    assert not durable.arena_champions.get(ARENA.bracket)


def test_pending_recovery_requires_matching_external_validator(tmp_path):
    ledger = Ledger()
    scope = make_chain_scope(genesis_hash="0x" + "3" * 64, netuid=1)
    ledger.bind_chain_scope(scope)
    _record_pending(
        ledger,
        hotkey="miner",
        content_hash="e" * 64,
        settlement_round=4,
        score_value=1.08,
    )
    ledger_path = str(tmp_path / "ledger.json")
    ledger.save(ledger_path)

    with pytest.raises(
        PendingSettlementError,
        match="failed authoritative crown verification",
    ):
        _recover_pending_settlements(
            ledger,
            ledger_path=ledger_path,
            arena=ARENA,
            margin=ARENA.settlement.dethrone_margin,
            host_attestation_verifier=_retained_verifier,
            validator_hotkey="different-validator",
        )

    durable = Ledger.load(ledger_path)
    assert durable.validator_hotkey == VALIDATOR_HOTKEY
    assert len(durable.pending_settlements) == 1
    assert not durable.arena_champions.get(ARENA.bracket)


@pytest.mark.parametrize("row_kind", ["score", "eval"])
@pytest.mark.parametrize(
    "field,value",
    [
        ("validator_hotkey", "different-validator"),
        ("evaluation_id", "9" * 64),
        ("qualification_evidence_sha256", "sha256:" + "9" * 64),
    ],
)
def test_pending_recovery_rejects_mutated_authority_fields(
    tmp_path, row_kind, field, value,
):
    ledger = Ledger()
    scope = make_chain_scope(genesis_hash="0x" + "4" * 64, netuid=1)
    ledger.bind_chain_scope(scope)
    _record_pending(
        ledger,
        hotkey="miner",
        content_hash="f" * 64,
        settlement_round=5,
        score_value=1.08,
    )
    ledger_path = str(tmp_path / "ledger.json")
    ledger.save(ledger_path)

    if row_kind == "score":
        setattr(ledger.scores[-1], field, value)
    else:
        key, record = next(iter(ledger.evals.items()))
        ledger.evals[key] = replace(record, **{field: value})

    with pytest.raises(
        PendingSettlementError,
        match="authoritative EvalRecord|exact Score/EvalRecord pair",
    ):
        _recover_pending_settlements(
            ledger,
            ledger_path=ledger_path,
            arena=ARENA,
            margin=ARENA.settlement.dethrone_margin,
            host_attestation_verifier=_retained_verifier,
            validator_hotkey=VALIDATOR_HOTKEY,
        )

    # The only on-disk state is the intact pre-mutation pending disposition.
    durable = Ledger.load(ledger_path)
    assert len(durable.pending_settlements) == 1
    assert not durable.arena_champions.get(ARENA.bracket)
