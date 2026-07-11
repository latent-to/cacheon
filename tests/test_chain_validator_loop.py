"""The validator loop end-to-end against a mock subtensor: chain commitments in,
fetch + copy-detection + evaluation + settlement, weights out. No network, no GPU —
the evaluator is stubbed; subprocess evaluators get their own contract tests."""
from __future__ import annotations
import json
import hashlib
import os
import stat
from dataclasses import replace
from types import SimpleNamespace

import pytest

from optima import chain
from optima.chain.fetch import package_bundle
from optima.chain.fetch import FetchTransientError
from optima.chain.payload import encode_payload_for_testing as encode_payload
from optima.chain.validator_loop import (
    EvaluationContext,
    EvalOutcome,
    LedgerLockError,
    QualificationAuthorityError,
    SubmissionPolicyError,
    WeightSafetyError,
    _atomic_write_weights_state,
    _exclusive_ledger_pass,
    _intake_fingerprints,
    _resolve_bundle_competition,
    command_evaluator,
    oci_evaluator,
    run_pass,
    run_validator,
)
from optima.arenas import (
    MINIMAX_M3_B300_TP4_DECODE_V1,
    derive_prompt_seed,
)
from optima.commit_reveal import (
    Ledger,
    LedgerAttestationError,
    RETRY_KIND_INFRASTRUCTURE,
    RETRY_KIND_NO_DECISION,
    RETRY_STATE_AUTOMATIC,
    RETRY_STATE_HELD,
    make_chain_scope,
)
from optima.competition import ResolvedCompetition
from optima.eval.qualification import QualificationReport
from tests.quality_report_helpers import (
    calibrated_test_arena,
    evaluation_report,
    patch_qualification_registry,
)
from optima.device_component import (
    UNTRUSTED_HOST_SYSTEM_TARGET,
    component_crown_rejection as _real_component_crown_rejection,
)
from optima.system_patch import SGLANG_INFERENCE_SYSTEM_V1


ARENA = calibrated_test_arena(MINIMAX_M3_B300_TP4_DECODE_V1)
HOST_ATTESTATION_SHA256 = "sha256:" + "3" * 64
QUALIFICATION_EVIDENCE_SHA256 = "sha256:" + "4" * 64
DIRECT_EVALUATION_ID = "5" * 64
VALIDATOR_WALLET = SimpleNamespace(
    hotkey=SimpleNamespace(ss58_address="val")
)


@pytest.fixture(autouse=True)
def _allow_legacy_test_component_lane(monkeypatch):
    """These loop fixtures test settlement plumbing, not CUDA pointer isolation."""

    monkeypatch.setattr(
        "optima.device_component.component_crown_rejection",
        lambda _manifest: None,
    )
    patch_qualification_registry(monkeypatch, ARENA)


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

class _MockMetagraph:
    def __init__(self, hotkeys, weights=None, last_update=None):
        self.uids = list(range(len(hotkeys)))
        self.hotkeys = list(hotkeys)
        self.last_update = list(last_update or ([0] * len(hotkeys)))
        self.validator_permit = [True] * len(hotkeys)
        self.W = (
            weights
            if weights is not None
            else [[0.0] * len(hotkeys) for _ in hotkeys]
        )


class _MockSubtensor:
    def __init__(self, *, hotkeys, revealed=None, block=100, weights=None):
        self._hotkeys = list(hotkeys)
        self.revealed = dict(revealed or {})  # hotkey -> ((block, data), ...)
        self._block = block
        self._weights = weights
        self._last_update = [0] * len(hotkeys)
        self.set_weights_calls: list[dict] = []

    def metagraph(self, netuid=None):
        return _MockMetagraph(
            self._hotkeys, self._weights, self._last_update
        )

    def weights(self, netuid=None):
        if self._weights is None:
            return []
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

    def get_current_block(self):
        return self._block

    def get_block_hash(self, block):
        return "0x" + hashlib.sha256(f"block:{block}".encode()).hexdigest()

    def get_finalized_block_number(self):
        return self._block

    def get_all_revealed_commitments(self, netuid=None, block=None):
        return {
            hotkey: tuple(
                entry for entry in history
                if block is None or entry[0] <= block
            )[-10:]
            for hotkey, history in self.revealed.items()
        }

    def set_weights(self, *, wallet, netuid, uids, weights, version_key,
                    wait_for_inclusion, wait_for_finalization):
        self.set_weights_calls.append(
            {"uids": uids, "weights": weights, "version_key": version_key})
        size = len(self._hotkeys)
        matrix = (
            [list(row) for row in self._weights]
            if self._weights is not None
            else [[0.0] * size for _ in range(size)]
        )
        matrix[0] = [0.0] * size
        for uid, weight in zip(uids, weights):
            matrix[0][int(uid)] = float(weight)
        self._weights = matrix
        self._last_update[0] = self._block
        return True


def _mini_bundle(root, name, body):
    """A minimal crownable validator-device bundle for chain plumbing tests."""
    b = root / name
    (b / "kernels").mkdir(parents=True)
    (b / "manifest.toml").write_text(
        f'bundle_id = "{name}"\n'
        'abi_version = "optima-op-abi-v0"\n\n'
        "[[ops]]\n"
        'slot = "activation.silu_and_mul"\n'
        'source = "kernels/k.cu"\n'
        'entry = "k"\n'
        'execution_class = "validator_device"\n'
        'device_abi = "activation.silu_and_mul.cuda.v1"\n'
        'dtypes = ["bfloat16"]\n'
    )
    # The evaluator is stubbed in these chain-state tests, so the text need only
    # participate in intake/copy identity; device compilation is covered by the
    # dedicated component-lane tests and B300 smoke.
    (b / "kernels" / "k.cu").write_text(body)
    return b


def _submission(root, name, body):
    """Package a mini-bundle; return (hotkey-agnostic) payload pieces."""
    bundle = _mini_bundle(root / "src", name, body)
    archive, ch = package_bundle(bundle, root / "hosted" / f"{name}.tar.gz")
    return ch, archive.as_uri()


def _atomic_submission(root):
    bundle = root / "src" / "atomic"
    (bundle / "kernels").mkdir(parents=True)
    (bundle / "manifest.toml").write_text(
        'bundle_id = "atomic"\n'
        'abi_version = "optima-op-abi-v0"\n\n'
        "[competition]\n"
        'target = "collective.moe_epilogue.v1"\n'
        'mode = "atomic"\n\n'
        "[[ops]]\n"
        'slot = "collective.moe_finalize_ar_rmsnorm"\n'
        'source = "kernels/k.py"\n'
        'entry = "deep"\n\n'
        "[[ops]]\n"
        'slot = "collective.ar_residual_rmsnorm"\n'
        'source = "kernels/k.py"\n'
        'entry = "shallow"\n'
    )
    (bundle / "kernels" / "k.py").write_text(
        "def deep(*args):\n    return None\n\n"
        "def shallow(*args):\n    return None\n"
    )
    archive, content = package_bundle(
        bundle,
        root / "hosted" / "atomic.tar.gz",
    )
    return content, archive.as_uri()


def _host_system_bundle(root, name, body):
    bundle = root / name
    (bundle / "kernels").mkdir(parents=True)
    (bundle / "manifest.toml").write_text(
        f'bundle_id = "{name}"\n'
        'abi_version = "optima-op-abi-v0"\n\n'
        "[competition]\n"
        f'target = "{UNTRUSTED_HOST_SYSTEM_TARGET}"\n'
        'mode = "system"\n\n'
        "[[ops]]\n"
        'slot = "activation.silu_and_mul"\n'
        'source = "kernels/k.py"\n'
        'entry = "k"\n'
        'execution_class = "untrusted_host"\n'
        'dtypes = ["bfloat16"]\n'
    )
    (bundle / "kernels" / "k.py").write_text(body)
    return bundle


def _host_system_submission(root, name, body):
    bundle = _host_system_bundle(root / "src", name, body)
    archive, content = package_bundle(
        bundle, root / "hosted" / f"{name}.tar.gz"
    )
    return content, archive.as_uri()


def _source_patch_system_bundle(root):
    bundle = root / "source-patch"
    (bundle / "patches").mkdir(parents=True)
    (bundle / "patches" / "runtime.patch").write_text(
        "--- a/sglang/srt/layers/example.py\n"
        "+++ b/sglang/srt/layers/example.py\n"
        "@@ -1 +1 @@\n"
        "-stock = True\n"
        "+stock = False\n"
    )
    (bundle / "manifest.toml").write_text(
        'bundle_id = "source-patch"\n'
        'abi_version = "optima-system-patch-v1"\n\n'
        "[competition]\n"
        f'target = "{SGLANG_INFERENCE_SYSTEM_V1}"\n'
        'mode = "system"\n\n'
        "[system]\n"
        'target = "sglang"\n'
        'region = "inference"\n'
        'patches = ["patches/runtime.patch"]\n'
    )
    return bundle


def _loop_env(tmp_path, revealed, hotkeys):
    st = _MockSubtensor(hotkeys=hotkeys, revealed=revealed, block=400)
    return st, dict(ledger_path=str(tmp_path / "ledger.json"),
                    bundles_dir=str(tmp_path / "cache"), arena=ARENA,
                    validator_hotkey="val",
                    test_only_allow_local_file_urls=True)


def _scope(context):
    return dict(
        arena_name=ARENA.name,
        arena_fingerprint=ARENA.fingerprint,
        arena_bracket=ARENA.bracket,
        regime=ARENA.workload.regime,
        bundle_hash=context.bundle_hash,
        sglang_version=ARENA.sglang_version,
        validator_image=ARENA.validator_image,
        referee_source_digest=ARENA.referee_source_digest,
        referee_tree_digest=ARENA.referee_tree_digest,
        model_revision=ARENA.model_revision,
        model_manifest_digest=ARENA.model_manifest_digest,
        model_content_digest=ARENA.model_content_digest,
        host_attestation_sha256=HOST_ATTESTATION_SHA256,
        chain_scope=context.chain_scope,
        validator_hotkey=context.validator_hotkey,
        evaluation_id=context.evaluation_id,
        miner_hotkey=context.miner_hotkey,
        settlement_round_id=context.settlement_round_id,
        evaluation_block=context.evaluation_block,
        qualification_evidence_sha256=QUALIFICATION_EVIDENCE_SHA256,
        prompt_seed=context.prompt_seed,
        prompt_engine_version=ARENA.workload.prompt_engine_version,
        prompt_seed_scheme=ARENA.workload.prompt_seed_scheme,
        seed_round_id=context.round_id,
        seed_block=context.block,
        seed_block_hash=context.block_hash,
        quality_evidence="controller paired top-k evidence: pass",
        passed_timed_quality=True,
        passed_warmup_quality=True,
        passed_speedup=True,
    )


def _validator_owned(fn):
    """Mark a test double as the trusted OCI controller capability."""
    setattr(fn, "validator_owned_oci", True)
    setattr(
        fn,
        "host_attestation_verifier",
        lambda reference, context: SimpleNamespace(
            qualification_evidence_sha256=(
                context["qualification_evidence_sha256"]
            )
        )
        if (
            reference == HOST_ATTESTATION_SHA256
            and context["arena_fingerprint"] == ARENA.fingerprint
        )
        else None,
    )
    return fn


@_validator_owned
def _pass_all(bundle_dir, context):
    return EvalOutcome(
        True,
        1.05,
        target="activation.silu_and_mul",
        mode="slot",
        member_slots=("activation.silu_and_mul",),
        crownable=True,
        **_scope(context),
    )


@_validator_owned
def _pass_system(bundle_dir, context):
    return EvalOutcome(
        True,
        1.05,
        target=UNTRUSTED_HOST_SYSTEM_TARGET,
        mode="system",
        member_slots=(),
        crownable=True,
        **_scope(context),
    )


def _context(bundle_hash: str) -> EvaluationContext:
    block = 400
    block_hash = "0x" + hashlib.sha256(f"block:{block}".encode()).hexdigest()
    round_id = block // ARENA.settlement.round_blocks
    return EvaluationContext(
        arena=ARENA,
        bundle_hash=bundle_hash,
        round_id=round_id,
        block=block,
        block_hash=block_hash,
        prompt_seed=derive_prompt_seed(
            ARENA,
            bundle_hash=bundle_hash,
            round_id=round_id,
            block_hash=block_hash,
        ),
        chain_scope=make_chain_scope(
            genesis_hash=(
                "0x" + hashlib.sha256(b"block:0").hexdigest()
            ),
            netuid=1,
            scheme=ARENA.settlement.chain_scope_scheme,
        ),
        validator_hotkey="val",
        evaluation_id=DIRECT_EVALUATION_ID,
        miner_hotkey="miner1",
        settlement_round_id=round_id,
        evaluation_block=block,
    )


def _qualification(context, *, target="moe.fused_experts", crownable=True):
    report = evaluation_report(
        ARENA, candidate_rate=107.0 if crownable else 100.0
    )
    return QualificationReport.from_evaluation(
        report,
        competition=ResolvedCompetition(
            target=target, mode="slot", members=(target,), crownable=True
        ),
        arena=ARENA,
        bundle_hash=context.bundle_hash,
        prompt_seed=context.prompt_seed,
        seed_round_id=context.round_id,
        seed_block=context.block,
        seed_block_hash=context.block_hash,
        chain_scope=context.chain_scope,
        validator_hotkey=context.validator_hotkey,
        evaluation_id=context.evaluation_id,
        miner_hotkey=context.miner_hotkey,
        settlement_round_id=context.settlement_round_id,
        evaluation_block=context.evaluation_block,
        host_attestation_sha256=HOST_ATTESTATION_SHA256,
    ).to_dict()


# --------------------------------------------------------------------------- #
# the referee cycle
# --------------------------------------------------------------------------- #

def test_full_pass_crowns_and_pushes_weights(tmp_path):
    ch, url = _submission(tmp_path, "m1", "def k(x):\n    return x\n")
    st, env = _loop_env(tmp_path, {"miner1": ((5, encode_payload(ch, url)),)},
                        hotkeys=["val", "miner1"])
    res = run_pass(st, VALIDATOR_WALLET, 1, evaluator=_pass_all, **env)
    assert res.seen == 1 and res.new == [ch]
    assert res.evaluated == {ch: True}
    assert res.weights == {"miner1": 1.0}
    assert res.weights_pushed and len(st.set_weights_calls) == 1
    assert st.set_weights_calls[0]["uids"] == [1]

    # ledger state: score + audit record + champion
    led = Ledger.load(env["ledger_path"])
    assert led.is_known("miner1", ch, arena_bracket=ARENA.bracket)
    scoped = led.arena_champions[ARENA.bracket]
    assert scoped and led.current_weights(
        arena=ARENA,
        host_attestation_verifier=_pass_all.host_attestation_verifier,
        validator_hotkey="val",
    ) == {"miner1": 1.0}


def test_plain_evaluator_cannot_mint_registered_arena_crown(tmp_path):
    ch, url = _submission(tmp_path, "plain", "def k(x):\n    return x\n")
    st, env = _loop_env(
        tmp_path,
        {"miner1": ((5, encode_payload(ch, url)),)},
        hotkeys=["val", "miner1"],
    )

    def plain(bundle_dir, context):
        return EvalOutcome(
            True,
            1.50,
            target="activation.silu_and_mul",
            mode="slot",
            member_slots=("activation.silu_and_mul",),
            crownable=True,
            **_scope(context),
        )

    with pytest.raises(QualificationAuthorityError, match="validator-owned OCI"):
        run_pass(st, VALIDATOR_WALLET, 1, evaluator=plain, **env)
    ledger = Ledger.load(env["ledger_path"])
    assert not ledger.scores
    assert ledger.retry_for("miner1", ch, arena_bracket=ARENA.bracket) is None


def test_development_eval_never_suppresses_later_production_qualification(tmp_path):
    ch, url = _submission(tmp_path, "dev-first", "def k(x):\n    return x\n")
    st, env = _loop_env(
        tmp_path,
        {"miner1": ((5, encode_payload(ch, url)),)},
        hotkeys=["val", "miner1"],
    )

    def development(bundle_dir, context):
        scope = _scope(context)
        scope.update(
            passed_timed_quality=True,
            passed_warmup_quality=True,
            passed_speedup=False,
        )
        return EvalOutcome(
            True,
            0.0,
            target="activation.silu_and_mul",
            mode="slot",
            member_slots=("activation.silu_and_mul",),
            crownable=False,
            **scope,
        )

    first = run_pass(st, VALIDATOR_WALLET, 1, evaluator=development, **env)
    assert first.evaluated == {ch: True} and first.weights == {}
    dev_record = Ledger.load(env["ledger_path"]).eval_for(
        "miner1", ch, arena_bracket=ARENA.bracket
    )
    assert dev_record is not None and dev_record.development_only
    dev_ledger = Ledger.load(env["ledger_path"])
    assert dev_ledger.is_known(
        "miner1", ch, arena_bracket=ARENA.bracket,
        require_authoritative=False,
    )
    assert not dev_ledger.is_known(
        "miner1", ch, arena_bracket=ARENA.bracket,
        require_authoritative=True,
    )

    calls = {"count": 0}

    @_validator_owned
    def production(bundle_dir, context):
        calls["count"] += 1
        return _pass_all(bundle_dir, context)

    second = run_pass(st, VALIDATOR_WALLET, 1, evaluator=production, **env)
    assert calls["count"] == 1
    assert second.weights == {"miner1": 1.0}
    assert not Ledger.load(env["ledger_path"]).eval_for(
        "miner1", ch, arena_bracket=ARENA.bracket
    ).development_only


def test_full_pass_crowns_atomic_bundle_once_under_canonical_target(tmp_path, monkeypatch):
    # This test isolates atomic ledger semantics; the current deep ABIs have not
    # yet been migrated to validator-owned device launchers.
    monkeypatch.setattr(
        "optima.device_component.component_crown_rejection", lambda _manifest: None
    )
    ch, url = _atomic_submission(tmp_path)
    st, env = _loop_env(
        tmp_path,
        {"miner1": ((5, encode_payload(ch, url)),)},
        hotkeys=["val", "miner1"],
    )
    members = (
        "collective.ar_residual_rmsnorm",
        "collective.moe_finalize_ar_rmsnorm",
    )

    res = run_pass(
        st,
        VALIDATOR_WALLET,
        1,
        evaluator=_validator_owned(
            lambda p, context: EvalOutcome(
                True,
                1.08,
                target="collective.moe_epilogue.v1",
                mode="atomic",
                member_slots=members,
                crownable=True,
                **_scope(context),
            )
        ),
        **env,
    )

    assert res.weights == {"miner1": 1.0}
    led = Ledger.load(env["ledger_path"])
    scoped = led.arena_champions[ARENA.bracket]
    assert set(scoped) == {"collective.moe_epilogue.v1"}
    assert not (set(members) & set(scoped))
    assert led.scores[0].slot == ""
    assert led.scores[0].member_slots == members


def test_chain_rejects_atomic_report_with_miner_manifest_member_order(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "optima.device_component.component_crown_rejection", lambda _manifest: None
    )
    ch, url = _atomic_submission(tmp_path)
    st, env = _loop_env(
        tmp_path,
        {"miner1": ((5, encode_payload(ch, url)),)},
        hotkeys=["val", "miner1"],
    )
    manifest_order = (
        "collective.moe_finalize_ar_rmsnorm",
        "collective.ar_residual_rmsnorm",
    )

    res = run_pass(
        st,
        VALIDATOR_WALLET,
        1,
        evaluator=lambda p, context: EvalOutcome(
            True,
            1.50,
            target="collective.moe_epilogue.v1",
            mode="atomic",
            member_slots=manifest_order,
            crownable=True,
            **_scope(context),
        ),
        **env,
    )

    assert ch in res.rejected
    assert "identity mismatch" in res.rejected[ch]
    assert not res.weights


def test_second_pass_is_idempotent(tmp_path):
    ch, url = _submission(tmp_path, "m1", "def k(x):\n    return x\n")
    st, env = _loop_env(tmp_path, {"miner1": ((5, encode_payload(ch, url)),)},
                        hotkeys=["val", "miner1"])
    calls = {"n": 0}

    @_validator_owned
    def counting(bundle_dir, context):
        calls["n"] += 1
        return _pass_all(bundle_dir, context)

    run_pass(st, VALIDATOR_WALLET, 1, evaluator=counting, **env)
    res2 = run_pass(st, VALIDATOR_WALLET, 1, evaluator=counting, **env)
    assert calls["n"] == 1          # not re-evaluated
    assert res2.new == []           # nothing new
    assert len(st.set_weights_calls) == 1  # unchanged weights not re-pushed


def test_invalidated_champion_never_silently_leaves_old_weights_active(tmp_path):
    ch, url = _submission(tmp_path, "weighted", "def k(x):\n    return x\n")
    st, env = _loop_env(
        tmp_path,
        {"miner1": ((5, encode_payload(ch, url)),)},
        hotkeys=["val", "miner1"],
    )
    run_pass(st, VALIDATOR_WALLET, 1, evaluator=_pass_all, **env)
    assert len(st.set_weights_calls) == 1

    ledger = Ledger.load(env["ledger_path"])
    key, record = next(iter(ledger.evals.items()))
    ledger.evals[key] = replace(
        record, host_attestation_sha256="sha256:" + "4" * 64
    )
    ledger.save(env["ledger_path"])

    with pytest.raises(LedgerAttestationError, match="refusing to redistribute"):
        run_pass(st, VALIDATOR_WALLET, 1, evaluator=_pass_all, **env)
    assert len(st.set_weights_calls) == 1


def test_live_on_chain_weights_are_authority_when_local_state_is_missing(tmp_path):
    st = _MockSubtensor(
        hotkeys=["val", "miner1"],
        revealed={},
        block=400,
        weights=[[0.0, 1.0], [0.0, 0.0]],
    )
    env = dict(
        ledger_path=str(tmp_path / "ledger.json"),
        bundles_dir=str(tmp_path / "cache"),
        arena=ARENA,
    )
    with pytest.raises(WeightSafetyError, match="stale emissions"):
        run_pass(
            st,
            VALIDATOR_WALLET,
            1,
            evaluator=_pass_all,
            validator_hotkey="val",
            **env,
        )


def test_weight_capable_pass_refuses_missing_validator_hotkey(tmp_path):
    st, env = _loop_env(tmp_path, {}, hotkeys=["val"])
    env.pop("validator_hotkey")
    with pytest.raises(
        chain.ChainWeightStateError, match="does not expose an exact hotkey"
    ):
        run_pass(st, object(), 1, evaluator=_pass_all, **env)


def test_weight_reconciliation_hotkey_must_match_signing_wallet(tmp_path):
    st, env = _loop_env(tmp_path, {}, hotkeys=["val", "other"])
    env["validator_hotkey"] = "other"
    with pytest.raises(
        chain.ChainWeightStateError, match="differs from reconciliation hotkey"
    ):
        run_pass(st, VALIDATOR_WALLET, 1, evaluator=_pass_all, **env)


def test_weight_state_publication_fsyncs_file_and_directory(tmp_path, monkeypatch):
    calls = []
    real_fsync = os.fsync

    def observed(fd):
        calls.append(os.fstat(fd).st_mode)
        return real_fsync(fd)

    monkeypatch.setattr(os, "fsync", observed)
    path = tmp_path / "weights.json"
    _atomic_write_weights_state(path, {"weights": {"miner": 1.0}})
    assert len(calls) == 2
    assert stat.S_ISREG(calls[0]) and stat.S_ISDIR(calls[1])
    assert not list(tmp_path.glob(".weights.json.tmp.*"))

def test_terminal_eval_cleans_contradictory_retry_debris(tmp_path):
    ch, url = _submission(tmp_path, "known-with-retry", "def k(x):\n    return x\n")
    st, env = _loop_env(
        tmp_path,
        {"miner1": ((5, encode_payload(ch, url)),)},
        hotkeys=["val", "miner1"],
    )
    run_pass(st, VALIDATOR_WALLET, 1, evaluator=_pass_all, **env)

    ledger = Ledger.load(env["ledger_path"])
    ledger.record_retry(
        hotkey="miner1",
        bundle_hash=ch,
        arena_bracket=ARENA.bracket,
        kind=RETRY_KIND_NO_DECISION,
        current_block=st._block,
        reason="simulated interrupted old build",
        base_backoff_blocks=ARENA.settlement.retry_backoff_blocks,
        max_backoff_blocks=ARENA.settlement.retry_max_backoff_blocks,
        max_automatic_infrastructure_attempts=(
            ARENA.settlement.retry_max_automatic_infrastructure_attempts
        ),
        max_automatic_no_decision_attempts=(
            ARENA.settlement.retry_max_automatic_no_decision_attempts
        ),
        max_total_attempts=ARENA.settlement.retry_max_total_attempts,
    )
    ledger.save(env["ledger_path"])

    calls = {"n": 0}

    def must_not_run(bundle_dir, context):
        calls["n"] += 1
        return _pass_all(bundle_dir, context)

    run_pass(
        st,
        VALIDATOR_WALLET,
        1,
        evaluator=must_not_run,
        host_attestation_verifier=_pass_all.host_attestation_verifier,
        **env,
    )
    assert calls["n"] == 0
    ledger = Ledger.load(env["ledger_path"])
    assert ledger.retry_for("miner1", ch, arena_bracket=ARENA.bracket) is None


def test_pre_sidecar_winner_is_requalified_and_replaces_legacy_evidence(tmp_path):
    ch, url = _submission(tmp_path, "legacy-host", "def k(x):\n    return x\n")
    st, env = _loop_env(
        tmp_path,
        {"miner1": ((5, encode_payload(ch, url)),)},
        hotkeys=["val", "miner1"],
    )
    calls = {"count": 0}

    @_validator_owned
    def counted(bundle_dir, context):
        calls["count"] += 1
        return _pass_all(bundle_dir, context)

    run_pass(st, VALIDATOR_WALLET, 1, evaluator=counted, **env)
    ledger = Ledger.load(env["ledger_path"])
    ledger.scores[-1].host_attestation_sha256 = ""
    key, record = next(iter(ledger.evals.items()))
    ledger.evals[key] = replace(record, host_attestation_sha256="")
    champion = next(iter(ledger.arena_champions[ARENA.bracket].values()))
    champion.host_attestation_sha256 = ""
    ledger.save(env["ledger_path"])

    rerun = run_pass(st, VALIDATOR_WALLET, 1, evaluator=counted, **env)
    assert calls["count"] == 2
    assert rerun.weights == {"miner1": 1.0}
    migrated = Ledger.load(env["ledger_path"])
    assert migrated.eval_for(
        "miner1", ch, arena_bracket=ARENA.bracket
    ).host_attestation_sha256 == HOST_ATTESTATION_SHA256
    assert migrated.current_weights(
        arena=ARENA,
        host_attestation_verifier=counted.host_attestation_verifier,
        validator_hotkey="val",
    ) == {"miner1": 1.0}


def test_no_decision_is_persistently_deferred_then_retried(tmp_path):
    ch, url = _submission(tmp_path, "retry", "def k(x):\n    return x\n")
    st, env = _loop_env(
        tmp_path,
        {"miner1": ((5, encode_payload(ch, url)),)},
        hotkeys=["val", "miner1"],
    )
    calls = {"n": 0}

    @_validator_owned
    def noisy_then_green(bundle_dir, context):
        calls["n"] += 1
        if calls["n"] == 1:
            return EvalOutcome(
                True,
                0.0,
                target="activation.silu_and_mul",
                mode="slot",
                member_slots=("activation.silu_and_mul",),
                crownable=False,
                confident=False,
                detail="B/B' exceeded the arena noise limit",
                **_scope(context),
            )
        return _pass_all(bundle_dir, context)

    first = run_pass(st, VALIDATOR_WALLET, 1, evaluator=noisy_then_green, **env)
    assert ch in first.deferred and ch not in first.evaluated
    ledger = Ledger.load(env["ledger_path"])
    retry = ledger.retry_for("miner1", ch, arena_bracket=ARENA.bracket)
    assert retry is not None and retry.attempts == 1
    assert retry.kind == RETRY_KIND_NO_DECISION
    assert retry.state == RETRY_STATE_AUTOMATIC
    assert retry.next_block == 400 + ARENA.settlement.retry_backoff_blocks
    assert not ledger.is_known("miner1", ch, arena_bracket=ARENA.bracket)

    second = run_pass(st, VALIDATOR_WALLET, 1, evaluator=noisy_then_green, **env)
    assert calls["n"] == 1
    assert "retry backoff" in second.deferred[ch]

    st._block = retry.next_block
    third = run_pass(st, VALIDATOR_WALLET, 1, evaluator=noisy_then_green, **env)
    assert calls["n"] == 2 and third.evaluated == {ch: True}
    ledger = Ledger.load(env["ledger_path"])
    assert ledger.retry_for("miner1", ch, arena_bracket=ARENA.bracket) is None
    assert ledger.current_weights(
        arena=ARENA,
        host_attestation_verifier=_pass_all.host_attestation_verifier,
        validator_hotkey="val",
    ) == {"miner1": 1.0}


def test_transient_bundle_fetch_uses_bounded_infrastructure_retry(
    tmp_path, monkeypatch
):
    ch, url = _submission(tmp_path, "fetch-retry", "def k(x):\n    return x\n")
    st, env = _loop_env(
        tmp_path,
        {"miner1": ((5, encode_payload(ch, url)),)},
        hotkeys=["val", "miner1"],
    )
    from optima.chain.fetch import fetch_bundle_from_local_file_for_testing

    calls = {"fetch": 0, "eval": 0}

    def flaky_fetch(*args, **kwargs):
        calls["fetch"] += 1
        if calls["fetch"] == 1:
            raise FetchTransientError("temporary archive host outage")
        return fetch_bundle_from_local_file_for_testing(*args, **kwargs)

    @_validator_owned
    def tracking(bundle_dir, context):
        calls["eval"] += 1
        return _pass_all(bundle_dir, context)

    monkeypatch.setattr(
        "optima.chain.validator_loop.fetch_bundle_from_local_file_for_testing",
        flaky_fetch,
    )
    first = run_pass(st, VALIDATOR_WALLET, 1, evaluator=tracking, **env)
    assert ch in first.deferred and ch not in first.rejected
    retry = Ledger.load(env["ledger_path"]).retry_for(
        "miner1", ch, arena_bracket=ARENA.bracket
    )
    assert retry is not None and retry.kind == RETRY_KIND_INFRASTRUCTURE
    assert calls == {"fetch": 1, "eval": 0}

    run_pass(st, VALIDATOR_WALLET, 1, evaluator=tracking, **env)
    assert calls == {"fetch": 1, "eval": 0}
    st._block = retry.next_block
    recovered = run_pass(st, VALIDATOR_WALLET, 1, evaluator=tracking, **env)
    assert recovered.evaluated == {ch: True}
    assert calls == {"fetch": 2, "eval": 1}


def test_no_decision_enters_its_own_bounded_hold(tmp_path):
    ch, url = _submission(tmp_path, "always-noisy", "def k(x):\n    return x\n")
    st, env = _loop_env(
        tmp_path,
        {"miner1": ((5, encode_payload(ch, url)),)},
        hotkeys=["val", "miner1"],
    )
    calls = {"n": 0}

    def always_noisy(bundle_dir, context):
        calls["n"] += 1
        return EvalOutcome(
            True,
            0.0,
            target="activation.silu_and_mul",
            mode="slot",
            member_slots=("activation.silu_and_mul",),
            crownable=False,
            confident=False,
            detail="unstable B/B' bracket",
            **_scope(context),
        )

    rounds = ARENA.settlement.retry_max_automatic_no_decision_attempts
    for expected_attempts in range(1, rounds + 1):
        result = run_pass(st, VALIDATOR_WALLET, 1, evaluator=always_noisy, **env)
        ledger = Ledger.load(env["ledger_path"])
        retry = ledger.retry_for("miner1", ch, arena_bracket=ARENA.bracket)
        assert retry is not None
        assert retry.attempts == expected_attempts
        assert retry.kind == RETRY_KIND_NO_DECISION
        assert retry.infrastructure_attempts == 0
        assert retry.no_decision_attempts == expected_attempts
        if expected_attempts < rounds:
            assert retry.state == RETRY_STATE_AUTOMATIC
            assert ch not in result.held
            st._block = retry.next_block
        else:
            assert retry.state == RETRY_STATE_HELD
            assert ch in result.held

    held = run_pass(st, VALIDATOR_WALLET, 1, evaluator=always_noisy, **env)
    assert ch in held.held
    assert calls["n"] == rounds

    assert calls["n"] == rounds
    assert not ledger.is_known("miner1", ch, arena_bracket=ARENA.bracket)


def test_infrastructure_retry_enters_hold_and_requires_trusted_release(tmp_path):
    ch, url = _submission(tmp_path, "infra", "def k(x):\n    return x\n")
    st, env = _loop_env(
        tmp_path,
        {"miner1": ((5, encode_payload(ch, url)),)},
        hotkeys=["val", "miner1"],
    )
    calls = {"n": 0}
    healthy = {"value": False}

    @_validator_owned
    def infrastructure_then_green(bundle_dir, context):
        calls["n"] += 1
        if healthy["value"]:
            return _pass_all(bundle_dir, context)
        scope = _scope(context)
        scope.update(
            passed_timed_quality=False,
            passed_warmup_quality=False,
            passed_speedup=False,
        )
        return EvalOutcome(
            False,
            0.0,
            target="activation.silu_and_mul",
            mode="slot",
            member_slots=("activation.silu_and_mul",),
            crownable=False,
            retryable=True,
            detail="validator launch infrastructure failed",
            **scope,
        )

    limit = ARENA.settlement.retry_max_automatic_infrastructure_attempts
    for expected_attempts in range(1, limit + 1):
        result = run_pass(
            st, VALIDATOR_WALLET, 1, evaluator=infrastructure_then_green, **env
        )
        ledger = Ledger.load(env["ledger_path"])
        retry = ledger.retry_for("miner1", ch, arena_bracket=ARENA.bracket)
        assert retry is not None
        assert retry.attempts == expected_attempts
        assert retry.kind == RETRY_KIND_INFRASTRUCTURE
        if expected_attempts < limit:
            assert retry.state == RETRY_STATE_AUTOMATIC
            assert ch not in result.held
            st._block = retry.next_block
        else:
            assert retry.state == RETRY_STATE_HELD
            assert ch in result.held

    st._block += ARENA.settlement.retry_max_backoff_blocks * 10
    held = run_pass(st, VALIDATOR_WALLET, 1, evaluator=infrastructure_then_green, **env)
    assert calls["n"] == limit
    assert ch in held.held and ch not in held.new
    assert not ledger.is_known("miner1", ch, arena_bracket=ARENA.bracket)

    released = ledger.release_held_retry(
        "miner1",
        ch,
        arena_bracket=ARENA.bracket,
        chain_scope=ledger.chain_scope,
    )
    assert released.state == RETRY_STATE_HELD
    ledger.save(env["ledger_path"])
    healthy["value"] = True
    passed = run_pass(st, VALIDATOR_WALLET, 1, evaluator=infrastructure_then_green, **env)
    assert calls["n"] == limit + 1
    assert passed.evaluated == {ch: True}


def test_run_pass_persists_lease_before_invoking_evaluator(tmp_path):
    ch, url = _submission(tmp_path, "leased", "def k(x):\n    return x\n")
    st, env = _loop_env(
        tmp_path,
        {"miner1": ((5, encode_payload(ch, url)),)},
        hotkeys=["val", "miner1"],
    )

    @_validator_owned
    def inspect_durable_lease(bundle_dir, context):
        durable = Ledger.load(env["ledger_path"])
        lease = durable.retry_for("miner1", ch, arena_bracket=ARENA.bracket)
        assert lease is not None
        assert lease.state == "in_progress"
        assert lease.attempts == 1
        assert lease.lease_id
        assert context.evaluation_id == lease.lease_id
        assert context.chain_scope == durable.chain_scope
        assert context.validator_hotkey == "val"
        return _pass_all(bundle_dir, context)

    result = run_pass(st, VALIDATOR_WALLET, 1, evaluator=inspect_durable_lease, **env)
    assert result.evaluated[ch]
    assert Ledger.load(env["ledger_path"]).retry_for(
        "miner1", ch, arena_bracket=ARENA.bracket
    ) is None


def test_crashed_lease_enters_durable_validator_hold_without_gpu_replay(tmp_path):
    ch, url = _submission(tmp_path, "crashed-lease", "def k(x):\n    return x\n")
    st, env = _loop_env(
        tmp_path,
        {"miner1": ((5, encode_payload(ch, url)),)},
        hotkeys=["val", "miner1"],
    )
    calls = {"n": 0}

    class SimulatedProcessDeath(BaseException):
        pass

    def crash(bundle_dir, context):
        calls["n"] += 1
        raise SimulatedProcessDeath("simulated validator death after lease persistence")

    with pytest.raises(SimulatedProcessDeath, match="simulated validator death"):
        run_pass(st, VALIDATOR_WALLET, 1, evaluator=crash, **env)
    crashed = Ledger.load(env["ledger_path"]).retry_for(
        "miner1", ch, arena_bracket=ARENA.bracket
    )
    assert crashed is not None and crashed.state == "in_progress"
    assert crashed.attempts == 1

    recovered = run_pass(st, VALIDATOR_WALLET, 1, evaluator=crash, **env)
    assert calls["n"] == 1
    assert ch in recovered.deferred
    held_ledger = Ledger.load(env["ledger_path"])
    assert held_ledger.retry_for(
        "miner1", ch, arena_bracket=ARENA.bracket
    ) is None
    hold = held_ledger.validator_fault_for(
        "miner1", ch, arena_bracket=ARENA.bracket
    )
    assert hold is not None and "abandoned in-progress" in hold.reason

    still_held = run_pass(st, VALIDATOR_WALLET, 1, evaluator=_pass_all, **env)
    assert ch in still_held.held and calls["n"] == 1
    held_ledger.release_validator_fault(
        "miner1",
        ch,
        arena_bracket=ARENA.bracket,
        chain_scope=held_ledger.chain_scope,
    )
    held_ledger.save(env["ledger_path"])
    passed = run_pass(st, VALIDATOR_WALLET, 1, evaluator=_pass_all, **env)
    assert passed.evaluated[ch]
    assert Ledger.load(env["ledger_path"]).retry_for(
        "miner1", ch, arena_bracket=ARENA.bracket
    ) is None


def test_untyped_evaluator_exception_is_validator_fault_not_miner_retry(tmp_path):
    ch, url = _submission(tmp_path, "untyped", "def k(x):\n    return x\n")
    st, env = _loop_env(
        tmp_path,
        {"miner1": ((5, encode_payload(ch, url)),)},
        hotkeys=["val", "miner1"],
    )

    def broken_controller(bundle_dir, context):
        raise TypeError("controller adapter bug")

    with pytest.raises(QualificationAuthorityError, match="untyped TypeError"):
        run_pass(st, VALIDATOR_WALLET, 1, evaluator=broken_controller, **env)
    faulted = Ledger.load(env["ledger_path"])
    assert faulted.retry_for(
        "miner1", ch, arena_bracket=ARENA.bracket
    ) is None
    assert faulted.validator_fault_for(
        "miner1", ch, arena_bracket=ARENA.bracket
    ) is not None


@pytest.mark.parametrize("malformed", [None, float("nan")])
def test_malformed_evaluator_result_enters_durable_validator_hold(
    tmp_path, malformed
):
    ch, url = _submission(tmp_path, "malformed-result", "def k(x):\n    return x\n")
    st, env = _loop_env(
        tmp_path,
        {"miner1": ((5, encode_payload(ch, url)),)},
        hotkeys=["val", "miner1"],
    )

    def broken_controller(bundle_dir, context):
        if malformed is None:
            return None
        return EvalOutcome(
            True,
            malformed,
            target="activation.silu_and_mul",
            mode="slot",
            member_slots=("activation.silu_and_mul",),
            **_scope(context),
        )

    with pytest.raises(QualificationAuthorityError, match="non-EvalOutcome|malformed"):
        run_pass(st, VALIDATOR_WALLET, 1, evaluator=broken_controller, **env)
    ledger = Ledger.load(env["ledger_path"])
    assert ledger.validator_fault_for(
        "miner1", ch, arena_bracket=ARENA.bracket
    ) is not None


def test_known_validator_fault_clears_miner_lease_and_aborts(tmp_path):
    ch, url = _submission(tmp_path, "validator-fault", "def k(x):\n    return x\n")
    st, env = _loop_env(
        tmp_path,
        {"miner1": ((5, encode_payload(ch, url)),)},
        hotkeys=["val", "miner1"],
    )

    class KnownValidatorFault(RuntimeError):
        validator_fault = True

    def fail_controller(bundle_dir, context):
        raise KnownValidatorFault("bad trusted source release")

    with pytest.raises(KnownValidatorFault):
        run_pass(st, VALIDATOR_WALLET, 1, evaluator=fail_controller, **env)
    assert Ledger.load(env["ledger_path"]).retry_for(
        "miner1", ch, arena_bracket=ARENA.bracket
    ) is None


def test_copy_is_demoted_not_evaluated(tmp_path):
    body = "def k(x):\n    return x + 1\n"
    ch, url = _submission(tmp_path, "orig", body)
    # the copycat commits the SAME content at a LATER block
    revealed = {
        "author": ((5, encode_payload(ch, url)),),
        "copycat": ((9, encode_payload(ch, url)),),
    }
    st, env = _loop_env(tmp_path, revealed, hotkeys=["val", "author", "copycat"])
    evaluated = []

    @_validator_owned
    def tracking(bundle_dir, context):
        evaluated.append(bundle_dir)
        return _pass_all(bundle_dir, context)

    res = run_pass(st, VALIDATOR_WALLET, 1, evaluator=tracking, **env)
    assert len(evaluated) == 1      # only the original ran
    assert res.copies == [ch]
    assert res.weights == {"author": 1.0}
    led = Ledger.load(env["ledger_path"])
    assert led.eval_for("copycat", ch, arena_bracket=ARENA.bracket).dq_reason == "copy"


def test_reveal_history_preserves_priority_across_validator_outage(tmp_path):
    """Alice-X, Bob-X, Alice-Y must not become Bob-X, Alice-Y after downtime."""

    hash_x, url_x = _submission(
        tmp_path, "alice-x", "def k(x):\n    return x + 1\n"
    )
    hash_y, url_y = _submission(
        tmp_path, "alice-y", "def k(x):\n    return x + 2\n"
    )
    revealed = {
        "alice": (
            (5, encode_payload(hash_x, url_x)),
            (9, encode_payload(hash_y, url_y)),
        ),
        "bob": ((7, encode_payload(hash_x, url_x)),),
    }
    st, env = _loop_env(
        tmp_path, revealed, hotkeys=["val", "alice", "bob"]
    )
    evaluated: list[tuple[str, str]] = []

    @_validator_owned
    def tracking(bundle_dir, context):
        evaluated.append((context.miner_hotkey, context.bundle_hash))
        return _pass_all(bundle_dir, context)

    result = run_pass(st, VALIDATOR_WALLET, 1, evaluator=tracking, **env)

    assert evaluated == [("alice", hash_x), ("alice", hash_y)]
    assert result.copies == [hash_x]
    ledger = Ledger.load(env["ledger_path"])
    assert ledger.eval_for(
        "bob", hash_x, arena_bracket=ARENA.bracket
    ).dq_reason == "copy"


def test_unfinalized_reveal_never_fetches_or_enters_submission_ledger(
    tmp_path, monkeypatch
):
    content, url = _submission(
        tmp_path, "unfinalized", "def k(x):\n    return x\n"
    )
    st, env = _loop_env(
        tmp_path,
        {"miner1": ((400, encode_payload(content, url)),)},
        hotkeys=["val", "miner1"],
    )
    st.get_finalized_block_number = lambda: 399
    calls = {"fetch": 0, "fingerprint": 0}

    def forbidden_fetch(*args, **kwargs):
        calls["fetch"] += 1
        raise AssertionError("unfinalized reveal reached transport")

    def forbidden_fingerprint(*args, **kwargs):
        calls["fingerprint"] += 1
        raise AssertionError("unfinalized reveal reached intake")

    monkeypatch.setattr(
        "optima.chain.validator_loop.fetch_bundle_from_local_file_for_testing",
        forbidden_fetch,
    )
    monkeypatch.setattr(
        "optima.chain.validator_loop._intake_fingerprints",
        forbidden_fingerprint,
    )

    result = run_pass(st, VALIDATOR_WALLET, 1, evaluator=_pass_all, **env)

    assert result.seen == 0 and content not in result.deferred
    assert calls == {"fetch": 0, "fingerprint": 0}
    ledger = Ledger.load(env["ledger_path"])
    assert not ledger.commitments and not ledger.reveals and not ledger.evals


def test_whole_pass_lock_rejects_a_second_local_owner(tmp_path):
    ledger_path = tmp_path / "ledger.json"
    with _exclusive_ledger_pass(ledger_path):
        with pytest.raises(LedgerLockError, match="another validator process"):
            with _exclusive_ledger_pass(ledger_path):
                raise AssertionError("second owner entered the protected pass")


def test_whole_pass_lock_rejects_shared_writable_parent(tmp_path):
    shared = tmp_path / "shared"
    shared.mkdir()
    shared.chmod(0o777)
    with pytest.raises(LedgerLockError, match="owner-controlled"):
        with _exclusive_ledger_pass(shared / "ledger.json"):
            raise AssertionError("unsafe ledger directory entered protected pass")


def test_reformatted_copy_is_demoted_by_fingerprint(tmp_path):
    ch1, url1 = _submission(
        tmp_path, "orig", '__global__ void k(){int x=1;}\n'
    )
    # different bytes (comment + spacing) -> different content hash, same normalized code
    ch2, url2 = _submission(
        tmp_path, "theft",
        '// totally my own work\n__global__   void k(){int x=1;}\n',
    )
    assert ch1 != ch2
    revealed = {
        "author": ((5, encode_payload(ch1, url1)),),
        "copycat": ((9, encode_payload(ch2, url2)),),
    }
    st, env = _loop_env(tmp_path, revealed, hotkeys=["val", "author", "copycat"])
    res = run_pass(st, VALIDATOR_WALLET, 1, evaluator=_pass_all, **env)
    assert ch2 in res.copies
    assert res.weights == {"author": 1.0}


def test_system_intake_selects_product_normalizer_and_clears_component_signals(
    tmp_path,
):
    host = _host_system_bundle(
        tmp_path / "host",
        "candidate",
        "def k(x, out):\n    out.copy_(x[..., :out.shape[-1]])\n",
    )
    host_signals = _intake_fingerprints(
        host, target=UNTRUSTED_HOST_SYSTEM_TARGET, mode="system"
    )
    assert host_signals["product_fingerprints"][UNTRUSTED_HOST_SYSTEM_TARGET]
    assert host_signals["fingerprint"] == ""
    assert host_signals["structural_fingerprint"] == ""
    assert host_signals["slot_fingerprints"] == {}
    assert host_signals["slot_file_fingerprints"] == {}

    source_patch = _source_patch_system_bundle(tmp_path / "patch")
    patch_signals = _intake_fingerprints(
        source_patch, target=SGLANG_INFERENCE_SYSTEM_V1, mode="system"
    )
    assert patch_signals["product_fingerprints"][SGLANG_INFERENCE_SYSTEM_V1]
    assert patch_signals["slot_fingerprints"] == {}
    assert patch_signals["slot_file_fingerprints"] == {}
    # Different product kinds and targets cannot accidentally share accounting.
    assert set(host_signals["product_fingerprints"]) == {
        UNTRUSTED_HOST_SYSTEM_TARGET
    }
    assert set(patch_signals["product_fingerprints"]) == {
        SGLANG_INFERENCE_SYSTEM_V1
    }


def test_chain_rejects_legacy_host_slot_with_exact_system_lane_migration(
    tmp_path, monkeypatch
):
    """Arbitrary scheduler Python must never inherit its legacy slot crown."""
    bundle = _host_system_bundle(
        tmp_path / "src",
        "legacy-host",
        "def k(x, out):\n    out.copy_(x[..., :out.shape[-1]])\n",
    )
    system = _resolve_bundle_competition(bundle)
    assert system.target == UNTRUSTED_HOST_SYSTEM_TARGET
    assert system.mode == "system"
    assert system.members == ()
    assert system.crownable

    manifest_path = bundle / "manifest.toml"
    manifest_path.write_text(
        manifest_path.read_text().replace(
            "[competition]\n"
            f'target = "{UNTRUSTED_HOST_SYSTEM_TARGET}"\n'
            'mode = "system"\n\n',
            "",
            1,
        )
    )
    monkeypatch.setattr(
        "optima.device_component.component_crown_rejection",
        _real_component_crown_rejection,
    )

    with pytest.raises(SubmissionPolicyError) as caught:
        _resolve_bundle_competition(bundle)
    reason = str(caught.value)
    assert "untrusted_host scheduler Python" in reason
    assert UNTRUSTED_HOST_SYSTEM_TARGET in reason
    assert "mode='system'" in reason


def test_exact_host_system_copy_is_demoted_without_component_champion(tmp_path):
    body = "def k(x, out):\n    out.copy_(x[..., :out.shape[-1]])\n"
    content, url = _host_system_submission(tmp_path, "original", body)
    revealed = {
        "author": ((5, encode_payload(content, url)),),
        "copycat": ((9, encode_payload(content, url)),),
    }
    st, env = _loop_env(
        tmp_path, revealed, hotkeys=["val", "author", "copycat"]
    )

    result = run_pass(st, VALIDATOR_WALLET, 1, evaluator=_pass_system, **env)

    assert result.copies == [content]
    assert result.weights == {"author": 1.0}
    ledger = Ledger.load(env["ledger_path"])
    assert len(ledger.reveals) == 2
    for reveal in ledger.reveals:
        assert reveal.fingerprint == ""
        assert reveal.structural_fingerprint == ""
        assert reveal.slot_fingerprints == {}
        assert reveal.slot_file_fingerprints == {}
        assert set(reveal.product_fingerprints) == {
            UNTRUSTED_HOST_SYSTEM_TARGET
        }
    scoped = ledger.arena_champions[ARENA.bracket]
    assert set(scoped) == {UNTRUSTED_HOST_SYSTEM_TARGET}
    champion = scoped[UNTRUSTED_HOST_SYSTEM_TARGET]
    assert champion.mode == "system" and champion.member_slots == ()
    assert "activation.silu_and_mul" not in scoped
    assert ledger.champions == {}
    assert ledger.scores[0].slot == ""
    assert ledger.scores[0].member_slots == ()


def test_reformatted_host_system_copy_is_demoted_at_product_level(tmp_path):
    first, first_url = _host_system_submission(
        tmp_path,
        "original",
        "def k(x, out):\n    out.copy_(x[..., :out.shape[-1]])\n",
    )
    copied, copied_url = _host_system_submission(
        tmp_path,
        "reformatted",
        "# presentation-only rewrite\n"
        "def k( x, out ):\n    out.copy_((x[..., :out.shape[-1]]))\n",
    )
    assert first != copied
    revealed = {
        "author": ((5, encode_payload(first, first_url)),),
        "copycat": ((9, encode_payload(copied, copied_url)),),
    }
    st, env = _loop_env(
        tmp_path, revealed, hotkeys=["val", "author", "copycat"]
    )

    result = run_pass(st, VALIDATOR_WALLET, 1, evaluator=_pass_system, **env)

    assert copied in result.copies
    assert result.weights == {"author": 1.0}
    ledger = Ledger.load(env["ledger_path"])
    original, copy = ledger.reveals
    assert original.product_fingerprints == copy.product_fingerprints
    assert original.slot_fingerprints == copy.slot_fingerprints == {}
    assert original.slot_file_fingerprints == copy.slot_file_fingerprints == {}
    assert set(ledger.arena_champions[ARENA.bracket]) == {
        UNTRUSTED_HOST_SYSTEM_TARGET
    }


def test_fetch_failure_is_recorded_and_not_retried(tmp_path):
    ch = "c" * 64
    revealed = {"miner1": ((5, encode_payload(ch, "file:///nonexistent/x.tar.gz")),)}
    st, env = _loop_env(tmp_path, revealed, hotkeys=["val", "miner1"])
    res = run_pass(st, VALIDATOR_WALLET, 1, evaluator=_pass_all, **env)
    assert ch in res.rejected and not res.weights
    led = Ledger.load(env["ledger_path"])
    assert led.eval_for("miner1", ch, arena_bracket=ARENA.bracket).dq_reason.startswith("fetch:")
    # second pass skips it entirely (no infinite refetch of a dead URL)
    res2 = run_pass(st, VALIDATOR_WALLET, 1, evaluator=_pass_all, **env)
    assert res2.new == [] and res2.rejected == {}


def test_hash_mismatch_rejects_submission(tmp_path):
    _, url = _submission(tmp_path, "m1", "def k(x):\n    return x\n")
    lie = "d" * 64  # committed hash does not match the hosted artifact
    st, env = _loop_env(tmp_path, {"miner1": ((5, encode_payload(lie, url)),)},
                        hotkeys=["val", "miner1"])
    res = run_pass(st, VALIDATOR_WALLET, 1, evaluator=_pass_all, **env)
    assert lie in res.rejected and "mismatch" in res.rejected[lie]


def test_garbage_payload_is_ignored(tmp_path):
    st, env = _loop_env(tmp_path, {"miner1": ((5, "not json at all"),)},
                        hotkeys=["val", "miner1"])
    res = run_pass(st, VALIDATOR_WALLET, 1, evaluator=_pass_all, **env)
    assert res.seen == 0 and res.new == []


def test_failed_gates_earn_no_weight(tmp_path):
    ch, url = _submission(tmp_path, "m1", "def k(x):\n    return x\n")
    st, env = _loop_env(tmp_path, {"miner1": ((5, encode_payload(ch, url)),)},
                        hotkeys=["val", "miner1"])
    res = run_pass(st, VALIDATOR_WALLET, 1,
                   evaluator=lambda p, context: EvalOutcome(
                       False,
                       0.0,
                       target="activation.silu_and_mul",
                       mode="slot",
                       member_slots=("activation.silu_and_mul",),
                       detail="verify failed",
                       **_scope(context),
                   ),
                   **env)
    assert res.evaluated == {ch: False}
    assert res.weights == {} and not st.set_weights_calls


@pytest.mark.parametrize(
    "change",
    [
        {"target": "norm.rmsnorm"},
        {"mode": "atomic"},
        {"member_slots": ("norm.rmsnorm",)},
    ],
)
def test_chain_rejects_report_identity_that_disagrees_with_fetched_manifest(
    tmp_path, change
):
    ch, url = _submission(tmp_path, "m1", "def k(x):\n    return x\n")
    st, env = _loop_env(
        tmp_path,
        {"miner1": ((5, encode_payload(ch, url)),)},
        hotkeys=["val", "miner1"],
    )
    identity = dict(
        target="activation.silu_and_mul",
        mode="slot",
        member_slots=("activation.silu_and_mul",),
    )
    identity.update(change)

    res = run_pass(
        st,
        VALIDATOR_WALLET,
        1,
        evaluator=lambda p, context: EvalOutcome(
            True, 1.50, crownable=True, **identity, **_scope(context)
        ),
        **env,
    )

    assert ch in res.rejected
    assert "identity mismatch" in res.rejected[ch]
    assert res.evaluated == {ch: False}
    assert not res.weights
    led = Ledger.load(env["ledger_path"])
    assert not led.scores
    rec = led.eval_for("miner1", ch, arena_bracket=ARENA.bracket)
    assert rec.target == "activation.silu_and_mul"
    assert rec.mode == "slot"
    assert rec.member_slots == ("activation.silu_and_mul",)


def test_chain_rejects_static_or_wrong_prompt_seed(tmp_path):
    ch, url = _submission(tmp_path, "seed", "def k(x):\n    return x\n")
    st, env = _loop_env(
        tmp_path,
        {"miner1": ((5, encode_payload(ch, url)),)},
        hotkeys=["val", "miner1"],
    )

    def wrong_seed(bundle_dir, context):
        scope = _scope(context)
        scope["prompt_seed"] = 1
        return EvalOutcome(
            True,
            1.50,
            target="activation.silu_and_mul",
            mode="slot",
            member_slots=("activation.silu_and_mul",),
            crownable=True,
            **scope,
        )

    result = run_pass(st, VALIDATOR_WALLET, 1, evaluator=wrong_seed, **env)
    assert ch in result.rejected
    assert "arena/bundle/seed mismatch" in result.rejected[ch]
    assert not result.weights


@pytest.mark.parametrize(
    "mismatch",
    [
        "competition",
        "scope",
        "chain_scope",
        "validator_hotkey",
        "evaluation_id",
    ],
)
def test_trusted_oci_identity_mismatch_aborts_as_validator_fault(
    tmp_path, mismatch
):
    ch, url = _submission(tmp_path, f"trusted-{mismatch}", "def k(x):\n    return x\n")
    st, env = _loop_env(
        tmp_path,
        {"miner1": ((5, encode_payload(ch, url)),)},
        hotkeys=["val", "miner1"],
    )

    @_validator_owned
    def inconsistent(bundle_dir, context):
        scope = _scope(context)
        if mismatch == "scope":
            scope["prompt_seed"] += 1
        elif mismatch == "chain_scope":
            scope["chain_scope"] = make_chain_scope(
                genesis_hash="0x" + "f" * 64,
                netuid=999,
                scheme=ARENA.settlement.chain_scope_scheme,
            )
        elif mismatch == "validator_hotkey":
            scope["validator_hotkey"] = "other-validator"
        elif mismatch == "evaluation_id":
            scope["evaluation_id"] = "f" * 64
        return EvalOutcome(
            True,
            1.50,
            target=(
                "norm.rmsnorm"
                if mismatch == "competition"
                else "activation.silu_and_mul"
            ),
            mode="slot",
            member_slots=(
                ("norm.rmsnorm",)
                if mismatch == "competition"
                else ("activation.silu_and_mul",)
            ),
            crownable=True,
            **scope,
        )

    with pytest.raises(QualificationAuthorityError, match="identity mismatch|arena/bundle/seed mismatch"):
        run_pass(st, VALIDATOR_WALLET, 1, evaluator=inconsistent, **env)
    ledger = Ledger.load(env["ledger_path"])
    assert ledger.retry_for("miner1", ch, arena_bracket=ARENA.bracket) is None
    assert not ledger.scores


def test_trusted_crownable_outcome_without_host_attestation_aborts_as_validator_fault(
    tmp_path,
):
    ch, url = _submission(tmp_path, "hostless", "def k(x):\n    return x\n")
    st, env = _loop_env(
        tmp_path,
        {"miner1": ((5, encode_payload(ch, url)),)},
        hotkeys=["val", "miner1"],
    )

    @_validator_owned
    def hostless(bundle_dir, context):
        scope = _scope(context)
        scope.pop("host_attestation_sha256")
        return EvalOutcome(
            True,
            1.50,
            target="activation.silu_and_mul",
            mode="slot",
            member_slots=("activation.silu_and_mul",),
            crownable=True,
            **scope,
        )

    with pytest.raises(QualificationAuthorityError, match=r"retained[- ]host"):
        run_pass(st, VALIDATOR_WALLET, 1, evaluator=hostless, **env)
    ledger = Ledger.load(env["ledger_path"])
    assert not ledger.scores
    assert ledger.retry_for("miner1", ch, arena_bracket=ARENA.bracket) is None
    assert ledger.current_weights(
        arena=ARENA,
        host_attestation_verifier=_pass_all.host_attestation_verifier,
        validator_hotkey="val",
    ) == {}


def test_dry_run_weights_never_submits(tmp_path):
    ch, url = _submission(tmp_path, "m1", "def k(x):\n    return x\n")
    st, env = _loop_env(tmp_path, {"miner1": ((5, encode_payload(ch, url)),)},
                        hotkeys=["val", "miner1"])
    res = run_pass(st, None, 1, evaluator=_pass_all, dry_run_weights=True, **env)
    assert res.weights == {"miner1": 1.0}
    assert not res.weights_pushed and not st.set_weights_calls


def test_weights_repushed_after_refresh_interval(tmp_path):
    ch, url = _submission(tmp_path, "m1", "def k(x):\n    return x\n")
    st, env = _loop_env(tmp_path, {"miner1": ((5, encode_payload(ch, url)),)},
                        hotkeys=["val", "miner1"])
    run_pass(st, VALIDATOR_WALLET, 1, evaluator=_pass_all, **env)
    st._block += 1000  # well past the refresh cadence
    run_pass(st, VALIDATOR_WALLET, 1, evaluator=_pass_all, **env)
    assert len(st.set_weights_calls) == 2


def test_run_validator_once_returns_pass_result(tmp_path):
    st, env = _loop_env(tmp_path, {}, hotkeys=["val"])
    env.pop("test_only_allow_local_file_urls")
    res = run_validator(st, VALIDATOR_WALLET, 1, evaluator=_pass_all, once=True, **env)
    assert res is not None and res.seen == 0


def test_run_validator_once_propagates_validator_fault(tmp_path):
    st, env = _loop_env(tmp_path, {}, hotkeys=["val"])
    env.pop("test_only_allow_local_file_urls")
    st.get_current_block = lambda: (_ for _ in ()).throw(RuntimeError("chain down"))
    with pytest.raises(RuntimeError, match="chain down"):
        run_validator(st, VALIDATOR_WALLET, 1, evaluator=_pass_all, once=True, **env)


# --------------------------------------------------------------------------- #
# subprocess evaluator contract
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "failure,expected_retryable",
    [
        ("candidate", False),
        ("infrastructure", True),
    ],
)
def test_oci_evaluator_preserves_typed_failure_classification(
    tmp_path, monkeypatch, failure, expected_retryable
):
    import optima.eval.oci_backend as backend

    bundle = _mini_bundle(tmp_path / "bundles", "typed-oci", "kernel")
    from optima.bundle_hash import content_hash

    context = _context(content_hash(bundle))
    source = tmp_path / "source"
    model = tmp_path / "model"
    source.mkdir()
    model.mkdir()
    if failure == "candidate":
        error = backend.OCICandidateArtifactError("bad candidate artifact")
    else:
        error = backend.OCIInfrastructureError("container runtime unavailable")

    class FailingLauncher:
        def __init__(self, profile):
            self.profile = profile

        def begin_evaluation(self, *, timeout_s):
            assert timeout_s is None

        def prebuild_candidate_artifacts(self):
            raise error

    monkeypatch.setattr(backend, "profile_for_arena", lambda *args, **kwargs: object())
    monkeypatch.setattr(backend, "OCILauncher", FailingLauncher)
    evaluate_bundle = oci_evaluator(
        arena=ARENA,
        source_dir=source,
        model_dir=model,
        artifact_root=tmp_path / "artifacts",
        scratch_root=tmp_path / "scratch",
        gpu_devices=(0, 1, 2, 3),
    )

    outcome = evaluate_bundle(bundle, context)
    assert not outcome.passed and not outcome.crownable
    assert outcome.retryable is expected_retryable
    assert ("terminal candidate" in outcome.detail) is (not expected_retryable)


def test_oci_evaluator_propagates_validator_profile_fault(tmp_path, monkeypatch):
    import optima.eval.oci_backend as backend

    bundle = _mini_bundle(tmp_path / "bundles", "profile-fault", "kernel")
    from optima.bundle_hash import content_hash

    context = _context(content_hash(bundle))
    source = tmp_path / "source"
    model = tmp_path / "model"
    source.mkdir()
    model.mkdir()

    def fail_profile(*args, **kwargs):
        raise backend.OCIBackendError("trusted source identity mismatch")

    monkeypatch.setattr(backend, "profile_for_arena", fail_profile)
    evaluate_bundle = oci_evaluator(
        arena=ARENA,
        source_dir=source,
        model_dir=model,
        artifact_root=tmp_path / "artifacts",
        scratch_root=tmp_path / "scratch",
        gpu_devices=(0, 1, 2, 3),
    )

    with pytest.raises(backend.OCIBackendError) as caught:
        evaluate_bundle(bundle, context)
    assert caught.value.validator_fault

def test_command_evaluator_report_and_exit_codes(tmp_path):
    bundle = tmp_path / "cache" / ("e" * 64)
    bundle.mkdir(parents=True)
    context = _context(bundle.name)

    # Exit zero is not evidence: no typed report means no qualification.
    missing = command_evaluator("true", arena=ARENA)(bundle, context)
    assert not missing.passed and missing.score == 0.0
    assert "no qualification report" in missing.detail

    bad = command_evaluator("exit 3", arena=ARENA)(bundle, context)
    assert not bad.passed and bad.score == 0.0

    report = _qualification(context)
    writer = tmp_path / "write_report.sh"
    writer.write_text(f"#!/bin/sh\necho '{json.dumps(report)}' > \"$1\"\n")
    writer.chmod(0o755)
    rich = command_evaluator(
        f"sh {writer} {{report}} # {{bundle}}", arena=ARENA
    )(bundle, context)
    assert rich.passed and rich.score == 0.0
    assert not rich.crownable
    assert rich.kl_mean == 0.002
    assert rich.target == "moe.fused_experts"
    assert rich.mode == "slot"
    assert rich.member_slots == ("moe.fused_experts",)
    assert "non-authoritative" in rich.quality_evidence

    wrong_seed = dict(report, prompt_seed=context.prompt_seed + 1)
    bad_writer = tmp_path / "write_wrong_seed.sh"
    bad_writer.write_text(
        f"#!/bin/sh\necho '{json.dumps(wrong_seed)}' > \"$1\"\n"
    )
    bad_writer.chmod(0o755)
    mismatch = command_evaluator(
        f"sh {bad_writer} {{report}}", arena=ARENA
    )(bundle, context)
    assert not mismatch.passed
    assert "block-hash derivation receipt" in mismatch.detail


def test_command_evaluator_rejects_stale_malformed_and_incomplete_reports(tmp_path):
    bundle = tmp_path / "cache" / ("f" * 64)
    bundle.mkdir(parents=True)
    context = _context(bundle.name)
    report_path = bundle.parent / f".{bundle.name}.report.json"

    # The evaluator deletes an old receipt before launch, so a crash cannot reuse it.
    report_path.write_text(json.dumps({"score": 99}))
    stale = command_evaluator("true", arena=ARENA)(bundle, context)
    assert not stale.passed and not report_path.exists()

    malformed_writer = tmp_path / "malformed.sh"
    malformed_writer.write_text("#!/bin/sh\nprintf '{' > \"$1\"\n")
    malformed_writer.chmod(0o755)
    malformed = command_evaluator(
        f"sh {malformed_writer} {{report}}", arena=ARENA
    )(bundle, context)
    assert not malformed.passed and "malformed JSON" in malformed.detail

    # Syntactically valid legacy/partial JSON is still invalid.
    incomplete_writer = tmp_path / "incomplete.sh"
    incomplete_writer.write_text(
        "#!/bin/sh\nprintf '%s' '{\"score\": 1.1}' > \"$1\"\n"
    )
    incomplete_writer.chmod(0o755)
    incomplete = command_evaluator(
        f"sh {incomplete_writer} {{report}}", arena=ARENA
    )(bundle, context)
    assert not incomplete.passed and "schema_version" in incomplete.detail


def test_command_evaluator_preserves_quality_without_minting_a_score(tmp_path):
    bundle = tmp_path / "cache" / ("a" * 64)
    bundle.mkdir(parents=True)
    context = _context(bundle.name)
    quality_only = _qualification(
        context, target="activation.silu_and_mul", crownable=False
    )
    writer = tmp_path / "quality_only.sh"
    writer.write_text(f"#!/bin/sh\necho '{json.dumps(quality_only)}' > \"$1\"\n")
    writer.chmod(0o755)

    outcome = command_evaluator(
        f"sh {writer} {{report}}", arena=ARENA
    )(bundle, context)
    assert outcome.passed  # quality is preserved for the audit ledger
    assert not outcome.crownable
    assert outcome.score == 0.0


def test_real_command_evaluator_rejects_zero_settlement_margin(tmp_path):
    evaluator = command_evaluator("true", arena=ARENA)
    with pytest.raises(ValueError, match="immutable arena policy"):
        run_pass(
            None, None, 1,
            ledger_path=str(tmp_path / "ledger.json"),
            bundles_dir=str(tmp_path / "bundles"),
            evaluator=evaluator,
            arena=ARENA,
            margin=0,
        )
