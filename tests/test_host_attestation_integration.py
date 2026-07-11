"""Positive production-shaped host-attestation flow through the real chain loop."""
from __future__ import annotations
import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace
import pytest
from optima.arenas import (
    MINIMAX_M3_B300_TP4_DECODE_V1,
    derive_prompt_seed,
)
from optima.chain.fetch import package_bundle
from optima.chain.payload import encode_payload_for_testing as encode_payload
from optima.chain.validator_loop import (
    EvaluationContext,
    oci_evaluator,
    run_pass,
)
from optima.commit_reveal import Ledger, make_chain_scope
from optima.eval.host_attestation import (
    HostAttestationError,
    host_attestation_context,
    verify_host_attestation,
)
from optima.eval.runtime_preflight import RuntimePreflightReceipt
from tests.quality_report_helpers import (
    calibrated_test_arena,
    evaluation_report,
    patch_qualification_registry,
)

ARENA = calibrated_test_arena(MINIMAX_M3_B300_TP4_DECODE_V1)
TARGET = "activation.silu_and_mul"
VALIDATOR_WALLET = SimpleNamespace(
    hotkey=SimpleNamespace(ss58_address="validator")
)
CONFIGURATION_SHA256 = "b" * 64
POLICY_SHA256 = "c" * 64
DIRECT_EVALUATION_ID = "d" * 64


@pytest.fixture(autouse=True)
def _test_quality_arena(monkeypatch):
    patch_qualification_registry(monkeypatch, ARENA)


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
        self.hotkeys = ["validator", "miner1"]
        self._weights = [[0.0, 0.0], [0.0, 0.0]]
        self._last_update = [0, 0]
        self.set_weights_calls = []

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

    def get_current_block(self):
        return self.block

    def get_finalized_block_number(self):
        return self.block

    def get_block_hash(self, block):
        return "0x" + hashlib.sha256(f"block:{block}".encode()).hexdigest()

    def get_all_revealed_commitments(self, netuid=None, block=None):
        return {
            hotkey: tuple(
                entry for entry in history
                if block is None or entry[0] <= block
            )[-10:]
            for hotkey, history in self.revealed.items()
        }

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
        self.set_weights_calls.append({"uids": uids, "weights": weights})
        self._weights[0] = [0.0, 0.0]
        for uid, weight in zip(uids, weights):
            self._weights[0][int(uid)] = float(weight)
        self._last_update[0] = self.block
        return True


def _mini_validator_device_bundle(root: Path) -> Path:
    bundle = root / "source" / "validator-device"
    (bundle / "kernels").mkdir(parents=True)
    (bundle / "manifest.toml").write_text(
        'bundle_id = "validator-device"\n'
        'abi_version = "optima-op-abi-v0"\n\n'
        "[[ops]]\n"
        f'slot = "{TARGET}"\n'
        'source = "kernels/k.cu"\n'
        'entry = "k"\n'
        'execution_class = "validator_device"\n'
        'device_abi = "activation.silu_and_mul.cuda.v1"\n'
        'dtypes = ["bfloat16"]\n',
        encoding="utf-8",
    )
    (bundle / "kernels" / "k.cu").write_text(
        'extern "C" __global__ void k() {}\n', encoding="utf-8"
    )
    return bundle


def _package_submission(root: Path) -> tuple[Path, str, str]:
    bundle = _mini_validator_device_bundle(root)
    archive, content_hash = package_bundle(
        bundle, root / "hosted" / "validator-device.tar.gz"
    )
    return bundle, content_hash, archive.as_uri()


def _runtime_receipt() -> RuntimePreflightReceipt:
    return RuntimePreflightReceipt(
        schema="optima-stock-runtime-preflight-v1",
        requested_image=ARENA.validator_image,
        requested_manifest_digest=ARENA.validator_image.rsplit("@", 1)[1],
        local_image_id="sha256:" + "6" * 64,
        repo_digests=(ARENA.validator_image,),
        docker_binary="/usr/bin/docker",
        uid=65532,
        gid=65532,
        sglang_version=ARENA.sglang_version,
        python_implementation="cpython",
        python_version="3.11.12",
        python_abi="cpython-311-x86_64-linux-gnu",
        python_platform="linux-x86_64",
        machine="x86_64",
        package_versions=(
            ("cuda-python", "13.0.1"),
            ("flashinfer-python", "0.4.1"),
            ("nvidia-cuda-runtime-cu12", "12.9.79"),
            ("torch", "2.8.0"),
            ("triton", "3.4.0"),
        ),
        cudart_library="libcudart.so.12",
        cuda_visible_devices="",
        nvidia_visible_devices="void",
        security_argv_sha256="7" * 64,
    )


def _telemetry() -> list[dict]:
    return [
        {
            "physical_id": gpu,
            "uuid": f"GPU-00000000-0000-0000-0000-{gpu:012d}",
            "pstate": "P0",
            "temperature_c": 35,
            "gpu_utilization_percent": 0,
            "memory_utilization_percent": 0,
            "current_graphics_clock_mhz": 210,
            "current_memory_clock_mhz": 3996,
            "power_draw_mw": 80_000,
        }
        for gpu in range(4)
    ]


def _device_receipt(sequence: int, arm: str, phase: str) -> dict:
    started = float(sequence * 10)
    return {
        "schema": "optima.device-state-receipt.v1",
        "sequence": sequence,
        "arm": arm,
        "phase": phase,
        "selected_physical_gpu_ids": [0, 1, 2, 3],
        "configuration_sha256": CONFIGURATION_SHA256,
        "policy_sha256": POLICY_SHA256,
        "started_monotonic_s": started,
        "completed_monotonic_s": started + 1.0,
        "consecutive_idle_samples": 2,
        "samples": [
            {
                "monotonic_s": started + offset,
                "telemetry": _telemetry(),
                "processes": [],
                "idle": True,
                "idle_reason": "no processes; temperature/utilization within policy",
                "active_envelope_passed": False,
                "active_envelope_reason": "not evaluated",
            }
            for offset in (0.25, 0.75)
        ],
    }


def _active_device_receipt(sequence: int, arm: str) -> dict:
    started = float(sequence * 10)
    processes = [
        {
            "physical_id": gpu,
            "pid": 20_000 + gpu,
            "kind": "C",
            "process_name": "sglang::scheduler",
        }
        for gpu in range(4)
    ]
    return {
        "schema": "optima.device-state-active-receipt.v2",
        "sequence": sequence,
        "arm": arm,
        "event": "final-warmup-conditioning",
        "selected_physical_gpu_ids": [0, 1, 2, 3],
        "configuration_sha256": CONFIGURATION_SHA256,
        "policy_sha256": POLICY_SHA256,
        "started_monotonic_s": started,
        "completed_monotonic_s": started + 1.0,
        "consecutive_active_samples": 2,
        "release_sample_index": 2,
        "post_release_ready_samples": 1,
        "samples": [
            {
                "monotonic_s": started + offset,
                "telemetry": _telemetry(),
                "processes": processes,
                "idle": False,
                "idle_reason": "active scheduler processes",
                "active_envelope_passed": True,
                "active_envelope_reason": "active envelope satisfied",
            }
            for offset in (0.2, 0.4, 0.75)
        ],
    }


def _device_receipts() -> list[dict]:
    receipts = []
    sequence = 1
    for arm in ("baseline-1", "candidate-1", "bookend-1"):
        receipts.append(_device_receipt(sequence, arm, "pre"))
        sequence += 1
        receipts.append(_active_device_receipt(sequence, arm))
        sequence += 1
        receipts.append(_device_receipt(sequence, arm, "post"))
        sequence += 1
    return receipts


def _qualification_report(*, malformed=False):
    return evaluation_report(
        ARENA,
        candidate_rate=107.0,
        malformed_baseline=malformed,
    )


def _context(
    bundle_hash: str,
    *,
    reveal_block=5,
    validator_hotkey="validator",
    evaluation_id=DIRECT_EVALUATION_ID,
    miner_hotkey="miner1",
    evaluation_block=400,
) -> EvaluationContext:
    block_hash = "0x" + hashlib.sha256(f"block:{reveal_block}".encode()).hexdigest()
    round_id = reveal_block // ARENA.settlement.round_blocks
    return EvaluationContext(
        arena=ARENA,
        bundle_hash=bundle_hash,
        round_id=round_id,
        block=reveal_block,
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
        validator_hotkey=validator_hotkey,
        evaluation_id=evaluation_id,
        miner_hotkey=miner_hotkey,
        settlement_round_id=(
            evaluation_block // ARENA.settlement.round_blocks
        ),
        evaluation_block=evaluation_block,
    )


def _host_context(context: EvaluationContext, outcome) -> dict[str, object]:
    return host_attestation_context(
        ARENA,
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
        target=outcome.target,
        mode=outcome.mode,
        member_slots=outcome.member_slots,
        score=outcome.score,
        passed_quality=outcome.passed,
        passed_timed_quality=outcome.passed_timed_quality,
        passed_warmup_quality=outcome.passed_warmup_quality,
        passed_speedup=outcome.passed_speedup,
        confident=outcome.confident,
        crownable=outcome.crownable,
        quality_evidence=outcome.quality_evidence,
        qualification_evidence_sha256=(
            outcome.qualification_evidence_sha256
        ),
    )


def _install_mocked_oci_edges(monkeypatch, *, malformed=False):
    runtime = _runtime_receipt()
    receipts = _device_receipts()

    class Launcher:
        def __init__(self, profile):
            self.profile = profile
            self.runtime_preflight_receipt = runtime
            self.attestation_receipts = copy.deepcopy(receipts)

        def begin_evaluation(self, *, timeout_s=None):
            self.timeout_s = timeout_s

        def prebuild_candidate_artifacts(self):
            return self.profile.artifact_dir / "prebuild-receipt.json"

    def profile_for_arena(arena, **kwargs):
        assert arena.fingerprint == ARENA.fingerprint
        assert kwargs["competition_target"] == TARGET
        return SimpleNamespace(
            artifact_dir=Path(kwargs["artifact_dir"]),
            bundle_dir=Path(kwargs["bundle_dir"]),
        )

    def evaluate(cfg, bundle_path, *, oci_launcher):
        assert cfg.prompt_seed > 0
        assert Path(bundle_path).is_dir()
        assert isinstance(oci_launcher, Launcher)
        return _qualification_report(malformed=malformed)

    monkeypatch.setattr(
        "optima.eval.oci_backend.profile_for_arena", profile_for_arena
    )
    monkeypatch.setattr(
        "optima.device_component.component_crown_rejection",
        lambda _manifest: None,
    )
    monkeypatch.setattr("optima.eval.oci_backend.OCILauncher", Launcher)
    monkeypatch.setattr("optima.eval.throughput_kl.evaluate", evaluate)


def _make_evaluator(tmp_path: Path):
    source = tmp_path / "referee-source"
    model = tmp_path / "model"
    artifact_root = tmp_path / "controller-artifacts"
    scratch = tmp_path / "scratch"
    source.mkdir()
    model.mkdir()
    evaluator = oci_evaluator(
        arena=ARENA,
        source_dir=source,
        model_dir=model,
        artifact_root=artifact_root,
        scratch_root=scratch,
        gpu_devices=(0, 1, 2, 3),
        timeout_s=60.0,
    )
    return evaluator, artifact_root


def test_real_sidecar_digest_flows_through_oci_chain_and_reopens_for_weights(
    tmp_path, monkeypatch,
):
    _install_mocked_oci_edges(monkeypatch)
    source_bundle, content_hash, url = _package_submission(tmp_path)
    evaluator, artifact_root = _make_evaluator(tmp_path)

    # Observe the exact real oci_evaluator outcome before the chain pass; the pass
    # evaluates the fetched copy and must carry the same content-addressed sidecar.
    context = _context(content_hash)
    direct = evaluator(source_bundle, context)
    assert direct.crownable and direct.host_attestation_sha256.startswith("sha256:")
    direct_context = _host_context(context, direct)
    verified = verify_host_attestation(
        artifact_root,
        direct.host_attestation_sha256,
        expected_context=direct_context,
    )
    assert verified.sha256 == direct.host_attestation_sha256
    assert (
        verified.qualification_evidence_sha256
        == direct.qualification_evidence_sha256
    )
    assert verified.receipt_count == 9

    verifier_calls = []
    real_verifier = evaluator.host_attestation_verifier

    def counted_verifier(reference, expected_context):
        verifier_calls.append((reference, dict(expected_context)))
        return real_verifier(reference, expected_context)

    subtensor = _Subtensor(
        revealed={
            "miner1": ((5, encode_payload(content_hash, url)),),
        }
    )
    ledger_path = tmp_path / "ledger.json"
    chain_contexts = []
    chain_outcomes = []

    def capturing_evaluator(bundle_dir, chain_context):
        durable = Ledger.load(ledger_path)
        lease = durable.retry_for(
            "miner1", content_hash, arena_bracket=ARENA.bracket
        )
        assert lease is not None and lease.state == "in_progress"
        assert chain_context.evaluation_id == lease.lease_id
        assert chain_context.chain_scope == durable.chain_scope
        assert chain_context.validator_hotkey == "validator"
        chain_contexts.append(chain_context)
        outcome = evaluator(bundle_dir, chain_context)
        chain_outcomes.append(outcome)
        return outcome

    capturing_evaluator.validator_owned_oci = True
    capturing_evaluator.requires_positive_margin = True
    capturing_evaluator.host_attestation_verifier = counted_verifier
    result = run_pass(
        subtensor,
        VALIDATOR_WALLET,
        1,
        ledger_path=str(ledger_path),
        bundles_dir=str(tmp_path / "bundle-cache"),
        evaluator=capturing_evaluator,
        arena=ARENA,
        validator_hotkey="validator",
        test_only_allow_local_file_urls=True,
    )

    assert result.evaluated == {content_hash: True}
    assert result.weights == {"miner1": 1.0}
    assert result.weights_pushed
    assert len(chain_contexts) == len(chain_outcomes) == 1
    chain_context = chain_contexts[0]
    chain_outcome = chain_outcomes[0]
    assert chain_context.evaluation_id != context.evaluation_id
    assert (
        chain_outcome.host_attestation_sha256
        != direct.host_attestation_sha256
    )
    assert (
        chain_outcome.qualification_evidence_sha256
        != direct.qualification_evidence_sha256
    )
    assert verifier_calls
    assert all(
        call[0] == chain_outcome.host_attestation_sha256
        for call in verifier_calls
    )

    ledger = Ledger.load(ledger_path)
    assert not ledger.pending_settlements
    score = ledger.scores[-1]
    record = ledger.eval_for(
        "miner1", content_hash, arena_bracket=ARENA.bracket
    )
    assert record is not None
    champion = ledger.arena_champions[ARENA.bracket][TARGET]
    assert {
        score.host_attestation_sha256,
        record.host_attestation_sha256,
        champion.host_attestation_sha256,
    } == {chain_outcome.host_attestation_sha256}
    assert {
        score.evaluation_id,
        record.evaluation_id,
        champion.evaluation_id,
    } == {chain_context.evaluation_id}
    assert {
        score.qualification_evidence_sha256,
        record.qualification_evidence_sha256,
        champion.qualification_evidence_sha256,
    } == {chain_outcome.qualification_evidence_sha256}
    assert {
        score.miner_hotkey,
        record.miner_hotkey,
        champion.miner_hotkey,
    } == {"miner1"}
    assert {
        score.settlement_round_id,
        record.settlement_round_id,
        champion.settlement_round_id,
    } == {chain_context.settlement_round_id}
    assert {
        score.evaluation_block,
        record.evaluation_block,
        champion.evaluation_block,
    } == {chain_context.evaluation_block}
    assert ledger.current_weights(
        arena=ARENA,
        host_attestation_verifier=counted_verifier,
        validator_hotkey="validator",
    ) == {"miner1": 1.0}


@pytest.mark.parametrize(
    "field,replacement",
    [
        (
            "chain_scope",
            make_chain_scope(
                genesis_hash="0x" + "f" * 64,
                netuid=999,
                scheme=ARENA.settlement.chain_scope_scheme,
            ),
        ),
        ("validator_hotkey", "another-validator"),
        ("evaluation_id", "f" * 64),
    ],
)
def test_real_sidecar_rejects_chain_validator_or_evaluation_transplant(
    tmp_path, monkeypatch, field, replacement,
):
    _install_mocked_oci_edges(monkeypatch)
    bundle, content_hash, _ = _package_submission(tmp_path)
    evaluator, artifact_root = _make_evaluator(tmp_path)
    context = _context(content_hash)
    outcome = evaluator(bundle, context)
    expected = _host_context(context, outcome)
    transplanted = dict(expected)
    transplanted[field] = replacement

    with pytest.raises(HostAttestationError, match="context differs"):
        verify_host_attestation(
            artifact_root,
            outcome.host_attestation_sha256,
            expected_context=transplanted,
        )


def test_real_oci_evaluator_treats_qualification_construction_failure_as_validator_fault(
    tmp_path, monkeypatch,
):
    _install_mocked_oci_edges(monkeypatch, malformed=True)
    bundle, content_hash, _ = _package_submission(tmp_path)
    evaluator, _ = _make_evaluator(tmp_path)

    from optima.eval.oci_backend import OCIBackendError

    with pytest.raises(OCIBackendError) as caught:
        evaluator(bundle, _context(content_hash))
    assert caught.value.validator_fault is True
    assert caught.value.retryable is False
    assert "qualification report construction failed" in str(caught.value)
