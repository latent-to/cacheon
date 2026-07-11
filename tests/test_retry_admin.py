"""Trusted operator inspection/release for non-terminal retry state."""

from optima.arenas import MINIMAX_M3_B300_TP4_DECODE_V1 as ARENA
from optima.chain.validator_loop import _exclusive_ledger_pass
from optima.cli import main
from optima.commit_reveal import (
    Ledger,
    RETRY_KIND_INFRASTRUCTURE,
    RETRY_STATE_HELD,
)


CHAIN_SCOPE = "genesis-netuid-v1:sha256:" + "d" * 64
BUNDLE_HASH = "e" * 64


def _held_ledger(path):
    led = Ledger()
    led.bind_chain_scope(CHAIN_SCOPE)
    retry = None
    for block in (10, 20, 40):
        retry = led.record_retry(
            hotkey="miner1",
            bundle_hash=BUNDLE_HASH,
            arena_bracket=ARENA.bracket,
            kind=RETRY_KIND_INFRASTRUCTURE,
            current_block=block,
            reason="runtime unavailable",
            base_backoff_blocks=10,
            max_backoff_blocks=100,
            max_automatic_infrastructure_attempts=3,
            max_automatic_no_decision_attempts=4,
            max_total_attempts=6,
        )
    assert retry is not None and retry.state == RETRY_STATE_HELD
    led.save(path)


def _args(path, *extra):
    return [
        "retries",
        "--ledger",
        str(path),
        "--arena",
        ARENA.name,
        "--chain-scope",
        CHAIN_SCOPE,
        *extra,
    ]


def _validator_fault_ledger(path):
    led = Ledger()
    led.bind_chain_scope(CHAIN_SCOPE)
    led.bind_validator_hotkey("validator")
    lease = led.begin_retry_attempt(
        hotkey="miner1",
        bundle_hash=BUNDLE_HASH,
        arena_bracket=ARENA.bracket,
        current_block=77,
        reason="evaluation lease acquired before GPU work",
        max_automatic_infrastructure_attempts=3,
        max_automatic_no_decision_attempts=4,
        max_total_attempts=6,
    )
    led.hold_validator_fault(
        hotkey="miner1",
        bundle_hash=BUNDLE_HASH,
        arena_bracket=ARENA.bracket,
        lease_id=lease.lease_id,
        current_block=78,
        reason="controller process died after durable lease",
    )
    led.save(path)


def test_retry_admin_inspects_and_releases_only_exact_scope(tmp_path, capsys):
    path = tmp_path / "ledger.json"
    _held_ledger(path)

    assert main(_args(path)) == 0
    output = capsys.readouterr().out
    assert "held" in output
    assert "infrastructure" in output
    assert BUNDLE_HASH in output

    wrong_scope = _args(path, "--release", "miner1", BUNDLE_HASH)
    wrong_scope[wrong_scope.index(CHAIN_SCOPE)] = (
        "genesis-netuid-v1:sha256:" + "f" * 64
    )
    assert main(wrong_scope) == 2
    assert "does not match" in capsys.readouterr().out
    assert Ledger.load(path).retry_for(
        "miner1", BUNDLE_HASH, arena_bracket=ARENA.bracket
    ) is not None

    assert main(_args(path, "--release", "miner1", BUNDLE_HASH)) == 0
    output = capsys.readouterr().out
    assert "released held miner retry" in output
    assert "miner retries: (none)" in output
    assert Ledger.load(path).retry_for(
        "miner1", BUNDLE_HASH, arena_bracket=ARENA.bracket
    ) is None


def test_retry_admin_refuses_to_reset_an_automatic_retry(tmp_path, capsys):
    path = tmp_path / "ledger.json"
    led = Ledger()
    led.bind_chain_scope(CHAIN_SCOPE)
    led.record_retry(
        hotkey="miner1",
        bundle_hash=BUNDLE_HASH,
        arena_bracket=ARENA.bracket,
        kind=RETRY_KIND_INFRASTRUCTURE,
        current_block=10,
        reason="first transient failure",
        base_backoff_blocks=10,
        max_backoff_blocks=100,
        max_automatic_infrastructure_attempts=3,
        max_automatic_no_decision_attempts=4,
        max_total_attempts=6,
    )
    led.save(path)

    assert main(_args(path, "--release", "miner1", BUNDLE_HASH)) == 2
    assert "only an operator-held retry" in capsys.readouterr().out
    assert Ledger.load(path).retry_for(
        "miner1", BUNDLE_HASH, arena_bracket=ARENA.bracket
    ) is not None


def test_retry_admin_identifies_a_durable_in_progress_lease(tmp_path, capsys):
    path = tmp_path / "ledger.json"
    led = Ledger()
    led.bind_chain_scope(CHAIN_SCOPE)
    lease = led.begin_retry_attempt(
        hotkey="miner1",
        bundle_hash=BUNDLE_HASH,
        arena_bracket=ARENA.bracket,
        current_block=77,
        reason="evaluation lease acquired before GPU work",
        max_automatic_infrastructure_attempts=3,
        max_automatic_no_decision_attempts=4,
        max_total_attempts=6,
    )
    led.save(path)

    assert main(_args(path)) == 0
    output = capsys.readouterr().out
    assert "in_progress" in output
    assert f"leased@{lease.lease_block}" in output
    assert "operator-release" not in output


def test_retry_admin_lists_and_releases_validator_fault_separately(
    tmp_path, capsys
):
    path = tmp_path / "ledger.json"
    _validator_fault_ledger(path)

    assert main(_args(path)) == 0
    output = capsys.readouterr().out
    assert "miner retries: (none)" in output
    assert "validator-fault held@78" in output
    assert "controller process died" in output

    assert main(_args(
        path,
        "--release-validator-fault",
        "miner1",
        BUNDLE_HASH,
    )) == 0
    output = capsys.readouterr().out
    assert "released validator-fault hold" in output
    assert "validator-fault holds: (none)" in output
    loaded = Ledger.load(path)
    assert loaded.validator_fault_for(
        "miner1", BUNDLE_HASH, arena_bracket=ARENA.bracket
    ) is None


def test_retry_admin_release_respects_whole_pass_lock(tmp_path, capsys):
    path = tmp_path / "ledger.json"
    _held_ledger(path)

    with _exclusive_ledger_pass(path):
        assert main(_args(path, "--release", "miner1", BUNDLE_HASH)) == 2
    assert "another validator process owns the whole pass" in capsys.readouterr().out
    assert Ledger.load(path).retry_for(
        "miner1", BUNDLE_HASH, arena_bracket=ARENA.bracket
    ) is not None


def test_manual_settle_respects_whole_pass_lock(tmp_path, capsys):
    path = tmp_path / "ledger.json"
    with _exclusive_ledger_pass(path):
        assert main([
            "settle",
            "--ledger", str(path),
            "--arena", ARENA.name,
            "--validator-hotkey-address", "validator",
        ]) == 2
    assert "another validator process owns the whole pass" in capsys.readouterr().out
