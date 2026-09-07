from __future__ import annotations

import argparse

import pytest

import cacheon.cli as cli


def test_chain_validate_refuses_implicit_fake_grading(monkeypatch):
    args = argparse.Namespace(intake_only=False)
    with pytest.raises(SystemExit, match="requires --intake-only or"):
        cli.cmd_chain_validate(args)


def test_chain_validate_intake_path_has_no_wallet_or_weight_arguments():
    # Global help routes commands rather than rendering subparser flags; inspect the
    # parser action directly without executing any chain code.
    parser = cli.build_parser()
    subparsers = next(
        action for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    )
    chain_validate = subparsers.choices["chain-validate"]
    options = {
        option
        for action in chain_validate._actions
        for option in action.option_strings
    }
    assert "--intake-only" in options
    assert "--arena-id" in options
    assert "--audit-log" in options
    assert not {
        "--eval-cmd", "--eval-device", "--eval-timeout", "--margin",
        "--wallet", "--hotkey", "--dry-run-weights",
    } & options


def test_chain_snapshot_surfaces_are_wallet_free_and_explicit() -> None:
    parser = cli.build_parser()
    subparsers = next(
        action
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    )
    snapshot = subparsers.choices["chain-snapshot"]
    verify = subparsers.choices["chain-snapshot-verify"]
    snapshot_options = {
        option for action in snapshot._actions for option in action.option_strings
    }
    verify_options = {
        option for action in verify._actions for option in action.option_strings
    }
    assert {
        "--intake-db",
        "--audit-log",
        "--sealed-input",
        "--object-store-endpoint",
        "--object-store-bucket",
    } <= snapshot_options
    assert {
        "--manifest-key",
        "--restore-root",
        "--object-store-endpoint",
        "--object-store-bucket",
    } <= verify_options
    assert not {"--wallet", "--hotkey", "--model", "--image"} & (
        snapshot_options | verify_options
    )


def test_chain_submit_can_reuse_an_unused_eval_cost_payment() -> None:
    parser = cli.build_parser()
    subparsers = next(
        action
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    )
    submit = subparsers.choices["chain-submit"]
    options = {
        option for action in submit._actions for option in action.option_strings
    }
    assert {
        "--pay",
        "--dry-run",
        "--eval-cost-payment-block",
        "--eval-cost-payment-extrinsic-index",
        "--eval-cost-tao-rao",
    } <= options


def test_chain_eval_cost_credit_is_wallet_free_and_intake_scoped() -> None:
    parser = cli.build_parser()
    subparsers = next(
        action
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    )
    credit = subparsers.choices["chain-eval-cost-credit"]
    options = {
        option for action in credit._actions for option in action.option_strings
    }
    assert {
        "--intake-db", "--hotkey", "--coldkey", "--amount-tao-rao",
        "--note", "--list",
    } <= options
    assert not {"--wallet", "--network", "--netuid"} & options


def test_chain_eval_cost_credit_grants_and_lists(tmp_path, capsys) -> None:
    import sqlite3

    db_path = tmp_path / "intake.sqlite3"
    sqlite3.connect(db_path).close()
    grant = argparse.Namespace(
        intake_db=str(db_path),
        hotkey="miner",
        coldkey="miner-cold",
        amount_tao_rao=1_000_000_000,
        note="ops make-good",
        list=False,
    )
    assert cli.cmd_chain_eval_cost_credit(grant) == 0
    granted = capsys.readouterr().out
    assert "credit_id:" in granted
    assert "miner" in granted
    listing = argparse.Namespace(
        intake_db=str(db_path),
        hotkey="miner",
        coldkey="",
        amount_tao_rao=1_000_000_000,
        note="",
        list=True,
    )
    assert cli.cmd_chain_eval_cost_credit(listing) == 0
    listed = capsys.readouterr().out
    assert "unspent" in listed
    assert "ops make-good" in listed


def test_chain_eval_cost_credit_refuses_a_grant_without_a_hotkey(
    tmp_path, capsys
) -> None:
    import sqlite3

    db_path = tmp_path / "intake.sqlite3"
    sqlite3.connect(db_path).close()
    args = argparse.Namespace(
        intake_db=str(db_path),
        hotkey="",
        coldkey="",
        amount_tao_rao=1_000_000_000,
        note="",
        list=False,
    )
    assert cli.cmd_chain_eval_cost_credit(args) == 2
    assert "REFUSED" in capsys.readouterr().out
