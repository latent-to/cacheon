"""Exact username identity and historical native-commitment integration contracts."""

import copy
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from scalecodec.utils.ss58 import ss58_encode

from cacheon import chain, cli
from cacheon.chain import okc_username as okc

HOTKEY = "5CzNs6w9tmimaUwddbqWjndz7LTuQMgtDnp5bND4AuDeVsjK"
OTHER = "5FxbrVDh7brAup3eneqGtyaUsCjXvgm6s6t5ygaikpRmQhq7"
NAME = "Cacheon_28ef5_exact"
HASH = "0x" + "ab" * 32


def metadata(name=NAME, block=80):
    raw = name.encode("utf-8")
    return {"block": block, "info": {"fields": [{f"Raw{len(raw)}": "0x" + raw.hex()}]}}


def subtensor():
    return SimpleNamespace(
        get_commitment_metadata=Mock(return_value=metadata()),
        substrate=SimpleNamespace(get_block_hash=Mock(return_value=HASH),
                                  query=Mock(return_value=SimpleNamespace(value=123456789000))),
        set_commitment=Mock(return_value=SimpleNamespace(success=True, message="")), close=Mock())


def test_tags_use_decoded_public_key_and_exact_names():
    assert okc.hotkey_tag(HOTKEY) == "28ef5"
    assert okc.hotkey_tag(OTHER) == "ac494"
    assert okc.validate_username(HOTKEY, NAME) == NAME
    assert okc.validate_username(OTHER, "ac494_Cacheon") == "ac494_Cacheon"
    assert okc.validate_username(HOTKEY, "Cacheon28ef5" + "x" * 116).endswith("x")
    with pytest.raises(ValueError, match="AccountId32"):
        okc.hotkey_tag(ss58_encode("01"))


@pytest.mark.parametrize("name", ["cacheon_28ef5", "Cacheon_28EF5", "Cacheon_ac494", "28ef5",
                                  "Cacheon28ef5é", "Cacheon28ef5" + "x" * 117, None])
def test_invalid_username_is_not_normalized(name):
    with pytest.raises(ValueError):
        okc.validate_username(HOTKEY, name)


def test_reader_pins_identity_block_and_timestamp():
    st = subtensor()
    assert okc.read_registration(st, 14, HOTKEY, 80, 90) == {
        "username": NAME, "block": 80, "block_hash": HASH, "timestamp_ms": 123456789000}
    st.get_commitment_metadata.assert_called_once_with(netuid=14, hotkey_ss58=HOTKEY, block=80)
    st.substrate.get_block_hash.assert_called_once_with(80)
    st.substrate.query.assert_called_once_with("Timestamp", "Now", block_hash=HASH)
    st.get_commitment_metadata.return_value = metadata("ac494_Cacheon", 81)
    assert okc.read_registration(st, 7, OTHER, 81, 90)["username"] == "ac494_Cacheon"


@pytest.mark.parametrize("block", [0, -1, 91, True, "80"])
def test_invalid_or_unfinalized_block_never_queries(block):
    st = subtensor()
    with pytest.raises(ValueError, match="finalized"):
        okc.read_registration(st, 14, HOTKEY, block, 90)
    st.get_commitment_metadata.assert_not_called()


@pytest.mark.parametrize("bad", ["", {}, metadata(block=79), metadata(block=True),
    {"block": 80, "info": {"fields": []}},
    {"block": 80, "info": {"fields": [{"Sha256": "0x" + "01" * 32}]}},
    {"block": 80, "info": {"fields": [{"Raw1": "0xff"}]}},
    {"block": 80, "info": {"fields": [{"Raw1": "0x0g"}]}},
    {"block": 80, "info": {"fields": [{"Raw1": "0x0"}]}},
    {"block": 80, "info": {"fields": [{"Raw1": "0x4142"}]}},
    {"block": 80, "info": {"fields": [({"Raw1": ((65,),)},)]}},
    {"block": 80, "info": {"fields": metadata()["info"]["fields"] * 2}}])
def test_malformed_metadata_is_rejected(bad):
    st = subtensor()
    st.get_commitment_metadata.return_value = copy.deepcopy(bad)
    with pytest.raises(ValueError):
        okc.read_registration(st, 14, HOTKEY, 80, 90)
    st.substrate.query.assert_not_called()


def test_rpc_errors_propagate_and_missing_timestamp_is_invalid():
    st = subtensor()
    st.get_commitment_metadata.side_effect = RuntimeError("archive unavailable")
    with pytest.raises(RuntimeError, match="archive unavailable"):
        okc.read_registration(st, 14, HOTKEY, 80, 90)
    st.get_commitment_metadata.side_effect = None
    st.substrate.query.return_value = None
    with pytest.raises(RuntimeError, match="timestamp"):
        okc.read_registration(st, 14, HOTKEY, 80, 90)


def command(monkeypatch, dry_run=False):
    parser = cli.build_parser()
    args = parser.parse_args(["chain-okc-register", NAME, "--netuid", "14", "--wallet-path", "/wallets"]
                             + (["--dry-run"] if dry_run else []))
    wallet = SimpleNamespace(hotkey=SimpleNamespace(ss58_address=HOTKEY))
    monkeypatch.setattr(cli, "_wallet_from_args", Mock(return_value=wallet))
    st = subtensor()
    monkeypatch.setattr(chain, "connect", Mock(return_value=st))
    monkeypatch.setattr(chain, "read_finalized_head", Mock(return_value=(90, HASH)))
    return args, wallet, st


def test_dry_run_uses_wallet_identity_without_chain(monkeypatch, capsys):
    args, _, st = command(monkeypatch, dry_run=True)
    assert args.func(args) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["tag"] == "28ef5" and result["registration_block"] is None
    assert result["submitted"] is False and args.wallet_path == "/wallets"
    cli._wallet_from_args.assert_called_once_with(args)
    chain.connect.assert_not_called()
    st.set_commitment.assert_not_called()


def test_producer_submits_native_username_and_reads_actual_block(monkeypatch, capsys):
    args, wallet, st = command(monkeypatch)
    assert args.func(args) == 0
    st.set_commitment.assert_called_once_with(wallet=wallet, netuid=14, data=NAME,
                                            wait_for_inclusion=True, wait_for_finalization=True)
    assert [call.kwargs["block"] for call in st.get_commitment_metadata.call_args_list] == [90, 80]
    result = json.loads(capsys.readouterr().out)
    assert result["submitted"] is True and result["registration_block"] == 80
    chain.connect.assert_called_once_with("finney")
    st.close.assert_called_once()


def test_submission_failure_keeps_cause_and_emits_no_success(monkeypatch, capsys):
    args, _, st = command(monkeypatch)
    st.set_commitment.return_value = SimpleNamespace(success=False, message="SpaceLimitExceeded")
    with pytest.raises(RuntimeError, match="SpaceLimitExceeded"):
        args.func(args)
    assert capsys.readouterr().out == ""
    st.get_commitment_metadata.assert_not_called()
    st.close.assert_called_once()


def test_concurrent_replacement_cannot_report_success(monkeypatch, capsys):
    args, _, st = command(monkeypatch)
    st.get_commitment_metadata.return_value = metadata("Cacheon_28ef5_replacement")
    with pytest.raises(ValueError, match="does not match"):
        args.func(args)
    assert capsys.readouterr().out == ""
    st.close.assert_called_once()


def test_claim_requires_original_registration_even_after_same_name_is_repeated():
    st = subtensor()
    owner = {"username": NAME, "block": 70}
    claim = {"hotkey": HOTKEY, "team": NAME, "block": 90, "okc_user_block": 80}
    with pytest.raises(okc.InvalidRegistration, match="oldest accepted"):
        okc.check_claim_registration(st, 14, claim, owner, 100)
    claim["okc_user_block"] = 70
    assert okc.check_claim_registration(st, 14, claim, owner, 100) is owner
    st.get_commitment_metadata.assert_not_called()
    claim["team"] = NAME.lower()
    with pytest.raises(okc.InvalidRegistration, match="exactly match"):
        okc.check_claim_registration(st, 14, claim, owner, 100)


@pytest.mark.parametrize("block", [None, True, 0, 91, 90, 110])
def test_registration_must_precede_claim_not_okc_job(block):
    st = subtensor()
    claim = {"hotkey": HOTKEY, "team": NAME, "block": 90, "okc_user_block": block}
    with pytest.raises(okc.InvalidRegistration, match="before the claim"):
        okc.check_claim_registration(st, 14, claim, {"username": NAME, "block": 70}, 100)
    st.get_commitment_metadata.assert_not_called()
