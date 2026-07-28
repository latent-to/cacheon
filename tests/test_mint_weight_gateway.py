"""Deploy helper for a dedicated serve-weights HTTP authority."""

from __future__ import annotations

import json
import stat
from argparse import Namespace
from pathlib import Path

import optima.cli as cli
from optima.chain.weight_push_auth import (
    PushCredentialSet,
    mint_push_credential,
    write_push_credentials,
)


class _FakeKeypair:
    def __init__(self, ss58: str = "5GatewayAuthorityHotkey"):
        self.ss58_address = ss58

    @staticmethod
    def generate_mnemonic(n_words: int = 12) -> str:
        assert n_words == 12
        return "abandon " * 11 + "about"

    @classmethod
    def create_from_mnemonic(cls, mnemonic: str):
        assert mnemonic
        return cls()


class _FakeWallet:
    def __init__(self, name, hotkey, path=None):
        self.name = name
        self.hotkey_name = hotkey
        self.path = path
        self.hotkey = _FakeKeypair()
        self._hotkey_written = False

    def set_coldkey(self, keypair, encrypt=False, overwrite=False):
        assert encrypt is False
        assert keypair is not None

    def set_hotkey(self, keypair, encrypt=False, overwrite=False):
        assert encrypt is False
        assert keypair is not None
        # Mimic wallet file layout so --force / refuse checks work.
        root = Path(self.path) / self.name / "hotkeys"
        root.mkdir(parents=True, exist_ok=True)
        (root / self.hotkey_name).write_text("fake-hotkey\n")
        self._hotkey_written = True
        self.hotkey = keypair


def _ns(tmp_path: Path, **overrides) -> Namespace:
    base = dict(
        wallet_path=str(tmp_path / "wallets"),
        wallet="gateway",
        hotkey="authority",
        secrets_out="",
        manifest_out="",
        push_credentials="",
        credential_id="",
        force=False,
    )
    base.update(overrides)
    return Namespace(**base)


def test_mint_weight_gateway_creates_separate_authority(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    monkeypatch.setattr(cli, "_resolve_wallet_stack", lambda: (_FakeWallet, _FakeKeypair))
    push_path = tmp_path / "push.json"
    assert cli.cmd_mint_weight_gateway(_ns(tmp_path, push_credentials=str(push_path), credential_id="deploy-1")) == 0
    out = capsys.readouterr().out
    assert "authority_ss58=5GatewayAuthorityHotkey" in out
    assert "--expected-authority" in out

    wallet_path = tmp_path / "wallets"
    manifest = json.loads((wallet_path / "gateway" / "AUTHORITY.json").read_text())
    secrets = json.loads(
        (wallet_path / "gateway" / "authority.mnemonics.json").read_text()
    )
    assert manifest["authority_ss58"] == secrets["authority_ss58"]
    assert manifest["role"] == "serve-weights-http-authority"
    assert secrets["hotkey_mnemonic"]
    assert "coldkey_mnemonic" not in secrets
    assert (wallet_path / "gateway" / "hotkeys" / "authority").exists()
    assert push_path.exists()
    mode = (wallet_path / "gateway" / "authority.mnemonics.json").stat().st_mode
    assert mode & (stat.S_IRWXG | stat.S_IRWXO) == 0
    # Shared wallets root must not be locked down to 0700.
    root_mode = wallet_path.stat().st_mode
    assert root_mode & stat.S_IRUSR


def test_mint_weight_gateway_refuses_existing_hotkey_without_force(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(cli, "_resolve_wallet_stack", lambda: (_FakeWallet, _FakeKeypair))
    assert cli.cmd_mint_weight_gateway(_ns(tmp_path)) == 0
    try:
        cli.cmd_mint_weight_gateway(_ns(tmp_path))
        raise AssertionError("expected SystemExit")
    except SystemExit as exc:
        assert "already exists" in str(exc)


def test_mint_weight_gateway_refuses_push_cred_clobber_without_force(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(cli, "_resolve_wallet_stack", lambda: (_FakeWallet, _FakeKeypair))
    push_path = tmp_path / "push.json"
    write_push_credentials(
        push_path, PushCredentialSet((mint_push_credential(credential_id="old"),))
    )
    # First mint with a different hotkey name so only push-cred path is tested.
    try:
        cli.cmd_mint_weight_gateway(
            _ns(
                tmp_path,
                hotkey="authority-2",
                push_credentials=str(push_path),
                credential_id="new",
                force=False,
            )
        )
        raise AssertionError("expected SystemExit")
    except SystemExit as exc:
        assert "already exist" in str(exc)
    # Original credential preserved.
    payload = json.loads(push_path.read_text())
    assert payload["credentials"][0]["credential_id"] == "old"
    assert payload["credentials"][0]["status"] == "active"


def test_mint_weight_gateway_force_appends_push_cred(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(cli, "_resolve_wallet_stack", lambda: (_FakeWallet, _FakeKeypair))
    push_path = tmp_path / "push.json"
    write_push_credentials(
        push_path, PushCredentialSet((mint_push_credential(credential_id="old"),))
    )
    assert (
        cli.cmd_mint_weight_gateway(
            _ns(
                tmp_path,
                hotkey="authority-3",
                push_credentials=str(push_path),
                credential_id="new",
                force=True,
            )
        )
        == 0
    )
    payload = json.loads(push_path.read_text())
    by_id = {row["credential_id"]: row for row in payload["credentials"]}
    assert by_id["old"]["status"] == "retired"
    assert by_id["new"]["status"] == "active"
