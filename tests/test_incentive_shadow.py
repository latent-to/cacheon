from __future__ import annotations

import argparse
import ast
import json
import sys
import types
from pathlib import Path

import pytest

from optima import chain
import optima.cli as cli
from optima.finite_debt import (
    CampaignBudgetShare,
    PPM,
    DebtClaimBalance,
    DebtClaimState,
    FiniteDebtPolicyManifest,
    RewardFamilyCampaign,
    issue_innovation_claim,
)
from optima.incentive_shadow import (
    MAX_SHADOW_INPUT_BYTES,
    IncentiveShadowError,
    SyntheticClaimStateFixture,
    execute_chain_incentive_shadow,
    load_shadow_inputs,
)
from optima.stack_identity import canonical_json_bytes, sha256_hex


FAMILY = sha256_hex(b"shadow-family")
CAMPAIGN = sha256_hex(b"minimax-m3-campaign")


def _digest(label: str) -> str:
    return sha256_hex(label.encode("utf-8"))


def _policy(*, reserve_hotkey: str = "reserve") -> FiniteDebtPolicyManifest:
    return FiniteDebtPolicyManifest(
        campaign_budget_shares=(CampaignBudgetShare(CAMPAIGN, PPM),),
        reward_family_campaigns=(RewardFamilyCampaign(FAMILY, CAMPAIGN),),
        selection_report_digest=_digest("d015-selection-report"),
        reserve_hotkey=reserve_hotkey,
        reserve_ppm=100_000,
        epoch_blocks=7_200,
        beta_ppm=100_000,
        tau_blocks=648_000,
        lifetime_blocks=648_000,
        k_ppm=PPM,
        improvement_basis="gross",
        clock_reset_threshold_log_units_ppm=1,
    )


def _state(
    policy: FiniteDebtPolicyManifest,
    *,
    label: str = "one",
    hotkey: str = "miner",
    settlement_block: int = 120,
) -> DebtClaimState:
    claim = issue_innovation_claim(
        policy,
        family_id=FAMILY,
        candidate_digest=_digest(f"candidate:{label}"),
        retained_evidence_digest=_digest(f"evidence:{label}"),
        hotkey=hotkey,
        settled_speedup="1.01",
        threshold_speedup="1",
        accepted_crown_block=100,
        prior_accepted_crown_block=None,
        settlement_block=settlement_block,
    )
    return DebtClaimState(claim, DebtClaimBalance.open(claim))


def _write_inputs(
    root: Path,
    policy: FiniteDebtPolicyManifest,
    states: tuple[DebtClaimState, ...],
) -> tuple[Path, Path, SyntheticClaimStateFixture]:
    fixture = SyntheticClaimStateFixture(
        policy.digest,
        tuple(sorted(states, key=lambda row: row.claim.digest)),
    )
    policy_path = root / "policy.json"
    claims_path = root / "claims.json"
    policy_path.write_bytes(canonical_json_bytes(policy.to_dict()))
    claims_path.write_bytes(canonical_json_bytes(fixture.to_dict()))
    return policy_path, claims_path, fixture


class _ReadOnlySubtensor:
    def __init__(
        self,
        *,
        hotkeys: tuple[str, ...],
        uids: tuple[int, ...] | None = None,
        block: int = 200,
        change_second_metagraph: bool = False,
    ) -> None:
        self.hotkeys = hotkeys
        self.uids = uids or tuple(range(len(hotkeys)))
        self.block = block
        self.change_second_metagraph = change_second_metagraph
        self.calls: list[tuple[str, object]] = []
        self.metagraph_count = 0

    def get_block_hash(self, block: int) -> str:
        self.calls.append(("get_block_hash", block))
        return "0x" + f"{block:064x}"

    def get_finalized_block_number(self) -> int:
        self.calls.append(("get_finalized_block_number", None))
        return self.block

    def metagraph(self, *, netuid: int, block: int):
        self.calls.append(("metagraph", (netuid, block)))
        self.metagraph_count += 1
        hotkeys = list(self.hotkeys)
        uids = list(self.uids)
        if self.change_second_metagraph and self.metagraph_count == 2:
            hotkeys.append("late-member")
            uids.append(max(uids, default=-1) + 1)
        return types.SimpleNamespace(
            netuid=netuid,
            block=block,
            uids=uids,
            hotkeys=hotkeys,
            validator_permit=[True] * len(hotkeys),
            last_update=[0] * len(hotkeys),
        )

    def _mutation(self, *args, **kwargs):  # pragma: no cover - must never run
        del args, kwargs
        raise AssertionError("shadow path touched a chain mutation surface")

    set_weights = _mutation
    set_commitment = _mutation
    set_reveal_commitment = _mutation
    burned_register = _mutation
    root_register = _mutation


def _execute(
    tmp_path: Path,
    *,
    policy: FiniteDebtPolicyManifest | None = None,
    states: tuple[DebtClaimState, ...] | None = None,
    subtensor: _ReadOnlySubtensor | None = None,
    output_name: str = "receipt.json",
    network: str = "wss://private-endpoint.invalid",
):
    selected_policy = policy or _policy()
    selected_states = (
        (_state(selected_policy),) if states is None else states
    )
    policy_path, claims_path, fixture = _write_inputs(
        tmp_path, selected_policy, selected_states
    )
    selected_subtensor = subtensor or _ReadOnlySubtensor(
        hotkeys=(selected_policy.reserve_hotkey, "miner"),
        uids=(9, 7),
    )
    connect_calls: list[str] = []

    def connect(value: str):
        connect_calls.append(value)
        return selected_subtensor

    output = tmp_path / output_name
    receipt = execute_chain_incentive_shadow(
        network=network,
        netuid=307,
        policy_path=policy_path,
        claims_fixture_path=claims_path,
        expected_policy_digest=selected_policy.digest,
        expected_claims_digest=fixture.digest,
        output_path=output,
        connect=connect,
        read_finalized_head=chain.read_finalized_head,
        fetch_metagraph=chain.fetch_metagraph,
    )
    return receipt, output, selected_subtensor, connect_calls


def test_shadow_is_deterministic_chain_bound_and_signer_free(tmp_path: Path) -> None:
    left_root = tmp_path / "left"
    right_root = tmp_path / "right"
    left_root.mkdir()
    right_root.mkdir()
    left, left_path, left_chain, left_connect = _execute(left_root)
    right, right_path, right_chain, right_connect = _execute(right_root)

    assert left.digest == right.digest
    assert left_path.read_bytes() == right_path.read_bytes()
    payload = json.loads(left_path.read_bytes())
    assert payload["receipt_digest"] == left.digest
    assert payload["receipt"]["submitted"] is False
    assert payload["receipt"]["non_authority"] == {
        "claims_source": "synthetic_fixture",
        "publication_authority": "none",
        "settlement_authority": "none",
    }
    assert payload["receipt"]["projection"]["miners"] == [
        {"hotkey": "miner", "ppm": 900_000, "uid": 7}
    ]
    assert payload["receipt"]["projection"]["reserve"] == {
        "hotkey": "reserve",
        "ppm": 100_000,
        "uid": 9,
    }
    assert payload["receipt"]["projection"]["total_ppm"] == PPM
    assert b"private-endpoint" not in left_path.read_bytes()
    assert left_connect == right_connect == ["wss://private-endpoint.invalid"]
    assert [call for call in left_chain.calls if call[0] == "metagraph"] == [
        ("metagraph", (307, 200)),
        ("metagraph", (307, 200)),
    ]
    assert sum(call == ("get_block_hash", 0) for call in left_chain.calls) == 2
    assert {name for name, _value in left_chain.calls} <= {
        "get_block_hash",
        "get_finalized_block_number",
        "metagraph",
    }
    assert {name for name, _value in right_chain.calls} <= {
        "get_block_hash",
        "get_finalized_block_number",
        "metagraph",
    }


def test_reserve_only_projection_still_requires_and_retains_reserve_uid(
    tmp_path: Path,
) -> None:
    policy = _policy()
    chain_view = _ReadOnlySubtensor(hotkeys=("reserve",), uids=(42,))
    receipt, output, _subtensor, _connect = _execute(
        tmp_path, policy=policy, states=(), subtensor=chain_view
    )
    projection = json.loads(output.read_bytes())["receipt"]["projection"]
    assert receipt.miners == ()
    assert projection["miners"] == []
    assert projection["payout_ppm"] == 0
    assert projection["reserve"] == {
        "hotkey": "reserve",
        "ppm": PPM,
        "uid": 42,
    }


@pytest.mark.parametrize(
    ("hotkeys", "match"),
    [
        (("reserve",), "positive finite-debt miner"),
        (("miner",), "reserve hotkey"),
    ],
)
def test_shadow_rejects_deregistered_miner_or_reserve(
    tmp_path: Path, hotkeys: tuple[str, ...], match: str
) -> None:
    policy = _policy()
    with pytest.raises(IncentiveShadowError, match=match):
        _execute(
            tmp_path,
            policy=policy,
            states=(_state(policy),),
            subtensor=_ReadOnlySubtensor(hotkeys=hotkeys),
        )
    assert not (tmp_path / "receipt.json").exists()


def test_shadow_rejects_future_claim_authority_before_writing(tmp_path: Path) -> None:
    policy = _policy()
    future = _state(policy, settlement_block=300)
    with pytest.raises(IncentiveShadowError, match="future chain authority"):
        _execute(tmp_path, policy=policy, states=(future,))
    assert not (tmp_path / "receipt.json").exists()


def test_shadow_rejects_changed_metagraph_on_exact_reopen(tmp_path: Path) -> None:
    policy = _policy()
    subtensor = _ReadOnlySubtensor(
        hotkeys=("reserve", "miner"),
        change_second_metagraph=True,
    )
    with pytest.raises(IncentiveShadowError, match="metagraph authority changed"):
        _execute(tmp_path, policy=policy, subtensor=subtensor)
    assert not (tmp_path / "receipt.json").exists()


@pytest.mark.parametrize(
    "bad_policy",
    [
        b'{"x":1,"x":2}',
        b'{"x":1.5}',
        b'{ "x":1}',
        b"x" * (MAX_SHADOW_INPUT_BYTES + 1),
    ],
    ids=("duplicate-key", "float", "noncanonical", "oversize"),
)
def test_shadow_rejects_noncanonical_or_unbounded_input_before_connect(
    tmp_path: Path, bad_policy: bytes
) -> None:
    policy = _policy()
    policy_path, claims_path, fixture = _write_inputs(tmp_path, policy, ())
    policy_path.write_bytes(bad_policy)
    connected = False

    def connect(_network: str):
        nonlocal connected
        connected = True
        raise AssertionError("invalid inputs must fail before network access")

    with pytest.raises(IncentiveShadowError):
        execute_chain_incentive_shadow(
            network="test",
            netuid=307,
            policy_path=policy_path,
            claims_fixture_path=claims_path,
            expected_policy_digest=policy.digest,
            expected_claims_digest=fixture.digest,
            output_path=tmp_path / "receipt.json",
            connect=connect,
            read_finalized_head=chain.read_finalized_head,
            fetch_metagraph=chain.fetch_metagraph,
        )
    assert connected is False


def test_claim_fixture_must_be_explicitly_synthetic_and_semantically_pinned(
    tmp_path: Path,
) -> None:
    policy = _policy()
    fixture = SyntheticClaimStateFixture(policy.digest, ())
    policy_path = tmp_path / "policy.json"
    claims_path = tmp_path / "claims.json"
    policy_path.write_bytes(canonical_json_bytes(policy.to_dict()))
    claims = fixture.to_dict()
    claims["fixture_kind"] = "retained"
    claims_path.write_bytes(canonical_json_bytes(claims))
    with pytest.raises(IncentiveShadowError, match="explicitly synthetic"):
        load_shadow_inputs(
            policy_path=policy_path,
            claims_fixture_path=claims_path,
            expected_policy_digest=policy.digest,
            expected_claims_digest=fixture.digest,
        )

    claims_path.write_bytes(canonical_json_bytes(fixture.to_dict()))
    with pytest.raises(IncentiveShadowError, match="semantic digest differs"):
        load_shadow_inputs(
            policy_path=policy_path,
            claims_fixture_path=claims_path,
            expected_policy_digest=policy.digest,
            expected_claims_digest=_digest("wrong-fixture"),
        )


@pytest.mark.parametrize("kind", ["file", "symlink"])
def test_shadow_refuses_existing_or_symlink_output_before_connect(
    tmp_path: Path, kind: str
) -> None:
    policy = _policy()
    policy_path, claims_path, fixture = _write_inputs(tmp_path, policy, ())
    output = tmp_path / "receipt.json"
    if kind == "file":
        output.write_text("owned", encoding="utf-8")
    else:
        target = tmp_path / "target"
        target.write_text("owned", encoding="utf-8")
        output.symlink_to(target)
    connected = False

    def connect(_network: str):
        nonlocal connected
        connected = True
        raise AssertionError("existing output must fail before network access")

    with pytest.raises(IncentiveShadowError, match="already exists"):
        execute_chain_incentive_shadow(
            network="test",
            netuid=307,
            policy_path=policy_path,
            claims_fixture_path=claims_path,
            expected_policy_digest=policy.digest,
            expected_claims_digest=fixture.digest,
            output_path=output,
            connect=connect,
            read_finalized_head=chain.read_finalized_head,
            fetch_metagraph=chain.fetch_metagraph,
        )
    assert connected is False
    if kind == "file":
        assert output.read_text(encoding="utf-8") == "owned"
    else:
        assert output.is_symlink()


def test_shadow_module_has_no_signer_storage_or_publication_imports() -> None:
    source_path = Path(__file__).parents[1] / "optima" / "incentive_shadow.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
    forbidden = {
        "bittensor",
        "sqlite3",
        "optima.chain.intake",
        "optima.chain.weights",
        "optima.economics",
    }
    assert not imports & forbidden


def test_cli_surface_has_no_signer_intake_or_publication_options() -> None:
    parser = cli.build_parser()
    subparsers = next(
        action
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    )
    command = subparsers.choices["chain-incentive-shadow"]
    options = {
        option
        for action in command._actions
        for option in action.option_strings
    }
    assert {
        "--network",
        "--netuid",
        "--policy",
        "--claims-fixture",
        "--expected-policy-digest",
        "--expected-claims-digest",
        "--output",
    } <= options
    assert not {
        "--wallet",
        "--hotkey",
        "--validator-hotkey",
        "--intake-db",
        "--dry-run",
        "--refresh-blocks",
        "--reconcile-only",
        "--release-hold",
    } & options


def test_cli_uses_subtensor_without_constructing_wallet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    policy = _policy()
    policy_path, claims_path, fixture = _write_inputs(
        tmp_path, policy, (_state(policy),)
    )
    subtensor = _ReadOnlySubtensor(
        hotkeys=("reserve", "miner"), uids=(9, 7)
    )
    fake_bt = types.ModuleType("bittensor")
    fake_bt.Subtensor = lambda *, network: subtensor  # type: ignore[attr-defined]

    def reject_wallet(*args, **kwargs):
        del args, kwargs
        raise AssertionError("shadow CLI constructed a wallet")

    fake_bt.Wallet = reject_wallet  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "bittensor", fake_bt)
    output = tmp_path / "receipt.json"
    rc = cli.main(
        [
            "chain-incentive-shadow",
            "--network",
            "test",
            "--netuid",
            "307",
            "--policy",
            str(policy_path),
            "--claims-fixture",
            str(claims_path),
            "--expected-policy-digest",
            policy.digest,
            "--expected-claims-digest",
            fixture.digest,
            "--output",
            str(output),
        ]
    )
    assert rc == 0
    assert json.loads(capsys.readouterr().out)["submitted"] is False
    assert output.exists()
