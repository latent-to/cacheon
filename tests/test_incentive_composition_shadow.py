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
    PPM,
    DebtClaimBalance,
    DebtClaimState,
    FamilyBudgetShare,
    FiniteDebtPolicyManifest,
    issue_innovation_claim,
)
from optima.incentive_composition import (
    DISCOVERY_BOUNTY_ONLY,
    DiscoveryClaimBalance,
    DiscoveryClaimState,
    IncentiveCompositionPolicyManifest,
    issue_discovery_claim,
    review_discovery_disposition,
)
from optima.incentive_composition_shadow import (
    IncentiveCompositionShadowError,
    SyntheticDiscoveryStateFixture,
    execute_chain_incentive_composition_shadow,
    load_composed_shadow_inputs,
)
from optima.incentive_shadow import SyntheticClaimStateFixture
from optima.stack_identity import canonical_json_bytes, sha256_hex


FAMILY = sha256_hex(b"composition-shadow-family")


def _digest(label: str) -> str:
    return sha256_hex(label.encode("utf-8"))


def _core_policy() -> FiniteDebtPolicyManifest:
    return FiniteDebtPolicyManifest(
        family_budget_shares=(FamilyBudgetShare(FAMILY, PPM),),
        reserve_hotkey="reserve",
        reserve_ppm=100_000,
        epoch_blocks=7_200,
        beta_ppm=100_000,
        tau_blocks=648_000,
        lifetime_blocks=648_000,
        k_ppm=PPM,
        improvement_basis="gross",
        clock_reset_threshold_log_units_ppm=1,
    )


def _composition_policy(
    core: FiniteDebtPolicyManifest,
) -> IncentiveCompositionPolicyManifest:
    return IncentiveCompositionPolicyManifest(
        innovation_policy_digest=core.digest,
        selection_report_digest=_digest("d013-selection-report"),
        reserve_ppm=core.reserve_ppm,
        epoch_blocks=core.epoch_blocks,
        discovery_cap_units=50_000,
        per_award_principal_cap_epochs=1,
        discovery_lifetime_blocks=648_000,
    )


def _core_state(
    policy: FiniteDebtPolicyManifest,
    *,
    hotkey: str = "core-miner",
    settlement_block: int = 120,
) -> DebtClaimState:
    claim = issue_innovation_claim(
        policy,
        family_id=FAMILY,
        candidate_digest=_digest(f"core-candidate:{hotkey}"),
        retained_evidence_digest=_digest(f"core-evidence:{hotkey}"),
        hotkey=hotkey,
        settled_speedup="1.01",
        threshold_speedup="1",
        accepted_crown_block=100,
        prior_accepted_crown_block=None,
        settlement_block=settlement_block,
    )
    return DebtClaimState(claim, DebtClaimBalance.open(claim))


def _discovery_state(
    policy: IncentiveCompositionPolicyManifest,
    *,
    hotkey: str = "discovery-miner",
    authority_block: int = 110,
) -> DiscoveryClaimState:
    disposition = review_discovery_disposition(
        policy,
        win_digest=_digest(f"discovery-win:{hotkey}"),
        proposal_digest=_digest(f"discovery-proposal:{hotkey}"),
        retained_evidence_digest=_digest(f"discovery-evidence:{hotkey}"),
        review_digest=_digest(f"discovery-review:{hotkey}"),
        hotkey=hotkey,
        win_block=authority_block,
        authority_block=authority_block,
        decision=DISCOVERY_BOUNTY_ONLY,
        requested_principal_epochs=1,
    )
    claim = issue_discovery_claim(policy, disposition)
    assert claim is not None
    return DiscoveryClaimState(claim, DiscoveryClaimBalance.open(claim))


def _write_inputs(
    root: Path,
    core: FiniteDebtPolicyManifest,
    composition: IncentiveCompositionPolicyManifest,
    core_states: tuple[DebtClaimState, ...],
    discovery_states: tuple[DiscoveryClaimState, ...],
) -> tuple[dict[str, Path], SyntheticClaimStateFixture, SyntheticDiscoveryStateFixture]:
    core_fixture = SyntheticClaimStateFixture(
        core.digest,
        tuple(sorted(core_states, key=lambda row: row.claim.digest)),
    )
    discovery_fixture = SyntheticDiscoveryStateFixture(
        composition.digest,
        tuple(sorted(discovery_states, key=lambda row: row.claim.digest)),
    )
    paths = {
        "core_policy": root / "core-policy.json",
        "core_claims": root / "core-claims.json",
        "discovery_policy": root / "discovery-policy.json",
        "discovery_claims": root / "discovery-claims.json",
    }
    paths["core_policy"].write_bytes(canonical_json_bytes(core.to_dict()))
    paths["core_claims"].write_bytes(canonical_json_bytes(core_fixture.to_dict()))
    paths["discovery_policy"].write_bytes(
        canonical_json_bytes(composition.to_dict())
    )
    paths["discovery_claims"].write_bytes(
        canonical_json_bytes(discovery_fixture.to_dict())
    )
    return paths, core_fixture, discovery_fixture


class _ReadOnlySubtensor:
    def __init__(
        self,
        *,
        hotkeys: tuple[str, ...],
        uids: tuple[int, ...] | None = None,
        block: int = 200,
        change_second_metagraph: bool = False,
        change_reopened_genesis: bool = False,
    ) -> None:
        self.hotkeys = hotkeys
        self.uids = uids or tuple(range(len(hotkeys)))
        self.block = block
        self.change_second_metagraph = change_second_metagraph
        self.change_reopened_genesis = change_reopened_genesis
        self.calls: list[tuple[str, object]] = []
        self.metagraph_count = 0
        self.genesis_count = 0

    def get_block_hash(self, block: int) -> str:
        self.calls.append(("get_block_hash", block))
        if block == 0:
            self.genesis_count += 1
            changed = self.change_reopened_genesis and self.genesis_count >= 2
            return "0x" + ("f" if changed else "0") * 64
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
        raise AssertionError("composed shadow touched a chain mutation surface")

    set_weights = _mutation
    set_commitment = _mutation
    set_reveal_commitment = _mutation
    burned_register = _mutation
    root_register = _mutation


def _execute(
    root: Path,
    *,
    core: FiniteDebtPolicyManifest | None = None,
    composition: IncentiveCompositionPolicyManifest | None = None,
    core_states: tuple[DebtClaimState, ...] | None = None,
    discovery_states: tuple[DiscoveryClaimState, ...] | None = None,
    subtensor: _ReadOnlySubtensor | None = None,
):
    selected_core = core or _core_policy()
    selected_composition = composition or _composition_policy(selected_core)
    selected_core_states = (
        (_core_state(selected_core),) if core_states is None else core_states
    )
    selected_discovery_states = (
        (_discovery_state(selected_composition),)
        if discovery_states is None
        else discovery_states
    )
    paths, core_fixture, discovery_fixture = _write_inputs(
        root,
        selected_core,
        selected_composition,
        selected_core_states,
        selected_discovery_states,
    )
    selected_subtensor = subtensor or _ReadOnlySubtensor(
        hotkeys=("reserve", "discovery-miner", "core-miner"),
        uids=(9, 5, 7),
    )
    connect_calls: list[str] = []

    def connect(network: str):
        connect_calls.append(network)
        return selected_subtensor

    output = root / "receipt.json"
    receipt = execute_chain_incentive_composition_shadow(
        network="wss://private-endpoint.invalid",
        netuid=307,
        core_policy_path=paths["core_policy"],
        core_claims_fixture_path=paths["core_claims"],
        discovery_policy_path=paths["discovery_policy"],
        discovery_claims_fixture_path=paths["discovery_claims"],
        expected_core_policy_digest=selected_core.digest,
        expected_core_claims_digest=core_fixture.digest,
        expected_discovery_policy_digest=selected_composition.digest,
        expected_discovery_claims_digest=discovery_fixture.digest,
        output_path=output,
        connect=connect,
        read_finalized_head=chain.read_finalized_head,
        fetch_metagraph=chain.fetch_metagraph,
    )
    return receipt, output, selected_subtensor, connect_calls


def test_composed_shadow_is_deterministic_exact_and_class_conserving(
    tmp_path: Path,
) -> None:
    left_root = tmp_path / "left"
    right_root = tmp_path / "right"
    left_root.mkdir()
    right_root.mkdir()
    left, left_path, left_chain, left_connect = _execute(left_root)
    right, right_path, _right_chain, right_connect = _execute(right_root)

    assert left.digest == right.digest
    assert left_path.read_bytes() == right_path.read_bytes()
    payload = json.loads(left_path.read_bytes())
    receipt = payload["receipt"]
    assert payload["receipt_digest"] == left.digest
    assert receipt["submitted"] is False
    assert receipt["mode"] == "synthetic"
    assert receipt["non_authority"] == {
        "core_claims_source": "synthetic_fixture",
        "discovery_claims_source": "synthetic_fixture",
        "publication_authority": "none",
        "review_authority": "none",
        "settlement_authority": "none",
    }
    classes = receipt["projection"]["classes"]
    assert classes["reviewed_discovery"]["payout_units"] == 50_000
    assert classes["registered_crown"]["payout_units"] == 850_000
    assert classes["reviewed_discovery"]["recipients"] == [
        {"hotkey": "discovery-miner", "uid": 5, "units": 50_000}
    ]
    assert classes["registered_crown"]["recipients"] == [
        {"hotkey": "core-miner", "uid": 7, "units": 850_000}
    ]
    assert receipt["projection"]["miners"] == [
        {"hotkey": "core-miner", "ppm": 850_000, "uid": 7},
        {"hotkey": "discovery-miner", "ppm": 50_000, "uid": 5},
    ]
    assert receipt["projection"]["reserve"] == {
        "hotkey": "reserve",
        "ppm": 100_000,
        "uid": 9,
    }
    assert sum(
        row["units"]
        for name in ("reviewed_discovery", "registered_crown")
        for row in classes[name]["allocations"]
    ) + receipt["projection"]["reserve"]["ppm"] == PPM
    assert b"private-endpoint" not in left_path.read_bytes()
    assert left_connect == right_connect == ["wss://private-endpoint.invalid"]
    assert [call for call in left_chain.calls if call[0] == "metagraph"] == [
        ("metagraph", (307, 200)),
        ("metagraph", (307, 200)),
    ]
    assert {name for name, _value in left_chain.calls} <= {
        "get_block_hash",
        "get_finalized_block_number",
        "metagraph",
    }


@pytest.mark.parametrize(
    ("hotkeys", "match"),
    [
        (("reserve", "core-miner"), "reviewed-discovery"),
        (("reserve", "discovery-miner"), "registered-CROWN"),
        (("core-miner", "discovery-miner"), "reserve hotkey"),
    ],
)
def test_composed_shadow_refuses_absent_or_deregistered_recipients(
    tmp_path: Path,
    hotkeys: tuple[str, ...],
    match: str,
) -> None:
    with pytest.raises(IncentiveCompositionShadowError, match=match):
        _execute(tmp_path, subtensor=_ReadOnlySubtensor(hotkeys=hotkeys))
    assert not (tmp_path / "receipt.json").exists()


@pytest.mark.parametrize("kind", ["core", "discovery"])
def test_composed_shadow_refuses_future_synthetic_authority(
    tmp_path: Path,
    kind: str,
) -> None:
    core = _core_policy()
    composition = _composition_policy(core)
    kwargs: dict[str, object] = {"core": core, "composition": composition}
    if kind == "core":
        kwargs["core_states"] = (_core_state(core, settlement_block=300),)
    else:
        kwargs["discovery_states"] = (
            _discovery_state(composition, authority_block=300),
        )
    with pytest.raises(IncentiveCompositionShadowError, match="future chain authority"):
        _execute(tmp_path, **kwargs)  # type: ignore[arg-type]
    assert not (tmp_path / "receipt.json").exists()


@pytest.mark.parametrize("change", ["metagraph", "genesis"])
def test_composed_shadow_refuses_changed_reopened_chain_authority(
    tmp_path: Path,
    change: str,
) -> None:
    subtensor = _ReadOnlySubtensor(
        hotkeys=("reserve", "core-miner", "discovery-miner"),
        change_second_metagraph=change == "metagraph",
        change_reopened_genesis=change == "genesis",
    )
    with pytest.raises(ValueError, match="changed|regressed"):
        _execute(tmp_path, subtensor=subtensor)
    assert not (tmp_path / "receipt.json").exists()


def test_composed_inputs_require_canonical_bytes_and_all_semantic_digests(
    tmp_path: Path,
) -> None:
    core = _core_policy()
    composition = _composition_policy(core)
    paths, core_fixture, discovery_fixture = _write_inputs(
        tmp_path,
        core,
        composition,
        (_core_state(core),),
        (_discovery_state(composition),),
    )
    paths["discovery_policy"].write_bytes(
        b" " + canonical_json_bytes(composition.to_dict())
    )
    with pytest.raises(ValueError, match="canonically encoded"):
        load_composed_shadow_inputs(
            core_policy_path=paths["core_policy"],
            core_claims_fixture_path=paths["core_claims"],
            discovery_policy_path=paths["discovery_policy"],
            discovery_claims_fixture_path=paths["discovery_claims"],
            expected_core_policy_digest=core.digest,
            expected_core_claims_digest=core_fixture.digest,
            expected_discovery_policy_digest=composition.digest,
            expected_discovery_claims_digest=discovery_fixture.digest,
        )


@pytest.mark.parametrize(
    ("wrong_field", "match"),
    [
        ("expected_core_policy_digest", "core policy semantic digest differs"),
        ("expected_core_claims_digest", "core claims semantic digest differs"),
        (
            "expected_discovery_policy_digest",
            "discovery policy semantic digest differs",
        ),
        (
            "expected_discovery_claims_digest",
            "discovery claims semantic digest differs",
        ),
    ],
)
def test_composed_inputs_pin_each_expected_semantic_digest(
    tmp_path: Path,
    wrong_field: str,
    match: str,
) -> None:
    core = _core_policy()
    composition = _composition_policy(core)
    paths, core_fixture, discovery_fixture = _write_inputs(
        tmp_path,
        core,
        composition,
        (_core_state(core),),
        (_discovery_state(composition),),
    )
    expected = {
        "expected_core_policy_digest": core.digest,
        "expected_core_claims_digest": core_fixture.digest,
        "expected_discovery_policy_digest": composition.digest,
        "expected_discovery_claims_digest": discovery_fixture.digest,
    }
    expected[wrong_field] = _digest(f"wrong:{wrong_field}")
    with pytest.raises(IncentiveCompositionShadowError, match=match):
        load_composed_shadow_inputs(
            core_policy_path=paths["core_policy"],
            core_claims_fixture_path=paths["core_claims"],
            discovery_policy_path=paths["discovery_policy"],
            discovery_claims_fixture_path=paths["discovery_claims"],
            **expected,
        )


def test_composed_shadow_never_replaces_output_or_connects(tmp_path: Path) -> None:
    core = _core_policy()
    composition = _composition_policy(core)
    paths, core_fixture, discovery_fixture = _write_inputs(
        tmp_path,
        core,
        composition,
        (_core_state(core),),
        (_discovery_state(composition),),
    )
    output = tmp_path / "receipt.json"
    output.write_text("owned", encoding="utf-8")
    connected = False

    def connect(_network: str):
        nonlocal connected
        connected = True
        raise AssertionError("existing output must fail before network access")

    with pytest.raises(ValueError, match="already exists"):
        execute_chain_incentive_composition_shadow(
            network="test",
            netuid=307,
            core_policy_path=paths["core_policy"],
            core_claims_fixture_path=paths["core_claims"],
            discovery_policy_path=paths["discovery_policy"],
            discovery_claims_fixture_path=paths["discovery_claims"],
            expected_core_policy_digest=core.digest,
            expected_core_claims_digest=core_fixture.digest,
            expected_discovery_policy_digest=composition.digest,
            expected_discovery_claims_digest=discovery_fixture.digest,
            output_path=output,
            connect=connect,
            read_finalized_head=chain.read_finalized_head,
            fetch_metagraph=chain.fetch_metagraph,
        )
    assert connected is False
    assert output.read_text(encoding="utf-8") == "owned"


def test_composed_shadow_module_has_no_signer_storage_or_publication_imports() -> None:
    source_path = (
        Path(__file__).parents[1] / "optima" / "incentive_composition_shadow.py"
    )
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imports: set[str] = set()
    calls: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                calls.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                calls.add(node.func.attr)
    forbidden_imports = {
        "bittensor",
        "sqlite3",
        "optima.chain.intake",
        "optima.chain.weights",
        "optima.economics",
    }
    assert not imports & forbidden_imports
    assert not calls & {
        "Wallet",
        "set_weights",
        "set_commitment",
        "set_reveal_commitment",
        "sign",
    }


def test_composed_cli_surface_is_explicit_and_signer_free() -> None:
    parser = cli.build_parser()
    subparsers = next(
        action
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    )
    command = subparsers.choices["chain-incentive-composition-shadow"]
    options = {
        option
        for action in command._actions
        for option in action.option_strings
    }
    assert {
        "--network",
        "--netuid",
        "--core-policy",
        "--core-claims-fixture",
        "--discovery-policy",
        "--discovery-claims-fixture",
        "--expected-core-policy-digest",
        "--expected-core-claims-digest",
        "--expected-discovery-policy-digest",
        "--expected-discovery-claims-digest",
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


def test_composed_cli_uses_subtensor_without_constructing_wallet(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    core = _core_policy()
    composition = _composition_policy(core)
    paths, core_fixture, discovery_fixture = _write_inputs(
        tmp_path,
        core,
        composition,
        (_core_state(core),),
        (_discovery_state(composition),),
    )
    subtensor = _ReadOnlySubtensor(
        hotkeys=("reserve", "core-miner", "discovery-miner"),
        uids=(9, 7, 5),
    )
    fake_bt = types.ModuleType("bittensor")
    fake_bt.Subtensor = lambda *, network: subtensor  # type: ignore[attr-defined]

    def reject_wallet(*args, **kwargs):
        del args, kwargs
        raise AssertionError("composed shadow CLI constructed a wallet")

    fake_bt.Wallet = reject_wallet  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "bittensor", fake_bt)
    output = tmp_path / "receipt.json"
    rc = cli.main(
        [
            "chain-incentive-composition-shadow",
            "--network",
            "test",
            "--netuid",
            "307",
            "--core-policy",
            str(paths["core_policy"]),
            "--core-claims-fixture",
            str(paths["core_claims"]),
            "--discovery-policy",
            str(paths["discovery_policy"]),
            "--discovery-claims-fixture",
            str(paths["discovery_claims"]),
            "--expected-core-policy-digest",
            core.digest,
            "--expected-core-claims-digest",
            core_fixture.digest,
            "--expected-discovery-policy-digest",
            composition.digest,
            "--expected-discovery-claims-digest",
            discovery_fixture.digest,
            "--output",
            str(output),
        ]
    )
    assert rc == 0
    assert json.loads(capsys.readouterr().out)["submitted"] is False
    assert output.exists()
