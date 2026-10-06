"""Exercise the standing supervisor's external commissioning boundary and restart."""

import json
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest

from cacheon.chain import baseline_loading as loading
from cacheon.chain.baseline_segments import commissioned_baseline
from cacheon.chain.recoverable_qualification_dispatcher import QualificationCommissionRequired
from cacheon.chain.standing_cpu_supervisor import (
    StandingCpuSupervisor, StandingCpuSupervisorError, build_standing_supervisor,
    load_standing_config, run_forever,
)
from cacheon.stack_manifest import EvaluationStackManifest, ProposalContributionRef
from tests.test_standing_supervisor_config import _h, _private_file, _rewrite, _setup


def setup_loader(tmp_path, monkeypatch):
    path, raw = _setup(tmp_path)
    stock = EvaluationStackManifest.from_dict(json.loads(
        load_standing_config(path).qualification_incumbent_stack_path.read_text()))
    winner = EvaluationStackManifest.from_dict({**stock.to_dict(), "entries": {"target.0": ProposalContributionRef(
        target_id="target.0", target_spec_digest=_h("spec"), artifact_digest=_h("winner"),
        selected_payload_digest=_h("payload"), attribution_digest=_h("attribution"),
    ).to_dict()}})
    target = commissioned_baseline(winner, _h("winner-tree"))
    prepared = path.parent / "prepared.json"
    request = path.parent / "request.json"
    command = [sys.executable, "-c", "import json,sys; from pathlib import Path; "
               "Path(sys.argv[1]).write_text(sys.stdin.read()); "
               "print(json.dumps({'standing_config':sys.argv[2]}))", str(request), str(prepared)]
    raw.update(enable_settlement=True, settlement_network="wss://example.invalid",
               baseline_loading={"command": command, "timeout_seconds": 10, "activation_block": 10})
    _rewrite(path, raw)
    stack = path.parent / "winner.json"
    _private_file(stack, json.dumps(winner.to_dict()).encode())
    new = {**raw, "qualification_incumbent_stack_path": str(stack),
           "qualification_incumbent_tree_digest": target.tree_digest}
    _private_file(prepared, json.dumps(new).encode())
    boundary = QualificationCommissionRequired(stock.digest, winner.digest, target.tree_digest)
    dispatcher = SimpleNamespace(
        qualification_incumbent_stack=stock, dispatch_once=lambda: boundary,
        _open_store=lambda: (SimpleNamespace(close=lambda: None), None),
    )
    monkeypatch.setattr(loading, "promotion_target", lambda store, incumbent: target)
    return loading.BaselineLoader(load_standing_config(path), dispatcher), prepared, request, winner


def test_queue_boundary_invokes_command_once_and_restarts_without_a_second_claim(tmp_path, monkeypatch):
    loader, prepared, request, winner = setup_loader(tmp_path, monkeypatch)
    supervisor = StandingCpuSupervisor(qualification_once=loader)
    run_forever(supervisor, stop=threading.Event(), wait=lambda _: pytest.fail("did not reload"))
    assert supervisor.status().last_disposition == "baseline_loaded"
    assert load_standing_config(loader.path).raw == load_standing_config(prepared).raw
    assert json.loads(request.read_text())["incumbent_stack"] == winner.to_dict()
    assert not loader.pending.exists()


@pytest.mark.parametrize("failure", ["exit", "timeout", "wrong_baseline", "changed_policy"])
def test_failed_cutover_preserves_original_config_and_is_not_repeated(tmp_path, monkeypatch, failure):
    loader, prepared, request, winner = setup_loader(tmp_path, monkeypatch)
    original = loader.path.read_bytes()
    if failure in {"exit", "timeout"}:
        def fail(*args, **kwargs):
            if failure == "exit":
                raise subprocess.CalledProcessError(1, args[0])
            raise subprocess.TimeoutExpired(args[0], 10)
        monkeypatch.setattr(loading.subprocess, "run", fail)
        error = subprocess.SubprocessError
    else:
        changed = load_standing_config(prepared).raw
        if failure == "wrong_baseline":
            changed["qualification_incumbent_stack_path"] = str(loader.config.qualification_incumbent_stack_path)
        else:
            changed["baseline_loading"] = {**changed["baseline_loading"], "activation_block": 99}
        _rewrite(prepared, changed)
        error = ValueError
    with pytest.raises(error):
        loader()
    assert loader.pending.exists()
    assert loader.path.read_bytes() == original
    with pytest.raises(RuntimeError, match="incomplete"):
        loader()
    assert loader.pending.stat().st_mode & 0o777 == 0o600


def test_restart_after_config_switch_clears_marker_without_repeating_commission(tmp_path, monkeypatch):
    loader, prepared, request, winner = setup_loader(tmp_path, monkeypatch)
    loader.pending.write_text(json.dumps({"incumbent_stack": winner.to_dict()}))
    loader.dispatcher.qualification_incumbent_stack = winner
    loader.dispatcher.dispatch_once = lambda: None
    assert loader() is None
    assert not loader.pending.exists()
    assert not request.exists()


def test_no_load_while_an_old_segment_is_unresolved(tmp_path, monkeypatch):
    loader, prepared, request, winner = setup_loader(tmp_path, monkeypatch)
    monkeypatch.setattr(loading, "promotion_target", lambda store, incumbent: None)
    assert isinstance(loader(), QualificationCommissionRequired)
    assert not request.exists() and not loader.pending.exists()


def test_enabled_loading_reaches_the_real_dispatcher_and_requires_both_stages(tmp_path, monkeypatch):
    from cacheon import chain

    loader, prepared, request, winner = setup_loader(tmp_path, monkeypatch)
    monkeypatch.setattr(chain, "connect", lambda *args, **kwargs: object())
    supervisor = build_standing_supervisor(loader.config)
    assert isinstance(supervisor.qualification_once, loading.BaselineLoader)
    assert callable(supervisor.qualification_once.dispatcher.baseline_admission)
    for key in ("enable_qualification", "enable_settlement"):
        _rewrite(loader.path, {**loader.config.raw, key: False})
        with pytest.raises(StandingCpuSupervisorError, match="requires qualification and settlement"):
            load_standing_config(loader.path)


def test_exact_chain_clock_is_cached_and_missing_time_is_an_error():
    reads = []
    substrate = SimpleNamespace(get_block_hash=lambda block: f"hash-{block}",
        query=lambda *args, **kwargs: reads.append(kwargs["block_hash"]) or SimpleNamespace(value=1234567))
    clock = loading.exact_block_clock(SimpleNamespace(substrate=substrate))
    assert clock(10) == clock(10) == {"unix": 1234, "estimated": False}
    assert reads == ["hash-10"]
    substrate.query = lambda *args, **kwargs: None
    with pytest.raises(RuntimeError, match="timestamp unavailable"):
        clock(11)
