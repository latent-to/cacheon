"""The real follower journal survives gateway loss and expires to owner burn."""

from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from cacheon import chain
from cacheon.chain import follower_fallback as fallback
from cacheon.chain.intake import FinalizedIntakeStore, IntakeScope
from cacheon.chain.weight_share import CurrentWeightOffer, WeightShareError
from tests.test_weight_publication import Chain, _projection, _wallet


def _setup(tmp_path, monkeypatch, *, apply=True):
    live = Chain(apply=apply)
    scope = IntakeScope(live.get_block_hash(0), 307)
    context = fallback._context(scope, "validator", chain.fetch_metagraph(live, scope.netuid))
    projection = replace(_projection(), chain_scope_digest=scope.digest, netuid=scope.netuid,
                         metagraph_digest=context.metagraph_digest)
    source = [CurrentWeightOffer.from_legacy_projection(projection)]

    def fetch(*args, **kwargs):
        if isinstance(source[0], Exception):
            raise source[0]
        return source[0]

    monkeypatch.setattr(fallback.weight_share, "fetch_current_weights", fetch)
    monkeypatch.setattr(chain, "resolve_subnet_owner_burn_target", lambda st, netuid: SimpleNamespace(
        metagraph=chain.fetch_metagraph(st, netuid), hotkey="validator",
        owner_coldkey="owner-cold", owner_hotkey="validator"))

    def follow(now=1000, **kwargs):
        with FinalizedIntakeStore(tmp_path / "journal.sqlite3", scope=scope) as store:
            result, mode = fallback.follow_with_fallback(
                store=store, subtensor=live, wallet=_wallet(), url="https://gateway",
                expected_authority="gateway-key", max_skew_seconds=60,
                refresh_blocks=20, now=now, **kwargs)
            head = store._db.execute(
                "SELECT offer_json FROM followed_weight_publications ORDER BY sequence DESC LIMIT 1"
            ).fetchone()
            cached = store._db.execute(
                "SELECT value FROM metadata WHERE key=?", (fallback._CACHE_KEY,)).fetchone()
        return result, mode, None if head is None else CurrentWeightOffer.from_dict(json.loads(head[0])), cached

    return live, source, follow


@pytest.mark.parametrize("failure", [OSError("offline"), WeightShareError("invalid signature"), "stale"])
def test_retained_vector_survives_restart_then_burns_after_twelve_hours(tmp_path, monkeypatch, failure):
    live, source, follow = _setup(tmp_path, monkeypatch)
    first, mode, original, cached = follow()
    assert mode == "source" and first.status == "confirmed"
    if failure != "stale":
        source[0] = failure
    live.block = 200
    result, mode, retained, _ = follow(1000 + fallback.MAX_RETAINED_SECONDS)
    assert mode == "retained" and result.status == "confirmed"
    assert retained.projection.weights_ppm == original.projection.weights_ppm
    assert retained.projection.effective_block == 200
    live.block = 201
    result, mode, burned, after = follow(1001 + fallback.MAX_RETAINED_SECONDS)
    assert mode == "burn" and result.status == "confirmed"
    assert burned.projection.weights_ppm == (("validator", 1_000_000),)
    assert after[0] == cached[0]  # neither stale GETs nor fallback writes renew the clock


def test_first_boot_without_a_source_burns_and_recovers(tmp_path, monkeypatch):
    live, source, follow = _setup(tmp_path, monkeypatch)
    good = source[0]
    source[0] = OSError("offline")
    result, mode, burned, cached = follow()
    assert mode == "burn" and result.status == "confirmed" and cached is None
    assert burned.projection.crown_count == 0
    # A valid producer can be behind our locally rebound burn head: rebind it
    # at the current head without treating it as a source rollback.
    live.block = 101
    source[0] = good
    result, mode, accepted, cached = follow(1001)
    assert mode == "source" and result.status == "confirmed"
    assert accepted.projection.weights_ppm == good.projection.weights_ppm
    assert accepted.projection.effective_block == 101 and cached is not None


def test_replayed_offer_does_not_extend_age_or_wait_for_reveal(tmp_path, monkeypatch):
    live, source, follow = _setup(tmp_path, monkeypatch, apply=False)
    first, _, original, cached = follow()
    assert first.status == "confirmed"
    _, _, _, repeated = follow(2000)
    assert repeated[0] == cached[0]
    source[0] = OSError("offline")
    live.block = 101
    result, mode, burned, _ = follow(1001 + fallback.MAX_RETAINED_SECONDS)
    assert mode == "burn" and result.status == "confirmed"
    assert burned != original and live.submit_calls == 2
    assert live.weight_reads == 0


def test_dry_run_does_not_seed_cache_or_sign(tmp_path, monkeypatch):
    live, _, follow = _setup(tmp_path, monkeypatch)
    result, mode, head, cached = follow(dry_run=True)
    assert mode == "source" and result.status == "dry_run"
    assert head is None and cached is None and live.submit_calls == 0


def test_reassigned_uid_follows_hotkey_and_missing_recipient_burns(tmp_path, monkeypatch):
    live, source, follow = _setup(tmp_path, monkeypatch)
    _, _, original, _ = follow()
    source[0] = OSError("offline")
    live.block = 200
    live.hotkeys = ["validator", "bob", "alice"]
    result, mode, offer, _ = follow(2000)
    assert mode == "retained" and result.status == "confirmed"
    assert offer.projection.weights_ppm == original.projection.weights_ppm
    assert live.row == [(2, 65535)]
    live.block = 201
    live.hotkeys = ["validator", "bob", "mallory"]
    result, mode, offer, _ = follow(2001)
    assert mode == "burn" and offer.projection.weights_ppm == (("validator", 1_000_000),)


def test_same_block_recovery_defers_one_block_without_equivocation(tmp_path, monkeypatch):
    live, source, follow = _setup(tmp_path, monkeypatch)
    good = source[0]
    source[0] = OSError("offline")
    _, _, burn, _ = follow()
    source[0] = good
    _, mode, deferred, _ = follow(1001)
    assert mode == "source" and deferred == burn
    live.block += 1
    _, _, recovered, _ = follow(1002)
    assert recovered.projection.weights_ppm == good.projection.weights_ppm


@pytest.mark.parametrize("accepted", [True, False])
def test_cli_uses_fallback_when_the_gateway_is_down(tmp_path, monkeypatch, capsys, accepted):
    import argparse
    from cacheon import cli

    live, source, _ = _setup(tmp_path, monkeypatch)
    if not accepted:
        original = live.set_weights

        def reject(**kwargs):
            original(**kwargs)
            return False

        live.set_weights = reject
    source[0] = OSError("offline")
    monkeypatch.setattr(cli, "_wallet_from_args", lambda args: _wallet())
    args = argparse.Namespace(netuid=307, expected_authority="gateway-key",
        url="https://gateway", journal_db=tmp_path / "journal.sqlite3",
        refresh_blocks=20, dry_run=False)
    assert cli._cmd_follow_weights_once(args, live) == 0
    assert "mode=burn" in capsys.readouterr().out and live.submit_calls == 1


@pytest.mark.parametrize("bad_source", ["scope", "same_block", "rollback"])
def test_bad_source_cannot_poison_or_renew_the_saved_vector(tmp_path, monkeypatch, bad_source):
    live, source, follow = _setup(tmp_path, monkeypatch)
    _, _, first, cached = follow()
    original = source[0].projection
    changes = {"scope": {"chain_scope_digest": "f" * 64},
               "same_block": {"weights_ppm": (("bob", 1_000_000),)},
               "rollback": {"effective_block": 99}}
    changed = replace(original, **changes[bad_source])
    if bad_source == "rollback":
        scope = IntakeScope(live.get_block_hash(0), 307)
        changed = replace(changed, metagraph_digest=fallback._context(
            scope, "validator", chain.fetch_metagraph(live, 307, block=99)).metagraph_digest)
    source[0] = CurrentWeightOffer.from_legacy_projection(changed)
    live.block = 101
    _, mode, retained, after = follow(2000)
    assert mode == "retained" and retained.projection.weights_ppm == first.projection.weights_ppm
    assert after[0] == cached[0]
