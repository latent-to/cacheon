"""Exercise username enforcement through the actual OKC share writer and intake rows."""

import json
import sqlite3
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from scripts.arena import okc_shares as shares

A = "5CzNs6w9tmimaUwddbqWjndz7LTuQMgtDnp5bND4AuDeVsjK"
B = "5FxbrVDh7brAup3eneqGtyaUsCjXvgm6s6t5ygaikpRmQhq7"
AN, BN = "Cacheon_28ef5_exact", "Cacheon_ac494_exact"
BOARD = shares.BOARDS[0]


@pytest.fixture
def arena(tmp_path, monkeypatch):
    db = tmp_path / "intake.sqlite3"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE reservations (reservation_id TEXT, hotkey TEXT, content_hash TEXT, "
                     "url TEXT, block INTEGER, event_index INTEGER, event_subindex INTEGER)")
    args = SimpleNamespace(intake_db=db, default_hotkey="default", cache=tmp_path / "cache",
                           out=tmp_path / "shares.json", private=tmp_path / "private.json",
                           public=tmp_path / "public.json", network="archive", netuid=14,
                           username_db=tmp_path / "usernames.sqlite3", history_batch_blocks=100, start_block=1)
    args.cache.mkdir()
    owners = {A: {"username": AN, "block": 5}, B: {"username": BN, "block": 6}}
    st = SimpleNamespace(close=Mock(), get_commitment_metadata=Mock())
    boards = {board: {} for board in shares.BOARDS}
    monkeypatch.setattr(shares.chain, "connect", Mock(return_value=st))
    monkeypatch.setattr(shares, "validator_key", Mock())
    monkeypatch.setattr(shares, "sync_registrations", Mock(return_value=(100, owners)))
    monkeypatch.setattr(shares, "top_jobs", lambda board: boards[board])
    monkeypatch.setattr(shares, "track_closes", lambda: {board: 2000 for board in shares.BOARDS})
    monkeypatch.setattr(shares.time, "time", lambda: 1000)
    fetch = Mock(side_effect=lambda claim, cache: {"kernel.py": claim["content_hash"]})
    monkeypatch.setattr(shares, "bundle_files", fetch)
    argv = ["okc_shares"]
    for key, value in vars(args).items():
        argv.extend(["--" + key.replace("_", "-"), str(value)])
    monkeypatch.setattr(sys, "argv", argv)
    return args, boards, fetch, st


def claim(args, ident, hotkey, block, job, registration, *, board=BOARD, code=None):
    url = f"https://example.test/{ident}?okc={board}:{job}"
    if registration is not None:
        url += f"&okc_user_block={registration}"
    with sqlite3.connect(args.intake_db) as db:
        db.execute("INSERT INTO reservations VALUES (?,?,?,?,?,?,?)",
                   (ident, hotkey, code or ident, url, block, 0, 0))


def row(team, rank=1):
    return {"team": team, "rank": rank, "value": 2.0}


def read(path):
    return json.loads(path.read_text())


def test_actual_writer_rejects_theft_before_code_or_job_ownership_and_keeps_equal_split(arena):
    args, boards, fetch, st = arena
    boards[BOARD] = {"a": row(AN), "renamed": row(BN + "_new", 2), "b": row(BN, 3)}
    boards[shares.BOARDS[1]] = {"other-board": row(AN)}
    claim(args, "thief", B, 10, "a", 6, code="shared-code")
    claim(args, "legacy", A, 11, "a", None)
    claim(args, "owner", A, 12, "a", 5, code="shared-code")
    claim(args, "duplicate", A, 13, "a", 5)
    claim(args, "another-job", A, 14, "other-board", 5, board=shares.BOARDS[1])
    claim(args, "rename", B, 15, "renamed", 6)
    claim(args, "other-owner", B, 16, "b", 6)
    shares.main()
    statuses = {c["reservation_id"]: c["status"] for c in read(args.private)["claims"]}
    assert statuses["thief"].startswith("username rejected:")
    assert "legacy" not in statuses
    assert statuses["rename"].startswith("username rejected:")
    assert statuses["duplicate"] == "job already claimed"
    assert [statuses[k] for k in ("owner", "another-job", "other-owner")] == ["paid"] * 3
    assert {c.args[0]["reservation_id"] for c in fetch.call_args_list} == {
        "owner", "duplicate", "another-job", "other-owner"}
    assert read(args.out)["shares"] == [{"hotkey": A, "parts": 666667}, {"hotkey": B, "parts": 333333}]
    assert all(c["files"] == {} for c in read(args.public)["claims"])
    assert read(args.private)["claims"][2]["files"]
    shares.sync_registrations.assert_called_once_with(st, 14, args.username_db, max_blocks=100, start_block=1)
    st.close.assert_called_once()


def test_verified_former_top_claim_still_owns_code_and_publication_waits_for_close(arena, monkeypatch):
    args, boards, _, _ = arena
    boards[BOARD] = {"a": row(AN)}
    claim(args, "first", A, 10, "a", 5, code="same")
    shares.main()
    (args.cache / "same").mkdir()
    boards[BOARD] = {"a": row(AN.lower())}
    shares.main()
    assert read(args.private)["claims"][0]["status"].startswith("username rejected:")
    assert read(args.private)["claims"][0]["verified_team"] == AN
    claim(args, "copy", B, 11, "b", 6, code="same")
    boards[BOARD] = {"b": row(BN)}
    monkeypatch.setattr(shares.time, "time", lambda: 3000)
    shares.main()
    claims = read(args.private)["claims"]
    assert claims[0]["status"] == "not in the top 3"
    assert claims[0]["username_verified"] is True
    assert claims[1]["status"] == "same code as earlier claim first"
    assert read(args.out)["shares"] == [{"hotkey": "default", "parts": 1000000}]
    assert read(args.public)["claims"][0]["files"] == {"kernel.py": "same"}


def test_failed_fetch_is_sticky_but_new_claim_can_succeed(arena):
    args, boards, fetch, _ = arena
    boards[BOARD] = {"a": row(AN)}
    claim(args, "failed", A, 10, "a", 5)
    fetch.side_effect = shares.FetchError("missing bundle")
    shares.main()
    fetch.reset_mock(side_effect=True)
    fetch.return_value = {"kernel.py": "new code"}
    boards[BOARD] = {}
    shares.main()
    boards[BOARD] = {"a": row(AN.lower())}
    shares.main()
    boards[BOARD] = {"a": row(AN)}
    claim(args, "new", A, 11, "a", 5)
    shares.main()
    assert [c["status"] for c in read(args.private)["claims"]] == ["code not fetched: missing bundle", "paid"]
    assert [c.args[0]["reservation_id"] for c in fetch.call_args_list] == ["new"]


@pytest.mark.parametrize("failure", [RuntimeError("username history incomplete"), ValueError("archive RPC failed")])
def test_history_failure_leaves_all_outputs_untouched(arena, failure):
    args, _, fetch, st = arena
    for path in (args.out, args.public, args.private):
        path.write_text("existing output")
    shares.sync_registrations.side_effect = failure
    with pytest.raises(type(failure), match=str(failure)):
        shares.main()
    assert all(path.read_text() == "existing output" for path in (args.out, args.public, args.private))
    fetch.assert_not_called()
    st.close.assert_called_once()


def test_historical_registration_rpc_failure_is_not_a_miner_rejection(arena):
    args, boards, _, st = arena
    boards[BOARD] = {"a": row(AN)}
    claim(args, "repeat", A, 10, "a", 7)
    st.get_commitment_metadata.side_effect = ValueError("archive unavailable")
    with pytest.raises(ValueError, match="archive unavailable"):
        shares.main()
    assert not args.out.exists()


def test_writer_indexes_unclaimed_registrations_before_processing_claim_arrival(arena, monkeypatch):
    from cacheon.chain.okc_history import sync_registrations

    args, boards, fetch, st = arena
    name = "Cacheon_28ef5_ac494"
    boards[BOARD] = {"a": row(name)}
    claim(args, "arrives-first", B, 10, "a", 6)
    claim(args, "oldest-registration", A, 11, "a", 5)
    st.get_finalized_block_number = lambda: 12
    st.get_block_hash = lambda block: "0x" + f"{block:064x}"
    def events(*, block_hash):
        block = int(block_hash[2:], 16)
        return [{"event": {"module_id": "Commitments", "event_id": "Commitment",
                           "attributes": {"netuid": 14, "who": A if block == 5 else B}}}]
    st.substrate = SimpleNamespace(get_events=lambda **kw: events(**kw)
                                   if int(kw["block_hash"][2:], 16) in (5, 6) else [])
    st.get_commitment_metadata.side_effect = lambda **kw: {
        "block": kw["block"], "info": {"fields": [{f"Raw{len(name)}": "0x" + name.encode().hex()}]}}
    monkeypatch.setattr(shares, "sync_registrations", sync_registrations)
    shares.main()
    assert read(args.out)["shares"] == [{"hotkey": A, "parts": 1000000}]
    assert read(args.private)["claims"][0]["status"].startswith("username rejected:")
    assert [c.args[0]["reservation_id"] for c in fetch.call_args_list] == ["oldest-registration"]
    with sqlite3.connect(args.username_db) as db:
        assert db.execute("SELECT hotkey,username,block FROM owners").fetchall() == [(A, name, 5)]


@pytest.mark.parametrize("proof", ["", "0", "5&okc_user_block=6", "5.0", "-5", "true", "9" * 30])
def test_ambiguous_or_malformed_url_registration_is_excluded(arena, proof):
    args, boards, fetch, _ = arena
    boards[BOARD] = {"a": row(AN)}
    claim(args, "bad-proof", A, 10, "a", proof)
    shares.main()
    assert read(args.private)["claims"] == []
    fetch.assert_not_called()


def test_fresh_round_excludes_legacy_pending_claims_and_preserves_shared_intake(arena):
    args, boards, fetch, st = arena
    args.start_block = 50
    sys.argv[sys.argv.index("--start-block") + 1] = "50"
    shares.sync_registrations.return_value = (100, {A: {"username": AN, "block": 50}})
    boards[BOARD] = {"a": row(AN)}
    claim(args, "old", A, 49, "a", 48, code="same")
    claim(args, "old-row-new-pointer", A, 49, "a", 50)
    claim(args, "pending", A, 60, "a", None)
    claim(args, "pending-old-proof", A, 61, "a", 49)
    args.private.write_text(json.dumps({"claims": [{"reservation_id": "old", "block": 49,
        "okc_user_block": 48, "status": "code not fetched: old", "fingerprint": "same"}]}))
    shares.main()
    for path in (args.private, args.public):
        assert read(path)["claims"] == [] and read(path)["start_block"] == 50
    assert read(args.out)["detail"]["start_block"] == 50
    assert read(args.out)["shares"] == [{"hotkey": "default", "parts": 1000000}]
    fetch.assert_not_called()
    with sqlite3.connect(args.intake_db) as db:
        assert db.execute("SELECT COUNT(*) FROM reservations").fetchone()[0] == 4
    claim(args, "forged-pointer", A, 62, "a", 55)
    claim(args, "fresh", A, 63, "a", 50, code="same")
    st.get_commitment_metadata.return_value = ""
    shares.main()
    claims = read(args.private)["claims"]
    assert [c["reservation_id"] for c in claims] == ["forged-pointer", "fresh"]
    assert claims[0]["status"].startswith("username rejected:") and claims[1]["status"] == "paid"
    assert read(args.out)["shares"] == [{"hotkey": A, "parts": 1000000}]
    assert [call.args[0]["reservation_id"] for call in fetch.call_args_list] == ["fresh"]
