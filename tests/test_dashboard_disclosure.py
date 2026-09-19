"""Public API disclosure starts at the result and also gates direct log URLs."""

import sqlite3

import pytest

from dashboard import disclosure
from cacheon.eval.remote_run_download import worker_log_download
from tests.test_dashboard_forensics import _retained_run
from tests.test_dashboard_metrics import _dashboard_db, client as dashboard_client

client = dashboard_client

RID, REQUEST = "7" * 64, "8" * 64
URL = "https://miner.example/private-kernel.tar.gz"
RESULT = 1_800_000_000


@pytest.fixture
def submission(tmp_path, client, monkeypatch):
    from dashboard import app

    db = tmp_path / "intake.sqlite3"
    _dashboard_db(db, "", tmp_path, "moe.fused_routed_experts")
    with sqlite3.connect(db) as con:
        con.execute("ALTER TABLE reservations ADD COLUMN url TEXT")
        con.execute("UPDATE reservations SET reservation_id=?,url=?", (RID, URL))
        con.execute("UPDATE qualification_dispositions SET reservation_id=?", (RID,))
        con.execute("INSERT INTO evaluation_lease_members VALUES(?, 'evaluation')", (RID,))
        con.execute("INSERT INTO evaluation_leases(lease_id,stage,state,claimed_block,"
                    "expires_block,completed_block) VALUES('evaluation','qualification',"
                    "'completed',9009700,9010000,9009800)")
    _retained_run(tmp_path / "spool", RID, REQUEST)
    monkeypatch.setattr(app.ENRICHER, "block_time", lambda block: {
        "unix": RESULT + (block - 9009800) * 12, "estimated": False})
    return db


@pytest.mark.parametrize("status", ["qualified", "failed"])
def test_result_plus_eight_hours_gates_detail_prefix_and_direct_download(
    submission, client, monkeypatch, status,
):
    with sqlite3.connect(submission) as con:
        con.execute("UPDATE reservations SET status=?", (status,))
    release = RESULT + 8 * 3600
    monkeypatch.setattr(disclosure.time, "time", lambda: release - 1)
    response = client.get(f"/api/submissions/{RID[:12]}")
    detail = response.json()
    assert response.headers["cache-control"] == "no-store"
    assert detail["bundle_visibility"] == {
        "available": False, "release_at": release, "result_block": 9009800}
    assert detail["url"] == "" and URL not in response.text
    assert detail["status"] == status and detail["qualification_attempts"]
    assert detail["forensics"][0]["worker_log"]["download_url"] is None
    assert detail["forensics"][0]["worker_log"]["explanation"] == []
    path = f"/api/submissions/{RID}/forensics/{REQUEST}.log"
    hidden = client.get(path)
    assert hidden.status_code == 403
    assert hidden.json()["detail"]["release_at"] == release
    with sqlite3.connect(submission) as con:
        assert con.execute("SELECT url FROM reservations").fetchone()[0] == URL
    private_log = worker_log_download(submission.parent / "spool", REQUEST)
    assert b"miner diagnostic output" in private_log.payload
    monkeypatch.setattr(disclosure.time, "time", lambda: release)
    assert client.get(f"/api/submissions/{RID}").json()["url"] == URL
    visible = client.get(path)
    assert visible.status_code == 200
    assert b"miner diagnostic output" in visible.content


@pytest.mark.parametrize("status", [
    "published", "screening", "promoted", "qualifying", "held", "no_decision",
])
def test_waiting_or_retry_does_not_release_an_old_result(submission, client, monkeypatch, status):
    monkeypatch.setattr(disclosure.time, "time", lambda: RESULT + 100_000)
    with sqlite3.connect(submission) as con:
        con.execute("UPDATE reservations SET status=?", (status,))
    detail = client.get(f"/api/submissions/{RID}").json()
    assert detail["url"] == ""
    assert detail["bundle_visibility"]["release_at"] is None


@pytest.mark.parametrize("stamp", [
    {"unix": RESULT, "estimated": True}, {"unix": None, "estimated": True},
])
def test_unknown_result_time_cannot_release_source(submission, client, monkeypatch, stamp):
    from dashboard import app

    monkeypatch.setattr(disclosure.time, "time", lambda: RESULT + 100_000)
    monkeypatch.setattr(app.ENRICHER, "block_time", lambda block: stamp)
    assert client.get(f"/api/submissions/{RID}").json()["url"] == ""


def test_active_lease_and_new_retained_result_reset_disclosure(submission, client, monkeypatch):
    monkeypatch.setattr(disclosure.time, "time", lambda: RESULT + 8 * 3600)
    with sqlite3.connect(submission) as con:
        con.execute("UPDATE evaluation_leases SET state='active'")
    assert client.get(f"/api/submissions/{RID}").json()["url"] == ""
    with sqlite3.connect(submission) as con:
        con.execute("UPDATE evaluation_leases SET state='completed'")
        con.execute("INSERT INTO settlement_qualifications VALUES(?,0,'{}','',9009900)", (RID,))
    detail = client.get(f"/api/submissions/{RID}").json()
    assert detail["url"] == ""
    assert detail["bundle_visibility"]["release_at"] == RESULT + 1200 + 8 * 3600


def test_no_completed_evaluation_keeps_source_hidden(submission, client, monkeypatch):
    monkeypatch.setattr(disclosure.time, "time", lambda: RESULT + 100_000)
    with sqlite3.connect(submission) as con:
        con.execute("DELETE FROM evaluation_leases")
    assert client.get(f"/api/submissions/{RID}").json()["bundle_visibility"]["release_at"] is None
    assert client.get(f"/api/submissions/{'9' * 64}/forensics/{REQUEST}.log").status_code == 404
