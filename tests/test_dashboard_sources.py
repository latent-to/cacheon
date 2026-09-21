"""Parallel chain listeners retain distinct, source-qualified submission histories."""

import sqlite3

import pytest

pytest.importorskip("fastapi")
from fastapi import FastAPI
from fastapi.testclient import TestClient

from dashboard.sources import DashboardSource, install_sources, selected


@pytest.fixture
def planes(tmp_path, monkeypatch):
    monkeypatch.delenv("CACHEON_DASH_SOURCES", raising=False)
    app = FastAPI()

    @app.get("/api/submissions/{reservation_id}")
    def detail(reservation_id):
        source = selected.get()
        with sqlite3.connect(source.values["DB_PATH"]) as con:
            status, reason = con.execute(
                "SELECT status,reason FROM reservations WHERE reservation_id=?",
                (reservation_id,),
            ).fetchone()
        return {"reservation_id": reservation_id, "status": status, "reason": reason,
                "log_url": f"/api/submissions/{reservation_id}/logs"}

    install_sources(app, {})
    sources = {}
    for key, status, reason, publication in (
        ("glm", "failed", "manifest: unknown arena", ""),
        ("qwen", "promoted", "screen_promoted", "published"),
    ):
        path = tmp_path / f"{key}.sqlite3"
        with sqlite3.connect(path) as con:
            con.execute("""CREATE TABLE reservations (
                reservation_id, status, reason, publication_digest,
                target_id DEFAULT '', arena_service_digest DEFAULT '',
                screen_status DEFAULT '', decision DEFAULT '')""")
            con.execute("INSERT INTO reservations (reservation_id,status,reason,publication_digest) "
                        "VALUES ('shared', ?, ?, ?)",
                        (status, reason, publication))
        sources[key] = DashboardSource(key, key, key, {"DB_PATH": path}, {}, False, None)
    app.state.dashboard_sources = sources
    app.state.dashboard_default = "glm"
    return TestClient(app), sources


@pytest.mark.parametrize("reason", ["missing_eval_cost_payment", "manifest: unknown arena"])
def test_prepublication_rejections_do_not_claim_the_other_arenas_submission(planes, reason):
    client, sources = planes
    with sqlite3.connect(sources["glm"].values["DB_PATH"]) as con:
        con.execute("UPDATE reservations SET reason=?, decision=?",
                    (reason, "FAIL" if reason.startswith("manifest:") else ""))
    for key, status in (("glm", "failed"), ("qwen", "promoted")):
        response = client.get(f"/api/submissions/shared?arena={key}")
        assert response.status_code == 200
        row = response.json()
        assert (row["source"], row["status"]) == (key, status)
        assert row["log_url"] == f"/api/submissions/shared/logs?arena={key}"
    assert client.get("/api/submissions/shared").json()["reason"] == reason


def test_two_published_owners_still_report_a_conflict(planes):
    client, sources = planes
    with sqlite3.connect(sources["glm"].values["DB_PATH"]) as con:
        con.execute("UPDATE reservations SET publication_digest='also-published'")
    for key in sources:
        assert client.get(f"/api/submissions/shared?arena={key}").status_code == 409


def test_unavailable_peer_does_not_hide_selected_history(planes):
    client, sources = planes
    sources["glm"].values["DB_PATH"].unlink()
    assert client.get("/api/submissions/shared?arena=qwen").status_code == 200
    response = client.get("/api/submissions/shared?arena=glm")
    assert response.status_code == 503
    assert response.json()["source"] == "glm"
