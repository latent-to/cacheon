"""Parallel chain listeners retain distinct, source-qualified submission histories."""

from contextlib import closing
import json
import sqlite3

import pytest

pytest.importorskip("fastapi")
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from dashboard.sources import DashboardSource, install_sources, selected
from dashboard import app as dashboard
from dashboard.disclosure import bundle_visibility


@pytest.fixture
def planes(tmp_path, monkeypatch):
    monkeypatch.delenv("CACHEON_DASH_SOURCES", raising=False)
    app = FastAPI()
    app.add_exception_handler(sqlite3.Error, dashboard._sqlite_error)

    @app.get("/api/queue")
    def queue():
        with closing(dashboard.intake_conn()) as con:
            return {"items": [dict(row) for row in con.execute("SELECT * FROM reservations")]}

    @app.get("/api/submissions/{reservation_id}")
    def detail(reservation_id):
        with closing(dashboard.intake_conn()) as con:
            row = con.execute(
                "SELECT status,reason FROM reservations WHERE reservation_id=?",
                (reservation_id,),
            ).fetchone()
        if row is None:
            raise HTTPException(404, "reservation not found")
        status, reason = row
        return {"reservation_id": reservation_id, "status": status, "reason": reason,
                "log_url": f"/api/submissions/{reservation_id}/logs"}

    install_sources(app, {"health": lambda: {"intake_finalized": {"block": 100}}})
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
                screen_status DEFAULT '', decision DEFAULT '', competition_arena DEFAULT '')""")
            con.execute("CREATE TABLE metadata (key, value)")
            if key == "glm":
                con.execute("INSERT INTO metadata VALUES ('legacy_arena_id', 'glm-arena')")
            con.execute("INSERT INTO reservations (reservation_id,status,reason,publication_digest) "
                        "VALUES ('shared', ?, ?, ?)",
                        (status, reason, publication))
            if key == "qwen":
                con.execute("UPDATE reservations SET competition_arena='qwen-arena'")
        registration = tmp_path / f"{key}-registration.json"
        registration.write_text(json.dumps({"worker_readiness": {"arena_id": f"{key}-arena"}}))
        sources[key] = DashboardSource(key, key, key,
                                      {"DB_PATH": path, "REGISTRATION_PATH": registration}, {}, False, None)
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


def test_replicated_publication_is_scoped_by_arena_not_database(planes):
    client, sources = planes
    for key, source in sources.items():
        with sqlite3.connect(source.values["DB_PATH"]) as con:
            con.execute("INSERT INTO reservations "
                        "(reservation_id,status,publication_digest,competition_arena) "
                        "VALUES ('replicated',?,'same-publication','qwen-arena')",
                        ("published" if key == "glm" else "failed",))
    assert client.get("/api/submissions/replicated?arena=glm").status_code == 404
    row = client.get("/api/submissions/replicated?arena=qwen").json()
    assert (row["source"], row["status"]) == ("qwen", "failed")
    assert row["log_url"].endswith("?arena=qwen")
    glm = client.get("/api/queue?arena=glm").json()["items"]
    assert [row["reservation_id"] for row in glm] == ["shared"]
    assert len(client.get("/api/queue?arena=qwen").json()["items"]) == 2
    token = selected.set(sources["glm"])
    try:
        with closing(dashboard.intake_conn()) as con, pytest.raises(HTTPException) as error:
            bundle_visibility(con, "replicated", lambda _: {})
        assert error.value.status_code == 404
    finally:
        selected.reset(token)


def test_legacy_history_follows_its_recorded_arena_alias(planes):
    client, sources = planes
    with sqlite3.connect(sources["glm"].values["DB_PATH"]) as con:
        con.execute("UPDATE metadata SET value='another-arena'")
    assert client.get("/api/submissions/shared?arena=glm").status_code == 404


def test_unavailable_peer_does_not_hide_selected_history(planes):
    client, sources = planes
    sources["glm"].values["DB_PATH"].unlink()
    assert client.get("/api/submissions/shared?arena=qwen").status_code == 200
    response = client.get("/api/submissions/shared?arena=glm")
    assert response.status_code == 503
    assert response.json()["arena"]["key"] == "glm"


@pytest.fixture
def target_settings(planes, tmp_path):
    client, sources = planes
    dispatcher, stage, producer = (tmp_path / name for name in ("dispatcher.json", "stage.json", "producer.json"))
    dispatcher.write_text(json.dumps({"intake_db": str(sources["glm"].values["DB_PATH"])}))
    stage.write_text("{}")
    producer.write_text(json.dumps({"weights_stage_config": str(stage), "screen_dispatcher_config": str(dispatcher)}))
    client.app.state.dashboard_weight_producer_config = producer
    return client, stage


def test_arena_targets_use_producer_settings_without_an_offer(target_settings):
    client, stage = target_settings
    rows = client.get("/api/arenas").json()["items"]
    assert {r["key"]: r["target_weight_ppm"] for r in rows} == {"glm": 1_000_000, "qwen": 0}
    # A target is independent of whether this source currently earns rewards.
    assert all(r["weights_status"].startswith("weights off") for r in rows)
    stage.unlink()
    assert all(r["target_weight_ppm"] is None for r in client.get("/api/arenas").json()["items"])


@pytest.mark.parametrize("weights,expected", [
    ({"glm": 600_000, "qwen": 400_000}, {"glm": 600_000, "qwen": 400_000}),
    ({"glm": 800_000, "qwen": 800_000}, {"glm": 500_000, "qwen": 500_000}),
    ({"glm": 200_000, "qwen": 300_000}, {"glm": 200_000, "qwen": 300_000}),
])
def test_arena_targets_refresh_current_schedule_not_future_or_actual_shares(target_settings, weights, expected):
    client, stage = target_settings
    allocation = stage.parent / "allocation.json"
    stage.write_text(json.dumps({"arena_allocation_path": str(allocation)}))
    history = [{"from_block": 0, "weights_ppm": {"glm": 1_000_000, "qwen": 0}},
               {"from_block": 100, "weights_ppm": weights},
               {"from_block": 500, "weights_ppm": {"glm": 0, "qwen": 1_000_000}}]
    allocation.write_text(json.dumps({"history": history}))
    rows = client.get("/api/arenas").json()["items"]
    assert {r["key"]: r["target_weight_ppm"] for r in rows} == expected
    history[1]["weights_ppm"] = {"glm": 100_000, "qwen": 900_000}
    allocation.write_text(json.dumps({"history": history}))
    assert {r["key"]: r["target_weight_ppm"] for r in client.get("/api/arenas").json()["items"]} == {"glm": 100_000, "qwen": 900_000}
    allocation.write_text("broken")
    assert all(r["target_weight_ppm"] is None for r in client.get("/api/arenas").json()["items"])
