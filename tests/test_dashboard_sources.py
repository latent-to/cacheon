"""Parallel chain listeners retain distinct, source-qualified submission histories."""

from contextlib import closing
import json
import sqlite3

import pytest

pytest.importorskip("fastapi")
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from dashboard.sources import DashboardSource, install_sources, selected, worker_heartbeats
from dashboard import app as dashboard
from dashboard import sources as source_module
from dashboard.disclosure import bundle_visibility


@pytest.mark.parametrize("key", ["qwen3.6", "glm-5.3"])
def test_competition_paths_serve_dashboard_without_shadowing_routes(monkeypatch, planes, key):
    monkeypatch.setattr(dashboard.app.state, "dashboard_sources", planes[1])
    monkeypatch.setattr(dashboard.app.state, "dashboard_default", "glm")
    client = TestClient(dashboard.app)
    root = client.get("/")
    response = client.get(f"/{key}#winners")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert response.content == root.content
    assert client.get(f"/{key}/").content == root.content
    assert client.get("/unknown-arena").status_code == 404
    for old in ("qwen", "glm"):
        assert client.get(f"/{old}").status_code == 404
    assert client.get(f"/{key}/extra").status_code == 404
    for path in ("/docs", "/openapi.json", "/favicon.ico", "/static/performance.js"):
        assert client.get(path).status_code == 200
    assert client.get("/api/queue?arena=unknown-arena").status_code == 404
    api_client, sources = planes
    old = "qwen" if key == "qwen3.6" else "glm"
    assert api_client.get(f"/api/queue?arena={key}").json() == api_client.get(f"/api/queue?arena={old}").json()
    assert sources[old].public()["slug"] == key


@pytest.mark.parametrize("case,state,fresh", [
    ("current", "running", True), ("boundary", "running", True),
    ("stale", "stale", False), ("missing", "unknown", False),
    ("corrupt", "unknown", False), ("future", "unknown", False),
    ("epoch", "epoch_mismatch", False), ("ready", "epoch_mismatch", False),
    ("registration_missing", "unknown", False),
])
def test_worker_observation_is_independent_of_fresh_relay(tmp_path, case, state, fresh):
    registration = {"worker_epoch": "epoch", "ready_receipt_digest": "ready", "worker_readiness_digest": "worker"}
    relay = {**registration, "time_unix": 999, "state": "running", "active_request_id": "old-request"}
    path = tmp_path / "heartbeat.json"
    path.write_text(json.dumps(relay))
    worker = {**registration, "time_unix": 999, "state": "running", "adapter_alive": True,
              "active_request_id": "worker-request", "private_path": "/private/worker"}
    if case == "boundary": worker["time_unix"] = 880
    if case == "stale": worker["time_unix"] = 879
    if case == "future": worker["time_unix"] = 1010
    if case == "epoch": worker["worker_epoch"] = "old-epoch"
    if case == "ready": worker["ready_receipt_digest"] = "old-ready"
    if case == "registration_missing": registration = {}
    if case != "missing":
        path.with_name("worker-heartbeat.json").write_text("broken" if case == "corrupt" else json.dumps(worker))
    result = worker_heartbeats(path, registration, 1000)
    observed = result["gpu_heartbeat"]
    assert (observed["state"], observed["fresh"]) == (state, fresh)
    assert observed["adapter_alive"] is fresh
    assert observed["active_request_id"] == ("worker-request" if fresh else None)
    assert result["relay_heartbeat"]["age_s"] == 1
    assert result["relay_heartbeat"]["fresh"] is (case != "registration_missing")
    assert "/private/worker" not in json.dumps(result)


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

    install_sources(app, {"health": lambda: {"intake_finalized": {"block": 100}},
                          "intake_conn": dashboard.intake_conn})
    sources = {}
    for key, status, reason, publication in (
        ("glm", "failed", "manifest: unknown arena", ""),
        ("qwen", "qualifying", "", "published"),
    ):
        path = tmp_path / f"{key}.sqlite3"
        with sqlite3.connect(path) as con:
            con.execute("""CREATE TABLE reservations (
                reservation_id, status, reason, publication_digest,
                target_id DEFAULT '', arena_service_digest DEFAULT '',
                decision DEFAULT '', competition_arena DEFAULT '')""")
            con.execute("CREATE TABLE metadata (key, value)")
            con.execute("CREATE TABLE eval_cost_payments (payment_block, payment_extrinsic_index, "
                        "reservation_id, content_hash, hotkey, amount_tao_rao)")
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
                                      {"DB_PATH": path, "REGISTRATION_PATH": registration}, {}, False, None,
                                      {"qwen": "qwen3.6", "glm": "glm-5.3"}[key])
    for source in sources.values():
        source.values["PEER_DB_PATHS"] = tuple(
            peer.values["DB_PATH"] for peer in sources.values() if peer is not source)
    app.state.dashboard_sources = sources
    app.state.dashboard_default = "glm"
    return TestClient(app), sources


@pytest.mark.parametrize("reason", ["missing_eval_cost_payment", "manifest: unknown arena"])
def test_prepublication_rejections_do_not_claim_the_other_arenas_submission(planes, reason):
    client, sources = planes
    with sqlite3.connect(sources["glm"].values["DB_PATH"]) as con:
        con.execute("UPDATE reservations SET reason=?, decision=?",
                    (reason, "FAIL" if reason.startswith("manifest:") else ""))
    response = client.get("/api/submissions/shared?arena=qwen")
    assert response.status_code == 200
    row = response.json()
    assert (row["source"], row["status"]) == ("qwen", "qualifying")
    assert row["log_url"] == "/api/submissions/shared/logs?arena=qwen"
    assert client.get("/api/submissions/shared?arena=glm").status_code == 404
    assert client.get("/api/submissions/shared").status_code == 404
    assert client.get("/api/queue?arena=glm").json()["items"] == []
    with sqlite3.connect(sources["glm"].values["DB_PATH"]) as con:
        assert con.execute("SELECT reason FROM reservations").fetchone()[0] == reason


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
    assert glm == []
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


@pytest.mark.parametrize("publication,arena", [("", "qwen-arena"), ("published", "glm-arena")])
def test_only_a_foreign_published_row_displaces_legacy_observation(planes, publication, arena):
    client, sources = planes
    with sqlite3.connect(sources["qwen"].values["DB_PATH"]) as con:
        con.execute("UPDATE reservations SET publication_digest=?,competition_arena=?", (publication, arena))
    row = client.get("/api/submissions/shared?arena=glm").json()
    assert (row["source"], row["status"]) == ("glm", "failed")
    assert len(client.get("/api/queue?arena=glm").json()["items"]) == 1


def test_local_publication_retains_its_legacy_history(planes):
    client, sources = planes
    with sqlite3.connect(sources["glm"].values["DB_PATH"]) as con:
        con.execute("UPDATE reservations SET publication_digest='local-publication'")
    assert client.get("/api/submissions/shared?arena=glm").status_code == 200
    assert client.get("/api/submissions/shared?arena=qwen").status_code == 200


def test_legacy_ownership_is_unavailable_when_peer_cannot_be_read(planes):
    client, sources = planes
    sources["qwen"].values["DB_PATH"].unlink()
    response = client.get("/api/submissions/shared?arena=glm")
    assert response.status_code == 503
    assert response.json()["arena"]["key"] == "glm"


def test_arena_fees_count_each_payment_once_under_the_arena_that_consumed_it(planes):
    client, sources = planes
    tao = 1_000_000_000
    with sqlite3.connect(sources["glm"].values["DB_PATH"]) as con:
        con.execute("INSERT INTO reservations (reservation_id,status,reason,publication_digest,competition_arena) "
                    "VALUES ('paid','published','','pub-paid','glm-arena')")
        con.execute("INSERT INTO eval_cost_payments VALUES (100,1,'paid','h','miner',?)", (tao,))
        # The GLM listener also observed the Qwen arrival and recorded it at its own 0.5 τ fee.
        con.execute("INSERT INTO eval_cost_payments VALUES (200,2,'shared','h','miner',?)", (tao // 2,))
    with sqlite3.connect(sources["qwen"].values["DB_PATH"]) as con:
        con.execute("INSERT INTO eval_cost_payments VALUES (200,2,'shared','h','miner',?)", (tao // 5,))

    def fees():
        data = client.get("/api/arenas").json()
        return {row["key"]: row["fees"] for row in data["items"]}, data["fees_count"], data["fees_total_tao"]

    by_arena, count, total = fees()
    assert by_arena["glm"] == {"count": 1, "tao": 1.0, "by_fee": [{"fee_tao": 1.0, "count": 1}]}
    assert by_arena["qwen"] == {"count": 1, "tao": 0.2, "by_fee": [{"fee_tao": 0.2, "count": 1}]}
    assert (count, total) == (2, 1.2)
    # A publication replicated into the other arena must not count its payment twice.
    with sqlite3.connect(sources["qwen"].values["DB_PATH"]) as con:
        con.execute("INSERT INTO reservations (reservation_id,status,reason,publication_digest,competition_arena) "
                    "VALUES ('paid','published','','pub-paid','qwen-arena')")
        con.execute("INSERT INTO eval_cost_payments VALUES (100,1,'paid','h','miner',?)", (tao // 5,))
    assert fees()[1:] == (2, 1.2)
    # With the GLM database unreadable, the replicated payment counts under the arena that can be read.
    sources["glm"].values["DB_PATH"].unlink()
    by_arena, count, total = fees()
    assert (by_arena["glm"], by_arena["qwen"]["tao"], count, total) == (None, 0.4, 2, 0.4)


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


def test_live_targets_follow_producer_restart_without_stale_config_fallback(target_settings, tmp_path, monkeypatch):
    client, stage = target_settings
    original = client.app.state.dashboard_weight_producer_config
    allocation, new_stage, new_producer = (tmp_path / name for name in ("targets.json", "new-stage.json", "new-producer.json"))
    allocation.write_text(json.dumps({"history": [{"from_block": 0, "weights_ppm": {"glm": 700_000, "qwen": 300_000}}]}))
    new_stage.write_text(json.dumps({"arena_allocation_path": str(allocation)}))
    new_producer.write_text(json.dumps({"weights_stage_config": str(new_stage)}))
    proc = tmp_path / "proc"
    for pid, config in (("11", original), ("12", new_producer)):
        (proc / pid).mkdir(parents=True)
        (proc / pid / "cmdline").write_bytes(f"python\0producer.py\0--config\0{config}\0".encode())
    pidfile = tmp_path / "producer.pid"
    client.app.state.dashboard_weight_producer_pidfile = pidfile
    resolve = source_module._producer_config
    monkeypatch.setattr(source_module, "_producer_config", lambda config, pid: resolve(config, pid, proc_root=proc))
    for pid, expected in (("11", 0), ("12", 300_000), ("99", None), ("0", None)):
        pidfile.write_text(pid)
        rows = client.get("/api/arenas").json()["items"]
        assert next(r["target_weight_ppm"] for r in rows if r["key"] == "qwen") == expected
    pidfile.unlink()
    assert all(r["target_weight_ppm"] is None for r in client.get("/api/arenas").json()["items"])


def test_single_arena_allocation_matches_dashboard_by_database(target_settings):
    client, stage = target_settings
    allocation = stage.parent / "allocation.json"
    stage.write_text(json.dumps({"arena_allocation_path": str(allocation)}))
    allocation.write_text(json.dumps({
        "sources": {"active_arena": str(stage.parent / "dispatcher.json")},
        "history": [{"from_block": 0, "weights_ppm": {"active_arena": 1_000_000}}],
    }))
    rows = client.get("/api/arenas").json()["items"]
    assert {r["key"]: r["target_weight_ppm"] for r in rows} == {"glm": 1_000_000, "qwen": 0}
