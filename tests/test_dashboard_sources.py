"""Dashboard source identity covers reads, health, links and disclosure."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
from pathlib import Path
import sqlite3
import time

import pytest
from fastapi.testclient import TestClient

from dashboard import app, sources, competition
from tests import test_chain_intake as intake
from tests.test_dashboard_forensics import _retained_run
from tests.test_dashboard_metrics import _dashboard_db


def _row(root, key, db):
    paths = {name: str(root / name) for name in sources._PATHS}
    paths.update(db=str(db), mission=str(root), heartbeat=str(root / "heartbeat.json"),
                 registration=str(root / "registration.json"), audit=str(root / "audit.jsonl"))
    return {"key": key, "label": key.upper(), "model": f"Model {key}", "paths": paths,
            "cache": str(root / "cache.sqlite3"), "evidence_roots": [str(root / "evidence")],
            "cutoff_reservation": "", "weights_included": key == "a",
            "processes": {role: [f"service.{role}", str(root)] for role in ("intake", "supervisor", "relay")},
            "checkpoint": {"same-engine": {"repo": f"models/{key}", "revision": "123",
                           "url": f"https://example.com/{key}"},
                           "previous-engine": {"repo": f"models/previous-{key}"}}}


def _configure(tmp_path, monkeypatch, rows):
    path = tmp_path / "sources.json"
    path.write_text(json.dumps({"default": "a", "sources": rows}))
    default, configured = sources.load_sources(path, "wss://unused.invalid", 14, False)
    monkeypatch.setattr(app.app.state, "dashboard_sources", configured)
    monkeypatch.setattr(app.app.state, "dashboard_default", default)
    monkeypatch.setattr(app, "OFFER_PATH", tmp_path / "missing-offer")
    for key, source in configured.items():
        enrich = source.values["ENRICHER"]
        monkeypatch.setattr(enrich, "block_time", lambda block: {"unix": 1_000_000 + block * 12, "estimated": False})
        source.values["HEARTBEAT_PATH"].write_text(json.dumps({
            "time_unix": int(time.time()) - (40000 if key == "a" else 1),
            "state": "running", "worker_epoch": key}))
        source.values["REGISTRATION_PATH"].write_text(json.dumps({"worker_epoch": key}))
    return configured, TestClient(app.app)


@pytest.fixture
def planes(tmp_path, monkeypatch):
    rows, ids = [], {}
    for index, key in enumerate(("a", "b")):
        root = tmp_path / key
        with intake._store(root) as store:
            row = intake._reserve_one(store, index=index, hotkey="shared", block=10+index)
            ids[key] = row.reservation_id
            store._db.execute("UPDATE reservations SET status='failed',decision='FAIL',reason=?", (key,))
            rows.append(_row(root, key, store.path))
    configured, client = _configure(tmp_path, monkeypatch, rows)
    return client, configured, ids


def test_all_reads_are_source_scoped_and_missing_source_stays_visible(planes):
    client, configured, ids = planes
    for key in configured:
        for route in ("overview", "queue", "submissions", "payments", "winners", "miners", "events", "health", "hotkey/shared"):
            response = client.get(f"/api/{route}?arena={key}")
            assert response.status_code == 200, response.text
            assert response.json()["arena"]["key"] == key
        rows = client.get(f"/api/submissions?arena={key}").json()["items"]
        assert [row["reservation_id"] for row in rows] == [ids[key]]
        assert rows[0]["source"] == key and rows[0]["competition"] == f"Model {key}"
        other = "b" if key == "a" else "a"
        assert client.get(f"/api/submissions/{ids[other]}?arena={key}").status_code == 404
    configured["b"].values["DB_PATH"] = configured["b"].values["DB_PATH"].with_name("absent.sqlite3")
    response = client.get("/api/arenas").json()
    assert [row["health"]["db"]["ok"] for row in response["items"]] == [True, False]
    assert client.get("/api/submissions?arena=a").status_code == 200
    assert client.get("/api/submissions?arena=b").status_code == 503
    assert not configured["b"].values["DB_PATH"].exists()
    assert client.get("/api/health?arena=unknown").status_code == 404


def test_health_uses_argv_and_exact_source_path(tmp_path, planes, monkeypatch):
    client, configured, _ = planes
    proc = tmp_path / "proc"
    process = proc / "123"
    process.mkdir(parents=True)
    process.joinpath("cwd").symlink_to(tmp_path, target_is_directory=True)
    process.joinpath("cmdline").write_bytes(b"python\0service.supervisor\0--config\0" +
        str(tmp_path / "b" / "supervisor.json").encode() + b"\0")
    real = sources.process_matches
    monkeypatch.setattr(app, "process_matches", lambda role, default: real(role, default, proc_root=proc))
    health_a = client.get("/api/health?arena=a").json()
    health_b = client.get("/api/health?arena=b").json()
    assert health_a["processes"][1]["up"] is False
    assert health_b["processes"][1]["up"] is True
    assert health_a["gpu_heartbeat"]["state"] == "stale"
    assert health_a["gpu_heartbeat"]["reported_state"] == "running"
    assert str(tmp_path) not in json.dumps(health_a)
    configured["b"].values["REGISTRATION_PATH"].write_text('{"worker_epoch":"new"}')
    assert client.get("/api/health?arena=b").json()["gpu_heartbeat"]["state"] == "epoch_mismatch"


def test_concurrent_requests_do_not_share_source_or_checkpoint(planes):
    client, configured, ids = planes
    def read(key):
        detail = client.get(f"/api/submissions/{ids[key]}?arena={key}").json()
        return detail["source"], detail["reason"]
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert list(pool.map(read, ["a", "b"] * 8)) == [(k, k) for k in ["a", "b"] * 8]
    for key in ("a", "b", "a"):
        token = sources.selected.set(configured[key])
        try:
            assert competition.checkpoint_for_engine("same-engine")["repo"] == f"models/{key}"
            assert competition.checkpoint_for_engine("previous-engine")["repo"] == f"models/previous-{key}"
            assert competition.checkpoint_for_engine("unknown") is None
        finally:
            sources.selected.reset(token)
    assert configured["a"].values["ENRICHER"].cache_db != configured["b"].values["ENRICHER"].cache_db


def test_duplicate_ownership_refuses_list_detail_and_download(planes):
    client, configured, ids = planes
    with intake.FinalizedIntakeStore(configured["b"].values["DB_PATH"], intake.IntakePolicy(), scope=intake.SCOPE) as store:
        intake._reserve(store, (intake._arrival(0, hotkey="shared", block=10),), block=11)
    for path in ("submissions", f"submissions/{ids['a']}", f"submissions/{ids['a']}/bundle.tar.gz"):
        assert client.get(f"/api/{path}?arena=a").status_code == 409
    assert client.get("/api/arena-events").status_code == 409


def test_pre_admission_payment_rejection_is_not_duplicate_ownership(tmp_path, monkeypatch):
    rows = []
    arrival = intake._arrival(0)
    for key in ("a", "b"):
        root = tmp_path / key
        with intake._store(root) as store:
            observed = replace(arrival, invalid_reason="missing_eval_cost_payment") if key == "a" else arrival
            reservation = intake._reserve(store, (observed,))[0]
            if key == "b":
                intake._publish(store, reservation.reservation_id,
                    intake._fingerprint("forward_pass", "model"), digest=intake._h("published"), root="/published/b")
            rows.append(_row(root, key, store.path))
    _, client = _configure(tmp_path, monkeypatch, rows)
    for key, status in (("a", "failed"), ("b", "published")):
        response = client.get(f"/api/submissions/{reservation.reservation_id}?arena={key}")
        assert response.status_code == 200, response.text
        assert response.json()["status"] == status
        assert response.json()["source"] == key
    assert client.get("/api/arena-events").status_code == 200


def test_global_offer_is_unchanged_by_arena_and_presentation(planes, monkeypatch):
    client, configured, _ = planes
    monkeypatch.setattr(app, "live_offer_shares", lambda path: ({"effective_block": 20}, {"shared": 0.75}))
    before = client.get("/api/weights").json()
    configured["b"].label = "Sponsored display name"
    assert client.get("/api/weights?arena=b").json() == before
    miners = client.get("/api/miners?arena=b").json()
    assert miners["items"][0]["weight_share"] is None
    assert client.get("/api/miners?arena=a").json()["items"][0]["weight_share"] == 0.75
    assert miners["arena"]["weights_status"] == "weights off / not yet in served vector"


def test_event_merge_orders_by_block_not_sqlite_sequence(planes):
    client, configured, _ = planes
    for key, sequence, block in (("a", 100, 10), ("b", 1, 20)):
        with sqlite3.connect(configured[key].values["DB_PATH"]) as con:
            con.execute("INSERT INTO settlement_events(sequence,event_id,event_type,reservation_id,arena_id,target_id,event_digest,event_json) VALUES(?,?,?,?,?,?,?,?)",
                        (sequence, key, "HOLD", "", key, "target", "0" * 64, json.dumps({"finalized_block": block})))
    events = client.get("/api/arena-events").json()["items"]
    assert [(row["source"], row["when"]["block"]) for row in events] == [("b", 20), ("a", 10)]


def test_disclosure_uses_selected_private_and_spool_roots(tmp_path, monkeypatch):
    from dashboard import disclosure
    from cacheon.chain.fetch import package_bundle, fetch_bundle_from_local_file_for_testing
    from tests.test_chain_fetch import _make_bundle

    rows, ids = [], {}
    for index, key in enumerate(("a", "b")):
        root = tmp_path / key
        root.mkdir()
        db = root / "intake.sqlite3"
        _dashboard_db(db, "", root, "moe.fused_routed_experts")
        rid, request = str(index + 1) * 64, str(index + 3) * 64
        ids[key] = (rid, request)
        bundle = _make_bundle(root)
        archive, digest = package_bundle(bundle, root / "input.tar.gz")
        if key == "a":
            fetch_bundle_from_local_file_for_testing(archive.as_uri(), digest, root / "private")
        with sqlite3.connect(db) as con:
            con.execute("ALTER TABLE reservations ADD COLUMN url TEXT")
            for column in ("publication_digest", "arena_service_digest", "screen_status", "decision", "reason"):
                if column not in {row[1] for row in con.execute("PRAGMA table_info(reservations)")}:
                    con.execute(f"ALTER TABLE reservations ADD COLUMN {column} TEXT NOT NULL DEFAULT ''")
            con.execute("UPDATE reservations SET reservation_id=?,content_hash=?", (rid, digest))
            con.execute("UPDATE qualification_dispositions SET reservation_id=?", (rid,))
            con.execute("INSERT INTO evaluation_lease_members VALUES(?, 'evaluation')", (rid,))
            con.execute("INSERT INTO evaluation_leases(lease_id,stage,state,claimed_block,expires_block,completed_block) VALUES('evaluation','qualification','completed',90,110,100)")
        _retained_run(root / "spool", rid, request)
        rows.append(_row(root, key, db))
    configured, client = _configure(tmp_path, monkeypatch, rows)
    release = 1_000_000 + 100 * 12 + disclosure.DISCLOSURE_DELAY_SECONDS
    monkeypatch.setattr(disclosure.time, "time", lambda: release - 1)
    for key, (rid, request) in ids.items():
        assert client.get(f"/api/submissions/{rid}/bundle.tar.gz?arena={key}").status_code == 403
        assert client.get(f"/api/submissions/{rid}/forensics/{request}.log?arena={key}").status_code == 403
    monkeypatch.setattr(disclosure.time, "time", lambda: release)
    rid, request = ids["a"]
    detail = client.get(f"/api/submissions/{rid}?arena=a").json()
    assert detail["url"].endswith("?arena=a")
    assert client.get(detail["url"]).status_code == 200
    assert client.get(f"/api/submissions/{rid}/forensics/{request}.log?arena=b").status_code == 404
    assert client.get(f"/api/submissions/{ids['b'][0]}/bundle.tar.gz?arena=b").status_code == 404


def test_config_never_opens_intake_as_enrichment_cache(tmp_path):
    row = _row(tmp_path, "a", tmp_path / "intake.sqlite3")
    row["cache"] = row["paths"]["db"]
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"default": "a", "sources": [row]}))
    with pytest.raises(ValueError, match="cannot be an intake"):
        sources.load_sources(path, "unused", 14, False)
    assert not Path(row["cache"]).exists()
