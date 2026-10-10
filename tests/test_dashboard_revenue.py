"""Evaluation fees count once, under the arena whose reservation consumed them."""

import json
import sqlite3

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from cacheon.chain.eval_cost_credit import EVAL_COST_CREDITS_DDL
from dashboard import app as dashboard
from dashboard.sources import DashboardSource

REFS = {"a": (100, 1), "b": (200, 2), "c": (300, 3)}
TAO = 1_000_000_000


class _Chain:
    """Enrichment stand-in with no chain access."""

    tip = {}
    metagraph = {}

    def block_time(self, block):
        return {"unix": None, "estimated": True}

    def extrinsic_signer(self, block, index):
        return {"signer": "", "call": ""}


def _database(path, legacy, fee, rows):
    con = sqlite3.connect(path)
    con.executescript(EVAL_COST_CREDITS_DDL + """
        CREATE TABLE metadata(key TEXT, value TEXT);
        CREATE TABLE reservations(reservation_id TEXT PRIMARY KEY, status TEXT, reason TEXT DEFAULT '',
            hotkey TEXT DEFAULT 'miner', block INTEGER, publication_digest TEXT DEFAULT '',
            competition_arena TEXT DEFAULT '', target_id TEXT DEFAULT '', decision TEXT DEFAULT '',
            arena_service_digest TEXT DEFAULT '', eval_cost_payment_block INTEGER);
        CREATE TABLE eval_cost_payments(payment_block INTEGER NOT NULL,
            payment_extrinsic_index INTEGER NOT NULL,
            reservation_id TEXT NOT NULL REFERENCES reservations(reservation_id),
            content_hash TEXT NOT NULL, hotkey TEXT NOT NULL, amount_tao_rao INTEGER NOT NULL,
            PRIMARY KEY(payment_block, payment_extrinsic_index));
        CREATE TABLE settlement_candidates(reservation_id TEXT, status TEXT);
        CREATE TABLE standing_reward_claims(status TEXT);
        CREATE TABLE evaluation_leases(state TEXT);
    """)
    if legacy:
        con.execute("INSERT INTO metadata VALUES('legacy_arena_id', ?)", (legacy,))
    for reservation, status, publication, arena, amount in rows:
        block, index = REFS[reservation]
        con.execute("INSERT INTO reservations(reservation_id,status,block,publication_digest,"
                    "competition_arena,eval_cost_payment_block) VALUES(?,?,?,?,?,?)",
                    (reservation, status, block, publication, arena, block))
        # Every listener records an arrival it observed at its own configured fee.
        con.execute("INSERT INTO eval_cost_payments VALUES(?,?,?,?,?,?)",
                    (block, index, reservation, "hash-" + reservation, "miner", amount or fee))
    con.commit()
    con.close()


@pytest.fixture
def arenas(tmp_path, monkeypatch):
    """Two listeners on one chain: GLM charged 1.0 τ then 0.5 τ, Qwen 0.2 τ, both saw b and c."""
    monkeypatch.delenv("CACHEON_DASH_SOURCES", raising=False)
    sources = {}
    for key, legacy, fee, rows in (
        ("glm", "glm-arena", TAO // 2, [("a", "published", "pub-a", "", TAO),
                                        ("b", "published", "pub-b", "", None),
                                        ("c", "failed", "", "", None)]),
        ("qwen", None, TAO // 5, [("b", "failed", "", "", None),
                                  ("c", "published", "pub-c", "qwen-arena", None)]),
    ):
        path = tmp_path / f"{key}.sqlite3"
        _database(path, legacy, fee, rows)
        registration = tmp_path / f"{key}-registration.json"
        registration.write_text(json.dumps({"worker_readiness": {"arena_id": f"{key}-arena"}}))
        sources[key] = DashboardSource(
            key, key.upper(), key, {"DB_PATH": path, "REGISTRATION_PATH": registration, "ENRICHER": _Chain()},
            {}, False, None, key)
    for source in sources.values():
        source.values["PEER_DB_PATHS"] = tuple(
            peer.values["DB_PATH"] for peer in sources.values() if peer is not source)
    monkeypatch.setattr(dashboard.app.state, "dashboard_sources", sources)
    monkeypatch.setattr(dashboard.app.state, "dashboard_default", "glm")
    return TestClient(dashboard.app), sources


def test_each_arena_lists_only_the_fees_its_reservations_consumed(arenas):
    client, _ = arenas
    paid = {key: [(row["ref"], row["amount_tao"]) for row in client.get(f"/api/payments?arena={key}").json()["items"]]
            for key in ("glm", "qwen")}
    assert paid == {"glm": [("200-2", 0.5), ("100-1", 1.0)], "qwen": [("300-3", 0.2)]}
    # The per-arena figure lives in /api/revenue alone; the overview no longer sums the shared table.
    totals = client.get("/api/overview?arena=glm").json()["totals"]
    assert "payments_tao" not in totals and totals["submissions"] == 2


def test_revenue_totals_every_fee_once_across_arenas(arenas):
    client, sources = arenas
    revenue = client.get("/api/revenue").json()
    assert "arena" not in revenue
    assert (revenue["payments_count"], revenue["unavailable_sources"]) == (3, [])
    assert revenue["total_tao"] == pytest.approx(1.7)
    items = {item["key"]: item for item in revenue["items"]}
    assert items["glm"]["by_fee"] == [{"fee_tao": 1.0, "count": 1}, {"fee_tao": 0.5, "count": 1}]
    assert items["qwen"]["by_fee"] == [{"fee_tao": 0.2, "count": 1}]
    assert (items["glm"]["payments_tao"], items["qwen"]["payments_tao"]) == (1.5, 0.2)
    # A publication replicated into the other arena must not count its payment twice.
    with sqlite3.connect(sources["qwen"].values["DB_PATH"]) as con:
        con.execute("UPDATE reservations SET competition_arena='qwen-arena', publication_digest='pub-b' "
                    "WHERE reservation_id='b'")
    replicated = client.get("/api/revenue").json()
    assert (replicated["payments_count"], replicated["total_tao"]) == (3, pytest.approx(1.7))
    assert [item["payments_count"] for item in replicated["items"]] == [2, 1]


def test_revenue_names_unavailable_arenas(arenas):
    client, sources = arenas
    sources["glm"].values["DB_PATH"].unlink()
    revenue = client.get("/api/revenue").json()
    assert revenue["unavailable_sources"] == ["glm"]
    assert [item["key"] for item in revenue["items"]] == ["qwen"]
    assert (revenue["payments_count"], revenue["total_tao"]) == (1, 0.2)


def test_single_database_revenue_reports_the_default_source(arenas, monkeypatch):
    client, sources = arenas
    monkeypatch.setattr(dashboard.app.state, "dashboard_sources", {})
    monkeypatch.setattr(dashboard.app.state, "dashboard_default", None)
    monkeypatch.setattr(dashboard, "DB_PATH", sources["glm"].values["DB_PATH"])
    revenue = client.get("/api/revenue").json()
    assert [item["key"] for item in revenue["items"]] == ["default"]
    assert (revenue["payments_count"], revenue["total_tao"]) == (3, 2.0)
