"""Dashboard chain enrichment reads and writes its cache through one connection held for the process."""

import sqlite3
from types import SimpleNamespace

from dashboard.enrichment import Enrichment


def test_lookups_and_the_worker_reuse_one_connection_and_persist(tmp_path, monkeypatch):
    block = {"extrinsics": [
        SimpleNamespace(value={"call": {"call_module": "Timestamp", "call_function": "set",
                                        "call_args": [{"name": "now", "value": 1_700_000_000_000}]}}),
        SimpleNamespace(value={"address": "5Signer", "call": {"call_module": "SubtensorModule",
                                                              "call_function": "burned_register"}}),
    ]}
    enricher = Enrichment(tmp_path / "enrichment.sqlite3", "ws://unused", 14)
    enricher._substrate = SimpleNamespace(get_block_hash=lambda number: "0x07", get_block=lambda block_hash: block)
    connects, connect = [], sqlite3.connect
    monkeypatch.setattr(sqlite3, "connect", lambda *args, **kwargs: connects.append(args) or connect(*args, **kwargs))
    assert enricher.block_time(7) == {"unix": None, "estimated": True}
    assert enricher.extrinsic_signer(7, 1) == {"signer": None, "call": None}
    enricher._fetch_block(7)
    assert enricher.block_time(7) == {"unix": 1_700_000_000, "estimated": False}
    assert enricher.extrinsic_signer(7, 1) == {"signer": "5Signer", "call": "SubtensorModule.burned_register"}
    assert connects == []
    assert Enrichment(tmp_path / "enrichment.sqlite3", "ws://unused", 14).block_time(7)["unix"] == 1_700_000_000
