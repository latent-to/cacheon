"""Failed enrichment connections release resources before the worker reconnects."""

from unittest.mock import Mock
import os
import sys
from types import ModuleType

import pytest

from dashboard.enrichment import Enrichment


class StopWorker(BaseException):
    """End the otherwise unbounded worker after a complete retry cycle."""


@pytest.mark.parametrize("configured,expected", [(None, "2"), ("4", "4")])
def test_runtime_cache_default_preserves_explicit_configuration(tmp_path, monkeypatch, configured, expected):
    monkeypatch.delenv("SUBSTRATE_RUNTIME_CACHE_SIZE", raising=False)
    if configured is not None:
        monkeypatch.setenv("SUBSTRATE_RUNTIME_CACHE_SIZE", configured)
    client = ModuleType("async_substrate_interface.sync_substrate")
    client.SubstrateInterface = Mock(side_effect=lambda **kw: os.environ["SUBSTRATE_RUNTIME_CACHE_SIZE"])
    monkeypatch.setitem(sys.modules, "async_substrate_interface", ModuleType("async_substrate_interface"))
    monkeypatch.setitem(sys.modules, client.__name__, client)
    worker = Enrichment(tmp_path / "cache.sqlite3", "test-chain", 7)
    worker._connect()
    assert worker._substrate == expected
    client.SubstrateInterface.assert_called_once_with(url="test-chain")


@pytest.mark.parametrize("stage", ["_refresh_tip", "_refresh_metagraph", "_drain_extrinsics", "_drain_blocks"])
@pytest.mark.parametrize("close_fails", [False, True])
def test_failed_connection_is_closed_before_reconnect(tmp_path, monkeypatch, stage, close_fails):
    worker = Enrichment(tmp_path / "cache.sqlite3", "test-chain", 7)
    old, replacement = Mock(), Mock()
    if close_fails:
        old.close.side_effect = OSError("close failed")
    worker._substrate = old
    for method in ("_refresh_tip", "_refresh_metagraph", "_drain_extrinsics", "_drain_blocks"):
        monkeypatch.setattr(worker, method, Mock())
    getattr(worker, stage).side_effect = [ValueError("chain failed"), None]
    pauses = []

    def sleep(seconds):
        pauses.append(seconds)
        if seconds == 10:
            old.close.assert_called_once_with()
            assert worker._substrate is None
            assert not worker.chain_ok
            assert worker.chain_error.startswith("ValueError: chain failed")
            assert ("close: OSError: close failed" in worker.chain_error) == close_fails
        if len(pauses) == 3:
            raise StopWorker

    def connect():
        old.close.assert_called_once_with()
        worker._substrate = replacement

    monkeypatch.setattr("dashboard.enrichment.time.sleep", sleep)
    monkeypatch.setattr(worker, "_connect", connect)
    with pytest.raises(StopWorker):
        worker._loop()
    assert pauses == [10, 5, 5]
    assert worker.chain_ok and worker.chain_error == ""
    assert worker._substrate is replacement
    replacement.close.assert_not_called()


def test_connect_failure_without_client_preserves_error(tmp_path, monkeypatch):
    worker = Enrichment(tmp_path / "cache.sqlite3", "other-chain", 19)
    monkeypatch.setattr(worker, "_connect", Mock(side_effect=OSError("connect failed")))
    monkeypatch.setattr("dashboard.enrichment.time.sleep", Mock(side_effect=StopWorker))
    with pytest.raises(StopWorker):
        worker._loop()
    assert worker._substrate is None
    assert not worker.chain_ok
    assert worker.chain_error == "OSError: connect failed"
