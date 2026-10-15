"""Health polling shares process reads without mixing arena process identities."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from dashboard import sources


@pytest.fixture
def process_table(tmp_path, monkeypatch):
    monkeypatch.setattr(sources, "_PROCESS_SNAPSHOT", (None, 0.0, ()))
    entry = tmp_path / "123"
    entry.mkdir()
    (entry / "cmdline").write_bytes(b"python\0-m\0relay\0--config=/arena/one/service.json\0")
    (entry / "cwd").symlink_to(tmp_path)
    return tmp_path, entry


def test_snapshot_preserves_exact_tokens_and_source_paths(process_table):
    root, _ = process_table
    assert sources.process_matches("relay", ("relay", "/arena/one"), proc_root=root)
    assert not sources.process_matches("relay", ("relay", "/arena/two"), proc_root=root)
    assert not sources.process_matches("relay", ("rel", "/arena/one"), proc_root=root)
    token = sources.selected.set(SimpleNamespace(processes={"relay": ("relay", "/arena/two")}))
    try:
        assert not sources.process_matches("relay", ("relay", "/arena/one"), proc_root=root)
    finally:
        sources.selected.reset(token)


def test_concurrent_pollers_scan_once_and_expiry_observes_stop(process_table, monkeypatch):
    root, entry = process_table
    now, scans = [10.0], []
    scan = sources._scan_processes
    def counted(path):
        scans.append(path)
        return scan(path)
    monkeypatch.setattr(sources, "_scan_processes", counted)
    monkeypatch.setattr(sources.time, "monotonic", lambda: now[0])
    with ThreadPoolExecutor(max_workers=12) as pool:
        results = list(pool.map(lambda _: sources.process_matches("relay", ("relay",), proc_root=root), range(36)))
    assert all(results)
    assert scans == [root]
    (entry / "cmdline").unlink()
    now[0] = 11.9
    assert sources.process_matches("relay", ("relay",), proc_root=root)
    now[0] = 12.0
    assert not sources.process_matches("relay", ("relay",), proc_root=root)
    assert scans == [root, root]


def test_different_process_root_and_failed_scan_do_not_reuse_snapshot(process_table, tmp_path):
    root, _ = process_table
    assert sources.process_matches("relay", ("relay",), proc_root=root)
    other = tmp_path / "other"
    other.mkdir()
    assert not sources.process_matches("relay", ("relay",), proc_root=other)
    with pytest.raises(FileNotFoundError):
        sources.process_matches("relay", ("relay",), proc_root=Path(other / "missing"))
