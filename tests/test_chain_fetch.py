"""Bundle transport: packaging roundtrip, hash verification, hostile archives."""

from __future__ import annotations

import io
import socket
import tarfile
import time
from types import SimpleNamespace

import pytest

from optima.bundle_hash import content_hash
from optima.chain.fetch import (
    FetchError,
    fetch_bundle as fetch_bundle_https,
    fetch_bundle_from_local_file_for_testing as fetch_bundle,
    package_bundle,
)


def _make_bundle(root, name="bundle"):
    b = root / name
    (b / "kernels").mkdir(parents=True)
    (b / "manifest.toml").write_text('bundle_id = "t"\n')
    (b / "kernels" / "k.py").write_text("def f():\n    return 1\n")
    return b


def test_package_roundtrip(tmp_path):
    bundle = _make_bundle(tmp_path)
    archive, ch = package_bundle(bundle, tmp_path / "out.tar.gz")
    assert ch == content_hash(bundle)
    fetched = fetch_bundle(archive.as_uri(), ch, tmp_path / "cache")
    assert content_hash(fetched) == ch
    assert (fetched / "manifest.toml").exists()


def test_package_excludes_junk(tmp_path):
    bundle = _make_bundle(tmp_path)
    (bundle / "__pycache__").mkdir()
    (bundle / "__pycache__" / "k.cpython-311.pyc").write_bytes(b"junk")
    archive, ch = package_bundle(bundle, tmp_path / "out.tar.gz")
    with tarfile.open(archive) as tar:
        names = tar.getnames()
    assert not any("__pycache__" in n for n in names)
    # junk does not perturb identity either
    assert ch == content_hash(bundle)


def test_fetch_is_idempotent_and_detects_corrupted_cache(tmp_path):
    bundle = _make_bundle(tmp_path)
    archive, ch = package_bundle(bundle, tmp_path / "out.tar.gz")
    cache = tmp_path / "cache"
    first = fetch_bundle(archive.as_uri(), ch, cache)
    again = fetch_bundle(archive.as_uri(), ch, cache)
    assert first == again
    (first / "kernels" / "k.py").write_text("tampered = True\n")
    with pytest.raises(FetchError, match="re-hashes"):
        fetch_bundle(archive.as_uri(), ch, cache)


def test_fetch_rejects_hash_mismatch(tmp_path):
    bundle = _make_bundle(tmp_path)
    archive, _ = package_bundle(bundle, tmp_path / "out.tar.gz")
    with pytest.raises(FetchError, match="mismatch"):
        fetch_bundle(archive.as_uri(), "b" * 64, tmp_path / "cache")
    # nothing cached under the bogus hash
    assert not (tmp_path / "cache" / ("b" * 64)).exists()


def _write_tar(path, members):
    """members: list of (TarInfo, bytes|None)"""
    with tarfile.open(path, "w:gz") as tar:
        for info, data in members:
            tar.addfile(info, io.BytesIO(data) if data is not None else None)


def _reg(name, data=b"x"):
    info = tarfile.TarInfo(name)
    info.size = len(data)
    return info, data


def test_extract_rejects_symlink_member(tmp_path):
    evil = tarfile.TarInfo("bundle/link")
    evil.type = tarfile.SYMTYPE
    evil.linkname = "/etc/passwd"
    path = tmp_path / "evil.tar.gz"
    _write_tar(path, [_reg("bundle/manifest.toml"), (evil, None)])
    with pytest.raises(FetchError, match="not a regular file"):
        fetch_bundle(path.as_uri(), "a" * 64, tmp_path / "cache")


def test_extract_rejects_path_escape(tmp_path):
    for name in (
        "../outside.py",
        "/abs.py",
        "bundle/../../outside.py",
        "bundle/control\nname.py",
    ):
        path = tmp_path / "evil.tar.gz"
        _write_tar(path, [_reg(name)])
        with pytest.raises(FetchError, match="escapes"):
            fetch_bundle(path.as_uri(), "a" * 64, tmp_path / "cache")
        assert not (tmp_path / "outside.py").exists()
        assert not (tmp_path / "cache" / "outside.py").exists()


def test_extract_rejects_hardlink_member(tmp_path):
    evil = tarfile.TarInfo("bundle/hard")
    evil.type = tarfile.LNKTYPE
    evil.linkname = "manifest.toml"
    path = tmp_path / "evil.tar.gz"
    _write_tar(path, [_reg("bundle/manifest.toml"), (evil, None)])
    with pytest.raises(FetchError, match="not a regular file"):
        fetch_bundle(path.as_uri(), "a" * 64, tmp_path / "cache")


def test_download_size_cap(tmp_path, monkeypatch):
    import optima.chain.fetch as fetch_mod

    bundle = _make_bundle(tmp_path)
    archive, ch = package_bundle(bundle, tmp_path / "out.tar.gz")
    monkeypatch.setattr(fetch_mod, "MAX_ARCHIVE_BYTES", 10)
    with pytest.raises(FetchError, match="exceeds"):
        fetch_bundle(archive.as_uri(), ch, tmp_path / "cache")


def test_fetch_rejects_unknown_scheme(tmp_path):
    with pytest.raises(FetchError, match="HTTPS"):
        fetch_bundle_https("ftp://example.com/x.tar.gz", "a" * 64, tmp_path / "cache")
    for index, url in enumerate((
        "https://example.com/x.tar.gz\r\nX-Evil: 1",
        "https://example.com/a b.tar.gz",
        "https://example.com/é.tar.gz",
    )):
        with pytest.raises(FetchError, match="canonical ASCII|control character"):
            fetch_bundle_https(
                url,
                "a" * 64,
                tmp_path / f"cache-control-{index}",
            )


def test_production_fetch_rejects_file_plain_http_and_private_dns(tmp_path, monkeypatch):
    for url in ("file:///tmp/x.tar.gz", "http://example.com/x.tar.gz"):
        with pytest.raises(FetchError, match="HTTPS"):
            fetch_bundle_https(url, "a" * 64, tmp_path / "cache")

    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *args, **kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("127.0.0.1", 443))
        ],
    )
    with pytest.raises(FetchError, match="non-public"):
        fetch_bundle_https(
            "https://miner.example/x.tar.gz", "a" * 64, tmp_path / "cache-private"
        )


def test_production_fetch_dns_obeys_absolute_deadline(tmp_path, monkeypatch):
    def blocked(*args, **kwargs):
        time.sleep(0.2)
        return []

    monkeypatch.setattr(socket, "getaddrinfo", blocked)
    started = time.monotonic()
    with pytest.raises(FetchError, match="deadline"):
        fetch_bundle_https(
            "https://miner.example/x.tar.gz",
            "a" * 64,
            tmp_path / "cache",
            transfer_timeout_s=0.02,
        )
    assert time.monotonic() - started < 0.15


def test_https_body_deadline_uses_response_socket_after_connection_close(
    tmp_path, monkeypatch
):
    import optima.chain.fetch as fetch_mod

    class FakeSocket:
        def __init__(self):
            self.closed = False
            self.timeouts = []

        def settimeout(self, value):
            if self.closed:
                raise OSError(9, "Bad file descriptor")
            self.timeouts.append(value)

    connection_socket = FakeSocket()
    body_socket = FakeSocket()

    class FakeResponse:
        status = 200
        fp = SimpleNamespace(raw=SimpleNamespace(_sock=body_socket))

        def __init__(self):
            self.chunks = iter((b"archive", b""))
            self.closed = False

        def getheader(self, name):
            return "7" if name == "Content-Length" else None

        def read(self, _size):
            chunk = next(self.chunks)
            if chunk:
                self.closed = True
                body_socket.closed = True
            return chunk

        def isclosed(self):
            return self.closed

    class FakeConnection:
        def __init__(self):
            self.sock = connection_socket

        def request(self, *_args, **_kwargs):
            pass

        def getresponse(self):
            connection_socket.closed = True
            self.sock = None
            return FakeResponse()

        def close(self):
            pass

    parsed = SimpleNamespace(
        hostname="miner.example", port=None, path="/bundle.tar.gz", query=""
    )
    monkeypatch.setattr(
        fetch_mod, "_validated_https_url", lambda *_args, **_kwargs: (parsed, ("8.8.8.8",))
    )
    monkeypatch.setattr(
        fetch_mod, "_open_pinned_https", lambda *_args, **_kwargs: FakeConnection()
    )
    destination = tmp_path / "bundle.tar.gz"
    fetch_mod._download_https(
        "https://miner.example/bundle.tar.gz",
        destination,
        1024,
        deadline=time.monotonic() + 5,
    )
    assert destination.read_bytes() == b"archive"
    assert body_socket.timeouts


def test_extract_rejects_duplicate_and_ancestor_path_conflicts(tmp_path):
    cases = (
        [_reg("bundle/a"), _reg("bundle/a")],
        [_reg("bundle/a"), _reg("bundle/a/child")],
        [_reg("bundle/a/child"), _reg("bundle/a")],
    )
    for index, members in enumerate(cases):
        archive = tmp_path / f"conflict-{index}.tar.gz"
        _write_tar(archive, members)
        with pytest.raises(FetchError, match="duplicate|conflict"):
            fetch_bundle(archive.as_uri(), "a" * 64, tmp_path / f"cache-{index}")
