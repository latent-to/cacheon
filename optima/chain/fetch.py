"""Bundle transport: pack a bundle directory into a tarball; fetch + safely extract one.

The tarball is *transport only* — identity is ``optima.bundle_hash.content_hash``
over the extracted DIRECTORY, so the same bundle hashes the same however it was
shipped. Packaging includes exactly the files the identity hash covers (same walk,
same skip rules); fetching re-hashes after extraction and refuses anything that
does not match the hash the miner committed on chain.

Extraction treats the archive as hostile: only regular files and directories are
accepted (no symlinks/hardlinks/devices), member paths must stay inside the
destination, and archive/extracted/member-count budgets are enforced. A rejected
archive leaves nothing behind.
"""

from __future__ import annotations

import http.client
import ipaddress
import logging
import os
import queue
import re
import socket
import ssl
import tarfile
import tempfile
import threading
import time
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urljoin, urlparse

from optima.bundle_hash import _iter_files, content_hash

logger = logging.getLogger("optima.chain.fetch")

MAX_ARCHIVE_BYTES = 64 * 1024 * 1024
MAX_EXTRACTED_BYTES = 256 * 1024 * 1024
MAX_MEMBERS = 4096
FETCH_TIMEOUT_S = 60.0
MAX_REDIRECTS = 5
_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
_TRANSFER_CHUNK_BYTES = 1024 * 1024


class FetchError(RuntimeError):
    """A submission artifact could not be fetched/extracted/verified. One bad
    submission must never take the validator loop down — callers catch this,
    record the rejection, and move on."""


class FetchTransientError(FetchError):
    """Transport/archive-host state may recover; do not terminally DQ the miner."""

    retryable = True


def package_bundle(bundle_dir: str | Path, out_path: str | Path | None = None) -> tuple[Path, str]:
    """Miner side: tar.gz the bundle and return ``(archive_path, content_hash)``.

    Contains exactly the files the identity hash covers, under a single top-level
    directory named after the bundle dir. The returned hash is what goes on chain.
    """
    root = Path(bundle_dir).resolve()
    ch = content_hash(root)  # raises if not a dir / empty
    out = Path(out_path) if out_path else Path(f"{root.name}.tar.gz")
    out.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out, "w:gz") as tar:
        for p in _iter_files(root):
            rel = p.relative_to(root).as_posix()
            tar.add(p, arcname=f"{root.name}/{rel}", recursive=False)
    return out, ch


def _remaining(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise FetchTransientError("bundle transfer exceeded its absolute deadline")
    return remaining


def _public_ip(value: str, *, context: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address:
    try:
        address = ipaddress.ip_address(value.split("%", 1)[0])
    except ValueError:
        raise FetchError(f"{context} is not a canonical IP address") from None
    if not address.is_global:
        raise FetchError(
            f"{context} resolves to a non-public destination ({address.compressed})"
        )
    return address


def _resolve_addresses(hostname: str, port: int, *, deadline: float) -> tuple[str, ...]:
    """Bound libc/NSS resolution by the same absolute transfer deadline."""

    result: queue.Queue[object] = queue.Queue(maxsize=1)

    def resolve() -> None:
        try:
            result.put(socket.getaddrinfo(
                hostname,
                port,
                family=socket.AF_UNSPEC,
                type=socket.SOCK_STREAM,
                proto=socket.IPPROTO_TCP,
            ))
        except BaseException as exc:  # handed back to the trusted caller
            result.put(exc)

    worker = threading.Thread(
        target=resolve,
        name="optima-bundle-dns",
        daemon=True,
    )
    worker.start()
    try:
        resolved = result.get(timeout=_remaining(deadline))
    except queue.Empty:
        raise FetchTransientError(
            "bundle host DNS resolution exceeded the transfer deadline"
        ) from None
    if isinstance(resolved, BaseException):
        raise FetchTransientError(
            f"bundle host DNS resolution failed: {resolved}"
        ) from None
    addresses = tuple(sorted({str(answer[4][0]) for answer in resolved}))
    if not addresses:
        raise FetchError("bundle host DNS resolution returned no addresses")
    for address in addresses:
        _public_ip(address, context="bundle host")
    return addresses


def _validated_https_url(
    url: str, *, deadline: float
) -> tuple[object, tuple[str, ...]]:
    """Validate syntax and every DNS answer before any request bytes are sent."""

    if not isinstance(url, str) or not url or len(url) > 8_192:
        raise FetchError("bundle URL is empty or oversized")
    if (
        not url.isascii()
        or any(ord(char) <= 32 or ord(char) == 127 for char in url)
    ):
        raise FetchError(
            "bundle URL must be canonical ASCII without spaces/control characters"
        )
    try:
        parsed = urlparse(url)
        port = parsed.port or 443
    except ValueError as exc:
        raise FetchError(f"bundle URL is malformed: {exc}") from None
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
        or not 1 <= port <= 65_535
    ):
        raise FetchError(
            "production bundle URLs must be HTTPS without credentials or fragments"
        )
    addresses = _resolve_addresses(parsed.hostname, port, deadline=deadline)
    return parsed, addresses


def _open_pinned_https(
    hostname: str,
    port: int,
    addresses: tuple[str, ...],
    *,
    deadline: float,
) -> http.client.HTTPSConnection:
    """Connect to an already-reviewed IP while retaining TLS SNI/hostname checks."""

    context = ssl.create_default_context()
    failures: list[str] = []
    for raw_address in addresses:
        address = _public_ip(raw_address, context="bundle host")
        family = socket.AF_INET6 if address.version == 6 else socket.AF_INET
        sock = socket.socket(family, socket.SOCK_STREAM, socket.IPPROTO_TCP)
        wrapped = None
        try:
            sock.settimeout(_remaining(deadline))
            destination = (
                (address.compressed, port, 0, 0)
                if family == socket.AF_INET6
                else (address.compressed, port)
            )
            sock.connect(destination)
            sock.settimeout(_remaining(deadline))
            wrapped = context.wrap_socket(sock, server_hostname=hostname)
            peer = wrapped.getpeername()[0]
            _public_ip(str(peer), context="connected bundle peer")
            wrapped.settimeout(_remaining(deadline))
            conn = http.client.HTTPSConnection(
                hostname,
                port,
                timeout=_remaining(deadline),
                context=context,
            )
            # Supplying the reviewed TLS socket prevents HTTPSConnection from doing
            # a second, attacker-raceable DNS resolution in ``request``.
            conn.sock = wrapped
            return conn
        except FetchError:
            if wrapped is not None:
                wrapped.close()
            else:
                sock.close()
            raise
        except (OSError, ssl.SSLError) as exc:
            failures.append(f"{address.compressed}:{type(exc).__name__}")
            if wrapped is not None:
                wrapped.close()
            else:
                sock.close()
    raise FetchTransientError(
        "could not establish pinned HTTPS connection to any reviewed address: "
        + ",".join(failures)[:1024]
    )


def _response_read_socket(response: http.client.HTTPResponse):
    """Return the socket still owned by an HTTPResponse body stream.

    ``HTTPConnection.getresponse`` closes ``conn.sock`` when the peer declares
    ``Connection: close``. The response's ``makefile`` keeps the transport alive,
    but the old connection socket object is no longer usable for deadline updates.
    """

    stream = getattr(response, "fp", None)
    raw = getattr(stream, "raw", stream)
    sock = getattr(raw, "_sock", None)
    if sock is None or not callable(getattr(sock, "settimeout", None)):
        raise FetchTransientError(
            "bundle response does not expose a deadline-controlled body socket"
        )
    return sock


def _download_https(
    url: str,
    dest: Path,
    max_bytes: int,
    *,
    deadline: float,
) -> None:
    """Download with no proxy, bounded redirects, destination checks and one deadline."""

    current = url
    for redirect_count in range(MAX_REDIRECTS + 1):
        parsed, addresses = _validated_https_url(current, deadline=deadline)
        port = parsed.port or 443
        conn = _open_pinned_https(
            parsed.hostname,
            port,
            addresses,
            deadline=deadline,
        )
        try:
            if conn.sock is None:
                raise FetchError("HTTPS connection exposed no peer socket")
            wire_sock = conn.sock
            wire_sock.settimeout(_remaining(deadline))
            path = parsed.path or "/"
            if parsed.query:
                path += "?" + parsed.query
            conn.request(
                "GET",
                path,
                headers={
                    "Accept": "application/octet-stream, application/gzip",
                    "Connection": "close",
                    "User-Agent": "optima-validator-bundle-fetch/1",
                },
            )
            wire_sock.settimeout(_remaining(deadline))
            response = conn.getresponse()
            if response.status in _REDIRECT_STATUSES:
                location = response.getheader("Location")
                response.close()
                if not location:
                    raise FetchError("HTTPS redirect omitted its Location header")
                if redirect_count >= MAX_REDIRECTS:
                    raise FetchError("bundle URL exceeded the redirect limit")
                current = urljoin(current, location)
                # Validate immediately, before the next connection attempt.
                _validated_https_url(current, deadline=deadline)
                continue
            if response.status != 200:
                error_cls = (
                    FetchTransientError
                    if response.status in {408, 425, 429} or response.status >= 500
                    else FetchError
                )
                raise error_cls(f"bundle host returned HTTP status {response.status}")
            body_sock = _response_read_socket(response)
            raw_length = response.getheader("Content-Length")
            declared: int | None = None
            if raw_length is not None:
                try:
                    declared = int(raw_length)
                except (TypeError, ValueError):
                    raise FetchError("bundle response Content-Length is invalid") from None
                if declared < 0 or declared > max_bytes:
                    raise FetchError(f"archive exceeds {max_bytes} bytes: {url}")
            total = 0
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            fd = os.open(dest, flags, 0o600)
            try:
                while True:
                    # HTTPResponse closes its makefile/socket as soon as a final
                    # fixed-length or chunked body read completes. Do not touch that
                    # already-closed socket merely to ask for one more EOF read.
                    if response.isclosed():
                        break
                    body_sock.settimeout(_remaining(deadline))
                    chunk = response.read(_TRANSFER_CHUNK_BYTES)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > max_bytes:
                        raise FetchError(f"archive exceeds {max_bytes} bytes: {url}")
                    view = memoryview(chunk)
                    while view:
                        written = os.write(fd, view)
                        if written <= 0:
                            raise FetchError("bundle archive write made no progress")
                        view = view[written:]
                if declared is not None and total != declared:
                    raise FetchTransientError(
                        "bundle response closed before its declared Content-Length"
                    )
                _remaining(deadline)
                os.fsync(fd)
            finally:
                os.close(fd)
            return
        except FetchError:
            raise
        except (OSError, ssl.SSLError, http.client.HTTPException) as exc:
            raise FetchTransientError(
                f"download failed for {current}: {type(exc).__name__}: {exc}"
            ) from None
        finally:
            conn.close()
    raise FetchError("bundle URL exceeded the redirect limit")  # pragma: no cover


def _copy_local_archive_for_testing(
    url: str, dest: Path, max_bytes: int, *, deadline: float
) -> None:
    """Explicit hermetic-test transport. Production code never selects this path."""

    parsed = urlparse(url)
    if parsed.scheme != "file" or parsed.netloc not in ("", "localhost"):
        raise FetchError("test-only local bundle URL must be file:// on this host")
    src = Path(unquote(parsed.path))
    try:
        size = src.stat().st_size
    except OSError as exc:
        raise FetchError(f"test-only file URL is unreadable: {exc}") from None
    if not src.is_file():
        raise FetchError(f"file url does not point at a file: {url}")
    if size > max_bytes:
        raise FetchError(f"archive exceeds {max_bytes} bytes: {url}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(dest, flags, 0o600)
    try:
        with src.open("rb") as stream:
            remaining = size
            while remaining:
                _remaining(deadline)
                chunk = stream.read(min(_TRANSFER_CHUNK_BYTES, remaining))
                if not chunk:
                    raise FetchError("test-only local archive was truncated")
                view = memoryview(chunk)
                while view:
                    written = os.write(fd, view)
                    if written <= 0:
                        raise FetchError("test-only archive copy made no progress")
                    view = view[written:]
                remaining -= len(chunk)
            if stream.read(1):
                raise FetchError("test-only local archive grew during copy")
        _remaining(deadline)
        os.fsync(fd)
    finally:
        os.close(fd)


def _safe_extract(archive: Path, dest: Path) -> None:
    """Extract accepting only regular files/dirs with in-tree relative paths."""
    budget = MAX_EXTRACTED_BYTES
    members = 0
    seen: dict[PurePosixPath, str] = {}
    try:
        with tarfile.open(archive, "r:*") as tar:
            for m in tar:
                members += 1
                if members > MAX_MEMBERS:
                    raise FetchError(f"archive has more than {MAX_MEMBERS} members")
                name = PurePosixPath(m.name)
                if (
                    not m.name
                    or "\\" in m.name
                    or any(ord(char) < 32 or ord(char) == 127 for char in m.name)
                    or name.is_absolute()
                    or not name.parts
                    or any(part in {"", ".", ".."} for part in name.parts)
                    or name.as_posix() != m.name.rstrip("/")
                ):
                    raise FetchError(f"archive member escapes destination: {m.name!r}")
                if name in seen:
                    raise FetchError(f"archive contains duplicate member path: {m.name!r}")
                ancestors = tuple(name.parents)[:-1]
                conflicting_parent = next(
                    (parent for parent in ancestors if seen.get(parent) == "file"),
                    None,
                )
                if conflicting_parent is not None:
                    raise FetchError(
                        "archive path conflicts with an earlier file: "
                        f"{conflicting_parent.as_posix()!r} vs {m.name!r}"
                    )
                if m.isdir():
                    seen[name] = "dir"
                    try:
                        (dest / Path(*name.parts)).mkdir(parents=True, exist_ok=True)
                    except OSError as exc:
                        raise FetchError(
                            f"archive directory path conflict: {m.name!r}: {exc}"
                        ) from None
                    continue
                if not m.isreg():
                    raise FetchError(f"archive member is not a regular file: {m.name!r}")
                if any(prior != name and name in prior.parents for prior in seen):
                    raise FetchError(
                        f"archive file path conflicts with an earlier child: {m.name!r}"
                    )
                seen[name] = "file"
                if type(m.size) is not int or m.size < 0:
                    raise FetchError(f"archive member has an invalid size: {m.name!r}")
                budget -= m.size
                if budget < 0:
                    raise FetchError(f"extracted size exceeds {MAX_EXTRACTED_BYTES} bytes")
                target = dest / Path(*name.parts)
                src = tar.extractfile(m)
                if src is None:
                    raise FetchError(f"unreadable archive member: {m.name!r}")
                try:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
                    if hasattr(os, "O_NOFOLLOW"):
                        flags |= os.O_NOFOLLOW
                    fd = os.open(target, flags, 0o600)
                    with src, os.fdopen(fd, "wb", closefd=True) as stream:
                        remaining = m.size
                        while remaining:
                            chunk = src.read(min(_TRANSFER_CHUNK_BYTES, remaining))
                            if not chunk:
                                raise FetchError(
                                    f"archive member was truncated: {m.name!r}"
                                )
                            stream.write(chunk)
                            remaining -= len(chunk)
                        if src.read(1):
                            raise FetchError(
                                f"archive member exceeded its declared size: {m.name!r}"
                            )
                except FetchError:
                    raise
                except OSError as exc:
                    raise FetchError(
                        f"archive file path conflict: {m.name!r}: {exc}"
                    ) from None
    except FetchError:
        raise
    except (tarfile.TarError, OSError, EOFError) as e:
        raise FetchError(f"corrupt archive: {e}") from e


def _bundle_root(extract_dir: Path) -> Path:
    """The bundle root is the single top-level dir if there is exactly one, else the
    extraction dir itself (manifest.toml at archive top level)."""
    entries = [p for p in extract_dir.iterdir() if p.name != "__MACOSX"]
    if len(entries) == 1 and entries[0].is_dir():
        return entries[0]
    return extract_dir


def _fetch_bundle(
    url: str,
    expected_hash: str,
    dest_root: str | Path,
    *,
    test_only_local_file: bool,
    transfer_timeout_s: float,
) -> Path:
    """Validator side: fetch, safely extract, and hash-verify a committed bundle.

    Returns the bundle directory at ``dest_root/<expected_hash>``. Idempotent: an
    existing directory for this hash is re-verified and reused. Raises FetchError
    on any transport, extraction, or hash failure — leaving no partial state.
    """
    if (
        not isinstance(expected_hash, str)
        or re.fullmatch(r"[0-9a-f]{64}", expected_hash) is None
    ):
        raise FetchError("expected bundle hash must be 64 lowercase hex characters")
    if (
        isinstance(transfer_timeout_s, bool)
        or not isinstance(transfer_timeout_s, (int, float))
        or not 0 < float(transfer_timeout_s) <= 600
    ):
        raise FetchError("bundle transfer timeout must be in (0, 600] seconds")
    deadline = time.monotonic() + float(transfer_timeout_s)
    dest_root = Path(dest_root)
    dest_root.mkdir(parents=True, exist_ok=True)
    final = dest_root / expected_hash
    if final.exists():
        actual = content_hash(final)
        if actual == expected_hash:
            return final
        # A corrupted/tampered cache entry: refuse to silently reuse it.
        raise FetchError(f"cached bundle at {final} re-hashes to {actual[:16]}…; "
                         "remove it manually to re-fetch")

    with tempfile.TemporaryDirectory(dir=dest_root, prefix=".fetch.") as tmp:
        tmp = Path(tmp)
        archive = tmp / "bundle.tar.gz"
        if test_only_local_file:
            _copy_local_archive_for_testing(
                url, archive, MAX_ARCHIVE_BYTES, deadline=deadline
            )
        else:
            _download_https(url, archive, MAX_ARCHIVE_BYTES, deadline=deadline)
        extract_dir = tmp / "extract"
        extract_dir.mkdir()
        _safe_extract(archive, extract_dir)
        root = _bundle_root(extract_dir)
        try:
            actual = content_hash(root)
        except (ValueError, NotADirectoryError) as e:
            raise FetchError(f"extracted archive is not a bundle: {e}") from e
        if actual != expected_hash:
            raise FetchError(
                f"content hash mismatch: committed {expected_hash[:16]}…, "
                f"fetched {actual[:16]}… — rejecting submission")
        root.rename(final)
    logger.info("fetched bundle %s… from %s", expected_hash[:16], url)
    return final


def fetch_bundle(
    url: str,
    expected_hash: str,
    dest_root: str | Path,
    *,
    transfer_timeout_s: float = FETCH_TIMEOUT_S,
) -> Path:
    """Production HTTPS-only bundle fetch."""

    return _fetch_bundle(
        url,
        expected_hash,
        dest_root,
        test_only_local_file=False,
        transfer_timeout_s=transfer_timeout_s,
    )


def fetch_bundle_from_local_file_for_testing(
    url: str,
    expected_hash: str,
    dest_root: str | Path,
    *,
    transfer_timeout_s: float = FETCH_TIMEOUT_S,
) -> Path:
    """Hermetic-test-only local-file transport; never selected by production APIs."""

    return _fetch_bundle(
        url,
        expected_hash,
        dest_root,
        test_only_local_file=True,
        transfer_timeout_s=transfer_timeout_s,
    )
