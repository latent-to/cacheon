"""Miner-side publication of content-addressed bundles to public object storage.

The validator never receives these credentials.  A miner uses them only to put
one gzip archive into the miner's own S3-compatible bucket.  The resulting
anonymous HTTPS URL and the independently computed extracted-tree hash are the
only values that cross the submission boundary.
"""

from __future__ import annotations

import hashlib
import re
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from urllib.parse import quote, urlparse, urlunparse

from cacheon.chain.fetch import (
    FETCH_TIMEOUT_S,
    MAX_ARCHIVE_BYTES,
    FetchError,
    fetch_bundle,
    fetch_bundle_from_local_file_for_testing,
)
from cacheon.object_store import (
    ObjectStoreConfig,
    ObjectStoreError,
    S3CompatibleObjectStore,
    open_object_store,
)

DEFAULT_BUNDLE_KEY_PREFIX = "optima/miner-bundles/sha256"
PUBLIC_READ_GROUP = "http://acs.amazonaws.com/groups/global/AllUsers"
_HASH_RE = re.compile(r"[0-9a-f]{64}\Z")
_MISSING_CODES = frozenset({"404", "NoSuchBucket", "NoSuchKey", "NotFound"})
_ACL_UNSUPPORTED_CODES = frozenset(
    {"AccessControlListNotSupported", "NotImplemented"}
)


class BundlePublishError(RuntimeError):
    """A public proposal archive could not be published or verified."""


@dataclass(frozen=True)
class PublicBundlePublication:
    """Verified result of one content-addressed public publication."""

    archive_path: Path
    content_hash: str
    stored_archive_sha256: str
    stored_archive_bytes: int
    object_key: str
    url: str
    reused: bool


def _error_code(exc: BaseException) -> str:
    response = getattr(exc, "response", None)
    if not isinstance(response, dict):
        return ""
    error = response.get("Error")
    if not isinstance(error, dict):
        return ""
    code = error.get("Code")
    return str(code) if code is not None else ""


def _require_content_hash(content_hash: str) -> str:
    if not isinstance(content_hash, str) or _HASH_RE.fullmatch(content_hash) is None:
        raise BundlePublishError(
            "bundle content hash must be 64 lowercase hexadecimal characters"
        )
    return content_hash


def bundle_object_name(content_hash: str) -> str:
    """Return the immutable logical object name for one extracted-tree hash."""

    return f"{_require_content_hash(content_hash)}.tar.gz"


def _canonical_https_base(value: str, *, field: str) -> tuple[str, str, int | None, str]:
    if (
        not isinstance(value, str)
        or not value
        or not value.isascii()
        or any(ord(char) <= 32 or ord(char) == 127 for char in value)
    ):
        raise BundlePublishError(f"{field} must be a canonical HTTPS URL")
    try:
        parsed = urlparse(value)
        port = parsed.port
    except ValueError:
        raise BundlePublishError(f"{field} must be a canonical HTTPS URL") from None
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or (port is not None and not 1 <= port <= 65_535)
    ):
        raise BundlePublishError(f"{field} must be a canonical HTTPS URL")
    return parsed.hostname, parsed.path.rstrip("/"), port, parsed.scheme


def public_object_url(
    config: ObjectStoreConfig,
    object_key: str,
    *,
    public_base_url: str | None = None,
) -> str:
    """Build the credential-free HTTPS URL for an S3-compatible object."""

    if type(config) is not ObjectStoreConfig or config.backend_kind() != "s3":
        raise BundlePublishError(
            "public bundle publication requires an S3-compatible object store"
        )
    if (
        not isinstance(object_key, str)
        or not object_key
        or object_key.startswith("/")
        or any(part in {"", ".", ".."} for part in object_key.split("/"))
    ):
        raise BundlePublishError("public bundle object key is malformed")
    encoded_key = quote(object_key, safe="/-._~")

    if public_base_url:
        host, base_path, port, scheme = _canonical_https_base(
            public_base_url, field="public base URL"
        )
        netloc = host if port is None else f"{host}:{port}"
        path = f"{base_path}/{encoded_key}" if base_path else f"/{encoded_key}"
        return urlunparse((scheme, netloc, path, "", "", ""))

    endpoint = config.resolved_endpoint_url()
    region = config.resolved_region_name() or "us-east-1"
    addressing = config.resolved_addressing_style() or "auto"
    if endpoint is None:
        endpoint = (
            "https://s3.amazonaws.com"
            if region == "us-east-1"
            else f"https://s3.{region}.amazonaws.com"
        )
    host, base_path, port, scheme = _canonical_https_base(
        endpoint, field="object-store endpoint"
    )
    if base_path:
        raise BundlePublishError(
            "object-store endpoint must not contain a path; use --public-base-url "
            "for a CDN or gateway prefix"
        )

    if addressing == "path":
        netloc = host if port is None else f"{host}:{port}"
        path = f"/{quote(config.bucket, safe='-._~')}/{encoded_key}"
    elif addressing in {"auto", "virtual"}:
        if re.fullmatch(r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]", config.bucket) is None:
            raise BundlePublishError(
                "virtual-hosted public URLs require a DNS-compatible bucket name"
            )
        netloc = f"{config.bucket}.{host}"
        if port is not None:
            netloc = f"{netloc}:{port}"
        path = f"/{encoded_key}"
    else:
        raise BundlePublishError(
            "object-store addressing style must be path, virtual, or auto"
        )
    return urlunparse((scheme, netloc, path, "", "", ""))


def _read_stable_archive(path: str | Path) -> bytes:
    archive = Path(path)
    try:
        before = archive.lstat()
    except OSError as exc:
        raise BundlePublishError(f"bundle archive is unreadable: {exc}") from None
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or before.st_size <= 0
        or before.st_size > MAX_ARCHIVE_BYTES
    ):
        raise BundlePublishError(
            f"bundle archive must be one regular file in (0, {MAX_ARCHIVE_BYTES}] bytes"
        )
    try:
        with archive.open("rb") as stream:
            data = stream.read(MAX_ARCHIVE_BYTES + 1)
        after = archive.lstat()
    except OSError as exc:
        raise BundlePublishError(f"bundle archive could not be read: {exc}") from None
    stable_fields = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_nlink",
        "st_uid",
        "st_gid",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    if (
        len(data) != before.st_size
        or len(data) > MAX_ARCHIVE_BYTES
        or any(getattr(before, field) != getattr(after, field) for field in stable_fields)
    ):
        raise BundlePublishError("bundle archive changed while it was being read")
    return data


def _validate_archive_bytes(data: bytes, content_hash: str, *, label: str) -> None:
    with tempfile.TemporaryDirectory(prefix="cacheon-publish-verify.") as temporary:
        root = Path(temporary)
        archive = root / "bundle.tar.gz"
        archive.write_bytes(data)
        archive.chmod(0o600)
        try:
            fetch_bundle_from_local_file_for_testing(
                archive.as_uri(),
                content_hash,
                root / "cache",
                transfer_timeout_s=FETCH_TIMEOUT_S,
            )
        except FetchError as exc:
            raise BundlePublishError(f"{label} is not the committed bundle: {exc}") from None


def _read_remote_bytes(client: object, bucket: str, key: str) -> bytes | None:
    try:
        response = client.get_object(Bucket=bucket, Key=key)  # type: ignore[attr-defined]
    except Exception as exc:
        if _error_code(exc) in _MISSING_CODES:
            return None
        raise BundlePublishError(
            f"cannot read existing object-store key {key!r}: {exc}"
        ) from None
    body = response.get("Body") if isinstance(response, dict) else None
    if body is None or not hasattr(body, "read"):
        raise BundlePublishError("object store returned a malformed object body")
    try:
        data = body.read(MAX_ARCHIVE_BYTES + 1)
    except Exception as exc:
        raise BundlePublishError(f"cannot read object-store response body: {exc}") from None
    finally:
        close = getattr(body, "close", None)
        if callable(close):
            close()
    if not isinstance(data, (bytes, bytearray)) or not 0 < len(data) <= MAX_ARCHIVE_BYTES:
        raise BundlePublishError("existing object is empty or exceeds the archive cap")
    return bytes(data)


@dataclass
class S3PublicBundlePublisher:
    """Publish and anonymously re-open bundles through one S3-compatible client."""

    config: ObjectStoreConfig
    client: object
    public_base_url: str | None = None
    public_fetch: Callable[..., Path] = fetch_bundle

    def __post_init__(self) -> None:
        if (
            type(self.config) is not ObjectStoreConfig
            or self.config.backend_kind() != "s3"
        ):
            raise BundlePublishError(
                "public bundle publication requires an S3-compatible provider"
            )

    def _ensure_bucket(self, *, create_bucket: bool) -> None:
        try:
            self.client.head_bucket(Bucket=self.config.bucket)  # type: ignore[attr-defined]
            return
        except Exception as exc:
            if _error_code(exc) not in _MISSING_CODES:
                raise BundlePublishError(
                    f"cannot access object-store bucket {self.config.bucket!r}: {exc}"
                ) from None
            if not create_bucket:
                raise BundlePublishError(
                    f"object-store bucket {self.config.bucket!r} does not exist; "
                    "create it first or pass --create-bucket"
                ) from None
        kwargs: dict[str, object] = {"Bucket": self.config.bucket}
        region = self.config.resolved_region_name()
        if (
            self.config.resolved_endpoint_url() is None
            and region
            and region != "us-east-1"
        ):
            kwargs["CreateBucketConfiguration"] = {
                "LocationConstraint": region
            }
        try:
            self.client.create_bucket(**kwargs)  # type: ignore[attr-defined]
            self.client.head_bucket(Bucket=self.config.bucket)  # type: ignore[attr-defined]
        except Exception as exc:
            raise BundlePublishError(
                f"cannot create object-store bucket {self.config.bucket!r}: {exc}"
            ) from None

    def _put_public(
        self,
        key: str,
        data: bytes,
        *,
        content_hash: str,
        archive_sha256: str,
    ) -> None:
        kwargs: dict[str, object] = {
            "Bucket": self.config.bucket,
            "Key": key,
            "Body": data,
            "ACL": "public-read",
            "ContentType": "application/gzip",
            "CacheControl": "public, max-age=31536000, immutable",
            "Metadata": {
                "optima-content-sha256": content_hash,
                "optima-archive-sha256": archive_sha256,
                "optima-schema": "bundle-archive-v1",
            },
        }
        try:
            self.client.put_object(**kwargs)  # type: ignore[attr-defined]
            return
        except Exception as exc:
            if _error_code(exc) not in _ACL_UNSUPPORTED_CODES:
                raise BundlePublishError(f"public object upload failed: {exc}") from None
        # Some modern S3 buckets disable ACLs and expose objects through a bucket
        # policy or CDN instead.  Upload without an ACL, then let the mandatory
        # anonymous production fetch below prove whether that policy is sufficient.
        kwargs.pop("ACL")
        try:
            self.client.put_object(**kwargs)  # type: ignore[attr-defined]
        except Exception as exc:
            raise BundlePublishError(f"object upload failed: {exc}") from None

    def _make_existing_public(self, key: str) -> None:
        try:
            self.client.put_object_acl(  # type: ignore[attr-defined]
                Bucket=self.config.bucket,
                Key=key,
                ACL="public-read",
            )
        except Exception as exc:
            if _error_code(exc) not in _ACL_UNSUPPORTED_CODES:
                raise BundlePublishError(
                    f"existing object could not be made public-read: {exc}"
                ) from None

    def _verify_public(self, url: str, content_hash: str, *, timeout_s: float) -> None:
        with tempfile.TemporaryDirectory(prefix="cacheon-public-fetch.") as temporary:
            try:
                self.public_fetch(
                    url,
                    content_hash,
                    Path(temporary) / "cache",
                    transfer_timeout_s=timeout_s,
                )
            except FetchError as exc:
                raise BundlePublishError(
                    f"published object is not anonymously fetchable as the "
                    f"committed bundle: {exc}"
                ) from None

    def publish_archive(
        self,
        archive_path: str | Path,
        content_hash: str,
        *,
        create_bucket: bool = False,
        verify_timeout_s: float = FETCH_TIMEOUT_S,
    ) -> PublicBundlePublication:
        """Publish, re-open, and anonymously validate one proposal archive."""

        content_hash = _require_content_hash(content_hash)
        if (
            isinstance(verify_timeout_s, bool)
            or not isinstance(verify_timeout_s, (int, float))
            or not 0 < float(verify_timeout_s) <= 600
        ):
            raise BundlePublishError("public verification timeout must be in (0, 600]")
        archive = Path(archive_path)
        data = _read_stable_archive(archive)
        _validate_archive_bytes(data, content_hash, label="local archive")
        archive_sha256 = hashlib.sha256(data).hexdigest()
        key = self.config.resolve_key(bundle_object_name(content_hash))
        url = public_object_url(
            self.config, key, public_base_url=self.public_base_url
        )

        self._ensure_bucket(create_bucket=create_bucket)
        existing = _read_remote_bytes(
            self.client, self.config.bucket, key
        )
        reused = existing is not None
        public_verified = False
        if existing is not None:
            _validate_archive_bytes(
                existing, content_hash, label="existing content-addressed object"
            )
            # A bucket policy or CDN may already expose this object even when the
            # caller lacks PutObjectAcl. Avoid an unnecessary ACL mutation in that
            # case. If anonymous fetch fails, try the object-level public-read path
            # and let the mandatory retry decide whether publication succeeded.
            try:
                self._verify_public(
                    url, content_hash, timeout_s=float(verify_timeout_s)
                )
                public_verified = True
            except BundlePublishError:
                self._make_existing_public(key)
        else:
            self._put_public(
                key,
                data,
                content_hash=content_hash,
                archive_sha256=archive_sha256,
            )

        remote = _read_remote_bytes(self.client, self.config.bucket, key)
        if remote is None:
            raise BundlePublishError("uploaded object disappeared before verification")
        if not reused and remote != data:
            raise BundlePublishError("uploaded object bytes differ from the local archive")
        _validate_archive_bytes(remote, content_hash, label="stored object")
        if not public_verified:
            self._verify_public(
                url, content_hash, timeout_s=float(verify_timeout_s)
            )
        return PublicBundlePublication(
            archive,
            content_hash,
            hashlib.sha256(remote).hexdigest(),
            len(remote),
            key,
            url,
            reused,
        )


def open_public_bundle_publisher(
    config: ObjectStoreConfig,
    *,
    public_base_url: str | None = None,
) -> S3PublicBundlePublisher:
    """Open the configured S3-compatible backend for miner publication."""

    try:
        store = open_object_store(config)
    except ObjectStoreError as exc:
        raise BundlePublishError(str(exc)) from None
    if not isinstance(store, S3CompatibleObjectStore):
        raise BundlePublishError(
            "public bundle publication requires an S3-compatible object store"
        )
    return S3PublicBundlePublisher(
        config,
        store.client,
        public_base_url=public_base_url,
    )


__all__ = [
    "BundlePublishError",
    "DEFAULT_BUNDLE_KEY_PREFIX",
    "PublicBundlePublication",
    "S3PublicBundlePublisher",
    "bundle_object_name",
    "open_public_bundle_publisher",
    "public_object_url",
]
