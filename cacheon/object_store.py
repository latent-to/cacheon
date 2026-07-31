"""Provider-agnostic object storage for Cacheon control-plane artifacts.

Backends are swappable through :class:`ObjectStoreConfig.provider` (and optional
endpoint overrides). The S3-compatible path uses boto3 only when selected; the
core package does not hard-depend on it. Hippius is one preset endpoint/region
profile over the same S3-compatible client — not a fork of any provider SDK.

boto3/botocore are Apache-2.0. This module contains no GPL code and does not
vendor third-party sources; it only calls the public S3 API shape.
"""

from __future__ import annotations

import os
import stat
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


class ObjectStoreError(RuntimeError):
    """Object-store configuration or I/O failed closed."""

    validator_fault = True
    retryable = False


class ObjectStoreRetryableError(ObjectStoreError):
    """Transient remote object-store failure."""

    retryable = True


class ObjectStoreNotFoundError(ObjectStoreError):
    """The requested object does not exist."""


class ObjectStore(Protocol):
    """Minimal byte-object API shared by every provider backend."""

    def put_bytes(
        self,
        key: str,
        data: bytes,
        *,
        content_type: str = "application/octet-stream",
    ) -> None: ...

    def get_bytes(self, key: str, *, max_bytes: int | None = None) -> bytes: ...


# Known S3-compatible presets. Operators swap providers by changing ``provider``
# (and credentials/endpoint when needed); call sites keep the same ObjectStore API.
_PROVIDER_DEFAULTS: dict[str, dict[str, str | None]] = {
    "hippius": {
        "backend": "s3",
        "endpoint_url": "https://s3.hippius.com",
        "region_name": "decentralized",
        "addressing_style": "path",
    },
    # Generic AWS S3 (or any path that accepts the AWS SDK defaults).
    "s3": {
        "backend": "s3",
        "endpoint_url": None,
        "region_name": "us-east-1",
        "addressing_style": "auto",
    },
    "minio": {
        "backend": "s3",
        "endpoint_url": "http://127.0.0.1:9000",
        "region_name": "us-east-1",
        "addressing_style": "path",
    },
    # Local filesystem rooted at ``root_dir`` (tests / air-gapped dry runs).
    "local": {
        "backend": "local",
        "endpoint_url": None,
        "region_name": None,
        "addressing_style": None,
    },
    # In-process map (unit tests only).
    "memory": {
        "backend": "memory",
        "endpoint_url": None,
        "region_name": None,
        "addressing_style": None,
    },
}

KNOWN_OBJECT_STORE_PROVIDERS = frozenset(_PROVIDER_DEFAULTS)


@dataclass(frozen=True)
class ObjectStoreConfig:
    """Declarative object-store selection. Swap providers without code changes."""

    provider: str
    bucket: str = ""
    key_prefix: str = ""
    endpoint_url: str | None = None
    region_name: str | None = None
    access_key_id: str | None = None
    secret_access_key: str | None = None
    addressing_style: str | None = None
    root_dir: str | None = None

    def __post_init__(self) -> None:
        provider = self.provider.strip().lower() if isinstance(self.provider, str) else ""
        if provider not in _PROVIDER_DEFAULTS:
            raise ObjectStoreError(
                "object-store provider must be one of: "
                + ", ".join(sorted(KNOWN_OBJECT_STORE_PROVIDERS))
            )
        object.__setattr__(self, "provider", provider)
        if self.backend_kind() == "s3":
            if not isinstance(self.bucket, str) or not self.bucket.strip():
                raise ObjectStoreError(f"{provider} object store requires a bucket")
            if self.bucket.strip() != self.bucket or "/" in self.bucket or ".." in self.bucket:
                raise ObjectStoreError("object-store bucket is malformed")
        if self.key_prefix is None or not isinstance(self.key_prefix, str):
            raise ObjectStoreError("object-store key_prefix is malformed")
        if any(part == ".." for part in self.key_prefix.split("/")):
            raise ObjectStoreError("object-store key_prefix must not contain '..'")
        if provider == "local":
            root = self.root_dir
            if not isinstance(root, str) or not root.strip():
                raise ObjectStoreError("local object store requires root_dir")

    def resolve_key(self, logical_key: str) -> str:
        key = _require_object_key(logical_key)
        prefix = self.key_prefix.strip("/")
        return f"{prefix}/{key}" if prefix else key

    def backend_kind(self) -> str:
        """Return the storage protocol implemented by the selected preset."""

        backend = _PROVIDER_DEFAULTS[self.provider]["backend"]
        assert backend is not None
        return backend

    def resolved_endpoint_url(self) -> str | None:
        """Return the configured endpoint after applying the provider preset."""

        if self.endpoint_url is not None:
            return self.endpoint_url
        return _PROVIDER_DEFAULTS[self.provider]["endpoint_url"]

    def resolved_region_name(self) -> str | None:
        """Return the configured region after applying the provider preset."""

        if self.region_name is not None:
            return self.region_name
        return _PROVIDER_DEFAULTS[self.provider]["region_name"]

    def resolved_addressing_style(self) -> str | None:
        """Return the configured addressing style after applying the preset."""

        if self.addressing_style is not None:
            return self.addressing_style
        return _PROVIDER_DEFAULTS[self.provider]["addressing_style"]

    @classmethod
    def from_env(
        cls,
        *,
        prefix: str = "CACHEON_OBJECT_STORE_",
        defaults: "ObjectStoreConfig | None" = None,
    ) -> "ObjectStoreConfig":
        """Build config from environment, overlaying optional CLI defaults."""

        def _env(name: str) -> str | None:
            value = os.environ.get(prefix + name)
            if value is None or value == "":
                return None
            return value

        provider = (
            _env("PROVIDER")
            or (defaults.provider if defaults is not None else "s3")
        ).lower()
        return cls(
            provider=provider,
            bucket=_env("BUCKET")
            or (defaults.bucket if defaults is not None else ""),
            key_prefix=(
                _env("KEY_PREFIX")
                if _env("KEY_PREFIX") is not None
                else (defaults.key_prefix if defaults is not None else "")
            ),
            endpoint_url=_env("ENDPOINT_URL")
            if _env("ENDPOINT_URL") is not None
            else (defaults.endpoint_url if defaults is not None else None),
            region_name=(
                _env("REGION")
                if _env("REGION") is not None
                else (defaults.region_name if defaults is not None else None)
            ),
            access_key_id=_env("ACCESS_KEY_ID")
            if _env("ACCESS_KEY_ID") is not None
            else (defaults.access_key_id if defaults is not None else None),
            secret_access_key=_env("SECRET_ACCESS_KEY")
            if _env("SECRET_ACCESS_KEY") is not None
            else (defaults.secret_access_key if defaults is not None else None),
            addressing_style=_env("ADDRESSING_STYLE")
            if _env("ADDRESSING_STYLE") is not None
            else (defaults.addressing_style if defaults is not None else None),
            root_dir=(
                _env("ROOT_DIR")
                if _env("ROOT_DIR") is not None
                else (defaults.root_dir if defaults is not None else None)
            ),
        )


def _require_object_key(key: str) -> str:
    if (
        not isinstance(key, str)
        or not key
        or key.strip() != key
        or key.startswith("/")
        or ".." in key.split("/")
        or len(key) > 1024
    ):
        raise ObjectStoreError("object-store key is malformed")
    return key


def _require_max_bytes(max_bytes: int | None) -> int | None:
    if max_bytes is not None and (
        type(max_bytes) is not int or max_bytes < 0
    ):
        raise ObjectStoreError("object-store max_bytes is malformed")
    return max_bytes


@dataclass
class MemoryObjectStore:
    """In-process store for tests; not durable."""

    _objects: dict[str, tuple[bytes, str]] | None = None
    _lock: threading.Lock | None = None

    def __post_init__(self) -> None:
        if self._objects is None:
            self._objects = {}
        if self._lock is None:
            self._lock = threading.Lock()

    def put_bytes(
        self,
        key: str,
        data: bytes,
        *,
        content_type: str = "application/octet-stream",
    ) -> None:
        key = _require_object_key(key)
        if not isinstance(data, (bytes, bytearray)):
            raise ObjectStoreError("object-store put requires bytes")
        assert self._lock is not None and self._objects is not None
        with self._lock:
            self._objects[key] = (bytes(data), content_type)

    def get_bytes(self, key: str, *, max_bytes: int | None = None) -> bytes:
        key = _require_object_key(key)
        max_bytes = _require_max_bytes(max_bytes)
        assert self._lock is not None and self._objects is not None
        with self._lock:
            row = self._objects.get(key)
        if row is None:
            raise ObjectStoreNotFoundError(f"object-store key is missing: {key}")
        if max_bytes is not None and len(row[0]) > max_bytes:
            raise ObjectStoreError("object-store object exceeds max_bytes")
        return row[0]


@dataclass(frozen=True)
class LocalDirectoryObjectStore:
    """Filesystem-backed store rooted at ``root_dir`` (provider=local)."""

    root_dir: Path

    def __post_init__(self) -> None:
        root = Path(self.root_dir)
        if root.exists() and not root.is_dir():
            raise ObjectStoreError("local object-store root_dir is not a directory")
        object.__setattr__(self, "root_dir", root)

    def _path_for(self, key: str) -> Path:
        key = _require_object_key(key)
        path = (self.root_dir / key).resolve()
        root = self.root_dir.resolve()
        if root not in path.parents and path != root:
            raise ObjectStoreError("local object-store key escapes root_dir")
        return path

    def put_bytes(
        self,
        key: str,
        data: bytes,
        *,
        content_type: str = "application/octet-stream",
    ) -> None:
        del content_type  # filesystem backend does not retain content-type
        if not isinstance(data, (bytes, bytearray)):
            raise ObjectStoreError("object-store put requires bytes")
        path = self._path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_bytes(bytes(data))
        os.replace(tmp, path)

    def get_bytes(self, key: str, *, max_bytes: int | None = None) -> bytes:
        path = self._path_for(key)
        max_bytes = _require_max_bytes(max_bytes)
        fd = -1
        try:
            flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
            flags |= getattr(os, "O_NOFOLLOW", 0)
            fd = os.open(path, flags)
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode):
                raise ObjectStoreError("local object-store value is not a regular file")
            if max_bytes is not None and info.st_size > max_bytes:
                raise ObjectStoreError("object-store object exceeds max_bytes")
            chunks: list[bytes] = []
            remaining = info.st_size
            while remaining:
                chunk = os.read(fd, min(4 << 20, remaining))
                if not chunk:
                    raise ObjectStoreError(
                        "local object-store value was truncated during read"
                    )
                chunks.append(chunk)
                remaining -= len(chunk)
            if os.read(fd, 1):
                raise ObjectStoreError(
                    "local object-store value changed during read"
                )
            return b"".join(chunks)
        except FileNotFoundError as exc:
            raise ObjectStoreNotFoundError(
                f"object-store key is missing: {key}"
            ) from exc
        except ObjectStoreError:
            raise
        except OSError as exc:
            raise ObjectStoreError(f"local object-store get failed: {exc}") from None
        finally:
            if fd >= 0:
                os.close(fd)


@dataclass(frozen=True)
class S3CompatibleObjectStore:
    """Any S3-compatible endpoint via boto3 (Hippius, AWS, MinIO, custom)."""

    client: object
    bucket: str

    def put_bytes(
        self,
        key: str,
        data: bytes,
        *,
        content_type: str = "application/octet-stream",
    ) -> None:
        key = _require_object_key(key)
        if not isinstance(data, (bytes, bytearray)):
            raise ObjectStoreError("object-store put requires bytes")
        try:
            self.client.put_object(  # type: ignore[attr-defined]
                Bucket=self.bucket,
                Key=key,
                Body=bytes(data),
                ContentType=content_type,
            )
        except Exception as exc:
            name = type(exc).__name__
            if name in {"EndpointConnectionError", "ConnectTimeoutError", "ReadTimeoutError"}:
                raise ObjectStoreRetryableError(
                    f"s3-compatible put failed transiently: {exc}"
                ) from None
            raise ObjectStoreError(f"s3-compatible put failed: {exc}") from None

    def get_bytes(self, key: str, *, max_bytes: int | None = None) -> bytes:
        key = _require_object_key(key)
        max_bytes = _require_max_bytes(max_bytes)
        body_object = None
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=key)  # type: ignore[attr-defined]
            body_object = response["Body"]
            content_length = response.get("ContentLength")
            if content_length is not None and (
                type(content_length) is not int or content_length < 0
            ):
                raise ObjectStoreError(
                    "s3-compatible get returned malformed ContentLength"
                )
            if (
                max_bytes is not None
                and content_length is not None
                and content_length > max_bytes
            ):
                raise ObjectStoreError("object-store object exceeds max_bytes")
            body = (
                body_object.read()
                if max_bytes is None
                else body_object.read(max_bytes + 1)
            )
        except ObjectStoreError:
            raise
        except Exception as exc:
            name = type(exc).__name__
            code = getattr(exc, "response", {}).get("Error", {}).get("Code", "")
            if code in {"NoSuchKey", "404", "NotFound"} or name == "NoSuchKey":
                raise ObjectStoreNotFoundError(
                    f"object-store key is missing: {key}"
                ) from None
            if name in {"EndpointConnectionError", "ConnectTimeoutError", "ReadTimeoutError"}:
                raise ObjectStoreRetryableError(
                    f"s3-compatible get failed transiently: {exc}"
                ) from None
            raise ObjectStoreError(f"s3-compatible get failed: {exc}") from None
        finally:
            close = getattr(body_object, "close", None)
            if callable(close):
                close()
        if not isinstance(body, (bytes, bytearray)):
            raise ObjectStoreError("s3-compatible get returned non-bytes body")
        if max_bytes is not None and len(body) > max_bytes:
            raise ObjectStoreError("object-store object exceeds max_bytes")
        return bytes(body)


def _boto3_client(config: ObjectStoreConfig):
    try:
        import boto3
        from botocore.config import Config as BotoConfig
    except ImportError as exc:
        raise ObjectStoreError(
            "s3-compatible object stores require boto3; "
            'install with pip install -e ".[object-store]"'
        ) from exc
    endpoint = config.resolved_endpoint_url()
    region = config.resolved_region_name()
    addressing = config.resolved_addressing_style()
    access_key = config.access_key_id or os.environ.get("AWS_ACCESS_KEY_ID")
    secret_key = config.secret_access_key or os.environ.get("AWS_SECRET_ACCESS_KEY")
    if not access_key or not secret_key:
        raise ObjectStoreError(
            f"{config.provider} object store requires access_key_id and secret_access_key"
        )
    boto_kwargs: dict[str, object] = {
        "service_name": "s3",
        "aws_access_key_id": access_key,
        "aws_secret_access_key": secret_key,
        "region_name": region or "us-east-1",
    }
    if endpoint:
        boto_kwargs["endpoint_url"] = endpoint
    if addressing and addressing != "auto":
        boto_kwargs["config"] = BotoConfig(
            signature_version="s3v4",
            s3={"addressing_style": addressing},
        )
    else:
        boto_kwargs["config"] = BotoConfig(signature_version="s3v4")
    return boto3.client(**boto_kwargs)


def open_object_store(config: ObjectStoreConfig) -> ObjectStore:
    """Construct the backend selected by ``config.provider``."""

    if type(config) is not ObjectStoreConfig:
        raise ObjectStoreError("object-store config is untyped")
    backend = config.backend_kind()
    if backend == "memory":
        return MemoryObjectStore()
    if backend == "local":
        assert config.root_dir is not None
        return LocalDirectoryObjectStore(Path(config.root_dir))
    if backend == "s3":
        return S3CompatibleObjectStore(_boto3_client(config), config.bucket)
    raise ObjectStoreError(f"unsupported object-store backend: {backend}")


def prefixed_store(store: ObjectStore, prefix: str) -> ObjectStore:
    """Return a view that prefixes every key (used with ObjectStoreConfig.key_prefix)."""

    prefix = prefix.strip("/")
    if not prefix:
        return store
    if ".." in prefix.split("/"):
        raise ObjectStoreError("object-store key prefix is malformed")

    class _Prefixed:
        def put_bytes(
            self,
            key: str,
            data: bytes,
            *,
            content_type: str = "application/octet-stream",
        ) -> None:
            store.put_bytes(
                f"{prefix}/{_require_object_key(key)}",
                data,
                content_type=content_type,
            )

        def get_bytes(
            self,
            key: str,
            *,
            max_bytes: int | None = None,
        ) -> bytes:
            return store.get_bytes(
                f"{prefix}/{_require_object_key(key)}",
                max_bytes=max_bytes,
            )

    return _Prefixed()


def open_configured_object_store(config: ObjectStoreConfig) -> ObjectStore:
    """Open the provider backend and apply ``key_prefix`` if set."""

    return prefixed_store(open_object_store(config), config.key_prefix)


__all__ = [
    "KNOWN_OBJECT_STORE_PROVIDERS",
    "LocalDirectoryObjectStore",
    "MemoryObjectStore",
    "ObjectStore",
    "ObjectStoreConfig",
    "ObjectStoreError",
    "ObjectStoreNotFoundError",
    "ObjectStoreRetryableError",
    "S3CompatibleObjectStore",
    "open_configured_object_store",
    "open_object_store",
    "prefixed_store",
]
