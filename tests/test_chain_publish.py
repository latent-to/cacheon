"""Miner-owned public object-store publication for proposal bundles."""

from __future__ import annotations

import io
from pathlib import Path
from types import SimpleNamespace

import pytest

import optima.cli as cli
from optima.chain.fetch import FetchError, package_bundle
from optima.chain.publish import (
    BundlePublishError,
    DEFAULT_BUNDLE_KEY_PREFIX,
    S3PublicBundlePublisher,
    public_object_url,
)
from optima.object_store import ObjectStoreConfig


class _S3Error(Exception):
    def __init__(self, code: str):
        super().__init__(code)
        self.response = {"Error": {"Code": code}}


class _FakeS3:
    def __init__(self, *, bucket_exists: bool = True):
        self.bucket_exists = bucket_exists
        self.objects: dict[str, bytes] = {}
        self.acls: dict[str, str] = {}
        self.puts: list[dict[str, object]] = []
        self.acl_updates: list[str] = []
        self.created = 0

    def head_bucket(self, *, Bucket):
        assert Bucket == "miner-bucket"
        if not self.bucket_exists:
            raise _S3Error("NoSuchBucket")
        return {}

    def create_bucket(self, **kwargs):
        assert kwargs == {"Bucket": "miner-bucket"}
        self.bucket_exists = True
        self.created += 1
        return {}

    def get_object(self, *, Bucket, Key):
        assert Bucket == "miner-bucket"
        if Key not in self.objects:
            raise _S3Error("NoSuchKey")
        return {"Body": io.BytesIO(self.objects[Key])}

    def put_object(self, **kwargs):
        assert kwargs["Bucket"] == "miner-bucket"
        key = str(kwargs["Key"])
        self.objects[key] = bytes(kwargs["Body"])
        self.acls[key] = str(kwargs.get("ACL", ""))
        self.puts.append(kwargs)
        return {"ETag": '"test"'}

    def put_object_acl(self, *, Bucket, Key, ACL):
        assert Bucket == "miner-bucket"
        assert Key in self.objects
        self.acls[Key] = ACL
        self.acl_updates.append(Key)
        return {}


def _bundle(root: Path) -> Path:
    bundle = root / "bundle"
    (bundle / "kernels").mkdir(parents=True)
    (bundle / "manifest.toml").write_text(
        'bundle_id = "publish-test"\n'
        'abi_version = "optima-op-abi-v0"\n\n'
        '[[ops]]\n'
        'slot = "activation.silu_and_mul"\n'
        'source = "kernels/k.py"\n'
        'entry = "silu_and_mul"\n'
        'dtypes = ["float32"]\n'
    )
    (bundle / "kernels" / "k.py").write_text(
        "def silu_and_mul(x, out):\n    out.copy_(x)\n"
    )
    return bundle


def _publisher(
    client: _FakeS3,
    fetched: list[tuple[str, str, float]],
) -> S3PublicBundlePublisher:
    config = ObjectStoreConfig(
        provider="hippius",
        bucket="miner-bucket",
        key_prefix=DEFAULT_BUNDLE_KEY_PREFIX,
    )

    def public_fetch(url, expected, _root, *, transfer_timeout_s):
        fetched.append((url, expected, transfer_timeout_s))
        return Path(_root) / expected

    return S3PublicBundlePublisher(
        config,
        client,
        public_fetch=public_fetch,
    )


def test_publish_new_bundle_is_content_addressed_public_and_reopened(tmp_path):
    archive, content_hash = package_bundle(
        _bundle(tmp_path), tmp_path / "bundle.tar.gz"
    )
    client = _FakeS3()
    fetched: list[tuple[str, str, float]] = []

    result = _publisher(client, fetched).publish_archive(
        archive, content_hash, verify_timeout_s=12
    )

    expected_key = f"{DEFAULT_BUNDLE_KEY_PREFIX}/{content_hash}.tar.gz"
    expected_url = f"https://s3.hippius.com/miner-bucket/{expected_key}"
    assert result.object_key == expected_key
    assert result.url == expected_url
    assert result.reused is False
    assert result.stored_archive_bytes == len(client.objects[expected_key])
    assert client.acls[expected_key] == "public-read"
    assert client.puts[0]["ContentType"] == "application/gzip"
    assert client.puts[0]["Metadata"]["optima-content-sha256"] == content_hash
    assert fetched == [(expected_url, content_hash, 12.0)]


def test_existing_equivalent_archive_is_reused_without_overwrite(tmp_path):
    archive, content_hash = package_bundle(
        _bundle(tmp_path), tmp_path / "bundle.tar.gz"
    )
    data = bytearray(archive.read_bytes())
    # Gzip header bytes 4..7 are the non-identity mtime. Change them without
    # changing the decompressed tar or its checksum.
    data[4:8] = b"\x01\x00\x00\x00"
    key = f"{DEFAULT_BUNDLE_KEY_PREFIX}/{content_hash}.tar.gz"
    client = _FakeS3()
    client.objects[key] = bytes(data)
    fetched: list[tuple[str, str, float]] = []

    result = _publisher(client, fetched).publish_archive(archive, content_hash)

    assert result.reused is True
    assert client.puts == []
    assert client.acl_updates == []


def test_existing_private_archive_is_made_public_then_reverified(tmp_path):
    archive, content_hash = package_bundle(
        _bundle(tmp_path), tmp_path / "bundle.tar.gz"
    )
    key = f"{DEFAULT_BUNDLE_KEY_PREFIX}/{content_hash}.tar.gz"
    client = _FakeS3()
    client.objects[key] = archive.read_bytes()
    calls = 0

    def fetch_after_acl(url, expected, _root, *, transfer_timeout_s):
        nonlocal calls
        del url, expected, _root, transfer_timeout_s
        calls += 1
        if calls == 1:
            raise FetchError("anonymous GET was denied")
        return Path("/verified")

    config = ObjectStoreConfig(
        provider="hippius",
        bucket="miner-bucket",
        key_prefix=DEFAULT_BUNDLE_KEY_PREFIX,
    )
    result = S3PublicBundlePublisher(
        config,
        client,
        public_fetch=fetch_after_acl,
    ).publish_archive(archive, content_hash)

    assert result.reused is True
    assert calls == 2
    assert client.acl_updates == [key]
    assert client.acls[key] == "public-read"


def test_existing_wrong_archive_fails_before_public_acl_change(tmp_path):
    archive, content_hash = package_bundle(
        _bundle(tmp_path), tmp_path / "bundle.tar.gz"
    )
    key = f"{DEFAULT_BUNDLE_KEY_PREFIX}/{content_hash}.tar.gz"
    client = _FakeS3()
    client.objects[key] = b"not a gzip archive"

    with pytest.raises(BundlePublishError, match="existing content-addressed"):
        _publisher(client, []).publish_archive(archive, content_hash)

    assert client.puts == []
    assert client.acl_updates == []


def test_bucket_creation_is_explicit(tmp_path):
    archive, content_hash = package_bundle(
        _bundle(tmp_path), tmp_path / "bundle.tar.gz"
    )
    missing = _FakeS3(bucket_exists=False)
    with pytest.raises(BundlePublishError, match="create it first"):
        _publisher(missing, []).publish_archive(archive, content_hash)
    assert missing.created == 0

    created = _FakeS3(bucket_exists=False)
    result = _publisher(created, []).publish_archive(
        archive, content_hash, create_bucket=True
    )
    assert created.created == 1
    assert result.reused is False


def test_public_url_uses_provider_style_and_rejects_non_https():
    key = "prefix/a.tar.gz"
    hippius = ObjectStoreConfig(provider="hippius", bucket="miner-bucket")
    assert public_object_url(hippius, key) == (
        "https://s3.hippius.com/miner-bucket/prefix/a.tar.gz"
    )
    aws = ObjectStoreConfig(
        provider="s3",
        bucket="miner-bucket",
        region_name="us-west-2",
    )
    assert public_object_url(aws, key) == (
        "https://miner-bucket.s3.us-west-2.amazonaws.com/prefix/a.tar.gz"
    )
    assert public_object_url(
        hippius,
        key,
        public_base_url="https://downloads.example/miner-bucket",
    ) == "https://downloads.example/miner-bucket/prefix/a.tar.gz"

    insecure = ObjectStoreConfig(
        provider="minio",
        bucket="miner-bucket",
    )
    with pytest.raises(BundlePublishError, match="HTTPS"):
        public_object_url(insecure, key)


def test_bundle_store_cli_defaults_to_generic_s3_environment(monkeypatch):
    monkeypatch.setenv("OPTIMA_OBJECT_STORE_BUCKET", "miner-bucket")
    monkeypatch.setenv("OPTIMA_OBJECT_STORE_ACCESS_KEY_ID", "hip_test")
    monkeypatch.setenv("OPTIMA_OBJECT_STORE_SECRET_ACCESS_KEY", "secret")

    config = cli._bundle_store_config_from_args(SimpleNamespace())

    assert config.provider == "s3"
    assert config.bucket == "miner-bucket"
    assert config.key_prefix == DEFAULT_BUNDLE_KEY_PREFIX
    assert config.access_key_id == "hip_test"
    assert config.secret_access_key == "secret"


def test_bundle_store_cli_endpoint_is_provider_agnostic(monkeypatch):
    monkeypatch.setenv("OPTIMA_OBJECT_STORE_BUCKET", "miner-bucket")
    monkeypatch.setenv(
        "OPTIMA_OBJECT_STORE_ENDPOINT_URL", "https://objects.example"
    )

    config = cli._bundle_store_config_from_args(SimpleNamespace())

    assert config.provider == "s3"
    assert config.resolved_endpoint_url() == "https://objects.example"
    assert config.resolved_addressing_style() == "path"
    assert public_object_url(config, "prefix/a.tar.gz") == (
        "https://objects.example/miner-bucket/prefix/a.tar.gz"
    )


def test_bundle_store_cli_hippius_preset_is_optional(monkeypatch):
    monkeypatch.setenv("OPTIMA_OBJECT_STORE_PROVIDER", "hippius")
    monkeypatch.setenv("OPTIMA_OBJECT_STORE_BUCKET", "miner-bucket")

    config = cli._bundle_store_config_from_args(SimpleNamespace())

    assert config.provider == "hippius"
    assert config.resolved_endpoint_url() == "https://s3.hippius.com"
    assert config.resolved_region_name() == "decentralized"
    assert config.resolved_addressing_style() == "path"
