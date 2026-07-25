"""Tests for provider-swappable object storage and remote weight-offer publish."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import optima.cli as cli
from optima.chain.weight_share import (
    CurrentWeightOffer,
    load_current_weight_offer_from_store,
    object_store_offer_loader,
    publish_current_weight_offer,
    read_current_weight_offer,
)
from optima.chain.weights import WeightProjection
from optima.object_store import (
    LocalDirectoryObjectStore,
    MemoryObjectStore,
    ObjectStoreConfig,
    ObjectStoreError,
    ObjectStoreNotFoundError,
    open_configured_object_store,
    open_object_store,
    prefixed_store,
)
from optima.stack_identity import canonical_digest, sha256_hex


def _d(label: str) -> str:
    return sha256_hex(label.encode())


def _projection(
    *,
    hotkey: str = "authority",
    block: int = 10,
) -> WeightProjection:
    scope = _d("scope")
    metagraph_digest = canonical_digest(
        "optima.economics.metagraph-membership",
        {
            "block": block,
            "block_hash": "0x" + f"{block:064x}",
            "chain_scope_digest": scope,
            "members": [
                {"hotkey": "authority", "uid": 0},
                {"hotkey": "miner", "uid": 1},
            ],
        },
    )
    return WeightProjection(
        scope,
        307,
        hotkey,
        _d("policy"),
        _d("settlement"),
        _d("evaluation"),
        metagraph_digest,
        (_d("arena"),),
        1,
        block,
        1,
        (_d("evidence"),),
        (("miner", 1_000_000),),
    )


def test_provider_presets_are_swappable_by_name() -> None:
    for provider in ("hippius", "s3", "minio"):
        cfg = ObjectStoreConfig(provider=provider, bucket="weights")
        assert cfg.provider == provider
    local = ObjectStoreConfig(provider="local", root_dir="/tmp/optima-store")
    assert local.provider == "local"
    with pytest.raises(ObjectStoreError, match="provider"):
        ObjectStoreConfig(provider="gpl-forbidden", bucket="x")


def test_memory_and_local_backends_roundtrip(tmp_path: Path) -> None:
    mem = MemoryObjectStore()
    mem.put_bytes("a.json", b'{"ok":true}\n', content_type="application/json")
    assert mem.get_bytes("a.json") == b'{"ok":true}\n'

    local = LocalDirectoryObjectStore(tmp_path / "root")
    local.put_bytes("dir/b.json", b"hello")
    assert local.get_bytes("dir/b.json") == b"hello"
    with pytest.raises(ObjectStoreError, match="missing"):
        local.get_bytes("missing.json")


def test_prefixed_store_and_open_configured(tmp_path: Path) -> None:
    cfg = ObjectStoreConfig(
        provider="local",
        root_dir=str(tmp_path / "root"),
        key_prefix="netuid/307",
    )
    store = open_configured_object_store(cfg)
    store.put_bytes("current_weights.json", b"payload")
    raw = (tmp_path / "root" / "netuid" / "307" / "current_weights.json").read_bytes()
    assert raw == b"payload"
    assert store.get_bytes("current_weights.json") == b"payload"


def test_publish_writes_local_and_remote_sync(tmp_path: Path) -> None:
    projection = _projection()
    local_path = tmp_path / "offer.json"
    remote = MemoryObjectStore()
    publish_current_weight_offer(
        projection,
        local_path=local_path,
        remote_store=remote,
        remote_key="current_weights.json",
        async_remote=False,
    )
    assert read_current_weight_offer(local_path).projection.digest == projection.digest
    assert load_current_weight_offer_from_store(remote).projection.digest == projection.digest


def test_publish_async_remote_does_not_block_local(tmp_path: Path) -> None:
    projection = _projection()
    local_path = tmp_path / "offer.json"
    started = threading.Event()
    released = threading.Event()

    class _SlowStore:
        def put_bytes(self, key, data, *, content_type="application/octet-stream"):
            started.set()
            assert released.wait(timeout=2)
            MemoryObjectStore().put_bytes(key, data, content_type=content_type)

        def get_bytes(self, key):
            raise ObjectStoreNotFoundError(f"missing: {key}")

    publish_current_weight_offer(
        projection,
        local_path=local_path,
        remote_store=_SlowStore(),
        async_remote=True,
    )
    # Local durability completes without waiting on the remote upload.
    assert read_current_weight_offer(local_path).projection.digest == projection.digest
    assert started.wait(timeout=2)
    released.set()
    time.sleep(0.05)


def test_async_remote_publish_cannot_regress_current_offer(
    tmp_path: Path,
) -> None:
    first_started = threading.Event()
    release_first = threading.Event()
    first_done = threading.Event()
    second_done = threading.Event()

    class _Store:
        value = b""

        def get_bytes(self, key):
            if not self.value:
                raise ObjectStoreNotFoundError(f"missing: {key}")
            return self.value

        def put_bytes(
            self,
            _key,
            data,
            *,
            content_type="application/octet-stream",
        ):
            del content_type
            block = CurrentWeightOffer.from_bytes(
                data
            ).projection.effective_block
            if block == 10:
                first_started.set()
                assert release_first.wait(timeout=3)
            self.value = bytes(data)
            (first_done if block == 10 else second_done).set()

    store = _Store()
    publish_current_weight_offer(
        CurrentWeightOffer.from_legacy_projection(_projection()),
        local_path=tmp_path / "offer-10.json",
        remote_store=store,
        async_remote=True,
    )
    assert first_started.wait(timeout=3)
    publish_current_weight_offer(
        CurrentWeightOffer.from_legacy_projection(
            _projection(block=11)
        ),
        local_path=tmp_path / "offer-11.json",
        remote_store=store,
        async_remote=True,
    )
    assert not second_done.wait(timeout=0.1)
    release_first.set()
    assert first_done.wait(timeout=3)
    assert second_done.wait(timeout=3)
    assert (
        CurrentWeightOffer.from_bytes(
            store.value
        ).projection.effective_block
        == 11
    )


def test_object_store_offer_loader(tmp_path: Path) -> None:
    projection = _projection()
    store = open_object_store(
        ObjectStoreConfig(provider="local", root_dir=str(tmp_path))
    )
    publish_current_weight_offer(
        projection,
        local_path=tmp_path / "local.json",
        remote_store=store,
        async_remote=False,
    )
    loaded = object_store_offer_loader(store)()
    assert loaded.projection.digest == projection.digest


def test_hippius_to_s3_swap_is_config_only() -> None:
    hippius = ObjectStoreConfig(provider="hippius", bucket="optima-weights")
    aws = ObjectStoreConfig(
        provider="s3",
        bucket="optima-weights",
        region_name="us-west-2",
        endpoint_url=None,
    )
    minio = ObjectStoreConfig(
        provider="minio",
        bucket="optima-weights",
        endpoint_url="http://minio.internal:9000",
    )
    assert hippius.provider != aws.provider != minio.provider
    # Same logical key resolution regardless of provider.
    assert hippius.resolve_key("current_weights.json") == "current_weights.json"
    prefixed = ObjectStoreConfig(
        provider="hippius", bucket="optima-weights", key_prefix="prod/sn307"
    )
    assert prefixed.resolve_key("current_weights.json") == "prod/sn307/current_weights.json"


def test_environment_only_object_store_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_root = tmp_path / "env-store"
    monkeypatch.setenv("OPTIMA_OBJECT_STORE_PROVIDER", "local")
    monkeypatch.setenv("OPTIMA_OBJECT_STORE_ROOT_DIR", str(env_root))
    store, key = cli._object_store_from_args(
        SimpleNamespace(object_store_provider="")
    )
    assert isinstance(store, LocalDirectoryObjectStore)
    assert store.root_dir == env_root
    assert key == "current_weights.json"

    explicit_root = tmp_path / "explicit-store"
    store, _key = cli._object_store_from_args(
        SimpleNamespace(
            object_store_provider="local",
            object_store_root=str(explicit_root),
        )
    )
    assert isinstance(store, LocalDirectoryObjectStore)
    assert store.root_dir == explicit_root


def test_open_s3_requires_boto3_message() -> None:
    cfg = ObjectStoreConfig(
        provider="hippius",
        bucket="optima-weights",
        access_key_id="hip_test",
        secret_access_key="secret",
    )
    try:
        import boto3  # noqa: F401
    except ImportError:
        with pytest.raises(ObjectStoreError, match="boto3"):
            open_object_store(cfg)
    else:
        # boto3 present in this environment: construction should at least reach client create.
        store = open_object_store(cfg)
        assert store.bucket == "optima-weights"
