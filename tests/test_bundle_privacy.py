"""Encrypted public hosting still feeds the ordinary validator's identity checks."""

import io
import json
from urllib.parse import urlparse

import pytest
from nacl.public import PrivateKey, SealedBox

from cacheon import cli
from cacheon.bundle_hash import content_hash
from cacheon.chain import fetch, publish
from cacheon.chain.bundle_privacy import MAGIC, KEY_ENV, encrypt_archive, recipient_key
from tests.test_chain_publish import _bundle, _publisher, _FakeS3


@pytest.fixture
def key(tmp_path, monkeypatch):
    key = PrivateKey.generate()
    path = tmp_path / "validator.key"
    path.write_text(bytes(key).hex());path.chmod(0o600)
    monkeypatch.setenv(KEY_ENV, str(path))
    return key


def test_publish_hides_plaintext_then_ordinary_fetch_recovers_exact_identity(tmp_path, monkeypatch, key):
    bundle = _bundle(tmp_path)
    archive, digest = fetch.package_bundle(bundle, tmp_path / "bundle.tar.gz")
    plaintext = archive.read_bytes()
    client = _FakeS3()

    def download(url, destination, max_bytes, *, deadline):
        payload = client.objects[urlparse(url).path.split('/miner-bucket/', 1)[1]]
        assert len(payload) <= max_bytes
        destination.write_bytes(payload);destination.chmod(0o600)

    monkeypatch.setattr(publish, '_download_https', download)
    monkeypatch.setattr(fetch, '_download_https', download)
    result = _publisher(client, []).publish_archive(archive, digest, encrypt_for=bytes(key.public_key).hex())
    wire = client.objects[result.object_key]
    assert wire.startswith(MAGIC) and not wire.startswith(b'\x1f\x8b')
    assert plaintext not in wire and archive.read_bytes() == wire
    assert SealedBox(key).decrypt(wire[len(MAGIC)+32:]) == plaintext
    assert result.object_key.endswith('.sealed')
    restored = fetch.fetch_bundle(result.url, digest, tmp_path / 'private')
    assert content_hash(restored) == digest
    assert (restored / 'kernels/k.py').read_bytes() == (bundle / 'kernels/k.py').read_bytes()
    client.objects[result.object_key] = wire[:-1] + bytes([wire[-1] ^ 1])
    with pytest.raises(fetch.FetchError, match='authentication'):
        fetch.fetch_bundle(result.url, digest, tmp_path / 'tampered')
    client.objects[result.object_key] = wire[:len(MAGIC)+32]
    with pytest.raises(fetch.FetchError, match='authentication'):
        fetch.fetch_bundle(result.url, digest, tmp_path / 'truncated')


def test_missing_key_retries_wrong_recipient_and_wrong_hash_fail(tmp_path, monkeypatch, key):
    archive, digest = fetch.package_bundle(_bundle(tmp_path), tmp_path / 'bundle.tar.gz')
    plain = archive.read_bytes()
    archive.write_bytes(encrypt_archive(plain, bytes(key.public_key).hex()))
    monkeypatch.delenv(KEY_ENV)
    with pytest.raises(fetch.FetchTransientError, match='not provisioned'):
        fetch.fetch_bundle_from_local_file_for_testing(archive.as_uri(), digest, tmp_path / 'missing')
    monkeypatch.setenv(KEY_ENV, str(tmp_path / 'validator.key'))
    with pytest.raises(fetch.FetchError, match='content hash mismatch'):
        fetch.fetch_bundle_from_local_file_for_testing(archive.as_uri(), 'b'*64, tmp_path / 'wronghash')
    archive.write_bytes(encrypt_archive(plain, bytes(PrivateKey.generate().public_key).hex()))
    with pytest.raises(fetch.FetchError, match='another validator'):
        fetch.fetch_bundle_from_local_file_for_testing(archive.as_uri(), digest, tmp_path / 'wrongkey')


def test_packaging_automatically_uses_the_published_key(tmp_path, monkeypatch, key, capsys):
    calls = []
    def key_response(request, **kwargs):
        calls.append(request)
        return io.BytesIO(json.dumps({'public_key': bytes(key.public_key).hex()}).encode())
    monkeypatch.delenv('CACHEON_BUNDLE_PUBLIC_KEY', raising=False)
    monkeypatch.setattr('urllib.request.urlopen', key_response)
    out = tmp_path / 'encrypted.tar.gz'
    bundle = _bundle(tmp_path)
    assert cli.main(['chain-package', str(bundle), '--out', str(out)]) == 0
    assert len(calls) == 1
    restored = fetch.fetch_bundle_from_local_file_for_testing(out.as_uri(), content_hash(bundle), tmp_path / 'private')
    assert restored.is_dir() and out.read_bytes().startswith(MAGIC)
    assert content_hash(bundle) in capsys.readouterr().out
    monkeypatch.setenv('CACHEON_BUNDLE_PUBLIC_KEY', bytes(key.public_key).hex())
    assert recipient_key() == bytes(key.public_key).hex() and len(calls) == 1
