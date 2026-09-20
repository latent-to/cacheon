"""Recipient-encrypted transport; decrypted bundles retain their original identity."""

import os
import stat
from pathlib import Path

MAGIC = b"cacheon-sealed-v1\0"
KEY_ENV = "CACHEON_BUNDLE_DECRYPTION_KEY"


def recipient_key(value: str | None = None) -> str:
    """Resolve the miner's recipient before creating or publishing an archive."""
    import json
    import urllib.request
    from nacl.public import PublicKey

    value = value or os.environ.get("CACHEON_BUNDLE_PUBLIC_KEY")
    if not value:
        url = os.environ.get("CACHEON_BUNDLE_KEY_URL", "https://dash.cacheon.ai/api/bundle-encryption-key")
        if not url.startswith("https://"):
            raise ValueError("validator key endpoint must use HTTPS")
        request = urllib.request.Request(url, headers={"User-Agent": "cacheon-miner/1", "Accept": "application/json"})
        with urllib.request.urlopen(request, timeout=15) as response:
            value = json.loads(response.read(4096))["public_key"]
    return bytes(PublicKey(bytes.fromhex(value))).hex()


def validator_key():
    """Read the operator's private key; missing provisioning is retryable infrastructure."""
    from nacl.public import PrivateKey
    from cacheon.chain.fetch import FetchTransientError

    try:
        path = Path(os.environ[KEY_ENV])
        with path.open("rb") as stream:
            info = os.fstat(stream.fileno())
            if (path.is_symlink() or not stat.S_ISREG(info.st_mode)
                    or info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) != 0o600):
                raise ValueError("key must be an owner-owned mode-0600 regular file")
            return PrivateKey(bytes.fromhex(stream.read(65).decode().strip()))
    except (OSError, KeyError, ValueError) as exc:
        raise FetchTransientError("bundle decryption key is not provisioned correctly") from exc


def encrypt_archive(data: bytes, public_key: str) -> bytes:
    """Seal an archive to one validator without changing the committed bundle hash."""
    from nacl.public import PublicKey, SealedBox
    from cacheon.chain.fetch import MAX_ARCHIVE_BYTES

    key = PublicKey(bytes.fromhex(public_key))
    wire = MAGIC + bytes(key) + SealedBox(key).encrypt(data)
    if len(wire) > MAX_ARCHIVE_BYTES:
        raise ValueError("encrypted archive exceeds the bundle transfer limit")
    return wire


def decrypt_archive(path: Path) -> None:
    """Decrypt before existing archive checks; old plaintext submissions remain readable."""
    with path.open("rb") as stream:
        if stream.read(len(MAGIC)) != MAGIC:
            return
    from nacl.exceptions import CryptoError
    from nacl.public import SealedBox
    from cacheon.chain.fetch import FetchError

    key = validator_key()
    wire = path.read_bytes()[len(MAGIC):]
    if wire[:32] != bytes(key.public_key):
        raise FetchError("encrypted bundle names another validator key")
    try:
        plaintext = SealedBox(key).decrypt(wire[32:])
    except (CryptoError, TypeError) as exc:
        raise FetchError("encrypted bundle authentication failed") from exc
    path.write_bytes(plaintext)
