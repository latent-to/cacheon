"""The on-chain submission payload: a tiny JSON binding a bundle hash to a fetch URL.

This is the ONLY thing a miner puts on chain — `{"v": 1, "h": <content_hash>,
"u": <url>}` — committed via the chain's timelock commit-reveal. Everything the
validator later trusts about the bundle flows from ``h``: the fetched artifact is
extracted and re-hashed with ``optima.bundle_hash.content_hash``, and a mismatch is
a rejected submission. Decoding is fail-quiet (returns ``None``), because payloads
arrive from arbitrary registered hotkeys: garbage on chain must never crash the
validator loop.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from urllib.parse import urlparse

logger = logging.getLogger("optima.chain.payload")

PAYLOAD_VERSION = 1
# Chain-side cap for TimelockEncrypted commitments (subtensor commitments pallet).
MAX_PAYLOAD_BYTES = 1024
# Production validators follow only authenticated HTTPS.  Local files are useful in
# hermetic tests, but must cross an explicitly named test-only API so an on-chain
# payload can never turn the validator into a local-file or plaintext-HTTP client.
ALLOWED_URL_SCHEMES = ("https",)
_TEST_ONLY_URL_SCHEMES = ("https", "file")

_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_SCHEME_RE = re.compile(r"^([a-z][a-z0-9+.-]*)://")


class PayloadError(ValueError):
    """A payload WE are building is invalid (miner-side, loud)."""


@dataclass(frozen=True)
class SubmissionRef:
    """A decoded, validated on-chain submission reference."""
    hotkey: str
    content_hash: str
    url: str
    block: int  # reveal block — the consensus anti-copy priority timestamp


def _url_allowed(url: object, *, schemes: tuple[str, ...]) -> bool:
    if (
        not isinstance(url, str)
        or not url
        or not url.isascii()
        or any(ord(char) <= 32 or ord(char) == 127 for char in url)
    ):
        return False
    scheme = _SCHEME_RE.match(url)
    if not scheme or scheme.group(1) not in schemes:
        return False
    try:
        parsed = urlparse(url)
        port = parsed.port
    except ValueError:
        return False
    if parsed.username is not None or parsed.password is not None or parsed.fragment:
        return False
    if parsed.scheme == "https":
        return bool(parsed.hostname) and (port is None or 1 <= port <= 65_535)
    if parsed.scheme == "file":
        return parsed.netloc in ("", "localhost") and bool(parsed.path) and not parsed.query
    return False


def _encode_payload(content_hash: str, url: str, *, schemes: tuple[str, ...]) -> str:
    """Miner side: build the commitment JSON. Raises PayloadError on anything that
    would be rejected by the validator or the chain — fail loud before signing."""
    if not _HASH_RE.match(content_hash or ""):
        raise PayloadError(f"content_hash must be 64 lowercase hex chars, got {content_hash!r}")
    if not _url_allowed(url, schemes=schemes):
        raise PayloadError(
            f"url must be canonical and use one of {schemes}, got {url!r}"
        )
    data = json.dumps({"v": PAYLOAD_VERSION, "h": content_hash, "u": url},
                      separators=(",", ":"))
    n = len(data.encode("utf-8"))
    if n > MAX_PAYLOAD_BYTES:
        raise PayloadError(f"payload is {n} bytes; chain cap is {MAX_PAYLOAD_BYTES}")
    return data


def encode_payload(content_hash: str, url: str) -> str:
    """Build a production submission payload; only canonical HTTPS is accepted."""

    return _encode_payload(content_hash, url, schemes=ALLOWED_URL_SCHEMES)


def encode_payload_for_testing(content_hash: str, url: str) -> str:
    """Hermetic-test helper which additionally permits ``file://`` archives.

    Production CLI/chain paths deliberately never call this function.
    """

    return _encode_payload(content_hash, url, schemes=_TEST_ONLY_URL_SCHEMES)


def _decode_payload(
    hotkey: str,
    block: int,
    data: str,
    *,
    schemes: tuple[str, ...],
) -> "SubmissionRef | None":
    """Validator side: parse an untrusted on-chain payload. Returns None (and logs)
    on anything malformed — never raises on chain-sourced data."""
    if not isinstance(data, str) or len(data.encode("utf-8", "replace")) > MAX_PAYLOAD_BYTES:
        logger.warning("payload from %s: oversized or non-string; ignored", hotkey)
        return None

    def unique_object(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate JSON key {key!r}")
            value[key] = item
        return value

    try:
        obj = json.loads(
            data,
            object_pairs_hook=unique_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"invalid JSON constant {value}")
            ),
        )
    except (
        json.JSONDecodeError,
        UnicodeDecodeError,
        RecursionError,
        TypeError,
        ValueError,
    ):
        logger.warning("payload from %s: not JSON; ignored", hotkey)
        return None
    if (
        not isinstance(obj, dict)
        or set(obj) != {"v", "h", "u"}
        or type(obj.get("v")) is not int
        or obj.get("v") != PAYLOAD_VERSION
    ):
        logger.warning("payload from %s: missing/unknown version; ignored", hotkey)
        return None
    content_hash, url = obj.get("h"), obj.get("u")
    if not isinstance(content_hash, str) or not _HASH_RE.match(content_hash):
        logger.warning("payload from %s: bad content hash; ignored", hotkey)
        return None
    if not _url_allowed(url, schemes=schemes):
        logger.warning("payload from %s: bad/disallowed url; ignored", hotkey)
        return None
    if type(block) is not int or block < 0:
        logger.warning("payload from %s: invalid reveal block; ignored", hotkey)
        return None
    if (
        not isinstance(hotkey, str)
        or not hotkey
        or len(hotkey) > 256
        or hotkey.strip() != hotkey
        or any(char in hotkey for char in "\x00\r\n")
    ):
        logger.warning("payload contains an invalid hotkey identity; ignored")
        return None
    return SubmissionRef(hotkey=hotkey, content_hash=content_hash, url=url, block=block)


def decode_payload(hotkey: str, block: int, data: str) -> "SubmissionRef | None":
    """Decode an untrusted production payload; HTTPS is the only fetch scheme."""

    return _decode_payload(
        hotkey, block, data, schemes=ALLOWED_URL_SCHEMES
    )


def decode_payload_for_testing(
    hotkey: str, block: int, data: str
) -> "SubmissionRef | None":
    """Hermetic-test decoder paired with :func:`encode_payload_for_testing`."""

    return _decode_payload(
        hotkey, block, data, schemes=_TEST_ONLY_URL_SCHEMES
    )
