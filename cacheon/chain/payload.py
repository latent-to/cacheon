"""Strict on-chain submission references for finalized validator intake.

Production payloads are exact JSON objects. Version 1 names a content hash
and a canonical HTTPS URL. Version 2 adds a compact eval-cost payment pointer
``{"b":block,"i":extrinsic_index}``. Identity remains ``h``/``u``; the pointer
only locates the transfer that paid for this reveal. ``file://`` exists behind
separate test-only functions so an on-chain value can never select local files
or plaintext HTTP.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from urllib.parse import urlparse

logger = logging.getLogger("cacheon.chain.payload")

PAYLOAD_VERSION = 1
PAID_PAYLOAD_VERSION = 2
MAX_PAYLOAD_BYTES = 1024
ALLOWED_URL_SCHEMES = ("https",)

_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_SCHEME_RE = re.compile(r"^([a-z][a-z0-9+.-]*)://")


class PayloadError(ValueError):
    """A locally constructed payload violates the production wire policy."""


@dataclass(frozen=True)
class SubmissionRef:
    hotkey: str
    content_hash: str
    url: str
    block: int
    payment_block: int = 0
    payment_extrinsic_index: int = 0


def _url_allowed(url: object, *, schemes: tuple[str, ...]) -> bool:
    if (
        not isinstance(url, str)
        or not url
        or len(url) > 8_192
        or not url.isascii()
        or any(ord(char) <= 32 or ord(char) == 127 for char in url)
    ):
        return False
    scheme = _SCHEME_RE.match(url)
    if scheme is None or scheme.group(1) not in schemes:
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


def _payment_pointer(payment_block: int, payment_extrinsic_index: int) -> dict[str, int] | None:
    if payment_block == 0 and payment_extrinsic_index == 0:
        return None
    if (
        type(payment_block) is not int
        or payment_block <= 0
        or type(payment_extrinsic_index) is not int
        or payment_extrinsic_index < 0
    ):
        raise PayloadError("eval-cost payment pointer is malformed")
    return {"b": payment_block, "i": payment_extrinsic_index}


def _encode_payload(
    content_hash: str,
    url: str,
    *,
    schemes: tuple[str, ...],
    payment_block: int = 0,
    payment_extrinsic_index: int = 0,
) -> str:
    if _HASH_RE.fullmatch(content_hash or "") is None:
        raise PayloadError(
            f"content_hash must be 64 lowercase hex chars, got {content_hash!r}"
        )
    if not _url_allowed(url, schemes=schemes):
        raise PayloadError(f"url must be canonical and use one of {schemes}, got {url!r}")
    pointer = _payment_pointer(payment_block, payment_extrinsic_index)
    if pointer is None:
        body: dict[str, object] = {"v": PAYLOAD_VERSION, "h": content_hash, "u": url}
    else:
        body = {
            "v": PAID_PAYLOAD_VERSION,
            "h": content_hash,
            "u": url,
            "p": pointer,
        }
    data = json.dumps(body, separators=(",", ":"))
    if len(data.encode("utf-8")) > MAX_PAYLOAD_BYTES:
        raise PayloadError(f"payload exceeds the {MAX_PAYLOAD_BYTES}-byte chain cap")
    return data


def encode_payload(
    content_hash: str,
    url: str,
    *,
    payment_block: int = 0,
    payment_extrinsic_index: int = 0,
) -> str:
    """Encode one production, HTTPS-only submission reference."""

    return _encode_payload(
        content_hash,
        url,
        schemes=ALLOWED_URL_SCHEMES,
        payment_block=payment_block,
        payment_extrinsic_index=payment_extrinsic_index,
    )


def _decode_payload(
    hotkey: str,
    block: int,
    data: object,
    *,
    schemes: tuple[str, ...],
) -> SubmissionRef | None:
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
        logger.warning("payload from %s: not canonical JSON; ignored", hotkey)
        return None
    parsed = _parse_payload_object(obj, data)
    if parsed is None:
        logger.warning("payload from %s: schema/version/encoding mismatch; ignored", hotkey)
        return None
    content_hash, url, payment_block, payment_extrinsic_index = parsed
    if not isinstance(content_hash, str) or _HASH_RE.fullmatch(content_hash) is None:
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
    return SubmissionRef(
        hotkey, content_hash, url, block,
        payment_block, payment_extrinsic_index,
    )


def _parse_payload_object(
    obj: object, data: str
) -> tuple[str, str, int, int] | None:
    if not isinstance(obj, dict) or type(obj.get("v")) is not int:
        return None
    version = obj["v"]
    if version == PAYLOAD_VERSION:
        if set(obj) != {"v", "h", "u"}:
            return None
        encoded = json.dumps(
            {"v": obj.get("v"), "h": obj.get("h"), "u": obj.get("u")},
            separators=(",", ":"),
        )
        if encoded != data:
            return None
        return obj["h"], obj["u"], 0, 0
    if version == PAID_PAYLOAD_VERSION:
        if set(obj) != {"v", "h", "u", "p"}:
            return None
        pointer = obj.get("p")
        if (
            not isinstance(pointer, dict)
            or set(pointer) != {"b", "i"}
            or type(pointer.get("b")) is not int
            or pointer["b"] <= 0
            or type(pointer.get("i")) is not int
            or pointer["i"] < 0
        ):
            return None
        encoded = json.dumps(
            {
                "v": obj.get("v"),
                "h": obj.get("h"),
                "u": obj.get("u"),
                "p": {"b": pointer["b"], "i": pointer["i"]},
            },
            separators=(",", ":"),
        )
        if encoded != data:
            return None
        return obj["h"], obj["u"], pointer["b"], pointer["i"]
    return None


def decode_payload(hotkey: str, block: int, data: object) -> SubmissionRef | None:
    """Decode hostile production bytes under the HTTPS-only policy."""

    return _decode_payload(hotkey, block, data, schemes=ALLOWED_URL_SCHEMES)


