"""Publish and read hotkey-bound OKC usernames through native commitments."""

from __future__ import annotations

import argparse
import json
import re


class InvalidRegistration(ValueError):
    """A claim's registration evidence fails the username rules."""


def _public_key(ss58: str) -> str:
    """Decode and validate an AccountId32 as lowercase hexadecimal."""
    from scalecodec.utils.ss58 import ss58_decode

    decoded = ss58_decode(ss58)
    if not isinstance(decoded, str) or re.fullmatch(r"[0-9a-fA-F]{64}", decoded) is None:
        raise ValueError("hotkey must decode to an AccountId32")
    return decoded.lower()


def hotkey_tag(ss58: str) -> str:
    """Return the minimum five-character username prefix of an AccountId32."""
    return _public_key(ss58)[:5]


def validate_username(hotkey: str, username: str) -> str:
    """Match an entire contiguous hexadecimal prefix without normalizing the username."""
    public_key = _public_key(hotkey)
    if not isinstance(username, str) or not username.isascii() or not 1 <= len(username) <= 128:
        raise ValueError("OKC username must contain 1–128 ASCII bytes")
    prefixes = re.findall(r"[0-9a-fA-F]{5,}", username)
    if "Cacheon" not in username or not any(public_key.startswith(prefix) for prefix in prefixes):
        raise ValueError("OKC username must contain exact 'Cacheon' and a full matching lowercase public-key prefix of 5–64 hex characters")
    return username


def _raw_username(metadata: dict) -> str:
    try:
        fields = metadata["info"]["fields"]
        if not isinstance(fields, (list, tuple)) or len(fields) != 1:
            raise ValueError("expected one Raw field")
        field = fields[0]
        if not isinstance(field, dict) or len(field) != 1:
            raise ValueError("expected one Raw field")
        kind, raw = next(iter(field.items()))
        match = re.fullmatch(r"Raw(0|[1-9][0-9]{0,2})", kind)
        if match is None or int(match[1]) > 128:
            raise ValueError("expected a native Raw0–Raw128 field")
        if not isinstance(raw, str) or re.fullmatch(r"0x(?:[0-9a-fA-F]{2})*", raw) is None:
            raise ValueError("invalid Raw hexadecimal bytes")
        data = bytes.fromhex(raw[2:])
        if len(data) != int(match[1]):
            raise ValueError("Raw length does not match its field type")
        return data.decode("utf-8", errors="strict")
    except (KeyError, TypeError, UnicodeDecodeError) as exc:
        raise ValueError(f"invalid username commitment metadata: {exc}") from exc


def read_registration(st, netuid: int, hotkey: str, block: int, finalized_block: int) -> dict:
    """Read the actual plaintext publication at a finalized historical block."""
    if type(block) is not int or type(finalized_block) is not int or not 0 < block <= finalized_block:
        raise InvalidRegistration("registration block must be positive and finalized")
    metadata = st.get_commitment_metadata(netuid=netuid, hotkey_ss58=hotkey, block=block)
    if not isinstance(metadata, dict) or type(metadata.get("block")) is not int or metadata["block"] != block:
        raise InvalidRegistration("commitment metadata must identify the requested registration block")
    try:
        username = validate_username(hotkey, _raw_username(metadata))
    except ValueError as exc:
        raise InvalidRegistration(str(exc)) from exc
    block_hash = st.substrate.get_block_hash(block)
    if not isinstance(block_hash, str) or not block_hash:
        raise RuntimeError("registration block hash is unavailable")
    result = st.substrate.query("Timestamp", "Now", block_hash=block_hash)
    timestamp_ms = getattr(result, "value", result)
    if type(timestamp_ms) is not int or timestamp_ms <= 0:
        raise RuntimeError("registration block timestamp is invalid")
    return {"username": username, "block": block, "block_hash": block_hash, "timestamp_ms": timestamp_ms}


def cmd_register(args: argparse.Namespace) -> int:
    """Publish the username, then report its verified finalized registration block."""
    from cacheon import chain
    from cacheon.cli import _wallet_from_args

    wallet = _wallet_from_args(args)
    hotkey = wallet.hotkey.ss58_address
    username = validate_username(hotkey, args.name)
    output = {"submitted": False, "dry_run": args.dry_run, "netuid": args.netuid,
              "hotkey": hotkey, "username": username, "tag": hotkey_tag(hotkey),
              "registration_block": None}
    if args.dry_run:
        print(json.dumps(output, sort_keys=True))
        return 0
    st = chain.connect(args.network)
    try:
        response = st.set_commitment(wallet=wallet, netuid=args.netuid, data=username,
                                     wait_for_inclusion=True, wait_for_finalization=True)
        if not response.success:
            raise RuntimeError(response.message)
        finalized_block, _ = chain.read_finalized_head(st)
        metadata = st.get_commitment_metadata(netuid=args.netuid, hotkey_ss58=hotkey, block=finalized_block)
        if not isinstance(metadata, dict):
            raise ValueError("published username commitment is unavailable at the finalized head")
        registration = read_registration(st, args.netuid, hotkey, metadata.get("block"), finalized_block)
        if registration["username"] != username:
            raise ValueError("finalized username commitment does not match the published username")
        output.update(submitted=True, registration_block=registration["block"],
                      block_hash=registration["block_hash"], timestamp_ms=registration["timestamp_ms"])
    finally:
        st.close()
    print(json.dumps(output, sort_keys=True))
    return 0


def check_claim_registration(st, netuid: int, claim: dict, owner: dict | None,
                             finalized_block: int) -> dict:
    """Bind a claim to the globally oldest accepted username, before reward ownership."""
    if owner is None:
        raise InvalidRegistration("hotkey has no accepted username commitment")
    block = claim.get("okc_user_block")
    if type(block) is not int or not 0 < block < claim["block"] or block > finalized_block:
        raise InvalidRegistration("username registration must be finalized before the claim")
    if block != owner["block"]:
        raise InvalidRegistration("claim must reference the oldest accepted username registration block")
    if claim.get("team") != owner["username"]:
        raise InvalidRegistration("OKC team does not exactly match the hotkey's committed username")
    try:
        validate_username(claim["hotkey"], owner["username"])
    except ValueError as exc:
        raise InvalidRegistration(str(exc)) from exc
    return owner


def add_register_parser(sub) -> None:
    """Add the standalone native username registration command to the real CLI."""
    parser = sub.add_parser("chain-okc-register", help="miner: register an exact OKC username on-chain")
    parser.add_argument("name", metavar="NAME")
    parser.add_argument("--netuid", type=int, required=True)
    parser.add_argument("--network", default="finney")
    parser.add_argument("--wallet", default="default")
    parser.add_argument("--hotkey", default="default")
    parser.add_argument("--wallet-path", default="", help="optional Bittensor wallets root")
    parser.add_argument("--dry-run", action="store_true", help="validate and print without chain calls")
    parser.set_defaults(func=cmd_register)
