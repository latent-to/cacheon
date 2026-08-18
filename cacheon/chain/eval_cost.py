"""Fixed, upgradeable evaluation-cost quotes and on-chain payment verification.

v1 ``quote_eval_cost`` ignores submission extras and returns a published TAO-rao
amount frozen from ``issued_block`` through ``expires_block`` (300 blocks, about
one hour) so the miner can quote once and pay that same amount. The freeze lives
in the on-chain remark. Later quote versions may inspect
``EvalCostRequest.submission`` without changing the transfer, remark, payload
pointer, or intake consume-once machinery.

The miner pays with ``Balances.transfer_keep_alive`` to the subnet owner
coldkey at payment time, batched with ``System.remark_with_event``. The
transfer may precede the reveal: intake consumes the pointer only when a
reveal is reserved or deferred, so an unused payment can be attached to a
later commit of the same proposal. Validators admit a paid reveal only after
the remark binds that transfer to this exact proposal and the destination
matches the owner at the inclusion block.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Mapping, Sequence

from cacheon.stack_identity import canonical_json_bytes, require_sha256_hex


EVAL_COST_QUOTE_VERSION = 1
EVAL_COST_PAYMENT_DOMAIN = "cacheon.chain.eval-cost-payment.v1"
EVAL_COST_ASSET = "tao"
EVAL_COST_INSTRUMENT = "transfer_keep_alive"
PUBLISHED_EVAL_COST_TAO_RAO = 1_000_000_000
DEFAULT_EVAL_COST_PAYMENT_WINDOW_BLOCKS = 7_200
# ~1 hour at 12s blocks. Ridges freezes a payment quote for 3600s of wall time.
DEFAULT_EVAL_COST_QUOTE_TTL_BLOCKS = 300

REASON_MISSING = "missing_eval_cost_payment"
REASON_INVALID = "eval_cost_payment_invalid"
REASON_WINDOW = "eval_cost_payment_window"
REASON_QUOTE_EXPIRED = "eval_cost_quote_expired"
REASON_USED = "eval_cost_payment_used"

_TRANSFER_CALL = "transfer_keep_alive"
_REMARK_CALL = "remark_with_event"
_REMARK_PAYLOAD_FIELDS = frozenset(
    {
        "amount_rao",
        "content_hash",
        "destination",
        "expires_block",
        "hotkey",
        "issued_block",
        "netuid",
    }
)


class EvalCostError(ValueError):
    """A quote, remark, or payment proof violates the eval-cost contract."""


class EvalCostCommitError(EvalCostError):
    """Reveal commit failed after an eval-cost transfer was already included."""

    def __init__(
        self,
        payment_block: int,
        payment_extrinsic_index: int,
        cause: BaseException,
    ) -> None:
        self.payment_block = payment_block
        self.payment_extrinsic_index = payment_extrinsic_index
        super().__init__(
            "reveal commit failed after eval-cost payment "
            f"{unused_eval_cost_retry_flags(payment_block, payment_extrinsic_index)}: "
            f"{cause}"
        )


class EvalCostFetchError(RuntimeError):
    """The chain payment lookup could not be completed; retry the pass."""


def unused_eval_cost_retry_flags(payment_block: int, payment_extrinsic_index: int) -> str:
    """CLI flags that attach one already-included unused eval-cost payment."""

    return (
        f"--eval-cost-payment-block {payment_block} "
        f"--eval-cost-payment-extrinsic-index {payment_extrinsic_index}"
    )


def _require_account(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value.strip() != value
        or len(value) > 256
        or any(char in value for char in "\x00\r\n")
    ):
        raise EvalCostError(f"eval-cost {field} is malformed")
    return value


@dataclass(frozen=True)
class EvalCostRequest:
    netuid: int
    hotkey: str
    content_hash: str = ""
    target_id: str = ""
    submission: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        if type(self.netuid) is not int or self.netuid < 0:
            raise EvalCostError("eval-cost netuid is malformed")
        _require_account(self.hotkey, field="hotkey")
        if self.content_hash:
            require_sha256_hex(self.content_hash, field="content_hash")
        if not isinstance(self.target_id, str) or any(
            char in self.target_id for char in "\x00\r\n"
        ):
            raise EvalCostError("eval-cost target_id is malformed")
        if self.submission is not None and not isinstance(self.submission, Mapping):
            raise EvalCostError("eval-cost submission context is malformed")


@dataclass(frozen=True)
class EvalCostPolicy:
    amount_rao: int = PUBLISHED_EVAL_COST_TAO_RAO
    destination: str = ""
    payment_window_blocks: int = DEFAULT_EVAL_COST_PAYMENT_WINDOW_BLOCKS
    quote_ttl_blocks: int = DEFAULT_EVAL_COST_QUOTE_TTL_BLOCKS

    def __post_init__(self) -> None:
        if type(self.amount_rao) is not int or self.amount_rao < 0:
            raise EvalCostError("eval-cost amount must be a non-negative integer")
        if self.destination:
            _require_account(self.destination, field="destination")
        elif not isinstance(self.destination, str):
            raise EvalCostError("eval-cost destination is malformed")
        if (
            type(self.payment_window_blocks) is not int
            or self.payment_window_blocks <= 0
        ):
            raise EvalCostError("eval-cost payment window must be a positive integer")
        if type(self.quote_ttl_blocks) is not int or self.quote_ttl_blocks <= 0:
            raise EvalCostError("eval-cost quote TTL must be a positive integer")


@dataclass(frozen=True)
class EvalCostQuote:
    version: int
    netuid: int
    amount_rao: int
    destination: str
    asset: str
    instrument: str
    issued_block: int
    expires_block: int

    def __post_init__(self) -> None:
        if type(self.version) is not int or self.version < 1:
            raise EvalCostError("eval-cost quote version is malformed")
        if type(self.netuid) is not int or self.netuid < 0:
            raise EvalCostError("eval-cost quote netuid is malformed")
        if type(self.amount_rao) is not int or self.amount_rao < 0:
            raise EvalCostError("eval-cost quote amount is malformed")
        if self.destination:
            _require_account(self.destination, field="destination")
        elif not isinstance(self.destination, str):
            raise EvalCostError("eval-cost quote destination is malformed")
        if self.asset != EVAL_COST_ASSET or self.instrument != EVAL_COST_INSTRUMENT:
            raise EvalCostError("eval-cost quote instrument is unsupported")
        if type(self.issued_block) is not int or self.issued_block < 0:
            raise EvalCostError("eval-cost quote issued block is malformed")
        if type(self.expires_block) is not int or self.expires_block <= self.issued_block:
            raise EvalCostError("eval-cost quote expiry is malformed")


@dataclass(frozen=True)
class EvalCostPaymentProof:
    block: int
    extrinsic_index: int
    signer: str
    payer: str
    destination: str
    amount_rao: int
    remark: str

    def __post_init__(self) -> None:
        if type(self.block) is not int or self.block <= 0:
            raise EvalCostError("payment block is malformed")
        if type(self.extrinsic_index) is not int or self.extrinsic_index < 0:
            raise EvalCostError("payment extrinsic index is malformed")
        for field in ("signer", "payer", "destination"):
            _require_account(getattr(self, field), field=field)
        if type(self.amount_rao) is not int or self.amount_rao < 0:
            raise EvalCostError("payment amount is malformed")
        if not isinstance(self.remark, str) or len(self.remark.encode("utf-8")) > 4_096:
            raise EvalCostError("payment remark is malformed")


def quote_eval_cost(
    request: EvalCostRequest,
    *,
    policy: EvalCostPolicy | None = None,
    at_block: int = 0,
) -> EvalCostQuote:
    """Return the v1 published TAO cost frozen until ``at_block + quote_ttl``.

    Submission extras are ignored until a later quote version. The amount is
    static from issuance through payment. The destination is the subnet owner
    coldkey resolved at payment, not a miner-supplied wallet.
    """

    if type(request) is not EvalCostRequest:
        raise EvalCostError("eval-cost request is not typed")
    if type(at_block) is not int or at_block < 0:
        raise EvalCostError("eval-cost quote block is malformed")
    resolved = policy if policy is not None else EvalCostPolicy()
    if type(resolved) is not EvalCostPolicy:
        raise EvalCostError("eval-cost policy is not typed")
    return EvalCostQuote(
        version=EVAL_COST_QUOTE_VERSION,
        netuid=request.netuid,
        amount_rao=resolved.amount_rao,
        destination=resolved.destination,
        asset=EVAL_COST_ASSET,
        instrument=EVAL_COST_INSTRUMENT,
        issued_block=at_block,
        expires_block=at_block + resolved.quote_ttl_blocks,
    )


def encode_payment_remark(request: EvalCostRequest, quote: EvalCostQuote) -> str:
    """Canonical remark bytes binding one transfer to one frozen quote."""

    if type(request) is not EvalCostRequest or type(quote) is not EvalCostQuote:
        raise EvalCostError("payment remark inputs are not typed")
    if not request.content_hash:
        raise EvalCostError("payment remark requires a content hash")
    if not quote.destination:
        raise EvalCostError("payment remark requires a destination wallet")
    if quote.amount_rao <= 0:
        raise EvalCostError("payment remark requires a positive amount")
    if quote.netuid != request.netuid:
        raise EvalCostError("payment remark quote netuid differs")
    return canonical_json_bytes(
        {
            "domain": EVAL_COST_PAYMENT_DOMAIN,
            "payload": {
                "amount_rao": quote.amount_rao,
                "content_hash": request.content_hash,
                "destination": quote.destination,
                "expires_block": quote.expires_block,
                "hotkey": request.hotkey,
                "issued_block": quote.issued_block,
                "netuid": request.netuid,
            },
            "schema_version": 1,
        }
    ).decode("utf-8")


def decode_payment_remark(text: object) -> dict[str, object] | None:
    """Return the closed remark payload, or ``None`` when it is not canonical."""

    if not isinstance(text, str) or not text or len(text.encode("utf-8")) > 4_096:
        return None

    def unique_object(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate JSON key {key!r}")
            value[key] = item
        return value

    try:
        obj = json.loads(text, object_pairs_hook=unique_object)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    if (
        not isinstance(obj, dict)
        or set(obj) != {"domain", "payload", "schema_version"}
        or obj.get("domain") != EVAL_COST_PAYMENT_DOMAIN
        or obj.get("schema_version") != 1
    ):
        return None
    payload = obj.get("payload")
    if not isinstance(payload, dict) or set(payload) != _REMARK_PAYLOAD_FIELDS:
        return None
    if canonical_json_bytes(obj).decode("utf-8") != text:
        return None
    issued = payload.get("issued_block")
    expires = payload.get("expires_block")
    amount = payload.get("amount_rao")
    netuid = payload.get("netuid")
    if (
        type(issued) is not int
        or issued < 0
        or type(expires) is not int
        or expires <= issued
        or type(amount) is not int
        or amount < 0
        or type(netuid) is not int
        or netuid < 0
        or not isinstance(payload.get("content_hash"), str)
        or not isinstance(payload.get("hotkey"), str)
        or not isinstance(payload.get("destination"), str)
        or not payload.get("destination")
    ):
        return None
    return payload


def verify_eval_cost_payment(
    *,
    request: EvalCostRequest,
    policy: EvalCostPolicy,
    proof: EvalCostPaymentProof | None,
    reveal_block: int,
) -> str:
    """Return ``""`` when the proof pays the quote frozen at issuance, else a reason token."""

    if type(request) is not EvalCostRequest or type(policy) is not EvalCostPolicy:
        raise EvalCostError("eval-cost verification inputs are not typed")
    if type(reveal_block) is not int or reveal_block < 0:
        raise EvalCostError("reveal block is malformed")
    if proof is None:
        return REASON_MISSING
    if type(proof) is not EvalCostPaymentProof:
        raise EvalCostError("eval-cost payment proof is not typed")
    if not policy.destination:
        raise EvalCostError("eval-cost destination is required")
    parsed = decode_payment_remark(proof.remark)
    if parsed is None:
        return REASON_INVALID
    expected = quote_eval_cost(
        request, policy=policy, at_block=int(parsed["issued_block"])
    )
    if proof.remark != encode_payment_remark(request, expected):
        return REASON_INVALID
    if proof.block < expected.issued_block or proof.block > expected.expires_block:
        return REASON_QUOTE_EXPIRED
    if (
        proof.block > reveal_block
        or reveal_block - proof.block > policy.payment_window_blocks
    ):
        return REASON_WINDOW
    if (
        proof.signer != proof.payer
        or proof.destination != expected.destination
        or proof.amount_rao < expected.amount_rao
    ):
        return REASON_INVALID
    return ""


def proof_from_decoded_extrinsic(
    *,
    block: int,
    extrinsic_index: int,
    signer: str,
    call_module: str,
    call_function: str,
    call_args: object,
    events: Sequence[Mapping[str, object]],
) -> EvalCostPaymentProof | None:
    """Build a proof from already-decoded extrinsic and event records.

    Accepts a top-level ``transfer_keep_alive`` or ``Utility.batch_all``
    containing ``transfer_keep_alive`` plus ``remark_with_event``.
    """

    if (
        type(block) is not int
        or block <= 0
        or type(extrinsic_index) is not int
        or extrinsic_index < 0
        or not isinstance(signer, str)
        or not signer
        or not isinstance(call_module, str)
        or not isinstance(call_function, str)
    ):
        return None
    transfer = None
    remark = ""
    args = _call_args_map(call_args)
    if args is None:
        return None
    if call_module == "Balances" and call_function == _TRANSFER_CALL:
        transfer = args
    elif call_module == "Utility" and call_function == "batch_all":
        calls = args.get("calls")
        if not isinstance(calls, Sequence) or isinstance(calls, (str, bytes)):
            return None
        for inner in calls:
            decoded = _inner_call(inner)
            if decoded is None:
                return None
            module, function, inner_args = decoded
            if module == "Balances" and function == _TRANSFER_CALL:
                if transfer is not None:
                    return None
                transfer = inner_args
            elif module == "System" and function == _REMARK_CALL:
                if remark:
                    return None
                remark = _remark_text(inner_args.get("remark"))
                if remark is None:
                    return None
            else:
                return None
    else:
        return None
    if not isinstance(transfer, Mapping):
        return None
    dest = _account_id(transfer.get("dest", transfer.get("dest_id")))
    value = _as_uint(transfer.get("value", transfer.get("amount")))
    if not dest or value is None:
        return None
    moved = _tao_transferred_event(events, extrinsic_index)
    if moved is None:
        return None
    payer, event_dest, amount = moved
    if event_dest != dest or amount != value:
        return None
    try:
        return EvalCostPaymentProof(
            block=block,
            extrinsic_index=extrinsic_index,
            signer=signer,
            payer=payer,
            destination=dest,
            amount_rao=amount,
            remark=remark,
        )
    except EvalCostError:
        return None


def _inner_call(value: object) -> tuple[str, str, Mapping[str, object]] | None:
    raw = value
    if not isinstance(raw, Mapping):
        raw = getattr(value, "value_serialized", None)
        if raw is None:
            raw = getattr(value, "value", None)
    if isinstance(raw, Mapping) and "call" in raw:
        raw = raw["call"]
    if not isinstance(raw, Mapping):
        return None
    module = raw.get("call_module", raw.get("module_id"))
    function = raw.get("call_function", raw.get("function"))
    args = _call_args_map(raw.get("call_args", raw.get("args", raw.get("call_params"))))
    if not isinstance(module, str) or not isinstance(function, str) or args is None:
        return None
    return module, function, args


def _call_args_map(value: object) -> Mapping[str, object] | None:
    if isinstance(value, Mapping):
        return value
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return None
    mapped: dict[str, object] = {}
    for item in value:
        if not isinstance(item, Mapping):
            return None
        name = item.get("name")
        if not isinstance(name, str) or not name or name in mapped:
            return None
        mapped[name] = item.get("value", item.get("val"))
    return mapped


def _remark_text(value: object) -> str | None:
    if isinstance(value, str):
        if value.startswith("0x"):
            try:
                return bytes.fromhex(value[2:]).decode("utf-8")
            except (UnicodeDecodeError, ValueError):
                return None
        return value
    if isinstance(value, (bytes, bytearray)):
        try:
            return bytes(value).decode("utf-8")
        except UnicodeDecodeError:
            return None
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if not all(type(item) is int and 0 <= item <= 255 for item in value):
            return None
        try:
            return bytes(value).decode("utf-8")
        except UnicodeDecodeError:
            return None
    return None


def _account_id(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        inner = value.get("Id", value.get("id", value.get("Address32", value.get("dest"))))
        if isinstance(inner, str):
            return inner
        if isinstance(inner, Mapping):
            nested = inner.get("Id", inner.get("id"))
            if isinstance(nested, str):
                return nested
    return ""


def _as_uint(value: object) -> int | None:
    if type(value) is int and value >= 0:
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _tao_transferred_event(
    events: Sequence[Mapping[str, object]],
    extrinsic_index: int,
) -> tuple[str, str, int] | None:
    for record in events:
        if not isinstance(record, Mapping):
            continue
        indexed = _as_uint(record.get("extrinsic_idx", record.get("extrinsic_index")))
        if indexed != extrinsic_index:
            continue
        inner = record.get("event", record)
        if not isinstance(inner, Mapping):
            continue
        module = inner.get("module_id", inner.get("module"))
        name = inner.get("event_id", inner.get("event"))
        if module != "Balances" or name != "Transfer":
            continue
        parsed = _parse_transfer(inner.get("attributes", inner))
        if parsed is None:
            continue
        return parsed
    return None


def _parse_transfer(attributes: object) -> tuple[str, str, int] | None:
    if isinstance(attributes, Sequence) and not isinstance(attributes, (str, bytes)):
        if len(attributes) < 3:
            return None
        source, dest, amount = attributes[:3]
        parsed_amount = _as_uint(amount)
        source_id = _account_id(source)
        dest_id = _account_id(dest)
        if not source_id or not dest_id or parsed_amount is None:
            return None
        return source_id, dest_id, parsed_amount
    if isinstance(attributes, Mapping):
        source = _account_id(
            attributes.get("from", attributes.get("From", attributes.get("source")))
        )
        dest = _account_id(
            attributes.get("to", attributes.get("To", attributes.get("dest")))
        )
        parsed_amount = _as_uint(
            attributes.get("amount", attributes.get("Amount", attributes.get("value")))
        )
        if not source or not dest or parsed_amount is None:
            return None
        return source, dest, parsed_amount
    return None
