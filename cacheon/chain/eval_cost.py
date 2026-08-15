"""Fixed, upgradeable evaluation-cost quotes and on-chain payment verification.

v1 ``quote_eval_cost`` ignores submission extras and returns a published alpha-rao
amount. Later quote versions may inspect ``EvalCostRequest.submission`` (target,
stack, estimated work) without changing the burn, remark, payload pointer, or
intake consume-once machinery.

The miner pays with ``SubtensorModule.burn_alpha`` batched with
``System.remark_with_event``. Validators admit a paid reveal only after the
remark binds the burn to this exact proposal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from cacheon.stack_identity import canonical_json_bytes, require_sha256_hex


EVAL_COST_QUOTE_VERSION = 1
EVAL_COST_PAYMENT_DOMAIN = "cacheon.chain.eval-cost-payment.v1"
EVAL_COST_ASSET = "alpha"
EVAL_COST_INSTRUMENT = "burn_alpha"
PUBLISHED_EVAL_COST_ALPHA_RAO = 1_000_000_000
DEFAULT_EVAL_COST_PAYMENT_WINDOW_BLOCKS = 7_200

REASON_MISSING = "missing_eval_cost_payment"
REASON_INVALID = "eval_cost_payment_invalid"
REASON_WINDOW = "eval_cost_payment_window"
REASON_USED = "eval_cost_payment_used"

_BURN_CALL = "burn_alpha"
_REMARK_CALL = "remark_with_event"


class EvalCostError(ValueError):
    """A quote, remark, or payment proof violates the eval-cost contract."""


class EvalCostFetchError(RuntimeError):
    """The chain payment lookup could not be completed; retry the pass."""


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
        if (
            not isinstance(self.hotkey, str)
            or not self.hotkey
            or self.hotkey.strip() != self.hotkey
            or len(self.hotkey) > 256
            or any(char in self.hotkey for char in "\x00\r\n")
        ):
            raise EvalCostError("eval-cost hotkey is malformed")
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
    amount_alpha_rao: int = PUBLISHED_EVAL_COST_ALPHA_RAO
    payment_window_blocks: int = DEFAULT_EVAL_COST_PAYMENT_WINDOW_BLOCKS

    def __post_init__(self) -> None:
        if type(self.amount_alpha_rao) is not int or self.amount_alpha_rao < 0:
            raise EvalCostError("eval-cost amount must be a non-negative integer")
        if (
            type(self.payment_window_blocks) is not int
            or self.payment_window_blocks <= 0
        ):
            raise EvalCostError("eval-cost payment window must be a positive integer")


@dataclass(frozen=True)
class EvalCostQuote:
    version: int
    netuid: int
    amount_alpha_rao: int
    asset: str
    instrument: str

    def __post_init__(self) -> None:
        if type(self.version) is not int or self.version < 1:
            raise EvalCostError("eval-cost quote version is malformed")
        if type(self.netuid) is not int or self.netuid < 0:
            raise EvalCostError("eval-cost quote netuid is malformed")
        if type(self.amount_alpha_rao) is not int or self.amount_alpha_rao < 0:
            raise EvalCostError("eval-cost quote amount is malformed")
        if self.asset != EVAL_COST_ASSET or self.instrument != EVAL_COST_INSTRUMENT:
            raise EvalCostError("eval-cost quote instrument is unsupported")


@dataclass(frozen=True)
class EvalCostPaymentProof:
    block: int
    extrinsic_index: int
    signer: str
    burn_coldkey: str
    burn_hotkey: str
    burn_netuid: int
    alpha_decrease: int
    remark: str

    def __post_init__(self) -> None:
        if type(self.block) is not int or self.block <= 0:
            raise EvalCostError("payment block is malformed")
        if type(self.extrinsic_index) is not int or self.extrinsic_index < 0:
            raise EvalCostError("payment extrinsic index is malformed")
        for field in ("signer", "burn_coldkey", "burn_hotkey"):
            value = getattr(self, field)
            if (
                not isinstance(value, str)
                or not value
                or value.strip() != value
                or len(value) > 256
            ):
                raise EvalCostError(f"payment {field} is malformed")
        if type(self.burn_netuid) is not int or self.burn_netuid < 0:
            raise EvalCostError("payment netuid is malformed")
        if type(self.alpha_decrease) is not int or self.alpha_decrease < 0:
            raise EvalCostError("payment alpha decrease is malformed")
        if not isinstance(self.remark, str) or len(self.remark.encode("utf-8")) > 4_096:
            raise EvalCostError("payment remark is malformed")


def quote_eval_cost(
    request: EvalCostRequest,
    *,
    policy: EvalCostPolicy | None = None,
) -> EvalCostQuote:
    """Return the v1 published cost. Submission extras are ignored until a later quote version."""

    if type(request) is not EvalCostRequest:
        raise EvalCostError("eval-cost request is not typed")
    resolved = policy if policy is not None else EvalCostPolicy()
    if type(resolved) is not EvalCostPolicy:
        raise EvalCostError("eval-cost policy is not typed")
    return EvalCostQuote(
        version=EVAL_COST_QUOTE_VERSION,
        netuid=request.netuid,
        amount_alpha_rao=resolved.amount_alpha_rao,
        asset=EVAL_COST_ASSET,
        instrument=EVAL_COST_INSTRUMENT,
    )


def encode_payment_remark(request: EvalCostRequest, quote: EvalCostQuote) -> str:
    """Canonical remark bytes binding one burn to one proposal."""

    if type(request) is not EvalCostRequest or type(quote) is not EvalCostQuote:
        raise EvalCostError("payment remark inputs are not typed")
    if not request.content_hash:
        raise EvalCostError("payment remark requires a content hash")
    if quote.netuid != request.netuid:
        raise EvalCostError("payment remark quote netuid differs")
    return canonical_json_bytes(
        {
            "domain": EVAL_COST_PAYMENT_DOMAIN,
            "payload": {
                "amount_alpha_rao": quote.amount_alpha_rao,
                "content_hash": request.content_hash,
                "hotkey": request.hotkey,
                "netuid": request.netuid,
            },
            "schema_version": 1,
        }
    ).decode("utf-8")


def verify_eval_cost_payment(
    *,
    request: EvalCostRequest,
    quote: EvalCostQuote,
    proof: EvalCostPaymentProof | None,
    reveal_block: int,
    window_blocks: int,
) -> str:
    """Return ``""`` when the proof pays this quote, else a stable invalid-reason token."""

    if type(request) is not EvalCostRequest or type(quote) is not EvalCostQuote:
        raise EvalCostError("eval-cost verification inputs are not typed")
    if type(reveal_block) is not int or reveal_block < 0:
        raise EvalCostError("reveal block is malformed")
    if type(window_blocks) is not int or window_blocks <= 0:
        raise EvalCostError("payment window is malformed")
    if proof is None:
        return REASON_MISSING
    if type(proof) is not EvalCostPaymentProof:
        raise EvalCostError("eval-cost payment proof is not typed")
    if proof.block > reveal_block or reveal_block - proof.block > window_blocks:
        return REASON_WINDOW
    expected = encode_payment_remark(request, quote)
    if (
        proof.signer != proof.burn_coldkey
        or proof.burn_hotkey != request.hotkey
        or proof.burn_netuid != request.netuid
        or proof.alpha_decrease < quote.amount_alpha_rao
        or proof.remark != expected
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

    Accepts a top-level ``burn_alpha`` or ``Utility.batch_all`` containing
    ``burn_alpha`` plus ``remark_with_event``. Returns ``None`` when the
    extrinsic is not a recognizable eval-cost payment.
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
    burn = None
    remark = ""
    args = _call_args_map(call_args)
    if args is None:
        return None
    if call_module == "SubtensorModule" and call_function == _BURN_CALL:
        burn = args
    elif call_module == "Utility" and call_function == "batch_all":
        calls = args.get("calls")
        if not isinstance(calls, Sequence) or isinstance(calls, (str, bytes)):
            return None
        for inner in calls:
            decoded = _inner_call(inner)
            if decoded is None:
                return None
            module, function, inner_args = decoded
            if module == "SubtensorModule" and function == _BURN_CALL:
                if burn is not None:
                    return None
                burn = inner_args
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
    if not isinstance(burn, Mapping):
        return None
    hotkey = burn.get("hotkey")
    netuid = _as_uint(burn.get("netuid"))
    if not isinstance(hotkey, str) or netuid is None:
        return None
    burned = _alpha_burned_event(events, extrinsic_index, netuid)
    if burned is None:
        return None
    coldkey, event_hotkey, alpha_decrease, event_netuid = burned
    if event_hotkey != hotkey or event_netuid != netuid:
        return None
    try:
        return EvalCostPaymentProof(
            block=block,
            extrinsic_index=extrinsic_index,
            signer=signer,
            burn_coldkey=coldkey,
            burn_hotkey=hotkey,
            burn_netuid=netuid,
            alpha_decrease=alpha_decrease,
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


def _as_uint(value: object) -> int | None:
    if type(value) is int and value >= 0:
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _alpha_burned_event(
    events: Sequence[Mapping[str, object]],
    extrinsic_index: int,
    netuid: int,
) -> tuple[str, str, int, int] | None:
    for record in events:
        if not isinstance(record, Mapping):
            continue
        indexed = _as_uint(record.get("extrinsic_idx", record.get("extrinsic_index")))
        if indexed is not None and indexed != extrinsic_index:
            continue
        inner = record.get("event", record)
        if not isinstance(inner, Mapping):
            continue
        module = inner.get("module_id", inner.get("module"))
        name = inner.get("event_id", inner.get("event"))
        if module != "SubtensorModule" or name != "AlphaBurned":
            continue
        parsed = _parse_alpha_burned(inner.get("attributes", inner))
        if parsed is None:
            continue
        coldkey, hotkey, amount, event_netuid = parsed
        if event_netuid != netuid:
            continue
        return coldkey, hotkey, amount, event_netuid
    return None


def _parse_alpha_burned(attributes: object) -> tuple[str, str, int, int] | None:
    if isinstance(attributes, Sequence) and not isinstance(attributes, (str, bytes)):
        if len(attributes) < 4:
            return None
        coldkey, hotkey, amount, netuid = attributes[:4]
        parsed_amount = _as_uint(amount)
        parsed_netuid = _as_uint(netuid)
        if (
            not isinstance(coldkey, str)
            or not isinstance(hotkey, str)
            or parsed_amount is None
            or parsed_netuid is None
        ):
            return None
        return coldkey, hotkey, parsed_amount, parsed_netuid
    if isinstance(attributes, Mapping):
        coldkey = attributes.get("Coldkey", attributes.get("coldkey"))
        hotkey = attributes.get("Hotkey", attributes.get("hotkey"))
        amount = attributes.get(
            "Actual Alpha Decrease",
            attributes.get("alpha_decrease", attributes.get("amount")),
        )
        netuid = attributes.get("Netuid", attributes.get("netuid"))
        parsed_amount = _as_uint(amount)
        parsed_netuid = _as_uint(netuid)
        if (
            not isinstance(coldkey, str)
            or not isinstance(hotkey, str)
            or parsed_amount is None
            or parsed_netuid is None
        ):
            return None
        return coldkey, hotkey, parsed_amount, parsed_netuid
    return None
