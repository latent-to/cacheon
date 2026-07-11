"""Non-executable streaming protocol across the untrusted SGLang boundary.

The entire candidate container, including its apparent Python "driver", is hostile:
SGLang deserializes scheduler-originated pickle in that process.  Consequently this
wire carries no secret and no Python object.  The outer controller sends one prompt
batch at a time and accepts only exact, bounded fixed-endian evidence tied to that
request's fresh public nonce and request id; JSON is limited to small control frames.
"""

from __future__ import annotations

import json
import math
import re
import struct
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from optima.eval.oci_protocol import (
    CONTAINER_MODEL_PATH,
    EVAL_CONFIG_FIELDS,
    MAX_PROMPT_CHARS,
    MAX_PROMPTS_PER_BATCH,
    MAX_TOTAL_PROMPT_CHARS,
    OCIProtocolError,
    config_to_wire,
    validate_eval_config,
)


SESSION_SCHEMA = "optima-outer-session-v3"
FRAME_MAGIC = b"OOS3"
EVIDENCE_MAGIC = b"OOE3"
TEACHER_EVIDENCE_MAGIC = b"OOT3"
FRAME_HEADER_BYTES = 8
MAX_INIT_BYTES = 2 * 1024 * 1024
MAX_BATCH_REQUEST_BYTES = 128 * 1024 * 1024
MAX_BATCH_RESPONSE_BYTES = 512 * 1024 * 1024
MAX_CONTROL_BYTES = 8 * 1024 * 1024
MAX_JSON_NESTING = 32
MAX_JSON_ITEMS = 32_000_000
MAX_TEXT_CHARS = 16_000_000
MAX_ERROR_CHARS = 16_384
_HEX_128 = re.compile(r"[0-9a-f]{32}\Z")
_HEX_256 = re.compile(r"[0-9a-f]{64}\Z")
_SHA256_ID = re.compile(r"sha256:[0-9a-f]{64}\Z")
_GPU_ARCH = re.compile(r"sm[0-9]{2,3}[a-z]?\Z")
_MODES = frozenset({"baseline", "candidate", "candidate_audit"})

RUNTIME_ATTESTATION_KEYS = frozenset({
    "verified",
    "sglang_version",
    "referee_source_digest",
    "model_revision",
    "model_manifest_digest",
    "model_content_digest",
    "environment_sha256",
    "gpu_count",
    "gpu_architectures",
    "topology_sha256",
})

# Only fields needed to construct and drive the engine cross the hostile boundary.
# In particular prompt_seed/input_len and every controller scoring threshold remain
# exclusively on the host.  Literal prompt batches are disclosed one at a time.
SESSION_ENGINE_CONFIG_FIELDS = (
    "model_path",
    "dtype",
    "max_new_tokens",
    "num_prompts",
    "timed_iters",
    "top_logprobs_num",
    "temperature",
    "ignore_eos",
    "warmup_iters",
    "conditioning_iters",
    "deterministic",
    "framework_mode",
    "isolate",
    "allow_unsafe_no_isolation",
    "seed",
    "attention_backend",
    "disable_cuda_graph",
    "mem_fraction_static",
    "log_level",
    "max_running_requests",
    "tp_size",
    "moe_runner_backend",
    "disable_custom_all_reduce",
    "candidate_attention_backend",
    "candidate_moe_runner_backend",
    "candidate_disable_custom_all_reduce",
    "extra_engine_kwargs",
    "candidate_extra_engine_kwargs",
    "audit_rate",
)


class SessionProtocolError(OCIProtocolError):
    """A streaming session message is malformed, ambiguous, or out of bounds."""


@dataclass(frozen=True)
class BatchEvidence:
    per_prompt: list[tuple[list[int], list]]
    texts: list[str]
    observed_tokens: int


_TEACHER_SOURCES = ("baseline", "candidate", "stock_control")
_TEACHER_PHASE = {"warmup": 0, "timed": 1}
_TEACHER_PHASE_BY_ID = {value: key for key, value in _TEACHER_PHASE.items()}


_EVIDENCE_BINDING = struct.Struct(">16s16s16sIIB3x")
_PROMPT_HEADER = struct.Struct(">III")  # tokens, top-k positions, UTF-8 text bytes
_TOPK_COUNT = struct.Struct(">H")
_TOPK_ENTRY = struct.Struct(">fI")
_TEACHER_BINDING = struct.Struct(">16s16s16sB3xII32s")
_TEACHER_PROMPT = struct.Struct(">I32s")
_TEACHER_LOGPROB = struct.Struct(">f")


@dataclass
class _Budget:
    remaining: int = MAX_JSON_ITEMS

    def take(self, count: int = 1) -> None:
        if count < 0 or self.remaining < count:
            raise SessionProtocolError("session JSON exceeds its item bound")
        self.remaining -= count


def _reject_duplicate_keys(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SessionProtocolError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _check_nesting(data: bytes) -> None:
    depth = 0
    quoted = False
    escaped = False
    for byte in data:
        char = chr(byte)
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
            continue
        if char == '"':
            quoted = True
        elif char in "[{":
            depth += 1
            if depth > MAX_JSON_NESTING:
                raise SessionProtocolError("session JSON nesting exceeds its bound")
        elif char in "]}":
            depth -= 1
            if depth < 0:
                raise SessionProtocolError("session JSON delimiters are unbalanced")
    if depth != 0 or quoted or escaped:
        raise SessionProtocolError("session JSON is truncated or unbalanced")


def _walk_json(value: Any, *, depth: int, budget: _Budget) -> None:
    if depth > MAX_JSON_NESTING:
        raise SessionProtocolError("session JSON semantic depth exceeds its bound")
    budget.take()
    if value is None or type(value) in (bool, int, float, str):
        if type(value) is float and not math.isfinite(value):
            raise SessionProtocolError("session JSON contains a non-finite number")
        return
    if isinstance(value, list):
        for item in value:
            _walk_json(item, depth=depth + 1, budget=budget)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str) or len(key) > 256:
                raise SessionProtocolError("session JSON has an invalid object key")
            budget.take()
            _walk_json(item, depth=depth + 1, budget=budget)
        return
    raise SessionProtocolError("session JSON contains a non-JSON value")


def encode_message(message: Mapping[str, Any], *, max_bytes: int) -> bytes:
    if not isinstance(message, Mapping):
        raise SessionProtocolError("session message must be an object")
    try:
        data = json.dumps(
            dict(message),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise SessionProtocolError(f"session message is not strict JSON: {exc}") from None
    if len(data) > max_bytes:
        raise SessionProtocolError(f"session message exceeds {max_bytes} bytes")
    return data


def decode_message(data: bytes, *, max_bytes: int) -> dict[str, Any]:
    if not isinstance(data, bytes) or len(data) > max_bytes:
        raise SessionProtocolError(f"session message exceeds {max_bytes} bytes")
    _check_nesting(data)
    try:
        text = data.decode("utf-8")
        decoder = json.JSONDecoder(
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                SessionProtocolError(f"non-finite JSON constant {value!r}")
            ),
        )
        value, end = decoder.raw_decode(text)
    except SessionProtocolError:
        raise
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise SessionProtocolError(f"session message is not valid JSON: {exc}") from None
    if end != len(text):
        raise SessionProtocolError("session message has trailing bytes/whitespace")
    if not isinstance(value, dict):
        raise SessionProtocolError("session message must decode to an object")
    _walk_json(value, depth=0, budget=_Budget())
    return value


def frame_message(message: Mapping[str, Any], *, max_bytes: int) -> bytes:
    payload = encode_message(message, max_bytes=max_bytes)
    return FRAME_MAGIC + struct.pack(">I", len(payload)) + payload


def parse_frame_bytes(frame: bytes, *, max_bytes: int) -> dict[str, Any]:
    if not isinstance(frame, bytes) or len(frame) < FRAME_HEADER_BYTES:
        raise SessionProtocolError("session frame is truncated")
    if frame[:4] != FRAME_MAGIC:
        raise SessionProtocolError("session frame magic/version mismatch")
    size = struct.unpack(">I", frame[4:8])[0]
    if size > max_bytes:
        raise SessionProtocolError(f"session frame declares more than {max_bytes} bytes")
    if len(frame) != FRAME_HEADER_BYTES + size:
        raise SessionProtocolError("session frame has trailing or missing bytes")
    return decode_message(frame[8:], max_bytes=max_bytes)


def _hex128(value: Any, name: str) -> str:
    if not isinstance(value, str) or _HEX_128.fullmatch(value) is None:
        raise SessionProtocolError(f"{name} must be 128 bits of lowercase hex")
    return value


def session_config_to_wire(config: Any) -> dict[str, Any]:
    """Project EvalConfig onto the reviewed engine-only session schema."""

    full = config_to_wire(config)
    if not set(SESSION_ENGINE_CONFIG_FIELDS) < set(EVAL_CONFIG_FIELDS):
        raise SessionProtocolError("engine session config schema is not a strict subset")
    forbidden = {
        "prompt_seed",
        "input_len",
        "kl_threshold",
        "argmax_disagree_rate_threshold",
        "p99_kl_threshold",
        "coverage_dev_threshold",
        "token_match_threshold",
        "speedup_margin",
        "bookend_baseline",
        "score_k",
        "max_noise",
        "fidelity_mode",
        "audit_min_calls",
    }
    if forbidden & set(SESSION_ENGINE_CONFIG_FIELDS):  # pragma: no cover - table invariant
        raise SessionProtocolError("controller-only config leaked into engine schema")
    return {name: full[name] for name in SESSION_ENGINE_CONFIG_FIELDS}


def validate_session_config(config: Any) -> dict[str, Any]:
    """Validate an exact engine-only mapping without accepting controller entropy."""

    if not isinstance(config, dict) or set(config) != set(SESSION_ENGINE_CONFIG_FIELDS):
        observed = set(config) if isinstance(config, dict) else set()
        expected = set(SESSION_ENGINE_CONFIG_FIELDS)
        raise SessionProtocolError(
            "engine_config fields do not match the session schema "
            f"(missing={sorted(expected - observed)}, extra={sorted(observed - expected)})"
        )
    # Reuse the rigorously bounded full EvalConfig validator by filling omitted
    # controller-only values from local class defaults. These defaults never cross
    # the wire and cannot reveal the host's real prompt plan seed.
    from optima.eval.throughput_kl import EvalConfig

    full = config_to_wire(EvalConfig(model_path=CONTAINER_MODEL_PATH))
    full.update(config)
    validated = validate_eval_config(full)
    return {name: validated[name] for name in SESSION_ENGINE_CONFIG_FIELDS}


def make_init(config: Any, *, mode: str, session_id: str) -> dict[str, Any]:
    if mode not in _MODES:
        raise SessionProtocolError(f"unsupported session mode {mode!r}")
    _hex128(session_id, "session_id")
    return {
        "schema": SESSION_SCHEMA,
        "type": "init",
        "session_id": session_id,
        "mode": mode,
        "engine_config": session_config_to_wire(config),
    }


def validate_init(message: Any) -> tuple[str, str, dict[str, Any]]:
    expected = {"schema", "type", "session_id", "mode", "engine_config"}
    if not isinstance(message, dict) or set(message) != expected:
        raise SessionProtocolError("init fields do not match the schema")
    if message["schema"] != SESSION_SCHEMA or message["type"] != "init":
        raise SessionProtocolError("init schema/type mismatch")
    session_id = _hex128(message["session_id"], "session_id")
    mode = message["mode"]
    if mode not in _MODES:
        raise SessionProtocolError(f"unsupported session mode {mode!r}")
    return session_id, mode, validate_session_config(message["engine_config"])


def _validated_runtime_attestation(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != RUNTIME_ATTESTATION_KEYS:
        raise SessionProtocolError("runtime attestation fields do not match the schema")
    if value["verified"] is not True:
        raise SessionProtocolError("runtime attestation is not verified")
    bounded = {}
    for name in ("sglang_version", "model_revision"):
        item = value[name]
        if (
            not isinstance(item, str)
            or not item
            or len(item) > 256
            or any(char in item for char in "\x00\r\n")
        ):
            raise SessionProtocolError(f"runtime attestation {name} is invalid")
        bounded[name] = item
    for name in (
        "referee_source_digest",
        "model_manifest_digest",
        "model_content_digest",
    ):
        item = value[name]
        if not isinstance(item, str) or _SHA256_ID.fullmatch(item) is None:
            raise SessionProtocolError(f"runtime attestation {name} is invalid")
        bounded[name] = item
    for name in ("environment_sha256", "topology_sha256"):
        item = value[name]
        if not isinstance(item, str) or _HEX_256.fullmatch(item) is None:
            raise SessionProtocolError(f"runtime attestation {name} is invalid")
        bounded[name] = item
    count = value["gpu_count"]
    architectures = value["gpu_architectures"]
    if (
        type(count) is not int
        or not 1 <= count <= 64
        or not isinstance(architectures, list)
        or len(architectures) != count
        or any(
            not isinstance(arch, str) or _GPU_ARCH.fullmatch(arch) is None
            for arch in architectures
        )
    ):
        raise SessionProtocolError("runtime attestation GPU identity is invalid")
    bounded["verified"] = True
    bounded["gpu_count"] = count
    bounded["gpu_architectures"] = list(architectures)
    return bounded


def preflight_message(
    *, session_id: str, mode: str, runtime_attestation: Mapping[str, Any]
) -> dict[str, Any]:
    if mode not in _MODES:
        raise SessionProtocolError(f"unsupported session mode {mode!r}")
    return {
        "schema": SESSION_SCHEMA,
        "type": "preflight",
        "session_id": _hex128(session_id, "session_id"),
        "mode": mode,
        "runtime_attestation": _validated_runtime_attestation(
            dict(runtime_attestation)
        ),
    }


def validate_preflight(
    message: Any,
    *,
    session_id: str,
    mode: str,
    expected_runtime: Mapping[str, Any],
) -> dict[str, Any]:
    expected_keys = {
        "schema", "type", "session_id", "mode", "runtime_attestation",
    }
    if not isinstance(message, dict) or set(message) != expected_keys:
        raise SessionProtocolError("worker preflight fields do not match the schema")
    if (
        message["schema"] != SESSION_SCHEMA
        or message["type"] != "preflight"
        or message["session_id"] != _hex128(session_id, "session_id")
        or message["mode"] != mode
    ):
        raise SessionProtocolError("worker preflight binding is stale or malformed")
    observed = _validated_runtime_attestation(message["runtime_attestation"])
    if not isinstance(expected_runtime, Mapping):
        raise SessionProtocolError("expected runtime attestation must be a mapping")
    unknown = set(expected_runtime) - RUNTIME_ATTESTATION_KEYS
    if unknown:
        raise SessionProtocolError(
            f"expected runtime attestation has unknown fields: {sorted(unknown)!r}"
        )
    for key, expected in expected_runtime.items():
        if observed[key] != expected:
            raise SessionProtocolError(
                f"runtime attestation {key} differs from host policy"
            )
    return observed


def ready_message(*, session_id: str, mode: str) -> dict[str, Any]:
    return {
        "schema": SESSION_SCHEMA,
        "type": "ready",
        "session_id": session_id,
        "mode": mode,
    }


def validate_ready(message: Any, *, session_id: str, mode: str) -> None:
    if message != ready_message(session_id=session_id, mode=mode):
        raise SessionProtocolError("worker ready marker is early, stale, or malformed")


def _validate_prompts(prompts: Any, *, expected_count: int) -> list[str]:
    if not isinstance(prompts, list) or len(prompts) != expected_count:
        raise SessionProtocolError(
            f"batch must contain exactly {expected_count} prompt strings"
        )
    if len(prompts) > MAX_PROMPTS_PER_BATCH:
        raise SessionProtocolError("batch exceeds its prompt-count bound")
    clean: list[str] = []
    total_chars = 0
    for prompt in prompts:
        if not isinstance(prompt, str) or len(prompt) > MAX_PROMPT_CHARS:
            raise SessionProtocolError("batch contains an invalid/oversized prompt")
        total_chars += len(prompt)
        if total_chars > MAX_TOTAL_PROMPT_CHARS:
            raise SessionProtocolError("batch exceeds its total prompt-character bound")
        clean.append(prompt)
    return clean


def batch_request(
    *,
    session_id: str,
    request_id: str,
    nonce: str,
    batch_index: int,
    warmup: bool,
    prompts: Sequence[str],
    expected_count: int,
) -> dict[str, Any]:
    message = {
        "schema": SESSION_SCHEMA,
        "type": "batch",
        "session_id": _hex128(session_id, "session_id"),
        "request_id": _hex128(request_id, "request_id"),
        "nonce": _hex128(nonce, "nonce"),
        "batch_index": batch_index,
        "warmup": warmup,
        "prompts": list(prompts),
    }
    validate_batch_request(message, expected_count=expected_count)
    return message


def validate_batch_request(
    message: Any, *, expected_count: int
) -> tuple[str, str, str, int, bool, list[str]]:
    expected = {
        "schema", "type", "session_id", "request_id", "nonce",
        "batch_index", "warmup", "prompts",
    }
    if not isinstance(message, dict) or set(message) != expected:
        raise SessionProtocolError("batch request fields do not match the schema")
    if message["schema"] != SESSION_SCHEMA or message["type"] != "batch":
        raise SessionProtocolError("batch request schema/type mismatch")
    session_id = _hex128(message["session_id"], "session_id")
    request_id = _hex128(message["request_id"], "request_id")
    nonce = _hex128(message["nonce"], "nonce")
    batch_index = message["batch_index"]
    if type(batch_index) is not int or not 0 <= batch_index < 128:
        raise SessionProtocolError("batch_index must be an integer in [0, 127]")
    warmup = message["warmup"]
    if type(warmup) is not bool:
        raise SessionProtocolError("warmup must be boolean")
    prompts = _validate_prompts(message["prompts"], expected_count=expected_count)
    return session_id, request_id, nonce, batch_index, warmup, prompts


def teacher_request(
    *,
    session_id: str,
    request_id: str,
    nonce: str,
    phase: str,
    batch_index: int,
    sealed_rollout_sha256: str,
    prompts: Sequence[str],
    baseline_ids: Sequence[Sequence[int]],
    candidate_ids: Sequence[Sequence[int]],
    stock_control_ids: Sequence[Sequence[int]],
    expected_count: int,
    expected_tokens: int,
) -> dict[str, Any]:
    message = {
        "schema": SESSION_SCHEMA,
        "type": "teacher",
        "session_id": _hex128(session_id, "session_id"),
        "request_id": _hex128(request_id, "request_id"),
        "nonce": _hex128(nonce, "nonce"),
        "phase": phase,
        "batch_index": batch_index,
        "sealed_rollout_sha256": sealed_rollout_sha256,
        "prompts": list(prompts),
        "baseline_ids": [list(ids) for ids in baseline_ids],
        "candidate_ids": [list(ids) for ids in candidate_ids],
        "stock_control_ids": [list(ids) for ids in stock_control_ids],
    }
    validate_teacher_request(
        message, expected_count=expected_count, expected_tokens=expected_tokens
    )
    return message


def validate_teacher_request(
    message: Any, *, expected_count: int, expected_tokens: int
) -> tuple[str, str, str, str, int, str, list[str], dict[str, list[list[int]]]]:
    expected = {
        "schema", "type", "session_id", "request_id", "nonce", "phase",
        "batch_index", "sealed_rollout_sha256", "prompts", *_TEACHER_SOURCES,
    }
    source_fields = {f"{source}_ids" for source in _TEACHER_SOURCES}
    expected -= set(_TEACHER_SOURCES)
    expected |= source_fields
    if not isinstance(message, dict) or set(message) != expected:
        raise SessionProtocolError("teacher request fields do not match the schema")
    if message["schema"] != SESSION_SCHEMA or message["type"] != "teacher":
        raise SessionProtocolError("teacher request schema/type mismatch")
    session_id = _hex128(message["session_id"], "session_id")
    request_id = _hex128(message["request_id"], "request_id")
    nonce = _hex128(message["nonce"], "nonce")
    phase = message["phase"]
    if phase not in _TEACHER_PHASE:
        raise SessionProtocolError("teacher phase is invalid")
    batch_index = message["batch_index"]
    if type(batch_index) is not int or not 0 <= batch_index < 128:
        raise SessionProtocolError("teacher batch_index is invalid")
    seal = message["sealed_rollout_sha256"]
    if (
        not isinstance(seal, str)
        or not seal.startswith("sha256:")
        or _HEX_256.fullmatch(seal[7:]) is None
    ):
        raise SessionProtocolError("teacher seal is not a SHA-256 identity")
    prompts = _validate_prompts(message["prompts"], expected_count=expected_count)
    sources: dict[str, list[list[int]]] = {}
    for source in _TEACHER_SOURCES:
        values = message[f"{source}_ids"]
        if not isinstance(values, list) or len(values) != expected_count:
            raise SessionProtocolError("teacher rollout prompt count mismatch")
        clean: list[list[int]] = []
        for ids in values:
            if (
                not isinstance(ids, list)
                or len(ids) != expected_tokens
                or any(
                    type(token) is not int or not 0 <= token <= 2_147_483_647
                    for token in ids
                )
            ):
                raise SessionProtocolError("teacher rollout token IDs are invalid")
            clean.append(list(ids))
        sources[source] = clean
    return (
        session_id, request_id, nonce, phase, batch_index, seal, prompts, sources
    )


def batch_response(
    *,
    session_id: str,
    request_id: str,
    nonce: str,
    batch_index: int,
    items: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema": SESSION_SCHEMA,
        "type": "batch_result",
        "session_id": session_id,
        "request_id": request_id,
        "nonce": nonce,
        "batch_index": batch_index,
        "items": items,
    }


def validate_batch_response(
    message: Any,
    *,
    session_id: str,
    request_id: str,
    nonce: str,
    batch_index: int,
    expected_prompts: int,
    max_new_tokens: int,
    top_logprobs_num: int,
    ignore_eos: bool,
    require_logprobs: bool,
    temperature: float = 0.0,
) -> BatchEvidence:
    expected = {
        "schema", "type", "session_id", "request_id", "nonce",
        "batch_index", "items",
    }
    if not isinstance(message, dict) or set(message) != expected:
        raise SessionProtocolError("batch result fields do not match the schema")
    exact = {
        "schema": SESSION_SCHEMA,
        "type": "batch_result",
        "session_id": session_id,
        "request_id": request_id,
        "nonce": nonce,
        "batch_index": batch_index,
    }
    if any(message.get(key) != value for key, value in exact.items()):
        raise SessionProtocolError("batch result nonce/request/session binding mismatch")
    items = message["items"]
    if not isinstance(items, list) or len(items) != expected_prompts:
        raise SessionProtocolError("batch result has the wrong number of prompt items")
    per_prompt: list[tuple[list[int], list]] = []
    texts: list[str] = []
    observed_tokens = 0
    for item in items:
        if not isinstance(item, dict) or set(item) != {
            "output_ids", "top_logprobs", "text"
        }:
            raise SessionProtocolError("batch result item fields do not match the schema")
        output_ids = item["output_ids"]
        if (
            not isinstance(output_ids, list)
            or len(output_ids) > max_new_tokens
            or any(type(token) is not int or not 0 <= token <= 2_147_483_647
                   for token in output_ids)
        ):
            raise SessionProtocolError("batch result output_ids are invalid")
        if require_logprobs and ignore_eos and len(output_ids) != max_new_tokens:
            raise SessionProtocolError("fixed-token batch returned an early/short output")
        if not require_logprobs and output_ids:
            raise SessionProtocolError("warmup acknowledgement must omit token evidence")
        raw_topk = item["top_logprobs"]
        if not isinstance(raw_topk, list):
            raise SessionProtocolError("batch result top_logprobs must be an array")
        if require_logprobs and len(raw_topk) != len(output_ids):
            raise SessionProtocolError("timed batch lacks per-token top-logprob evidence")
        if not require_logprobs and raw_topk:
            raise SessionProtocolError("warmup batch unexpectedly returned timed evidence")
        clean_topk: list[list[tuple[float, int, None]]] = []
        for position in raw_topk:
            if (
                not isinstance(position, list)
                or not position
                or len(position) != top_logprobs_num
            ):
                raise SessionProtocolError("one top-logprob position is invalid")
            clean_position: list[tuple[float, int, None]] = []
            seen_tokens: set[int] = set()
            for entry in position:
                if not isinstance(entry, list) or len(entry) != 2:
                    raise SessionProtocolError("top-logprob entry must be [logprob, token_id]")
                logprob, token_id = entry
                if (
                    isinstance(logprob, bool)
                    or not isinstance(logprob, (int, float))
                    or not math.isfinite(float(logprob))
                    or not -1_000_000.0 <= float(logprob) <= 1e-4
                    or type(token_id) is not int
                    or not 0 <= token_id <= 2_147_483_647
                    or token_id in seen_tokens
                ):
                    raise SessionProtocolError("top-logprob entry value is invalid")
                seen_tokens.add(token_id)
                clean_position.append((float(logprob), token_id, None))
            if any(
                clean_position[index][0] < clean_position[index + 1][0]
                for index in range(len(clean_position) - 1)
            ):
                raise SessionProtocolError("top-logprob entries are not descending")
            if sum(math.exp(entry[0]) for entry in clean_position) > 1.0001:
                raise SessionProtocolError("top-logprob probability mass exceeds one")
            clean_topk.append(clean_position)
        # Preserve sampled/model-consumed token IDs and raw reported top-k as
        # independent evidence. SGLang may apply sampler transforms after the
        # logits used for output_top_logprobs, so even stock temperature-zero M3
        # does not guarantee their argmax identity. The trusted controller grades
        # token match and paired top-k fidelity separately.
        text = item["text"]
        if text != "":
            raise SessionProtocolError("timed protocol forbids variable-length text")
        ids = list(output_ids)
        per_prompt.append((ids, clean_topk))
        texts.append(text)
        observed_tokens += len(ids)
    return BatchEvidence(per_prompt, texts, observed_tokens)


def _hex_bytes(value: str, name: str) -> bytes:
    return bytes.fromhex(_hex128(value, name))


def evidence_frame(
    evidence: BatchEvidence,
    *,
    session_id: str,
    request_id: str,
    nonce: str,
    batch_index: int,
    require_logprobs: bool,
) -> bytes:
    """Encode fixed-endian token/top-k evidence without JSON decoding overhead."""

    if len(evidence.per_prompt) != len(evidence.texts):
        raise SessionProtocolError("evidence prompt/text counts differ")
    payload = bytearray(_EVIDENCE_BINDING.pack(
        _hex_bytes(session_id, "session_id"),
        _hex_bytes(request_id, "request_id"),
        _hex_bytes(nonce, "nonce"),
        batch_index,
        len(evidence.per_prompt),
        1 if require_logprobs else 0,
    ))
    for (output_ids, topk), text in zip(evidence.per_prompt, evidence.texts):
        if text != "":
            raise SessionProtocolError("binary evidence forbids variable-length text")
        text_bytes = b""
        positions = len(topk)
        if require_logprobs and positions != len(output_ids):
            raise SessionProtocolError("evidence top-k count differs from token count")
        if not require_logprobs and positions:
            raise SessionProtocolError("warmup evidence unexpectedly contains top-k")
        payload.extend(_PROMPT_HEADER.pack(len(output_ids), positions, len(text_bytes)))
        for token_id in output_ids:
            if type(token_id) is not int or not 0 <= token_id <= 2_147_483_647:
                raise SessionProtocolError("evidence output token is invalid")
            payload.extend(struct.pack(">I", token_id))
        for position in topk:
            if not position or len(position) > 65_535:
                raise SessionProtocolError("evidence top-k position is invalid")
            payload.extend(_TOPK_COUNT.pack(len(position)))
            for entry in position:
                if not isinstance(entry, (list, tuple)) or len(entry) < 2:
                    raise SessionProtocolError("evidence top-k entry is invalid")
                logprob = float(entry[0])
                token_id = int(entry[1])
                if (
                    not math.isfinite(logprob)
                    or not -1_000_000.0 <= logprob <= 100.0
                    or not 0 <= token_id <= 2_147_483_647
                ):
                    raise SessionProtocolError("evidence top-k value is invalid")
                payload.extend(_TOPK_ENTRY.pack(logprob, token_id))
        payload.extend(text_bytes)
        if len(payload) > MAX_BATCH_RESPONSE_BYTES:
            raise SessionProtocolError("binary evidence exceeds its hard bound")
    return EVIDENCE_MAGIC + struct.pack(">I", len(payload)) + bytes(payload)


def decode_evidence_payload(
    payload: bytes,
    *,
    session_id: str,
    request_id: str,
    nonce: str,
    batch_index: int,
    expected_prompts: int,
    max_new_tokens: int,
    top_logprobs_num: int,
    ignore_eos: bool,
    require_logprobs: bool,
    temperature: float = 0.0,
) -> BatchEvidence:
    if not isinstance(payload, bytes) or not _EVIDENCE_BINDING.size <= len(payload) <= MAX_BATCH_RESPONSE_BYTES:
        raise SessionProtocolError("binary evidence size is invalid")
    try:
        (
            got_session, got_request, got_nonce, got_batch,
            prompt_count, flags,
        ) = _EVIDENCE_BINDING.unpack_from(payload, 0)
    except struct.error:
        raise SessionProtocolError("binary evidence binding is truncated") from None
    if (
        got_session != _hex_bytes(session_id, "session_id")
        or got_request != _hex_bytes(request_id, "request_id")
        or got_nonce != _hex_bytes(nonce, "nonce")
        or got_batch != batch_index
        or prompt_count != expected_prompts
        or flags != (1 if require_logprobs else 0)
    ):
        raise SessionProtocolError("binary evidence nonce/request/session binding mismatch")
    offset = _EVIDENCE_BINDING.size
    per_prompt: list[tuple[list[int], list]] = []
    texts: list[str] = []
    observed_tokens = 0

    def take(size: int) -> bytes:
        nonlocal offset
        if size < 0 or offset + size > len(payload):
            raise SessionProtocolError("binary evidence is truncated")
        result = payload[offset:offset + size]
        offset += size
        return result

    for _ in range(prompt_count):
        try:
            token_count, position_count, text_bytes = _PROMPT_HEADER.unpack(
                take(_PROMPT_HEADER.size)
            )
        except struct.error:
            raise SessionProtocolError("binary prompt header is truncated") from None
        if (
            token_count > max_new_tokens
            or (require_logprobs and ignore_eos and token_count != max_new_tokens)
            or (not require_logprobs and token_count != 0)
        ):
            raise SessionProtocolError("binary evidence returned an early/oversized output")
        if (
            (require_logprobs and position_count != token_count)
            or (not require_logprobs and position_count != 0)
        ):
            raise SessionProtocolError("binary evidence has wrong top-k position count")
        if text_bytes != 0:
            raise SessionProtocolError("binary evidence contains forbidden text bytes")
        output_ids = [struct.unpack(">I", take(4))[0] for _ in range(token_count)]
        topk: list[list[tuple[float, int, None]]] = []
        for _ in range(position_count):
            k_count = _TOPK_COUNT.unpack(take(_TOPK_COUNT.size))[0]
            if k_count != top_logprobs_num:
                raise SessionProtocolError("binary evidence top-k width is invalid")
            position: list[tuple[float, int, None]] = []
            seen: set[int] = set()
            for _ in range(k_count):
                logprob, token_id = _TOPK_ENTRY.unpack(take(_TOPK_ENTRY.size))
                if (
                    not math.isfinite(logprob)
                    or not -1_000_000.0 <= logprob <= 1e-4
                    or token_id > 2_147_483_647
                    or token_id in seen
                ):
                    raise SessionProtocolError("binary evidence top-k value is invalid")
                seen.add(token_id)
                position.append((float(logprob), int(token_id), None))
            if any(
                position[index][0] < position[index + 1][0]
                for index in range(len(position) - 1)
            ):
                raise SessionProtocolError("binary top-k entries are not descending")
            if sum(math.exp(entry[0]) for entry in position) > 1.0001:
                raise SessionProtocolError("binary top-k probability mass exceeds one")
            topk.append(position)
        try:
            text = take(text_bytes).decode("utf-8")
        except UnicodeDecodeError:
            raise SessionProtocolError("binary evidence text is not UTF-8") from None
        if len(text) > MAX_TEXT_CHARS:
            raise SessionProtocolError("binary evidence text character count exceeds its bound")
        per_prompt.append((output_ids, topk))
        texts.append(text)
        observed_tokens += token_count
    if offset != len(payload):
        raise SessionProtocolError("binary evidence contains trailing bytes")
    return BatchEvidence(per_prompt, texts, observed_tokens)


def expected_evidence_payload_bytes(
    *,
    prompt_count: int,
    max_new_tokens: int,
    top_logprobs_num: int,
    require_logprobs: bool,
    ignore_eos: bool,
) -> int | None:
    """Exact crownable response size, or ``None`` for natural-length debug mode."""

    if not require_logprobs:
        return _EVIDENCE_BINDING.size + prompt_count * _PROMPT_HEADER.size
    if not ignore_eos:
        return None
    per_prompt = (
        _PROMPT_HEADER.size
        + max_new_tokens * 4
        + max_new_tokens * (_TOPK_COUNT.size + top_logprobs_num * _TOPK_ENTRY.size)
    )
    return _EVIDENCE_BINDING.size + prompt_count * per_prompt


def expected_teacher_payload_bytes(
    *, prompt_count: int, token_count: int, top_logprobs_num: int
) -> int:
    per_trace = token_count * (
        _TEACHER_LOGPROB.size
        + _TOPK_COUNT.size
        + top_logprobs_num * _TOPK_ENTRY.size
    )
    return (
        _TEACHER_BINDING.size
        + prompt_count * (_TEACHER_PROMPT.size + len(_TEACHER_SOURCES) * per_trace)
    )


def teacher_evidence_frame(
    traces,
    *,
    session_id: str,
    request_id: str,
    nonce: str,
    phase: str,
    batch_index: int,
    sealed_rollout_sha256: str,
    token_count: int,
    top_logprobs_num: int,
) -> bytes:
    from optima.eval.external_quality import TeacherForcedPromptTrace

    if phase not in _TEACHER_PHASE or not isinstance(traces, (list, tuple)):
        raise SessionProtocolError("teacher evidence binding is invalid")
    if (
        not isinstance(sealed_rollout_sha256, str)
        or not sealed_rollout_sha256.startswith("sha256:")
        or _HEX_256.fullmatch(sealed_rollout_sha256[7:]) is None
    ):
        raise SessionProtocolError("teacher evidence seal is invalid")
    payload = bytearray(_TEACHER_BINDING.pack(
        _hex_bytes(session_id, "session_id"),
        _hex_bytes(request_id, "request_id"),
        _hex_bytes(nonce, "nonce"),
        _TEACHER_PHASE[phase],
        batch_index,
        len(traces),
        bytes.fromhex(sealed_rollout_sha256[7:]),
    ))
    for prompt in traces:
        if type(prompt) is not TeacherForcedPromptTrace:
            raise SessionProtocolError("teacher prompt trace is not typed")
        if (
            type(prompt.prompt_token_count) is not int
            or prompt.prompt_token_count <= 0
            or not isinstance(prompt.prompt_token_sha256, str)
            or _HEX_256.fullmatch(prompt.prompt_token_sha256) is None
        ):
            raise SessionProtocolError("teacher prompt-token receipt is invalid")
        payload.extend(_TEACHER_PROMPT.pack(
            prompt.prompt_token_count,
            bytes.fromhex(prompt.prompt_token_sha256),
        ))
        for source in _TEACHER_SOURCES:
            trace = getattr(prompt, source)
            try:
                trace.validated(
                    expected_tokens=token_count, topk_num=top_logprobs_num
                )
            except ValueError as exc:
                raise SessionProtocolError(str(exc)) from None
            for logprob, position in zip(
                trace.target_logprobs, trace.trusted_topk, strict=True
            ):
                payload.extend(_TEACHER_LOGPROB.pack(float(logprob)))
                payload.extend(_TOPK_COUNT.pack(len(position)))
                for top_logprob, token_id, _none in position:
                    payload.extend(_TOPK_ENTRY.pack(float(top_logprob), int(token_id)))
    if len(payload) > MAX_BATCH_RESPONSE_BYTES:
        raise SessionProtocolError("teacher evidence exceeds its hard bound")
    return TEACHER_EVIDENCE_MAGIC + struct.pack(">I", len(payload)) + bytes(payload)


def decode_teacher_evidence_payload(
    payload: bytes,
    *,
    session_id: str,
    request_id: str,
    nonce: str,
    phase: str,
    batch_index: int,
    sealed_rollout_sha256: str,
    expected_prompts: int,
    token_count: int,
    top_logprobs_num: int,
):
    from optima.eval.external_quality import TeacherForcedPromptTrace, TeacherForcedTrace

    exact = expected_teacher_payload_bytes(
        prompt_count=expected_prompts,
        token_count=token_count,
        top_logprobs_num=top_logprobs_num,
    )
    if not isinstance(payload, bytes) or len(payload) != exact:
        raise SessionProtocolError("teacher evidence has the wrong exact size")
    try:
        (
            got_session, got_request, got_nonce, phase_id, got_batch,
            prompt_count, seal,
        ) = _TEACHER_BINDING.unpack_from(payload, 0)
    except struct.error:
        raise SessionProtocolError("teacher evidence binding is truncated") from None
    if (
        phase not in _TEACHER_PHASE
        or got_session != _hex_bytes(session_id, "session_id")
        or got_request != _hex_bytes(request_id, "request_id")
        or got_nonce != _hex_bytes(nonce, "nonce")
        or phase_id != _TEACHER_PHASE[phase]
        or got_batch != batch_index
        or prompt_count != expected_prompts
        or seal.hex() != sealed_rollout_sha256.removeprefix("sha256:")
    ):
        raise SessionProtocolError("teacher evidence nonce/request/seal binding mismatch")
    offset = _TEACHER_BINDING.size

    def take(size: int) -> bytes:
        nonlocal offset
        if size < 0 or offset + size > len(payload):
            raise SessionProtocolError("teacher evidence is truncated")
        result = payload[offset:offset + size]
        offset += size
        return result

    prompts = []
    for _ in range(prompt_count):
        prompt_tokens, prompt_digest = _TEACHER_PROMPT.unpack(
            take(_TEACHER_PROMPT.size)
        )
        if prompt_tokens <= 0:
            raise SessionProtocolError("teacher prompt token count is invalid")
        sources = {}
        for source in _TEACHER_SOURCES:
            target_logprobs: list[float] = []
            trusted_topk = []
            for _position in range(token_count):
                (target_logprob,) = _TEACHER_LOGPROB.unpack(
                    take(_TEACHER_LOGPROB.size)
                )
                if not math.isfinite(target_logprob) or not -1_000.0 <= target_logprob <= 1e-4:
                    raise SessionProtocolError("teacher target logprob is invalid")
                target_logprobs.append(float(target_logprob))
                (width,) = _TOPK_COUNT.unpack(take(_TOPK_COUNT.size))
                if width != top_logprobs_num:
                    raise SessionProtocolError("teacher top-k width mismatch")
                position = []
                seen: set[int] = set()
                for _entry in range(width):
                    logprob, token_id = _TOPK_ENTRY.unpack(take(_TOPK_ENTRY.size))
                    if (
                        not math.isfinite(logprob)
                        or not -1_000.0 <= logprob <= 1e-4
                        or token_id > 2_147_483_647
                        or token_id in seen
                    ):
                        raise SessionProtocolError("teacher top-k entry is invalid")
                    seen.add(token_id)
                    position.append((float(logprob), int(token_id), None))
                if any(
                    position[index][0] < position[index + 1][0]
                    for index in range(len(position) - 1)
                ) or sum(math.exp(entry[0]) for entry in position) > 1.0001:
                    raise SessionProtocolError("teacher top-k ordering/mass is invalid")
                trusted_topk.append(tuple(position))
            sources[source] = TeacherForcedTrace(
                tuple(target_logprobs), tuple(trusted_topk)
            )
        prompts.append(TeacherForcedPromptTrace(
            prompt_token_count=prompt_tokens,
            prompt_token_sha256=prompt_digest.hex(),
            baseline=sources["baseline"],
            candidate=sources["candidate"],
            stock_control=sources["stock_control"],
        ))
    if offset != len(payload):
        raise SessionProtocolError("teacher evidence contains trailing bytes")
    return tuple(prompts)


def parse_evidence_frame_bytes(frame: bytes, **kwargs) -> BatchEvidence:
    if not isinstance(frame, bytes) or len(frame) < FRAME_HEADER_BYTES:
        raise SessionProtocolError("binary evidence frame is truncated")
    if frame[:4] != EVIDENCE_MAGIC:
        raise SessionProtocolError("binary evidence magic/version mismatch")
    size = struct.unpack(">I", frame[4:8])[0]
    if size > MAX_BATCH_RESPONSE_BYTES:
        raise SessionProtocolError("binary evidence declares an oversized payload")
    if len(frame) != FRAME_HEADER_BYTES + size:
        raise SessionProtocolError("binary evidence frame has trailing/missing bytes")
    return decode_evidence_payload(frame[8:], **kwargs)


def close_request(*, session_id: str, request_id: str, nonce: str) -> dict[str, Any]:
    return {
        "schema": SESSION_SCHEMA,
        "type": "close",
        "session_id": _hex128(session_id, "session_id"),
        "request_id": _hex128(request_id, "request_id"),
        "nonce": _hex128(nonce, "nonce"),
    }


def validate_close_request(message: Any) -> tuple[str, str, str]:
    expected = {"schema", "type", "session_id", "request_id", "nonce"}
    if not isinstance(message, dict) or set(message) != expected:
        raise SessionProtocolError("close request fields do not match the schema")
    if message["schema"] != SESSION_SCHEMA or message["type"] != "close":
        raise SessionProtocolError("close request schema/type mismatch")
    return (
        _hex128(message["session_id"], "session_id"),
        _hex128(message["request_id"], "request_id"),
        _hex128(message["nonce"], "nonce"),
    )


def closed_message(
    *,
    session_id: str,
    request_id: str,
    nonce: str,
    audit_receipts: list,
    audit_members: list,
) -> dict[str, Any]:
    return {
        "schema": SESSION_SCHEMA,
        "type": "closed",
        "session_id": session_id,
        "request_id": request_id,
        "nonce": nonce,
        "audit_receipts": audit_receipts,
        "audit_members": audit_members,
    }


def validate_closed(
    message: Any, *, session_id: str, request_id: str, nonce: str
) -> tuple[list, list]:
    expected = {
        "schema", "type", "session_id", "request_id", "nonce",
        "audit_receipts", "audit_members",
    }
    if not isinstance(message, dict) or set(message) != expected:
        raise SessionProtocolError("closed response fields do not match the schema")
    exact = {
        "schema": SESSION_SCHEMA,
        "type": "closed",
        "session_id": session_id,
        "request_id": request_id,
        "nonce": nonce,
    }
    if any(message.get(key) != value for key, value in exact.items()):
        raise SessionProtocolError("closed response binding mismatch")
    audit_receipts = message["audit_receipts"]
    audit_members = message["audit_members"]
    if (
        not isinstance(audit_receipts, list)
        or not isinstance(audit_members, list)
        or len(audit_receipts) > 16_384
        or len(audit_members) > 4096
        or any(not isinstance(item, dict) for item in [*audit_receipts, *audit_members])
    ):
        raise SessionProtocolError("closed response audit evidence is malformed")
    return audit_receipts, audit_members


def error_message(*, session_id: str, stage: str, error: BaseException) -> dict[str, Any]:
    return {
        "schema": SESSION_SCHEMA,
        "type": "session_error",
        "session_id": session_id,
        "stage": str(stage)[:128],
        "error_type": type(error).__name__[:128],
        "message": str(error)[:MAX_ERROR_CHARS],
    }


def parse_error_message(
    message: Any, *, expected_session_id: str | None = None
) -> tuple[str, str, str] | None:
    """Return one bounded worker diagnostic, or ``None`` for a non-error frame."""

    if not isinstance(message, dict) or (
        message.get("schema") != SESSION_SCHEMA
        or message.get("type") != "session_error"
    ):
        return None
    expected = {
        "schema", "type", "session_id", "stage", "error_type", "message",
    }
    if set(message) != expected or any(
        not isinstance(message.get(field), str)
        for field in ("session_id", "stage", "error_type", "message")
    ):
        raise SessionProtocolError("worker error marker is malformed")
    session_id = message["session_id"]
    if len(session_id) != 32 or (
        expected_session_id is not None and session_id != expected_session_id
    ):
        raise SessionProtocolError("worker error marker has a stale session binding")
    if (
        len(message["stage"]) > 128
        or len(message["error_type"]) > 128
        or len(message["message"]) > MAX_ERROR_CHARS
    ):
        raise SessionProtocolError("worker error marker exceeds its bounds")
    return message["stage"], message["error_type"], message["message"]
