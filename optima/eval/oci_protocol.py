"""Strict stdin protocol for one isolated OCI engine launch.

The controller and worker exchange no Python objects.  A request is a bounded JSON
document with an exact schema, followed by a fixed-size per-launch HMAC key.  Both
travel over the container's stdin; the worker closes that descriptor before it
starts SGLang, so scheduler descendants cannot recover the key from an inherited
pipe.  Results use :mod:`optima.ipc`'s authenticated, non-executable JSON format.

This module intentionally carries an explicit ``EvalConfig`` field allowlist.  A
new evaluator knob must be reviewed and added here instead of silently widening the
production worker's input language.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import re
import struct
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence


REQUEST_SCHEMA = "optima-oci-engine-launch-v2"
FRAME_MAGIC = b"OPTIMA-OCI-STDIN-V2\x00"
AUTH_KEY_BYTES = 32
AUTH_NONCE_BYTES = 16
MAX_REQUEST_BYTES = 128 * 1024 * 1024
MAX_REQUEST_NESTING = 24
MAX_BATCHES = 128
MAX_PROMPTS_PER_BATCH = 4096
MAX_TOTAL_PROMPTS = 32_768
MAX_PROMPT_CHARS = 2_000_000
MAX_TOTAL_PROMPT_CHARS = 96_000_000
MAX_SMALL_STRING_CHARS = 4096

CONTAINER_SOURCE_PATH = "/optima/input/source"
CONTAINER_MODEL_PATH = "/optima/input/model"
CONTAINER_ARTIFACT_PATH = "/optima/input/artifacts"
CONTAINER_BUNDLE_PATH = "/optima/input/bundle"
CONTAINER_JIT_PATH = "/optima/jit"
CONTAINER_OUTPUT_PATH = "/optima/output"
CONTAINER_RESULT_PATH = f"{CONTAINER_OUTPUT_PATH}/result.auth"

_MODES = frozenset({"baseline", "candidate", "candidate_audit"})
_HEX_128 = re.compile(r"[0-9a-f]{32}\Z")
_TOKEN = re.compile(r"[A-Za-z0-9_.:+/@-]{1,256}\Z")
_ENV_KEY = re.compile(r"[A-Z_][A-Z0-9_]{0,127}\Z")


class OCIProtocolError(ValueError):
    """The OCI launch request/frame is malformed or outside a hard bound."""


# Keep this literal and ordered.  Comparing it with dataclasses.fields(EvalConfig)
# in ``config_to_wire`` makes schema drift fail closed.
EVAL_CONFIG_FIELDS = (
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
    "kl_threshold",
    "argmax_disagree_rate_threshold",
    "p99_kl_threshold",
    "coverage_dev_threshold",
    "framework_mode",
    "token_match_threshold",
    "isolate",
    "allow_unsafe_no_isolation",
    "seed",
    "prompt_seed",
    "input_len",
    "speedup_margin",
    "bookend_baseline",
    "score_k",
    "max_noise",
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
    "fidelity_mode",
    "audit_rate",
    "audit_min_calls",
)


# These are the engine kwargs currently pinned by validator-owned arenas.  They are
# deliberately not an arbitrary JSON escape hatch into ``sglang.Engine``.
_ENGINE_KWARG_KINDS: dict[str, str] = {
    "quantization": "token",
    "page_size": "positive_int",
    "trust_remote_code": "bool",
    "cuda_graph_backend_prefill": "token",
    "kv_cache_dtype": "token",
    "context_length": "positive_int",
    "chunked_prefill_size": "positive_int",
    "max_prefill_tokens": "positive_int",
    "disable_radix_cache": "bool",
    "enable_flashinfer_allreduce_fusion": "bool",
}


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _bounded_int(value: Any, name: str, low: int, high: int) -> int:
    if not _is_int(value) or not low <= value <= high:
        raise OCIProtocolError(f"{name} must be an integer in [{low}, {high}]")
    return value


def _bounded_float(value: Any, name: str, low: float, high: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise OCIProtocolError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not low <= result <= high:
        raise OCIProtocolError(f"{name} must be finite and in [{low}, {high}]")
    return result


def _optional_float(value: Any, name: str, low: float, high: float) -> None:
    if value is not None:
        _bounded_float(value, name, low, high)


def _optional_int(value: Any, name: str, low: int, high: int) -> None:
    if value is not None:
        _bounded_int(value, name, low, high)


def _bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise OCIProtocolError(f"{name} must be boolean")
    return value


def _small_string(value: Any, name: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise OCIProtocolError(f"{name} must be a string")
    if (not allow_empty and not value) or len(value) > MAX_SMALL_STRING_CHARS:
        raise OCIProtocolError(
            f"{name} must be {'at most' if allow_empty else '1..'} "
            f"{MAX_SMALL_STRING_CHARS} characters"
        )
    if "\x00" in value or "\n" in value or "\r" in value:
        raise OCIProtocolError(f"{name} contains a forbidden control character")
    return value


def _optional_token(value: Any, name: str) -> None:
    if value is None:
        return
    if not isinstance(value, str) or _TOKEN.fullmatch(value) is None:
        raise OCIProtocolError(f"{name} must be null or a simple backend token")


def _validate_engine_kwargs(value: Any, name: str) -> None:
    if not isinstance(value, dict):
        raise OCIProtocolError(f"{name} must be an object")
    unknown = set(value) - set(_ENGINE_KWARG_KINDS)
    if unknown:
        raise OCIProtocolError(f"{name} contains unsupported keys: {sorted(unknown)}")
    for key, item in value.items():
        kind = _ENGINE_KWARG_KINDS[key]
        field = f"{name}.{key}"
        if kind == "bool":
            _bool(item, field)
        elif kind == "positive_int":
            _bounded_int(item, field, 1, 16_777_216)
        elif kind == "token":
            if not isinstance(item, str) or _TOKEN.fullmatch(item) is None:
                raise OCIProtocolError(f"{field} must be a simple token")
        else:  # pragma: no cover - validator-owned table invariant
            raise AssertionError(f"unknown engine-kwarg kind {kind!r}")


def validate_eval_config(config: Any) -> dict[str, Any]:
    """Validate and return an exact JSON-only ``EvalConfig`` mapping."""

    if not isinstance(config, dict):
        raise OCIProtocolError("eval_config must be an object")
    expected = set(EVAL_CONFIG_FIELDS)
    if set(config) != expected:
        raise OCIProtocolError(
            "eval_config fields do not match the OCI schema "
            f"(missing={sorted(expected - set(config))}, "
            f"extra={sorted(set(config) - expected)})"
        )

    model_path = _small_string(config["model_path"], "eval_config.model_path")
    if model_path != CONTAINER_MODEL_PATH:
        raise OCIProtocolError(
            f"eval_config.model_path must be profile-owned {CONTAINER_MODEL_PATH!r}"
        )
    if config["dtype"] not in {"bfloat16", "float16", "float32"}:
        raise OCIProtocolError("eval_config.dtype is unsupported")
    _bounded_int(config["max_new_tokens"], "eval_config.max_new_tokens", 1, 32_768)
    _bounded_int(config["num_prompts"], "eval_config.num_prompts", 1, MAX_PROMPTS_PER_BATCH)
    _bounded_int(config["timed_iters"], "eval_config.timed_iters", 1, 64)
    _bounded_int(config["top_logprobs_num"], "eval_config.top_logprobs_num", 0, 4096)
    _bounded_float(config["temperature"], "eval_config.temperature", 0.0, 100.0)
    _bool(config["ignore_eos"], "eval_config.ignore_eos")
    _bounded_int(config["warmup_iters"], "eval_config.warmup_iters", 0, 64)
    _bounded_int(
        config["conditioning_iters"], "eval_config.conditioning_iters", 1, 64
    )
    if config["conditioning_iters"] > config["warmup_iters"]:
        raise OCIProtocolError(
            "eval_config.conditioning_iters cannot exceed warmup_iters"
        )
    _bool(config["deterministic"], "eval_config.deterministic")
    for field in (
        "kl_threshold",
        "argmax_disagree_rate_threshold",
        "p99_kl_threshold",
        "coverage_dev_threshold",
    ):
        _optional_float(config[field], f"eval_config.{field}", 0.0, 1_000_000.0)
    _bool(config["framework_mode"], "eval_config.framework_mode")
    _bounded_float(
        config["token_match_threshold"], "eval_config.token_match_threshold", 0.0, 1.0
    )
    if not _bool(config["isolate"], "eval_config.isolate"):
        raise OCIProtocolError("production OCI launch requires isolate=true")
    if _bool(
        config["allow_unsafe_no_isolation"],
        "eval_config.allow_unsafe_no_isolation",
    ):
        raise OCIProtocolError("production OCI launch forbids unsafe isolation bypass")
    _bounded_int(config["seed"], "eval_config.seed", 0, (1 << 63) - 1)
    _bounded_int(config["prompt_seed"], "eval_config.prompt_seed", 0, (1 << 63) - 1)
    _optional_int(config["input_len"], "eval_config.input_len", 1, 4_194_304)
    _bounded_float(config["speedup_margin"], "eval_config.speedup_margin", 0.0, 10.0)
    _bool(config["bookend_baseline"], "eval_config.bookend_baseline")
    _bounded_float(config["score_k"], "eval_config.score_k", 0.0, 1000.0)
    _bounded_float(config["max_noise"], "eval_config.max_noise", 0.0, 10.0)
    _optional_token(config["attention_backend"], "eval_config.attention_backend")
    _bool(config["disable_cuda_graph"], "eval_config.disable_cuda_graph")
    _bounded_float(
        config["mem_fraction_static"], "eval_config.mem_fraction_static", 0.000001, 0.999999
    )
    _small_string(config["log_level"], "eval_config.log_level")
    _optional_int(
        config["max_running_requests"], "eval_config.max_running_requests", 1, 1_048_576
    )
    _optional_int(config["tp_size"], "eval_config.tp_size", 1, 64)
    _optional_token(config["moe_runner_backend"], "eval_config.moe_runner_backend")
    _bool(
        config["disable_custom_all_reduce"],
        "eval_config.disable_custom_all_reduce",
    )
    _optional_token(
        config["candidate_attention_backend"],
        "eval_config.candidate_attention_backend",
    )
    _optional_token(
        config["candidate_moe_runner_backend"],
        "eval_config.candidate_moe_runner_backend",
    )
    candidate_car = config["candidate_disable_custom_all_reduce"]
    if candidate_car is not None:
        _bool(candidate_car, "eval_config.candidate_disable_custom_all_reduce")
    _validate_engine_kwargs(config["extra_engine_kwargs"], "eval_config.extra_engine_kwargs")
    _validate_engine_kwargs(
        config["candidate_extra_engine_kwargs"],
        "eval_config.candidate_extra_engine_kwargs",
    )
    if config["fidelity_mode"] not in {"kl", "audit"}:
        raise OCIProtocolError("eval_config.fidelity_mode must be 'kl' or 'audit'")
    _bounded_float(config["audit_rate"], "eval_config.audit_rate", 0.0, 1.0)
    _bounded_int(config["audit_min_calls"], "eval_config.audit_min_calls", 1, 10_000_000)

    # Round-trip through strict JSON to detach from arbitrary mapping subclasses and
    # prove no dynamic/non-finite value survived the validators.
    try:
        return json.loads(json.dumps(config, allow_nan=False))
    except (TypeError, ValueError) as exc:  # pragma: no cover - defense in depth
        raise OCIProtocolError(f"eval_config is not strict JSON: {exc}") from None


def config_to_wire(config: Any) -> dict[str, Any]:
    """Copy an exact ``EvalConfig`` into the profile-owned container namespace."""

    from optima.eval.throughput_kl import EvalConfig

    if type(config) is not EvalConfig:
        raise OCIProtocolError("OCI launch requires an exact EvalConfig instance")
    actual_fields = tuple(field.name for field in dataclasses.fields(EvalConfig))
    if actual_fields != EVAL_CONFIG_FIELDS:
        raise OCIProtocolError(
            "EvalConfig changed without a reviewed OCI request-schema update"
        )
    result = {name: getattr(config, name) for name in EVAL_CONFIG_FIELDS}
    result["model_path"] = CONTAINER_MODEL_PATH
    return validate_eval_config(result)


def _validate_prompt_batches(
    prompt_batches: Any, *, config: Mapping[str, Any], mode: str
) -> tuple[tuple[str, ...], ...]:
    if not isinstance(prompt_batches, (list, tuple)):
        raise OCIProtocolError("prompt_batches must be an array")
    expected_batches = (
        2 if mode == "candidate_audit"
        else int(config["warmup_iters"]) + int(config["timed_iters"])
    )
    if len(prompt_batches) != expected_batches:
        raise OCIProtocolError(
            f"prompt_batches has {len(prompt_batches)} batches; expected {expected_batches}"
        )
    if len(prompt_batches) > MAX_BATCHES:
        raise OCIProtocolError(f"prompt_batches exceeds {MAX_BATCHES} batches")
    expected_prompts = int(config["num_prompts"])
    total_prompts = 0
    total_chars = 0
    result: list[tuple[str, ...]] = []
    for batch_index, batch in enumerate(prompt_batches):
        if not isinstance(batch, (list, tuple)):
            raise OCIProtocolError(f"prompt_batches[{batch_index}] must be an array")
        if len(batch) != expected_prompts:
            raise OCIProtocolError(
                f"prompt_batches[{batch_index}] has {len(batch)} prompts; "
                f"expected {expected_prompts}"
            )
        if len(batch) > MAX_PROMPTS_PER_BATCH:
            raise OCIProtocolError("prompt batch exceeds its hard prompt bound")
        clean_batch: list[str] = []
        for prompt_index, prompt in enumerate(batch):
            if not isinstance(prompt, str):
                raise OCIProtocolError(
                    f"prompt_batches[{batch_index}][{prompt_index}] must be a string"
                )
            if len(prompt) > MAX_PROMPT_CHARS:
                raise OCIProtocolError("one prompt exceeds its hard character bound")
            total_chars += len(prompt)
            total_prompts += 1
            if total_chars > MAX_TOTAL_PROMPT_CHARS:
                raise OCIProtocolError("prompt plan exceeds its total character bound")
            if total_prompts > MAX_TOTAL_PROMPTS:
                raise OCIProtocolError("prompt plan exceeds its total prompt bound")
            clean_batch.append(prompt)
        result.append(tuple(clean_batch))
    return tuple(result)


@dataclass(frozen=True)
class OCILaunchRequest:
    request_id: str
    nonce: bytes
    mode: str
    eval_config: Mapping[str, Any]
    prompt_batches: tuple[tuple[str, ...], ...]

    @property
    def active(self) -> bool:
        return self.mode != "baseline"

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema": REQUEST_SCHEMA,
            "request_id": self.request_id,
            "nonce": self.nonce.hex(),
            "mode": self.mode,
            "eval_config": dict(self.eval_config),
            "prompt_batches": [list(batch) for batch in self.prompt_batches],
        }


def make_request(
    config: Any,
    prompt_batches: Sequence[Sequence[str]],
    *,
    mode: str,
    request_id: str,
    nonce: bytes,
) -> OCILaunchRequest:
    if mode not in _MODES:
        raise OCIProtocolError(f"unsupported OCI launch mode {mode!r}")
    if not isinstance(request_id, str) or _HEX_128.fullmatch(request_id) is None:
        raise OCIProtocolError("request_id must be 128 bits of lowercase hex")
    if not isinstance(nonce, bytes) or len(nonce) != AUTH_NONCE_BYTES:
        raise OCIProtocolError(f"nonce must contain exactly {AUTH_NONCE_BYTES} bytes")
    wire_config = config_to_wire(config)
    batches = _validate_prompt_batches(prompt_batches, config=wire_config, mode=mode)
    return OCILaunchRequest(request_id, nonce, mode, wire_config, batches)


def _reject_duplicate_keys(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise OCIProtocolError(f"duplicate JSON key {key!r}")
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
            if depth > MAX_REQUEST_NESTING:
                raise OCIProtocolError("request JSON nesting exceeds its hard bound")
        elif char in "]}":
            depth -= 1
            if depth < 0:
                raise OCIProtocolError("request JSON delimiters are unbalanced")
    if depth != 0 or quoted or escaped:
        raise OCIProtocolError("request JSON is truncated or unbalanced")


def encode_request(request: OCILaunchRequest) -> bytes:
    try:
        data = json.dumps(
            request.to_payload(),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise OCIProtocolError(f"request is not strict JSON: {exc}") from None
    if len(data) > MAX_REQUEST_BYTES:
        raise OCIProtocolError(f"request exceeds {MAX_REQUEST_BYTES} bytes")
    return data


def decode_request(data: bytes) -> OCILaunchRequest:
    if not isinstance(data, bytes) or len(data) > MAX_REQUEST_BYTES:
        raise OCIProtocolError("request is not bounded bytes")
    _check_nesting(data)
    try:
        payload = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                OCIProtocolError(f"non-finite JSON constant {value!r}")
            ),
        )
    except OCIProtocolError:
        raise
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise OCIProtocolError(f"request is not valid JSON: {exc}") from None
    expected = {"schema", "request_id", "nonce", "mode", "eval_config", "prompt_batches"}
    if not isinstance(payload, dict) or set(payload) != expected:
        raise OCIProtocolError("request envelope fields do not match the schema")
    if payload["schema"] != REQUEST_SCHEMA:
        raise OCIProtocolError("request schema/version mismatch")
    request_id = payload["request_id"]
    if not isinstance(request_id, str) or _HEX_128.fullmatch(request_id) is None:
        raise OCIProtocolError("request_id must be 128 bits of lowercase hex")
    nonce_hex = payload["nonce"]
    if not isinstance(nonce_hex, str) or _HEX_128.fullmatch(nonce_hex) is None:
        raise OCIProtocolError("nonce must be 128 bits of lowercase hex")
    mode = payload["mode"]
    if mode not in _MODES:
        raise OCIProtocolError(f"unsupported OCI launch mode {mode!r}")
    config = validate_eval_config(payload["eval_config"])
    batches = _validate_prompt_batches(payload["prompt_batches"], config=config, mode=mode)
    return OCILaunchRequest(request_id, bytes.fromhex(nonce_hex), mode, config, batches)


def encode_stdin_frame(request: OCILaunchRequest, *, auth_key: bytes) -> bytes:
    if not isinstance(auth_key, bytes) or len(auth_key) != AUTH_KEY_BYTES:
        raise OCIProtocolError(f"auth_key must contain exactly {AUTH_KEY_BYTES} bytes")
    payload = encode_request(request)
    return FRAME_MAGIC + struct.pack(">I", len(payload)) + payload + auth_key


def decode_stdin_frame(frame: bytes) -> tuple[OCILaunchRequest, bytes]:
    if not isinstance(frame, bytes):
        raise OCIProtocolError("stdin frame must be bytes")
    header_size = len(FRAME_MAGIC) + 4
    if len(frame) < header_size + AUTH_KEY_BYTES or not frame.startswith(FRAME_MAGIC):
        raise OCIProtocolError("stdin frame magic/header is invalid")
    size = struct.unpack(">I", frame[len(FRAME_MAGIC):header_size])[0]
    if size > MAX_REQUEST_BYTES:
        raise OCIProtocolError("stdin request length exceeds its hard bound")
    expected = header_size + size + AUTH_KEY_BYTES
    if len(frame) != expected:
        raise OCIProtocolError("stdin frame length does not match its header")
    payload = frame[header_size:header_size + size]
    return decode_request(payload), frame[-AUTH_KEY_BYTES:]


def _read_exact(fd: int, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        try:
            chunk = os.read(fd, min(remaining, 1024 * 1024))
        except InterruptedError:
            continue
        if not chunk:
            raise OCIProtocolError("stdin frame ended early")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _read_exact_into(fd: int, target: bytearray) -> None:
    """Read secret bytes without creating an immutable intermediate copy."""

    view = memoryview(target)
    offset = 0
    while offset < len(target):
        try:
            count = os.readv(fd, [view[offset:]])
        except InterruptedError:
            continue
        if count == 0:
            raise OCIProtocolError("stdin frame ended before the HMAC key")
        offset += count


def read_stdin_frame(fd: int = 0) -> tuple[OCILaunchRequest, bytearray]:
    """Read exactly one frame and require EOF after the fixed-size key."""

    magic = _read_exact(fd, len(FRAME_MAGIC))
    if magic != FRAME_MAGIC:
        raise OCIProtocolError("stdin frame magic/version mismatch")
    size = struct.unpack(">I", _read_exact(fd, 4))[0]
    if size > MAX_REQUEST_BYTES:
        raise OCIProtocolError("stdin request length exceeds its hard bound")
    payload = _read_exact(fd, size)
    key = bytearray(AUTH_KEY_BYTES)
    _read_exact_into(fd, key)
    while True:
        try:
            trailing = os.read(fd, 1)
            break
        except InterruptedError:
            continue
    if trailing:
        key[:] = b"\x00" * len(key)
        raise OCIProtocolError("stdin frame contains trailing bytes")
    return decode_request(payload), key


def env_is_safe(name: str, value: str) -> bool:
    """Shared profile validator for non-secret, profile-owned environment values."""

    if not isinstance(name, str) or _ENV_KEY.fullmatch(name) is None:
        return False
    if not isinstance(value, str) or len(value) > MAX_SMALL_STRING_CHARS:
        return False
    if any(char in value for char in ("\x00", "\n", "\r")):
        return False
    secret_words = ("HMAC", "AUTH_KEY", "SECRET", "PASSWORD", "ACCESS_TOKEN")
    return not any(word in name for word in secret_words)


def environment_fingerprint(environment: Mapping[str, str]) -> str:
    """Canonical fingerprint of the exact validator-declared environment subset."""

    if not isinstance(environment, Mapping):
        raise OCIProtocolError("environment policy must be a mapping")
    clean: dict[str, str] = {}
    for key, value in environment.items():
        if not env_is_safe(key, value):
            raise OCIProtocolError(f"unsafe environment policy entry {key!r}")
        clean[key] = value
    encoded = json.dumps(
        clean, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def topology_fingerprint(text: str) -> str:
    """Hash the score-relevant GPU interconnect matrix from ``nvidia-smi topo -m``.

    NIC enumeration, CPU-affinity formatting, ANSI underlining, and host GPU IDs
    differ between otherwise-equivalent four-GPU slices.  They must not split a
    serving bracket.  The GPU-to-GPU transport matrix (NVLink/PCIe class) is the
    load-bearing topology, so canonicalize exactly that square submatrix and
    relabel devices by visible rank.
    """

    if not isinstance(text, str) or len(text) > 1_000_000 or "\x00" in text:
        raise OCIProtocolError("GPU topology text is invalid or oversized")
    clean = re.sub(r"\x1b\[[0-9;]*m", "", text)
    lines = [
        line.rstrip()
        for line in clean.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        if line.strip()
    ]
    header: list[str] | None = None
    for line in lines:
        cells = line.split()
        start = 0
        gpu_cells: list[str] = []
        for cell in cells[start:]:
            if re.fullmatch(r"GPU[0-9]+", cell) is None:
                break
            gpu_cells.append(cell)
        if gpu_cells:
            header = gpu_cells
            break
    if not header:
        raise OCIProtocolError("GPU topology table header is missing")
    rows: dict[str, list[str]] = {}
    for line in lines:
        cells = line.split()
        if not cells or cells[0] not in header:
            continue
        if len(cells) > 1 and re.fullmatch(r"GPU[0-9]+", cells[1]):
            continue  # header row begins with GPU0 after whitespace tokenization
        if len(cells) < 1 + len(header):
            raise OCIProtocolError("GPU topology row is truncated")
        links = cells[1:1 + len(header)]
        if any(re.fullmatch(r"X|NV[0-9]+|PIX|PXB|PHB|NODE|SYS", link) is None
               for link in links):
            raise OCIProtocolError("GPU topology row contains an unknown link class")
        rows[cells[0]] = links
    if set(rows) != set(header):
        raise OCIProtocolError("GPU topology matrix is incomplete")
    matrix = [rows[label] for label in header]
    for index, row in enumerate(matrix):
        if row[index] != "X":
            raise OCIProtocolError("GPU topology matrix diagonal is malformed")
    encoded = json.dumps(
        {"schema": "optima-gpu-topology-v1", "matrix": matrix},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
