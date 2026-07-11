"""Hostile worker-result bytes never become trusted-parent code execution."""

from __future__ import annotations

import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path

import pytest

from optima import ipc
from optima.eval import _launch
from optima.eval.throughput_kl import ModeResult
from optima.verify import ShapeResult, VerifyResult


def _gadget_effect(marker: str) -> None:
    Path(marker).write_text("pickle gadget executed")


class _PickleGadget:
    def __init__(self, marker: str) -> None:
        self.marker = marker

    def __reduce__(self):
        return (_gadget_effect, (self.marker,))


def _return_pickle_gadget(marker: str):
    return _PickleGadget(marker)


@dataclass
class _UnlistedDataclass:
    value: int


def test_allowlisted_result_dataclasses_round_trip_with_exact_public_types():
    result = VerifyResult(
        slot="activation.silu_and_mul",
        dtype="float32",
        passed=False,
        shape_results=[
            ShapeResult(
                shape={"num_tokens": 3},
                dtype="float32",
                passed=False,
                max_abs_err=float("inf"),
                max_rel_err=float("nan"),
                pass_ratio=0.5,
                detail="diagnostic",
                metric="ratio",
                graph_replays=0,
                applicable=False,
            )
        ],
        graph_required=True,
        graph_verified=False,
        coverage_required=3,
    )

    decoded = ipc.loads(ipc.dumps(result))

    assert type(decoded) is VerifyResult
    assert type(decoded.shape_results[0]) is ShapeResult
    assert math.isinf(decoded.shape_results[0].max_abs_err)
    assert math.isnan(decoded.shape_results[0].max_rel_err)
    assert not decoded.shape_results[0].applicable
    assert decoded.coverage_required == 3


def test_mode_result_preserves_tuples_without_generic_object_reconstruction():
    result = ModeResult(
        tok_per_s=123.5,
        tok_per_s_samples=[122.0, 125.0],
        tokens=2,
        per_prompt=[([7, 8], [[(-0.25, 7, None)]])],
    )

    decoded = ipc.loads(ipc.dumps(result))

    assert type(decoded) is ModeResult
    assert isinstance(decoded.per_prompt[0], tuple)
    assert isinstance(decoded.per_prompt[0][1][0][0], tuple)
    assert decoded == result


def test_unlisted_dataclass_is_rejected_without_calling_user_protocols():
    with pytest.raises(ipc.WireError, match="not allowed"):
        ipc.dumps(_UnlistedDataclass(1))


def test_pickle_gadget_returned_by_child_cannot_execute_in_parent(tmp_path):
    marker = tmp_path / "executed"

    with pytest.raises(RuntimeError, match="serialization failed"):
        _launch.call_in_subprocess(
            _return_pickle_gadget, str(marker), timeout_s=10
        )

    assert not marker.exists()


def test_pickle_payload_is_never_deserialized(tmp_path):
    marker = tmp_path / "executed"
    hostile = pickle.dumps(_PickleGadget(str(marker)))

    with pytest.raises(ipc.WireError):
        ipc.loads(hostile)

    assert not marker.exists()


def test_payload_cannot_select_a_module_or_class_to_import():
    hostile = json.dumps({
        "schema": ipc.WIRE_SCHEMA,
        "payload": {
            "$": "dataclass",
            "name": "os.system",
            "fields": {"command": "touch /tmp/should-not-exist"},
        },
    }).encode()

    with pytest.raises(ipc.WireError, match="unknown .* tag"):
        ipc.loads(hostile)


def test_allowlisted_dataclass_requires_exact_fields():
    encoded = json.loads(ipc.dumps(ShapeResult(
        shape={}, dtype="float32", passed=True,
        max_abs_err=0.0, max_rel_err=0.0,
    )))
    del encoded["payload"]["fields"]["detail"]

    with pytest.raises(ipc.WireError, match="fields do not match schema"):
        ipc.loads(json.dumps(encoded).encode())


@pytest.mark.parametrize("payload", [
    b"",
    b"not json",
    b'[]',
    b'{"schema":"wrong","payload":null}',
    b'{"schema":"optima-worker-result-v1","payload":NaN}',
    b'{"schema":"optima-worker-result-v1","payload":{"$":"wat"}}',
])
def test_malformed_payloads_fail_closed(payload):
    with pytest.raises(ipc.WireError):
        ipc.loads(payload)


def test_duplicate_encoded_mapping_keys_fail_closed():
    payload = json.dumps({
        "schema": ipc.WIRE_SCHEMA,
        "payload": {"$": "dict", "items": [["x", 1], ["x", 2]]},
    }).encode()

    with pytest.raises(ipc.WireError, match="duplicate mapping key"):
        ipc.loads(payload)


def test_oversized_and_overdeep_payloads_fail_closed(tmp_path, monkeypatch):
    with pytest.raises(ipc.WireError, match="exceeds 32 bytes"):
        ipc.loads(b"x" * 33, max_bytes=32)

    value = None
    for _ in range(ipc.MAX_DEPTH + 2):
        value = [value]
    with pytest.raises(ipc.WireError, match="exceeds depth"):
        ipc.dumps(value)

    oversized = tmp_path / "oversized.json"
    oversized.write_bytes(b"x" * 65)
    monkeypatch.setattr(ipc, "MAX_WIRE_BYTES", 64)
    with pytest.raises(ipc.WireError, match="exceeds 64 bytes"):
        ipc.load_file(oversized)

    item_payload = ipc.dumps([1, 2, 3, 4])
    monkeypatch.setattr(ipc, "MAX_ITEMS", 4)
    with pytest.raises(ipc.WireError, match="exceeds 4 items"):
        ipc.loads(item_payload)


def test_collective_verdict_schema_rejects_wrong_field_types():
    encoded = json.loads(ipc.dumps(ipc.CollectiveVerdict(
        rank=0, passed=True, score=1.0, max_abs=0.0,
        detail="", metric="ratio", err=None, graph_replays=0,
    )))
    encoded["payload"]["fields"]["rank"] = "zero"

    with pytest.raises(ipc.WireError, match="rank must be"):
        ipc.loads(json.dumps(encoded).encode())


def test_authenticated_launch_result_rejects_tampering_wrong_key_and_replay():
    key = b"k" * 32
    nonce = b"n" * 16
    result = ipc.LaunchOutcome(value=17, error=None)
    encoded = ipc.authenticated_dumps(result, key=key, nonce=nonce)

    assert ipc.authenticated_loads(encoded, key=key, nonce=nonce) == result

    tampered = encoded[:-1] + bytes([encoded[-1] ^ 1])
    with pytest.raises(ipc.WireError, match="authentication failed"):
        ipc.authenticated_loads(tampered, key=key, nonce=nonce)
    with pytest.raises(ipc.WireError, match="authentication failed"):
        ipc.authenticated_loads(encoded, key=b"x" * 32, nonce=nonce)
    with pytest.raises(ipc.WireError, match="nonce mismatch"):
        ipc.authenticated_loads(encoded, key=key, nonce=b"z" * 16)


def test_authenticated_result_file_cannot_be_replaced_with_valid_unauthenticated_json(tmp_path):
    path = tmp_path / "launch.json"
    path.touch()
    key = b"s" * 32
    nonce = b"f" * 16
    ipc.dump_authenticated_file(
        path, ipc.LaunchOutcome(value=1, error=None), key=key, nonce=nonce
    )
    # Candidate-controlled code can write the shared file, but cannot mint its MAC.
    path.write_bytes(ipc.dumps(ipc.LaunchOutcome(value=999, error=None)))

    with pytest.raises(ipc.WireError, match="authentication header"):
        ipc.load_authenticated_file(path, key=key, nonce=nonce)
