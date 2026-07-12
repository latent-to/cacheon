import dataclasses
import math
import struct

import pytest

from optima.eval.reference_protocol import (
    EVIDENCE_MAGIC,
    FRAME_HEADER_BYTES,
    REQUEST_MAGIC,
    ReferenceEvidence,
    ReferenceFaultCode,
    ReferencePromptEvidence,
    ReferencePromptInput,
    ReferenceProtocolError,
    ReferenceRequest,
    ReferenceRoleEvidence,
    ReferenceRoleInput,
    ReferenceTokenEvidence,
    decode_reference_evidence,
    decode_reference_request,
    encode_reference_evidence,
    encode_reference_request,
    expected_evidence_payload_bytes,
    request_sha256,
)


SESSION = "1" * 32
LAUNCH = "2" * 64
PLAN = "3" * 64
REQUEST = "4" * 32
NONCE = "5" * 32


def _role(offset=0):
    return ReferenceRoleInput(
        (7 + offset, 8 + offset),
        ((7 + offset, 9 + offset), (8 + offset, 10 + offset)),
    )


def _request():
    prompts = (
        ReferencePromptInput("6" * 64, "first prompt", (_role(), _role(10), _role(20))),
        ReferencePromptInput("7" * 64, "second λ", (_role(30), _role(40), _role(50))),
    )
    return ReferenceRequest(SESSION, LAUNCH, PLAN, REQUEST, NONCE, 7, 2, 2, prompts)


def _token(offset=0):
    return ReferenceTokenEvidence(-0.1 - offset / 100, 7 + offset, (-0.2, -2.0))


def _evidence(request=None):
    request = request or _request()
    prompts = []
    for index, prompt in enumerate(request.prompts):
        roles = tuple(
            ReferenceRoleEvidence(tuple(_token(index * 3 + role) for _ in range(2)))
            for role in range(3)
        )
        prompts.append(ReferencePromptEvidence(
            prompt.prompt_digest, 3 + index, f"{8 + index:x}" * 64, roles
        ))
    return ReferenceEvidence(
        request.session_id,
        request.launch_digest,
        request.plan_digest,
        request_sha256(request),
        request.request_id,
        request.nonce,
        request.request_index,
        128,
        tuple(prompts),
    )


def _replace_bytes(frame, start, value):
    result = bytearray(frame)
    result[start:start + len(value)] = value
    return bytes(result)


def test_request_roundtrip_is_exact_binary_and_canonical():
    request = _request()
    frame = encode_reference_request(request)
    assert frame[:4] == REQUEST_MAGIC
    assert struct.unpack(">I", frame[4:8])[0] == len(frame) - FRAME_HEADER_BYTES
    assert decode_reference_request(frame) == request
    assert encode_reference_request(decode_reference_request(frame)) == frame
    assert request_sha256(request) == __import__("hashlib").sha256(frame).hexdigest()
    assert b"first prompt" in frame and b"second \xce\xbb" in frame


@pytest.mark.parametrize("mutation,match", [
    (lambda frame: b"BAD!" + frame[4:], "magic"),
    (lambda frame: frame[:-1], "size"),
    (lambda frame: frame + b"x", "size"),
    (lambda frame: _replace_bytes(frame, 4, struct.pack(">I", 0)), "size"),
])
def test_request_rejects_bad_envelope(mutation, match):
    with pytest.raises(ReferenceProtocolError, match=match):
        decode_reference_request(mutation(encode_reference_request(_request())))


def test_request_rejects_noncanonical_padding_and_trailing_payload():
    frame = bytearray(encode_reference_request(_request()))
    # The two reserved bytes terminate the 128-byte request header.
    frame[FRAME_HEADER_BYTES + 126] = 1
    with pytest.raises(ReferenceProtocolError, match="canonical"):
        decode_reference_request(bytes(frame))

    frame = bytearray(encode_reference_request(_request()))
    frame.extend(b"x")
    frame[4:8] = struct.pack(">I", len(frame) - FRAME_HEADER_BYTES)
    with pytest.raises(ReferenceProtocolError, match="trailing"):
        decode_reference_request(bytes(frame))


@pytest.mark.parametrize("change,match", [
    ({"session_id": "0" * 32}, "session ID"),
    ({"request_id": SESSION}, "distinct"),
    ({"tokens_per_prompt": 3}, "geometry"),
    ({"support_width": 3}, "geometry"),
    ({"prompts": ()}, "coverage"),
    ({"request_index": -1}, "request index"),
])
def test_request_header_and_geometry_fail_closed(change, match):
    with pytest.raises(ReferenceProtocolError, match=match):
        dataclasses.replace(_request(), **change)


def test_request_rejects_prompt_reorder_and_bad_support_domains():
    request = _request()
    with pytest.raises(ReferenceProtocolError, match="digest-sorted"):
        dataclasses.replace(request, prompts=tuple(reversed(request.prompts)))
    with pytest.raises(ReferenceProtocolError, match="support rows"):
        ReferenceRoleInput((1, 2), ((4, 4), (2, 3)))
    with pytest.raises(ReferenceProtocolError, match="support rows"):
        ReferenceRoleInput((1, 2), ((4, 3), (2, 3)))
    with pytest.raises(ReferenceProtocolError, match="token ID"):
        ReferenceRoleInput((-1, 2), ((3, 4), (2, 3)))


def test_request_decoder_rechecks_geometry_before_accepting_hostile_counts():
    frame = bytearray(encode_reference_request(_request()))
    # token_count and support_width offsets inside the fixed request header.
    frame[FRAME_HEADER_BYTES + 120:FRAME_HEADER_BYTES + 124] = struct.pack(">I", 0)
    with pytest.raises(ReferenceProtocolError):
        decode_reference_request(bytes(frame))


def test_evidence_roundtrip_echoes_exact_request_hash_and_has_exact_size():
    request = _request()
    evidence = _evidence(request)
    frame = encode_reference_evidence(evidence, request)
    assert frame[:4] == EVIDENCE_MAGIC
    assert len(frame) - FRAME_HEADER_BYTES == expected_evidence_payload_bytes(request)
    decoded = decode_reference_evidence(frame, request)
    assert decoded == evidence
    assert decoded.request_sha256 == request_sha256(request)
    assert encode_reference_evidence(decoded, request) == frame


@pytest.mark.parametrize("field,value", [
    ("session_id", "a" * 32),
    ("launch_digest", "a" * 64),
    ("plan_digest", "a" * 64),
    ("request_sha256", "a" * 64),
    ("request_id", "a" * 32),
    ("nonce", "a" * 32),
    ("request_index", 8),
])
def test_evidence_rejects_every_replay_binding(field, value):
    request = _request()
    with pytest.raises(ReferenceProtocolError, match="another request"):
        encode_reference_evidence(dataclasses.replace(_evidence(request), **{field: value}), request)


def test_evidence_rejects_prompt_or_geometry_relabeling():
    request = _request()
    evidence = _evidence(request)
    prompt = evidence.prompts[0]
    with pytest.raises(ReferenceProtocolError, match="prompt order"):
        encode_reference_evidence(dataclasses.replace(
            evidence,
            prompts=(dataclasses.replace(prompt, prompt_digest="a" * 64), evidence.prompts[1]),
        ), request)
    short = ReferenceRoleEvidence((prompt.roles[0].tokens[0],))
    with pytest.raises(ReferenceProtocolError, match="geometry"):
        encode_reference_evidence(dataclasses.replace(
            evidence,
            prompts=(dataclasses.replace(prompt, roles=(short, *prompt.roles[1:])), evidence.prompts[1]),
        ), request)


def test_evidence_rejects_argmax_outside_vocab_and_invalid_logprobs():
    request = _request()
    evidence = _evidence(request)
    token = evidence.prompts[0].roles[0].tokens[0]
    bad = dataclasses.replace(token, true_argmax_token_id=evidence.vocab_size)
    role = dataclasses.replace(
        evidence.prompts[0].roles[0], tokens=(bad, *evidence.prompts[0].roles[0].tokens[1:])
    )
    prompt = dataclasses.replace(evidence.prompts[0], roles=(role, *evidence.prompts[0].roles[1:]))
    with pytest.raises(ReferenceProtocolError, match="vocabulary"):
        encode_reference_evidence(dataclasses.replace(
            evidence, prompts=(prompt, evidence.prompts[1])
        ), request)
    for value in (math.nan, math.inf, 1.0, -1_000_001.0):
        with pytest.raises(ReferenceProtocolError, match="logprob"):
            ReferenceTokenEvidence(value, 1, (-0.2, -0.3))


@pytest.mark.parametrize("mutation,match", [
    (lambda frame: b"BAD!" + frame[4:], "magic"),
    (lambda frame: frame[:-1], "size"),
    (lambda frame: frame + b"x", "size"),
])
def test_evidence_rejects_bad_envelope(mutation, match):
    request = _request()
    with pytest.raises(ReferenceProtocolError, match=match):
        decode_reference_evidence(mutation(encode_reference_evidence(_evidence(request), request)), request)


def test_evidence_rejects_request_transplant_even_with_equal_geometry():
    request = _request()
    frame = encode_reference_evidence(_evidence(request), request)
    other = dataclasses.replace(request, nonce="a" * 32)
    with pytest.raises(ReferenceProtocolError, match="another request"):
        decode_reference_evidence(frame, other)


def test_wire_records_contain_no_verdict_timing_or_hidden_authority_fields():
    names = set(ReferenceRequest.__dataclass_fields__) | set(ReferenceEvidence.__dataclass_fields__)
    assert not names & {
        "decision", "score", "passed", "elapsed", "hidden_tasks", "judge",
        "calibration", "miner", "t_session_digest",
    }


def test_fault_code_namespace_is_closed_and_contains_no_verdicts():
    assert {code.value for code in ReferenceFaultCode} == set(range(1, 7))
    assert all("PASS" not in code.name and "FAIL" not in code.name for code in ReferenceFaultCode)
