import pytest

from optima.arenas import (
    MINIMAX_M3_B300_TP4_DECODE_V1,
    MINIMAX_M3_B300_TP4_LONGPREFILL_V1,
)
from optima.eval.external_quality import TeacherForcedPromptTrace, TeacherForcedTrace
from optima.eval.oci_session_protocol import (
    FRAME_HEADER_BYTES,
    MAX_BATCH_REQUEST_BYTES,
    MAX_BATCH_RESPONSE_BYTES,
    SessionProtocolError,
    decode_teacher_evidence_payload,
    encode_message,
    expected_teacher_payload_bytes,
    teacher_evidence_frame,
    teacher_request,
    validate_teacher_request,
)
from optima.eval.oci_session_worker import (
    CandidateTeacherInputError,
    _teacher_forced_traces,
)


SESSION = "1" * 32
REQUEST = "2" * 32
NONCE = "3" * 32
SEAL = "sha256:" + "4" * 64


def _trace():
    topk = (
        ((-0.1, 7, None), (-2.4, 8, None)),
        ((-0.2, 8, None), (-1.8, 7, None)),
    )
    return TeacherForcedTrace((-0.1, -0.2), topk)


def test_teacher_request_is_exact_bounded_and_three_way_bound():
    request = teacher_request(
        session_id=SESSION,
        request_id=REQUEST,
        nonce=NONCE,
        phase="timed",
        batch_index=1,
        sealed_rollout_sha256=SEAL,
        prompts=["ordinary prompt", "second prompt"],
        baseline_ids=[[7, 8], [7, 8]],
        candidate_ids=[[8, 8], [8, 8]],
        stock_control_ids=[[7, 8], [7, 8]],
        expected_count=2,
        expected_tokens=2,
    )
    decoded = validate_teacher_request(
        request, expected_count=2, expected_tokens=2
    )
    assert decoded[:6] == (SESSION, REQUEST, NONCE, "timed", 1, SEAL)
    broken = dict(request)
    broken["candidate_ids"] = [[8], [8, 8]]
    with pytest.raises(SessionProtocolError, match="token IDs"):
        validate_teacher_request(broken, expected_count=2, expected_tokens=2)


def test_teacher_binary_evidence_roundtrip_binds_seal_and_exact_size():
    trace = _trace()
    prompt = TeacherForcedPromptTrace(
        prompt_token_count=3,
        prompt_token_sha256="5" * 64,
        baseline=trace,
        candidate=trace,
        stock_control=trace,
    )
    frame = teacher_evidence_frame(
        (prompt, prompt),
        session_id=SESSION,
        request_id=REQUEST,
        nonce=NONCE,
        phase="warmup",
        batch_index=0,
        sealed_rollout_sha256=SEAL,
        token_count=2,
        top_logprobs_num=2,
    )
    payload = frame[FRAME_HEADER_BYTES:]
    assert len(payload) == expected_teacher_payload_bytes(
        prompt_count=2, token_count=2, top_logprobs_num=2
    )
    decoded = decode_teacher_evidence_payload(
        payload,
        session_id=SESSION,
        request_id=REQUEST,
        nonce=NONCE,
        phase="warmup",
        batch_index=0,
        sealed_rollout_sha256=SEAL,
        expected_prompts=2,
        token_count=2,
        top_logprobs_num=2,
    )
    assert len(decoded) == 2
    assert decoded[0].prompt_token_sha256 == prompt.prompt_token_sha256
    assert decoded[0].baseline.target_logprobs == pytest.approx(
        prompt.baseline.target_logprobs
    )
    assert decoded[0].candidate.trusted_topk[0][0][1] == 7
    with pytest.raises(SessionProtocolError, match="binding"):
        decode_teacher_evidence_payload(
            payload,
            session_id=SESSION,
            request_id=REQUEST,
            nonce=NONCE,
            phase="warmup",
            batch_index=0,
            sealed_rollout_sha256="sha256:" + "6" * 64,
            expected_prompts=2,
            token_count=2,
            top_logprobs_num=2,
        )


def test_worker_uses_prefill_only_input_ids_and_cross_checks_target_sentinel():
    class Tokenizer:
        def __len__(self):
            return 128

        def encode(self, prompt):
            return [1, 2]

    class Manager:
        tokenizer = Tokenizer()

    class Engine:
        tokenizer_manager = Manager()

        def __init__(self):
            self.calls = []

        def generate(self, **kwargs):
            self.calls.append(kwargs)
            outputs = []
            for full_ids, sentinel in zip(
                kwargs["input_ids"], kwargs["token_ids_logprob"], strict=True
            ):
                targets = full_ids[-2:]
                target_rows = [(-0.1, token, None) for token in targets]
                topk = [
                    [(-0.1, token, None), (-3.0, 99, None)]
                    for token in targets
                ]
                sentinel_rows = [
                    [(-0.1 if token == sentinel[0] else -4.0, sentinel[0], None)]
                    for token in targets
                ]
                outputs.append({
                    "meta_info": {
                        "input_token_logprobs": target_rows,
                        "input_top_logprobs": topk,
                        "input_token_ids_logprobs": sentinel_rows,
                    }
                })
            return outputs

    engine = Engine()
    sources = {
        "baseline": [[7, 8], [7, 8]],
        "candidate": [[8, 8], [8, 8]],
        "stock_control": [[7, 8], [7, 8]],
    }
    traces = _teacher_forced_traces(
        engine, ["a", "b"], sources, top_logprobs_num=2
    )
    assert len(traces) == 2
    assert len(engine.calls) == 3
    assert all(call["sampling_params"]["max_new_tokens"] == 0 for call in engine.calls)
    assert all(call["logprob_start_len"] == [1, 1] for call in engine.calls)
    assert engine.calls[1]["input_ids"][0] == [1, 2, 8, 8]
    assert engine.calls[1]["token_ids_logprob"][0] == [8]

    broken = dict(sources)
    broken["candidate"] = [[128, 8], [8, 8]]
    with pytest.raises(CandidateTeacherInputError, match="out-of-vocabulary"):
        _teacher_forced_traces(
            engine, ["a", "b"], broken, top_logprobs_num=2
        )


@pytest.mark.parametrize(
    "arena",
    (MINIMAX_M3_B300_TP4_DECODE_V1, MINIMAX_M3_B300_TP4_LONGPREFILL_V1),
)
def test_registered_teacher_frames_fit_fixed_controller_bounds(arena):
    count = arena.fidelity.teacher_forced_policy.clusters_per_batch
    tokens = arena.workload.max_new_tokens
    prompt_chars = arena.workload.input_len or 256
    request = teacher_request(
        session_id=SESSION,
        request_id=REQUEST,
        nonce=NONCE,
        phase="timed",
        batch_index=0,
        sealed_rollout_sha256=SEAL,
        prompts=["p" * prompt_chars for _ in range(count)],
        baseline_ids=[[1] * tokens for _ in range(count)],
        candidate_ids=[[1] * tokens for _ in range(count)],
        stock_control_ids=[[1] * tokens for _ in range(count)],
        expected_count=count,
        expected_tokens=tokens,
    )
    assert len(encode_message(request, max_bytes=MAX_BATCH_REQUEST_BYTES)) < (
        MAX_BATCH_REQUEST_BYTES
    )
    assert expected_teacher_payload_bytes(
        prompt_count=count,
        token_count=tokens,
        top_logprobs_num=arena.workload.top_logprobs,
    ) < MAX_BATCH_RESPONSE_BYTES
