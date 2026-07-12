from __future__ import annotations

import ast
import hashlib
import os
import struct
import time
from dataclasses import fields, replace
from pathlib import Path

import pytest

import optima.eval.oci_reference_session as reference_session
from optima.eval.oci_outer_session import (
    OuterSessionProtocolError,
    OuterSessionWorkerError,
)
from optima.eval.oci_reference_session import (
    AttachedReferenceTransport,
    ReferenceSessionError,
    ReferenceSessionEvidence,
    ReferenceSessionPlan,
    run_reference_session,
)
from optima.eval.oci_session_protocol import (
    MAX_CONTROL_BYTES,
    EngineSessionConfig,
    RuntimePreflightFacts,
    error_message,
    frame_message,
    parse_frame_bytes,
    preflight_message,
    ready_message,
    validate_init,
    validate_preflight_accept,
)
from optima.eval.qualification import ReferenceManifest
from optima.eval.reference_protocol import (
    EVIDENCE_MAGIC,
    ReferenceEvidence,
    ReferencePromptEvidence,
    ReferencePromptInput,
    ReferenceRequest,
    ReferenceRoleEvidence,
    ReferenceRoleInput,
    ReferenceTokenEvidence,
    decode_reference_request,
    encode_reference_evidence,
    request_sha256,
)
from optima.stack_identity import canonical_digest
from optima.stack_manifest import EvaluationStackManifest, ProposalContributionRef


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _stack(*, entries=()) -> EvaluationStackManifest:
    catalog = {
        "schema_version": 1,
        "policy_version": "target-catalog.v1",
        "targets": [{"target_id": "activation.silu_and_mul", "marker": "base"}],
        "composition_rules": [],
    }
    return EvaluationStackManifest(
        runtime_digest=_digest("runtime"),
        base_engine_digest=_digest("base"),
        arena_digest=_digest("arena"),
        catalog_snapshot=catalog,
        catalog_digest=canonical_digest("optima.target-catalog", catalog),
        entries=entries,
    )


def _config() -> EngineSessionConfig:
    return EngineSessionConfig(
        model_path="/optima/input/model",
        dtype="bfloat16",
        deterministic=False,
        attention_backend="flashinfer",
        disable_cuda_graph=False,
        mem_fraction_static=0.82,
        log_level="error",
        max_running_requests=64,
        tp_size=1,
        moe_runner_backend="flashinfer_trtllm",
        disable_custom_all_reduce=False,
        engine_kwargs={},
    )


def _reference(stack: EvaluationStackManifest, config: EngineSessionConfig) -> ReferenceManifest:
    return ReferenceManifest(
        stack.digest,
        _digest("tree"),
        _digest("launch"),
        stack.runtime_digest,
        stack.base_engine_digest,
        stack.arena_digest,
        stack.catalog_digest,
        _digest("controller"),
        _digest("worker"),
        _digest("model-revision"),
        _digest("model-manifest"),
        _digest("model-content"),
        _digest("hardware"),
        _digest("workload"),
        _digest("tokenizer"),
        _digest("corpus"),
        _digest("judge"),
        _digest("selection"),
    )


def _facts(reference: ReferenceManifest, config: EngineSessionConfig) -> RuntimePreflightFacts:
    return RuntimePreflightFacts(
        launch_digest=reference.pristine_launch_digest,
        runtime_digest=reference.runtime_digest,
        stack_digest=reference.pristine_stack_digest,
        tree_digest=reference.pristine_tree_digest,
        engine_config_digest=config.digest,
        worker_distribution_digest=reference.worker_distribution_digest,
        model_revision_digest=reference.model_revision_digest,
        model_manifest_digest=reference.model_manifest_digest,
        model_content_digest=reference.model_content_digest,
        sglang_version="0.0.0.dev1+g56e290315",
        gpu_architectures=("sm120",),
        topology_digest=_digest("topology"),
        loopback_only=True,
        read_only_inputs=True,
        private_writable_cache=True,
    )


def _request(
    reference: ReferenceManifest,
    *,
    session_id: str,
    plan_digest: str,
    index: int,
) -> ReferenceRequest:
    roles = tuple(
        ReferenceRoleInput(
            (10 + role, 20 + role),
            ((10, 11), (20, 21)),
        )
        for role in range(3)
    )
    prompt = ReferencePromptInput(_digest(f"prompt-{index}"), f"prompt {index}", roles)
    return ReferenceRequest(
        session_id,
        reference.pristine_launch_digest,
        plan_digest,
        f"{10 + index:032x}",
        f"{20 + index:032x}",
        index,
        2,
        2,
        (prompt,),
    )


def _raw(request: ReferenceRequest, *, request_digest: str | None = None) -> ReferenceEvidence:
    roles = tuple(
        ReferenceRoleEvidence(
            tuple(
                ReferenceTokenEvidence(-0.25 - position, 100 + position, (-0.5, -1.5))
                for position in range(request.tokens_per_prompt)
            )
        )
        for _role in range(3)
    )
    prompts = tuple(
        ReferencePromptEvidence(row.prompt_digest, 3, _digest("prompt-tokens"), roles)
        for row in request.prompts
    )
    return ReferenceEvidence(
        request.session_id,
        request.launch_digest,
        request.plan_digest,
        request_digest or request_sha256(request),
        request.request_id,
        request.nonce,
        request.request_index,
        32000,
        prompts,
    )


def _plan() -> ReferenceSessionPlan:
    stack = _stack()
    config = _config()
    reference = _reference(stack, config)
    request_plan = _digest("request-plan")
    session = "1" * 32
    requests = tuple(
        _request(reference, session_id=session, plan_digest=request_plan, index=index)
        for index in range(2)
    )
    return ReferenceSessionPlan(
        reference,
        stack,
        config.digest,
        config,
        _facts(reference, config),
        request_plan,
        requests,
    )


class _Clock:
    def __init__(self) -> None:
        self.value = 100.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


class _FakeTransport:
    def __init__(self, plan: ReferenceSessionPlan, clock: _Clock) -> None:
        self.plan = plan
        self.clock = clock
        self.control_reads = 0
        self.accepted = False
        self.requests: list[ReferenceRequest] = []
        self.started = self.finalized = self.aborted = False
        self.pending = False

    def start(self) -> None:
        self.started = True

    def has_pending_output(self) -> bool:
        return self.pending

    def write_frame(self, frame: bytes, *, deadline: float) -> None:
        del deadline
        if frame[:4] == b"ORQ1":
            self.requests.append(decode_reference_request(frame))
        else:
            message = parse_frame_bytes(frame, max_bytes=1 << 20)
            if message["type"] == "init":
                session, launch, _ = validate_init(message)
                assert (session, launch) == (
                    self.plan.requests[0].session_id,
                    self.plan.reference.pristine_launch_digest,
                )
            else:
                validate_preflight_accept(
                    message,
                    session_id=self.plan.requests[0].session_id,
                    launch_digest=self.plan.reference.pristine_launch_digest,
                    expected_facts_digest=self.plan.expected_preflight.digest,
                )
                self.accepted = True
        self.clock.advance(0.25)

    def read_control(self, *, max_bytes: int, deadline: float) -> dict:
        del max_bytes, deadline
        self.clock.advance(0.25)
        self.control_reads += 1
        if self.control_reads == 1:
            return preflight_message(
                session_id=self.plan.requests[0].session_id,
                launch_digest=self.plan.reference.pristine_launch_digest,
                facts=self.plan.expected_preflight,
            )
        assert self.accepted
        return ready_message(
            session_id=self.plan.requests[0].session_id,
            launch_digest=self.plan.reference.pristine_launch_digest,
        )

    def read_reference_evidence(
        self, request: ReferenceRequest, *, deadline: float
    ) -> ReferenceEvidence:
        del deadline
        assert request == self.requests[-1]
        self.clock.advance(1.0)
        return _raw(request)

    def finalize(self) -> None:
        self.finalized = True
        self.clock.advance(0.25)

    def abort(self) -> None:
        self.aborted = True


def test_plan_closes_empty_stack_identity_and_ordered_cohort() -> None:
    plan = _plan()
    assert len(plan.requests) == 2
    assert plan.digest != plan.request_plan_digest

    proposal = ProposalContributionRef(
        "activation.silu_and_mul",
        canonical_digest(
            "optima.target-spec",
            {"target_id": "activation.silu_and_mul", "marker": "base"},
        ),
        _digest("artifact"),
        _digest("payload"),
        _digest("attribution"),
    )
    with pytest.raises(ReferenceSessionError, match="contributions"):
        replace(plan, pristine_stack=_stack(entries=(("activation.silu_and_mul", proposal),)))
    with pytest.raises(ReferenceSessionError, match="indices"):
        replace(plan, requests=(replace(plan.requests[0], request_index=1),))
    with pytest.raises(ReferenceSessionError, match="cohort binding"):
        replace(plan, requests=(replace(plan.requests[0], plan_digest=_digest("wrong")),))


def test_one_session_returns_raw_evidence_for_ordered_multi_candidate_cohort() -> None:
    plan = _plan()
    clock = _Clock()
    transport = _FakeTransport(plan, clock)
    result = run_reference_session(
        plan,
        transport=transport,
        deadline=200.0,
        init_timeout_s=10.0,
        batch_timeout_s=10.0,
        clock=clock,
    )

    assert type(result) is ReferenceSessionEvidence
    assert transport.started and transport.finalized and not transport.aborted
    assert transport.requests == list(plan.requests)
    assert tuple(row.request_sha256 for row in result.exchanges) == tuple(
        request_sha256(row) for row in plan.requests
    )
    assert all(row.evidence_frame_sha256 for row in result.exchanges)
    assert tuple(row.evidence for row in result.exchanges) == tuple(
        _raw(row) for row in plan.requests
    )
    assert result.reference_manifest_digest == plan.reference.digest
    assert result.session_plan_digest == plan.digest
    changed = replace(
        result.exchanges[0].evidence.prompts[0].roles[0].tokens[0],
        target_logprob=-0.75,
    )
    changed_role = replace(
        result.exchanges[0].evidence.prompts[0].roles[0],
        tokens=(changed, *result.exchanges[0].evidence.prompts[0].roles[0].tokens[1:]),
    )
    changed_prompt = replace(
        result.exchanges[0].evidence.prompts[0],
        roles=(changed_role, *result.exchanges[0].evidence.prompts[0].roles[1:]),
    )
    changed_evidence = replace(
        result.exchanges[0].evidence,
        prompts=(changed_prompt, *result.exchanges[0].evidence.prompts[1:]),
    )
    with pytest.raises(ReferenceSessionError, match="malformed"):
        replace(result.exchanges[0], evidence=changed_evidence)
    for forbidden in ("pass", "score", "crown", "verdict", "quality"):
        assert all(forbidden not in field.name for field in fields(type(result)))


class _PipeClient:
    def __init__(self) -> None:
        request_read, request_write = os.pipe()
        response_read, response_write = os.pipe()
        self.stdin = os.fdopen(request_write, "wb", buffering=0)
        self.stdout = os.fdopen(response_read, "rb", buffering=0)
        self.request_read = request_read
        self.response_write = response_write
        self.closed = self.finalized = self.aborted = False

    def finalize(self) -> None:
        self.finalized = self.closed = True

    def abort(self) -> None:
        self.aborted = self.closed = True

    def close(self) -> None:
        for stream in (self.stdin, self.stdout):
            if not stream.closed:
                stream.close()
        for fd in (self.request_read, self.response_write):
            try:
                os.close(fd)
            except OSError:
                pass


class _PipeManager:
    def __init__(self, client: _PipeClient) -> None:
        self.client = client

    def spawn_attached(self, _lease, _argv):
        return self.client


def _attached() -> tuple[AttachedReferenceTransport, _PipeClient]:
    client = _PipeClient()
    transport = AttachedReferenceTransport(
        _PipeManager(client), object(), ("/usr/bin/docker", "run")  # type: ignore[arg-type]
    )
    transport.start()
    return transport, client


def test_attached_reference_transport_enforces_exact_ore1_and_request_sha() -> None:
    request = _plan().requests[0]
    transport, client = _attached()
    try:
        frame = encode_reference_evidence(_raw(request), request)
        os.write(client.response_write, frame)
        assert transport.read_reference_evidence(
            request, deadline=time.monotonic() + 1
        ) == _raw(request)
    finally:
        transport.abort()
        client.close()

    transport, client = _attached()
    try:
        os.write(client.response_write, EVIDENCE_MAGIC + struct.pack(">I", 1) + b"x")
        with pytest.raises(OuterSessionProtocolError, match="exact size"):
            transport.read_reference_evidence(request, deadline=time.monotonic() + 1)
    finally:
        transport.abort()
        client.close()

    transport, client = _attached()
    try:
        error = error_message(
            session_id=request.session_id,
            launch_digest=request.launch_digest,
            stage="reference",
            error=RuntimeError("teacher failed"),
        )
        os.write(client.response_write, frame_message(error, max_bytes=MAX_CONTROL_BYTES))
        with pytest.raises(OuterSessionWorkerError, match="teacher failed"):
            transport.read_reference_evidence(request, deadline=time.monotonic() + 1)
    finally:
        transport.abort()
        client.close()


def test_reference_authority_import_closure_stays_data_only() -> None:
    source = Path(reference_session.__file__).read_text(encoding="utf-8")
    imported: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    for forbidden in (
        "optima.eval._launch",
        "optima.audit",
        "optima.eval.throughput_kl",
        "optima.eval.capability",
        "torch",
        "sglang",
        "bittensor",
    ):
        assert forbidden not in imported
