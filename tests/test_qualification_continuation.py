from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from types import SimpleNamespace

import pytest

import cacheon.eval.native_artifact as native_artifact
import cacheon.eval.qualification_runner as runner
from cacheon.audit_gate import gate
from cacheon.eval.continuation_codec import (
    ContinuationCodec,
    ContinuationCodecError,
)
from cacheon.eval.device_state import (
    DeviceStateActiveReceipt,
    DeviceStateReceipt,
    DeviceStateSample,
)
from cacheon.eval.evidence_store import EvidenceArtifactRef
from cacheon.eval.native_artifact import NativeArtifactPublication
from cacheon.eval.oci_backend import (
    EngineExecutionEvidence,
    OCIEngineExecutor,
    PristineReferenceExecutionEvidence,
    runtime_identity_from_preflight,
)
from cacheon.eval.oci_outer_session import SessionExecutionEvidence
from cacheon.eval.oci_prebuild import OCIPrebuildResult
from cacheon.eval.oci_reference_session import (
    ReferenceExchangeEvidence,
    ReferenceSessionEvidence,
)
from cacheon.eval.qualification import QualificationDecision, SelectionEntropyReceipt
from cacheon.eval.qualification_continuation import (
    AuditContinuation,
    QualificationContinuationError,
    QualificationContinuationStore,
    QualityContinuation,
)
from cacheon.eval.reference_protocol import encode_reference_evidence, request_sha256
from cacheon.stack_identity import canonical_digest, canonical_json_bytes
from tests.test_marginal_runtime import (
    _batch_evidence,
    _binding_id,
    _case,
    _prepared,
)
from tests.test_oci_reference_session import (
    _config,
    _facts,
    _raw,
    _reference,
    _request,
    _stack,
)
from tests.test_qualification_runner import (
    _Harness,
    _install_resident_runner_path,
    _quiescence,
)


def _d(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


# --------------------------------------------------------------------------
# continuation codec
# --------------------------------------------------------------------------


class _Color(Enum):
    RED = "red"
    BLUE = "blue"


@dataclass(frozen=True)
class _Leaf:
    digest: str

    def __post_init__(self) -> None:
        if len(self.digest) != 64:
            raise ValueError("leaf digest must be 64 hex characters")


@dataclass(frozen=True)
class _Node:
    name: str
    count: int
    ratio: float
    flag: bool
    location: Path
    color: _Color
    leaf: _Leaf
    rows: tuple[_Leaf, ...]
    pair: tuple[str, int]
    maybe: _Leaf | None


def _node(*, maybe: _Leaf | None = None) -> _Node:
    return _Node(
        "node",
        3,
        1.5,
        True,
        Path("/evidence/root"),
        _Color.BLUE,
        _Leaf(_d("leaf")),
        (_Leaf(_d("row-0")), _Leaf(_d("row-1"))),
        ("pair", 7),
        maybe,
    )


def test_codec_round_trips_exact_types_and_reruns_validation() -> None:
    codec = ContinuationCodec((_Node,))
    for value in (_node(), _node(maybe=_Leaf(_d("maybe")))):
        encoded = codec.encode(value)
        # The wire form must be plain JSON.
        json.loads(json.dumps(encoded))
        assert codec.decode(encoded) == value

    # Reconstruction goes through __post_init__: a well-typed but invalid
    # field value must fail closed instead of resurrecting silently.
    tampered = codec.encode(_node())
    tampered["value"]["leaf"]["digest"] = "not-a-digest"
    with pytest.raises(ContinuationCodecError, match="construction validation"):
        codec.decode(tampered)


def test_codec_rejects_open_schemas_and_foreign_roots() -> None:
    @dataclass(frozen=True)
    class _Open:
        mapping: dict

    with pytest.raises(ContinuationCodecError, match="unsupported type"):
        ContinuationCodec((_Open,))
    with pytest.raises(ContinuationCodecError, match="not a dataclass or enum"):
        ContinuationCodec((object,))

    codec = ContinuationCodec((_Node,))
    with pytest.raises(ContinuationCodecError, match="not a registered codec root"):
        codec.encode(_Leaf(_d("leaf")))  # nested class, not a root
    with pytest.raises(ContinuationCodecError, match="not a registered codec root"):
        codec.decode({"type": "builtins.object", "value": {}})


def test_codec_rejects_shape_and_type_tampers() -> None:
    codec = ContinuationCodec((_Node,))
    base = codec.encode(_node())

    missing = json.loads(json.dumps(base))
    del missing["value"]["count"]
    with pytest.raises(ContinuationCodecError, match="fields are not closed"):
        codec.decode(missing)

    extra = json.loads(json.dumps(base))
    extra["value"]["injected"] = 1
    with pytest.raises(ContinuationCodecError, match="fields are not closed"):
        codec.decode(extra)

    wrong_scalar = json.loads(json.dumps(base))
    wrong_scalar["value"]["count"] = "3"
    with pytest.raises(ContinuationCodecError, match="not exactly int"):
        codec.decode(wrong_scalar)

    wrong_enum = json.loads(json.dumps(base))
    wrong_enum["value"]["color"] = "green"
    with pytest.raises(ContinuationCodecError):
        codec.decode(wrong_enum)

    wrong_arity = json.loads(json.dumps(base))
    wrong_arity["value"]["pair"] = ["pair"]
    with pytest.raises(ContinuationCodecError, match="tuple arity"):
        codec.decode(wrong_arity)


# --------------------------------------------------------------------------
# fully typed evidence builders (real classes end to end)
# --------------------------------------------------------------------------


def _publication(artifact_root: Path, build_spec_digest: str) -> NativeArtifactPublication:
    payload = native_artifact._identity_payload(build_spec_digest, (), ())
    return NativeArtifactPublication(
        artifact_root / build_spec_digest[:2] / build_spec_digest,
        build_spec_digest,
        native_artifact._publication_digest(payload),
        (),
        (),
    )


def _device_receipts(
    label: str, policy_digest: str
) -> tuple[DeviceStateReceipt, DeviceStateActiveReceipt, DeviceStateReceipt]:
    sample = DeviceStateSample(1.0, (), (), True, "idle")
    def receipt(sequence: int, phase: str, start: float) -> DeviceStateReceipt:
        return DeviceStateReceipt(
            "cacheon.device-state-receipt.v1",
            sequence,
            "launch-" + label,
            phase,
            (0,),
            _d("device-config"),
            policy_digest,
            start,
            start + 0.1,
            1,
            (sample,),
        )
    active = DeviceStateActiveReceipt(
        "cacheon.device-state-active-receipt.v1",
        2,
        "launch-" + label,
        "warmup",
        (0,),
        _d("device-config"),
        policy_digest,
        1.2,
        1.3,
        1,
        0,
        1,
        (sample,),
    )
    return (receipt(1, "pre", 1.0), active, receipt(3, "post", 1.4))


def _typed_execution(
    launch, binding, mount, plan, *, label: str, artifact_root: Path
) -> EngineExecutionEvidence:
    batches = tuple(
        _batch_evidence(plan, index, label=label)
        for index in range(len(plan.prompt_batches))
    )
    session = SessionExecutionEvidence(
        _binding_id("session:" + label),
        launch.digest,
        plan.expected_preflight,
        0.5,
        batches,
        plan.warmup_count,
        plan.conditioning_count,
        0.5,
        3.0,
        sum(row.token_numerator for row in batches[: plan.warmup_count + 1]),
        4.0,
    )
    receipt = binding.runtime_preflight_receipt
    native = binding.native_build_spec
    publication = _publication(artifact_root, native.digest)
    prebuild = OCIPrebuildResult(
        launch.digest,
        native.digest,
        publication,
        1.0,
        _d("security-argv:" + label),
    )
    return EngineExecutionEvidence(
        "cacheon.oci-engine-execution.v1",
        launch.digest,
        runtime_identity_from_preflight(receipt),
        receipt.sha256,
        mount.digest,
        launch.resource_policy_digest,
        prebuild,
        publication.publication_digest,
        _d("argv:" + label),
        (),
        _device_receipts(label, launch.hardware.device_policy_digest),
        session,
    )


class _TypedExecutor:
    """Executor double emitting fully typed EngineExecutionEvidence."""

    def __init__(self, artifact_root: Path) -> None:
        self.artifact_root = artifact_root
        self.calls = 0

    def execute(self, launch, binding, mount, plan, *, deadline):
        del deadline
        label = f"typed-{self.calls}"
        self.calls += 1
        return _typed_execution(
            launch, binding, mount, plan, label=label, artifact_root=self.artifact_root
        )


def _scope(
    tmp_path: Path, *, request: str = "request", source: str = "source"
):
    store = QualificationContinuationStore(tmp_path / "continuation")
    return store.scope(
        request_digest=_d(request),
        authority_digest=_d("qualification-authority"), source_digest=_d(source)
    )


def _final_ref(label: str) -> EvidenceArtifactRef:
    return EvidenceArtifactRef(
        runner.ATTEMPT_DOMAIN,
        _d(label),
        2,
        "application/json",
        runner.ATTEMPT_SCHEMA,
    )


# --------------------------------------------------------------------------
# continuation store
# --------------------------------------------------------------------------


def test_final_record_round_trip_is_idempotent_and_conflict_closed(
    tmp_path: Path,
) -> None:
    continuation = _scope(tmp_path)
    assert continuation.load_final() is None
    reference = _final_ref("final-artifact")
    continuation.record_final(reference)
    continuation.record_final(reference)  # identical re-record is a no-op
    assert continuation.load_final() == reference

    with pytest.raises(
        QualificationContinuationError, match="already exists with other content"
    ):
        continuation.record_final(_final_ref("another-final-artifact"))

    with pytest.raises(QualificationContinuationError, match="exact evidence reference"):
        continuation.record_final(SimpleNamespace(sha256=_d("untyped")))


def test_record_files_reject_tamper_symlink_and_foreign_identity(
    tmp_path: Path,
) -> None:
    continuation = _scope(tmp_path)
    continuation.record_final(_final_ref("final-artifact"))
    record_path = continuation.directory / "final.json"

    original = record_path.read_bytes()
    record_path.chmod(0o600)
    record_path.write_bytes(original.replace(b'"size":2', b'"size":3'))
    with pytest.raises(QualificationContinuationError):
        continuation.load_final()

    # A forged record that recomputes its own digest still names identities.
    forged = json.loads(original.decode())
    forged["source_digest"] = _d("another-source")
    body = {key: forged[key] for key in forged if key != "record_digest"}
    forged["record_digest"] = canonical_digest(
        "cacheon.eval.qualification-continuation-record.v2", body
    )
    record_path.write_bytes(canonical_json_bytes(forged))
    with pytest.raises(
        QualificationContinuationError, match="names another sealed identity"
    ):
        continuation.load_final()

    # The same bytes cannot be replayed under another sealed cohort either.
    foreign = _scope(tmp_path, source="another-source-entirely")
    (foreign.directory / "final.json").write_bytes(original)
    with pytest.raises(
        QualificationContinuationError, match="names another sealed identity"
    ):
        foreign.load_final()

    record_path.unlink()
    (continuation.directory / "target.json").write_bytes(original)
    record_path.symlink_to(continuation.directory / "target.json")
    with pytest.raises(QualificationContinuationError, match="regular file"):
        continuation.load_final()


def test_request_binding_reopens_exact_checkpoint_and_isolates_other_request(
    tmp_path: Path,
) -> None:
    first = _scope(tmp_path, request="request-one")
    reference = _final_ref("request-one-final")
    first.record_final(reference)

    reopened = _scope(tmp_path, request="request-one")
    second = _scope(tmp_path, request="request-two")
    assert reopened.directory == first.directory
    assert reopened.load_final() == reference
    assert second.directory != first.directory
    assert second.load_final() is None

    (second.directory / "final.json").write_bytes(
        (first.directory / "final.json").read_bytes()
    )
    with pytest.raises(
        QualificationContinuationError, match="names another sealed identity"
    ):
        second.load_final()

    store = QualificationContinuationStore(tmp_path / "missing-request")
    with pytest.raises(
        QualificationContinuationError, match="authenticated request digest"
    ):
        store.scope(
            authority_digest=_d("qualification-authority"),
            source_digest=_d("source"),
        )


def _audit_witness() -> tuple[str, object]:
    policy = runner.SlotAuditPolicy(
        f"{700:032x}", 250_000, 32, ("norm.rmsnorm",), 1
    )
    receipts = (
        runner.AuditReceiptFacts(
            "norm.rmsnorm", 32, 0, 0, 0, 1.0, 0.995, "allclose", 900, 0, 1
        ),
    )
    passed, detail = gate(
        [row.to_gate_dict() for row in receipts],
        min_calls=policy.minimum_calls,
        expected_slots=policy.expected_slots,
        expected_member_count=policy.expected_member_count,
    )
    assert passed
    witness = runner.AuditWitness(
        _d("delta-0"),
        _d("candidate-launch"),
        _d("audit-execution"),
        f"{900:032x}",
        _d("runtime-policy"),
        policy,
        receipts,
        QualificationDecision.PASS,
        detail,
    )
    return witness.selected_delta_digest, witness


def _pristine_reference_execution(
    engine_execution: EngineExecutionEvidence,
) -> PristineReferenceExecutionEvidence:
    stack = _stack()
    config = _config()
    reference = _reference(stack, config)
    session_id = "1" * 32
    plan_digest = _d("continuation-request-plan")
    request = _request(
        reference, session_id=session_id, plan_digest=plan_digest, index=0
    )
    raw = _raw(request)
    exchange = ReferenceExchangeEvidence(
        0,
        request,
        request_sha256(request),
        hashlib.sha256(encode_reference_evidence(raw, request)).hexdigest(),
        5.0,
        5.5,
        raw,
    )
    session = ReferenceSessionEvidence(
        "cacheon.pristine-reference-session.v1",
        session_id,
        reference.pristine_launch_digest,
        reference.digest,
        _d("continuation-session-plan"),
        plan_digest,
        _facts(reference, config),
        4.5,
        (exchange,),
        6.0,
    )
    return PristineReferenceExecutionEvidence(
        "cacheon.oci-pristine-reference.v1",
        reference.pristine_launch_digest,
        engine_execution.runtime_identity,
        engine_execution.runtime_preflight_receipt_sha256,
        engine_execution.arena_model_receipt_digest,
        engine_execution.resource_policy_digest,
        engine_execution.prebuild,
        engine_execution.native_publication_digest,
        engine_execution.runtime_argv_sha256,
        (),
        (
            engine_execution.device_receipts[0],
            engine_execution.device_receipts[2],
        ),
        session,
    )


def test_quality_record_round_trips_real_evidence(tmp_path: Path) -> None:
    case = _case(tmp_path / "runtime")
    prepared = _prepared(case)
    baseline_execution = _TypedExecutor(tmp_path / "artifacts").execute(
        prepared.baseline_launch,
        case.baseline_binding.launch_binding,
        case.mount,
        prepared.baseline_session_plan,
        deadline=999.0,
    )
    reference_execution = _pristine_reference_execution(baseline_execution)
    entropy = SelectionEntropyReceipt(
        _d("entropy-source"), _d("commitment"), _d("entropy"), _d("entropy-authority")
    )
    value = QualityContinuation(
        teardown_before=_quiescence(1, 3.0),
        entropy=entropy,
        entropy_observed=3.5,
        requests=tuple(
            row.request for row in reference_execution.session.exchanges
        ),
        reference_execution=reference_execution,
        teardown_after=_quiescence(2, 6.0),
        t_nonce="pending",
        t_operation_digest=_d("t-operation"),
    )

    continuation = _scope(tmp_path)
    assert continuation.load_quality() is None
    value = replace(
        value,
        t_nonce=continuation.arm_evaluator("t", value.t_operation_digest),
    )
    continuation.record_quality(value)
    reopened = continuation.load_quality()
    assert reopened == value
    assert type(reopened.reference_execution) is PristineReferenceExecutionEvidence
    assert reopened.reference_execution.session.digest == (
        reference_execution.session.digest
    )

    with pytest.raises(QualificationContinuationError, match="exact checkpoint"):
        continuation.record_quality(SimpleNamespace())


def test_evaluator_claim_is_unique_and_audit_completion_reopens(
    tmp_path: Path,
) -> None:
    operation = _d("audit-operation")
    continuation = _scope(tmp_path / "complete")
    nonce = continuation.arm_evaluator("audit", operation)
    audit = AuditContinuation(
        nonce, operation, (_audit_witness(),), 2.1, 2.5, 2.4
    )
    continuation.record_audit(audit)
    assert continuation.load_audit(operation) == audit

    ambiguous = _scope(tmp_path / "ambiguous")
    ambiguous.arm_evaluator("audit", operation)
    assert ambiguous.load_audit(operation) is None
    with pytest.raises(QualificationContinuationError, match="other content"):
        ambiguous.arm_evaluator("audit", operation)


# --------------------------------------------------------------------------
# at-most-once across crashes (callback counts through the runner)
# --------------------------------------------------------------------------


class _MemoryContinuation:
    """In-memory continuation double with the store's exact surface."""

    def __init__(
        self,
        *,
        authority_digest: str | None = None,
        source_digest: str | None = None,
    ) -> None:
        self.authority_digest = authority_digest or _d("qualification-authority")
        self.source_digest = source_digest or _d("source")
        self.records: dict[str, object] = {}

    def load_final(self):
        return self.records.get("final")

    def load_quality(self):
        return self.records.get("quality")

    def load_resident_speed(self):
        return self.records.get("speed")

    def arm_evaluator(self, stage, operation_digest):
        key = f"{stage}_armed"
        if key in self.records:
            raise QualificationContinuationError(
                f"continuation {key} record already exists with other content"
            )
        nonce = f"{stage}-nonce"
        self.records[key] = (nonce, operation_digest)
        return nonce

    def load_audit(self, operation_digest):
        value = self.records.get("audit")
        if value is not None and value.operation_digest != operation_digest:
            raise QualificationContinuationError("audit operation differs")
        return value

    def record_audit(self, value):
        assert self.records["audit_armed"] == (value.nonce, value.operation_digest)
        self.records["audit"] = value

    def load_marginal_speed(self, prepared):
        del prepared
        return self.records.get("speed")

    def record_resident_speed(self, crossover):
        self.records["speed"] = crossover

    def record_marginal_speed(self, lifecycle):
        self.records["speed"] = lifecycle

    def record_quality(self, value):
        assert self.records["t_armed"] == (value.t_nonce, value.t_operation_digest)
        self.records["quality"] = value

    def record_final(self, reference):
        self.records["final"] = reference


def _fresh_quiescence(monkeypatch):
    """Install pre/post-T receipts; returns a reset for each retried attempt."""

    state = {"count": 0}

    def prove(_executor):
        state["count"] += 1
        return _quiescence(state["count"], 3.0 if state["count"] == 1 else 6.0)

    monkeypatch.setattr(OCIEngineExecutor, "prove_quiescent", prove)
    return lambda: state.update(count=0)


def _run(
    harness: _Harness,
    continuation: _MemoryContinuation,
    resident_baseline_executor=None,
):
    ids = iter(f"{index + 1:032x}" for index in range(64))
    return runner.run_causal_qualification(
        harness.value,
        executor=harness.executor,
        resident_baseline_executor=resident_baseline_executor,
        entropy_provider=lambda _commitment, _teardown: harness.entropy,
        hidden_judge=harness.hidden_judge,
        deadline=100.0,
        id_factory=lambda: next(ids),
        continuation=continuation,
    )


def _pass_harness(monkeypatch) -> _Harness:
    harness = _Harness(
        monkeypatch,
        graph=(QualificationDecision.PASS,),
        speed=(QualificationDecision.PASS,),
        quality=(QualificationDecision.PASS,),
    )
    monkeypatch.setattr(runner, "QualificationContinuation", _MemoryContinuation)
    return harness


def _crash_after_completion(producer, message):
    def wrapped(*args, completion_sink=None, **kwargs):
        assert completion_sink is not None

        def crash(*values):
            completion_sink(*values)
            raise RuntimeError(message)

        return producer(*args, completion_sink=crash, **kwargs)

    return wrapped


def _without_completion(producer):
    def wrapped(*args, completion_sink=None, **kwargs):
        assert completion_sink is not None
        return producer(*args, completion_sink=None, **kwargs)

    return wrapped


def _resident_pass_harness(monkeypatch):
    """Resident continuation scaffolding: the only mode with durable writers."""

    harness = _Harness(
        monkeypatch,
        graph=(QualificationDecision.PASS,),
        speed=(QualificationDecision.PASS,),
        quality=(QualificationDecision.PASS,),
    )
    baseline, _stage_reference, exits = _install_resident_runner_path(
        monkeypatch,
        harness,
        speed_decision=QualificationDecision.PASS,
        escalated=False,
    )
    monkeypatch.setattr(runner, "QualificationContinuation", _MemoryContinuation)
    clock = iter(3.05 + index * 0.05 for index in range(64))
    harness.executor.manager.clock = lambda: next(clock)
    return harness, baseline, exits


def _run_resident(harness, baseline, continuation, *, entropy_provider=None):
    ids = iter(f"{index + 1:032x}" for index in range(64))
    return runner.run_causal_qualification(
        harness.value,
        executor=harness.executor,
        resident_baseline_executor=baseline,
        entropy_provider=entropy_provider or harness.entropy_provider,
        hidden_judge=harness.hidden_judge,
        deadline=100.0,
        id_factory=lambda: next(ids),
        continuation=continuation,
    )


def test_pristine_t_completion_survives_a_crash_before_producer_return(
    monkeypatch,
) -> None:
    harness, baseline, _exits = _resident_pass_harness(monkeypatch)
    continuation = _MemoryContinuation()
    producer = OCIEngineExecutor.execute_reference
    monkeypatch.setattr(
        OCIEngineExecutor,
        "execute_reference",
        _crash_after_completion(producer, "crash after durable pristine T"),
    )

    with pytest.raises(RuntimeError, match="after durable pristine T"):
        _run_resident(harness, baseline, continuation)
    assert harness.calls.count("resident.speed") == 1
    assert harness.calls.count("audit") == 1
    assert harness.reference_calls == 1
    assert "quality" in continuation.records

    assert _run_resident(harness, baseline, continuation) == harness.attempt_reference
    assert harness.calls.count("resident.speed") == 1
    assert harness.reference_calls == 1
    assert harness.calls.count("audit") == 1
    assert continuation.records["final"] == harness.attempt_reference


def test_quality_executes_at_most_once_across_a_finalization_crash(
    monkeypatch,
) -> None:
    harness, baseline, _exits = _resident_pass_harness(monkeypatch)
    reset_quiescence = _fresh_quiescence(monkeypatch)
    continuation = _MemoryContinuation()

    # Downstream grading re-runs on resume, so give it a stateless scorer
    # instead of the harness's one-shot iterator.
    from tests.test_qualification_runner import _quality_verdict

    monkeypatch.setattr(
        runner,
        "score_reference_quality",
        lambda *_args, **_kwargs: _quality_verdict(
            QualificationDecision.PASS, 0, harness.calibration.digest
        ),
    )

    publish = runner.publish_causal_qualification
    state = {"crashes": 1}

    def crashing_publish(root, attempt):
        if state["crashes"]:
            state["crashes"] -= 1
            raise RuntimeError("simulated crash before the attempt became durable")
        return publish(root, attempt)

    monkeypatch.setattr(runner, "publish_causal_qualification", crashing_publish)

    entropy = lambda _commitment, _teardown: harness.entropy
    with pytest.raises(RuntimeError, match="simulated crash"):
        _run_resident(harness, baseline, continuation, entropy_provider=entropy)
    assert harness.reference_calls == 1
    assert "quality" in continuation.records
    assert "final" not in continuation.records

    reset_quiescence()
    reference = _run_resident(harness, baseline, continuation, entropy_provider=entropy)
    assert reference == harness.attempt_reference
    # Neither expensive stage ran again: no new resident speed, no new
    # pristine T, no second audit session.
    assert harness.calls.count("resident.speed") == 1
    assert harness.reference_calls == 1
    assert harness.calls.count("audit") == 1
    assert continuation.records["final"] == harness.attempt_reference


def test_resident_audit_completion_survives_a_crash_before_producer_return(
    monkeypatch,
) -> None:
    harness = _Harness(
        monkeypatch,
        graph=(QualificationDecision.PASS,),
        speed=(QualificationDecision.PASS,),
        quality=(QualificationDecision.PASS,),
    )
    baseline, _stage_reference, exits = _install_resident_runner_path(
        monkeypatch,
        harness,
        speed_decision=QualificationDecision.PASS,
        escalated=False,
    )
    monkeypatch.setattr(runner, "QualificationContinuation", _MemoryContinuation)
    clock = iter(3.05 + index * 0.05 for index in range(16))
    harness.executor.manager.clock = lambda: next(clock)
    continuation = _MemoryContinuation()
    producer = runner._run_slot_audits
    monkeypatch.setattr(
        runner,
        "_run_slot_audits",
        _crash_after_completion(producer, "crash after durable audit"),
    )

    def run_resident():
        ids = iter(f"{index + 1:032x}" for index in range(16))
        return runner.run_causal_qualification(
            harness.value,
            executor=harness.executor,
            resident_baseline_executor=baseline,
            entropy_provider=harness.entropy_provider,
            hidden_judge=harness.hidden_judge,
            deadline=100.0,
            id_factory=lambda: next(ids),
            continuation=continuation,
        )

    with pytest.raises(RuntimeError, match="after durable audit"):
        run_resident()
    assert harness.calls.count("resident.speed") == 1
    assert harness.calls.count("audit") == 1
    assert "audit" in continuation.records
    assert harness.reference_calls == 0

    assert run_resident() == harness.attempt_reference
    assert exits == []
    assert harness.calls.count("resident.speed") == 1
    assert harness.calls.count("audit") == 1
    assert harness.reference_calls == 1
    assert continuation.records["final"] == harness.attempt_reference


def test_continuation_rejects_producers_that_skip_the_completion_sink(
    monkeypatch,
) -> None:
    audit_harness, audit_baseline, _exits = _resident_pass_harness(monkeypatch)
    producer = runner._run_slot_audits
    monkeypatch.setattr(runner, "_run_slot_audits", _without_completion(producer))
    with pytest.raises(runner.QualificationRunnerError, match="audit sink"):
        _run_resident(audit_harness, audit_baseline, _MemoryContinuation())
    assert audit_harness.calls.count("audit") == 1
    assert audit_harness.reference_calls == 0

    t_harness, t_baseline, _exits = _resident_pass_harness(monkeypatch)
    producer = OCIEngineExecutor.execute_reference
    monkeypatch.setattr(
        OCIEngineExecutor, "execute_reference", _without_completion(producer)
    )
    continuation = _MemoryContinuation()
    with pytest.raises(runner.QualificationRunnerError, match="pristine T returned"):
        _run_resident(t_harness, t_baseline, continuation)
    assert t_harness.reference_calls == 1
    assert "quality" not in continuation.records


def test_continuation_identity_mismatch_holds_before_any_execution(
    monkeypatch,
) -> None:
    harness, baseline, _exits = _resident_pass_harness(monkeypatch)
    continuation = _MemoryContinuation(authority_digest=_d("foreign-authority"))
    with pytest.raises(
        QualificationContinuationError, match="differs from the sealed cohort"
    ):
        _run_resident(harness, baseline, continuation)
    assert "resident.speed" not in harness.calls
    assert harness.reference_calls == 0


def test_nonresident_qualification_is_refused_at_entry(monkeypatch) -> None:
    # D8: the marginal (nonresident) execution path is retired; a non-v3 speed
    # policy is refused before prevalidation, execution, or any durable write.
    harness = _pass_harness(monkeypatch)
    with pytest.raises(
        runner.QualificationRunnerError,
        match="causal qualification requires the resident speed policy",
    ):
        _run(harness, _MemoryContinuation())
    assert harness.calls == []
    assert harness.reference_calls == 0


def test_untyped_continuation_is_rejected_before_any_execution(monkeypatch) -> None:
    harness = _Harness(
        monkeypatch,
        graph=(QualificationDecision.PASS,),
        speed=(QualificationDecision.PASS,),
        quality=(QualificationDecision.PASS,),
    )
    baseline, _stage_reference, _exits = _install_resident_runner_path(
        monkeypatch,
        harness,
        speed_decision=QualificationDecision.PASS,
        escalated=False,
    )
    with pytest.raises(
        runner.QualificationRunnerError, match="not exactly typed"
    ):
        _run(harness, _MemoryContinuation(), baseline)
    assert "resident.speed" not in harness.calls


def test_quality_record_without_speed_record_holds(
    monkeypatch,
) -> None:
    resident_harness = _Harness(
        monkeypatch,
        graph=(QualificationDecision.PASS,),
        speed=(QualificationDecision.PASS,),
        quality=(QualificationDecision.PASS,),
    )
    baseline, _stage_reference, _exits = _install_resident_runner_path(
        monkeypatch,
        resident_harness,
        speed_decision=QualificationDecision.PASS,
        escalated=False,
    )
    monkeypatch.setattr(runner, "QualificationContinuation", _MemoryContinuation)
    resident_continuation = _MemoryContinuation()
    resident_continuation.records["quality"] = object()
    with pytest.raises(
        QualificationContinuationError, match="without its speed continuation"
    ):
        runner.run_causal_qualification(
            resident_harness.value,
            executor=resident_harness.executor,
            resident_baseline_executor=baseline,
            entropy_provider=resident_harness.entropy_provider,
            hidden_judge=resident_harness.hidden_judge,
            deadline=100.0,
            continuation=resident_continuation,
        )
    assert "resident.speed" not in resident_harness.calls
