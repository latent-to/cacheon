"""Request-pinned qualification refuses before entry and survives CPU restart."""

from types import SimpleNamespace

import pytest

import test_b300_remote_worker_adapter as worker_tests
import test_recoverable_qualification_dispatcher as recovery_tests
import test_b300_remote_qualification_adapter as qualification_tests
from cacheon.chain.qualification_request import INCUMBENT_FIELDS
from cacheon.chain.remote_evaluation_dispatcher import (
    RemoteEvaluationDispatcherError, _validate_request_body,
)
from cacheon.chain.recoverable_qualification_dispatcher import (
    RecoverableQualificationDispatcherError,
)
from cacheon.eval import b300_remote_worker_adapter as worker

configured = qualification_tests.configured


@pytest.mark.parametrize("lane", ["primary", "reproduction"])
@pytest.mark.parametrize("drift", ["stack", "tree", "legacy"])
def test_wrong_incumbent_stops_before_publication_and_resident_entry(
    tmp_path, monkeypatch, lane, drift,
):
    primary = SimpleNamespace(construction=SimpleNamespace(
        incumbent_stack=SimpleNamespace(digest="a" * 64),
        incumbent_tree_digest="b" * 64,
    ))
    reproduction = SimpleNamespace(construction=SimpleNamespace(
        incumbent_stack=SimpleNamespace(digest="c" * 64),
        incumbent_tree_digest="d" * 64,
    ))
    runtime = worker_tests._runtime_shell(
        worker_tests._adapter_paths(tmp_path), qualification_commission=primary,
    )
    runtime._commissioned_service = SimpleNamespace(reproduction_commission=reproduction)
    construction = (primary if lane == "primary" else reproduction).construction
    body = {
        "candidates": [{"publication": {}}], "screen_lane": lane,
        "incumbent_stack_digest": construction.incumbent_stack.digest,
        "incumbent_tree_digest": construction.incumbent_tree_digest,
    }
    if drift == "legacy":
        for field in INCUMBENT_FIELDS:
            body.pop(field)
    else:
        body[f"incumbent_{drift}_digest"] = "f" * 64
    worker_tests._patch_authenticated_carrier(
        monkeypatch, stage="qualification", wire=SimpleNamespace(body=body),
    )
    calls = []
    monkeypatch.setattr(worker, "resolve_cohort_publications", lambda *_: calls.append("resolve"))
    monkeypatch.setattr(runtime, "qualification_adapter_for", lambda *_: calls.append("adapter"))
    monkeypatch.setattr(worker, "publish_resident_entry", lambda *_: calls.append("resident"))

    with pytest.raises(worker.AdapterRequestFailed) as error:
        worker.run_with_runtime(tmp_path / ("1" * 64), tmp_path / "result", runtime)

    assert "incumbent differs" in str(error.value.__cause__)
    assert calls == []
    assert runtime.worker.retire_calls == runtime.worker.calls == 0
    assert not (tmp_path / "result" / "RESIDENT_ENTRY_ARMED.json").exists()


def test_request_reader_preserves_legacy_and_rejects_partial_or_invalid_pin(tmp_path):
    fixtures = recovery_tests._fixtures()
    authority = fixtures._authority(tmp_path, recoverable=True)
    bound = authority.request.body
    _validate_request_body("qualification", bound)
    legacy = {key: value for key, value in bound.items() if key not in INCUMBENT_FIELDS}
    _validate_request_body("qualification", legacy)
    for field in INCUMBENT_FIELDS:
        with pytest.raises(RemoteEvaluationDispatcherError):
            _validate_request_body("qualification", {**legacy, field: bound[field]})
        with pytest.raises(RemoteEvaluationDispatcherError):
            _validate_request_body("qualification", {**bound, field: "invalid"})


@pytest.mark.parametrize("field", sorted(INCUMBENT_FIELDS))
def test_direct_qualification_adapter_checks_pin_before_resolving_publication(
    configured, monkeypatch, field,
):
    body = qualification_tests._body(configured)
    body[field] = "f" * 64
    request = qualification_tests._request(configured, body=body)
    monkeypatch.setattr(
        type(configured.adapter.publications), "resolve",
        lambda *_: pytest.fail("mismatched incumbent resolved a publication"),
    )
    with pytest.raises(RemoteEvaluationDispatcherError, match="incumbent differs"):
        configured.adapter.run(request)


@pytest.mark.parametrize("legacy", [False, True])
def test_completed_request_reopens_without_changing_its_pin_or_grammar(
    tmp_path, legacy,
):
    fixtures = recovery_tests._fixtures()
    authority = fixtures._authority(tmp_path, recoverable=True)
    transport = recovery_tests._Transport(authority, fixtures, fail_resume=True)
    first = recovery_tests._dispatcher(authority, transport)
    if legacy:
        expected_request = first._expected_request
        first._expected_request = lambda claim, retained_body=None: expected_request(claim, {})
    with pytest.raises(RecoverableQualificationDispatcherError, match="not ready"):
        first.dispatch_once()
    retained = transport.plan.remote_request.to_dict()
    request_id = transport.plan.request_id
    fixtures._write_completed_result(authority, transport.plan)
    transport.fail_resume = False
    restarted = recovery_tests._dispatcher(authority, transport)
    if not legacy:
        # A completed bound result belongs to its request, not this later CPU pin.
        restarted.qualification_incumbent_tree_digest = authority.fixtures._h("later-cpu-pin")
    result = restarted.dispatch_once()
    assert result.disposition == "completed"
    assert transport.plan.request_id == request_id
    assert transport.plan.remote_request.to_dict() == retained
    assert (transport.plans, transport.materializations, transport.publications) == (1, 1, 1)


def test_explicit_missing_service_does_not_borrow_the_only_other_stack(tmp_path):
    fixtures = recovery_tests._fixtures()
    authority = fixtures._authority(tmp_path, recoverable=True)
    with recovery_tests._store(authority) as store:
        original = authority.fixtures._incumbent(authority.service)
        store.initialize_evaluation_stack(original, tree_digest=authority.fixtures._h("tree"))
        assert store._unambiguous_evaluation_stack("").manifest == original
        assert store._unambiguous_evaluation_stack(original.arena_digest).manifest == original
        assert store._unambiguous_evaluation_stack(authority.fixtures._h("other-arena")) is None
