"""Bind AIPerf source/credit metadata to real controller evidence before scoring."""

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from cacheon.eval.agent_replay import AgentReplayPlan, collect_read, _rank
from cacheon.eval.oci_outer_session import BatchExecutionEvidence
from cacheon.eval.oci_session_protocol import BatchEvidence, PromptEvidence
from cacheon.eval.service_capacity import ServiceContract, ServiceEvidenceError
from tests.test_agent_slice import _session, _write_slice
from tests.test_oci_outer_session import _plan
from cacheon.eval.scoring import marginal_workload_digest


def _inputs(tmp_path):
    session = _session(1, k=1, inner=1)
    manifest = _write_slice(tmp_path, [session])
    plan = AgentReplayPlan(manifest, 1, Path('/bin/aiperf'), Path('/model'), tmp_path / 'out',
                           ServiceContract(20, 2, .8), 'candidate', 1, 'second')
    export = plan.output_directory / 'aiperf'
    export.mkdir(parents=True)
    rows, metadata = {}, []
    for i, kind in enumerate(('main', 'inner')):
        request_id = f'{i + 1:032x}'
        rows[request_id] = BatchExecutionEvidence(
            i, request_id, 'a' * 32, 1.1 + i, 1.5 + i, 4,
            BatchEvidence((PromptEvidence((1, 2, 3, 4), ((),) * 4, 64),)), (), ((.1, .39),),
        )
        metadata.append({'metadata': {
            'x_request_id': request_id, 'source_trace_id': session['id'],
            'source_inner_idx': None if kind == 'main' else 0,
            # Main nonextending requests also carry weka_flat: source_kind is
            # not the distinction between a root turn and a child request.
            'source_kind': 'weka_flat' if kind == 'main' else 'weka_subagent',
            'request_start_ns': 1_000_000_000 + i * 1_000_000_000,
            'credit_issued_ns': 1_000_000_000 + i * 1_000_000_000,
            'benchmark_phase': 'profiling',
        }})
    path = export / 'profile_export.jsonl'
    path.write_text('\n'.join(json.dumps(m) for m in reversed(metadata)))
    return plan, SimpleNamespace(rows=rows, _ns=lambda t: round(t * 1e9)), path


def test_join_keeps_source_kind_clock_and_exact_work(tmp_path):
    plan, bridge, _ = _inputs(tmp_path)
    read = collect_read(plan, bridge)
    assert [(r.kind, r.ordinal) for r in read.records] == [('main', 0), ('inner', 0)]
    assert read.records[0].ttft_s == pytest.approx(.2)
    assert read.records[0].request_end_ns == 1_500_000_000
    summary = json.loads((plan.output_directory / 'read.json').read_text())
    assert summary['turns'] == 2 and summary['elapsed_s'] == 1.5
    assert len((plan.output_directory / 'turns.jsonl').read_text().splitlines()) == 2


@pytest.mark.parametrize('fault', ['duplicate', 'missing', 'wrong-root', 'credit-clock'])
def test_invalid_join_never_becomes_a_goodput_result(tmp_path, fault):
    plan, bridge, path = _inputs(tmp_path)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if fault == 'duplicate':
        rows.append(rows[0])
    elif fault == 'missing':
        rows.pop()
    elif fault == 'wrong-root':
        rows[0]['metadata']['source_trace_id'] = 'another-root'
    else:
        rows[0]['metadata']['credit_issued_ns'] *= 1000
    path.write_text('\n'.join(json.dumps(row) for row in rows))
    with pytest.raises(ServiceEvidenceError):
        collect_read(plan, bridge)
    assert not (plan.output_directory / 'read.json').exists()


@pytest.mark.parametrize('ranks', [1, 4])
def test_routing_and_workload_identity_do_not_depend_on_followup_text(tmp_path, ranks):
    opening = [{'role': 'system', 'content': 'coding'}, {'role': 'user', 'content': 'session-a'}]
    later = opening + [{'role': 'assistant', 'content': 'answer'}, {'role': 'user', 'content': 'followup'}]
    assert _rank(opening, ranks) == _rank(later, ranks) < ranks
    replay, _, _ = _inputs(tmp_path)
    first = _plan(replay=replay)
    changed = replace(first, replay=replace(replay, ramp_duration_s=2))
    assert marginal_workload_digest(first) != marginal_workload_digest(changed)
