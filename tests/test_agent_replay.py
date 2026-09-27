"""Bind AIPerf source/credit metadata to real controller evidence before scoring."""

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from cacheon.eval.agent_replay import AgentReplayPlan, collect_read, _Placement, _chat_input_ids
from cacheon.eval.oci_outer_session import BatchExecutionEvidence, OuterSessionInfrastructureError
from cacheon.eval.oci_session_protocol import BatchEvidence, PromptEvidence
from cacheon.eval.service_capacity import ServiceContract, ServiceEvidenceError
from tests.test_agent_slice import _session, _write_slice
from tests.test_oci_outer_session import _plan
from cacheon.eval.scoring import marginal_workload_digest
from cacheon.arena_service import WorkloadCell
from cacheon.eval.b300_arena_definition import engine_config, B300DeploymentError


def test_chat_input_ids_use_the_tokenizer_ids_not_its_mapping_keys():
    transformers = pytest.importorskip("transformers")
    from tokenizers import Tokenizer, models, pre_tokenizers

    backend = Tokenizer(models.WordLevel({"[UNK]": 0, "user": 1, "hello": 2, "assistant": 3},
                                         unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = transformers.PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")
    tokenizer.chat_template = (
        "{% for m in messages %}{{ m.role }} {{ m.content }} {% endfor %}"
        "{% if add_generation_prompt %}assistant{% endif %}"
    )
    ids = _chat_input_ids(tokenizer, {"messages": [{"role": "user", "content": "hello"}]})
    assert ids == [1, 2, 3]
    assert all(type(token) is int for token in ids)


def _inputs(tmp_path):
    session = _session(1, k=1, inner=1)
    manifest = _write_slice(tmp_path, [session])
    plan = AgentReplayPlan(manifest, (1,), Path('/bin/aiperf'), Path('/model'), tmp_path / 'out',
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
def test_routing_is_session_sticky_and_balanced(tmp_path, ranks):
    placement = _Placement(ranks)
    opening = [{'role': 'system', 'content': 'coding'}, {'role': 'user', 'content': '[rid:0a1b2c]\n\nsession-a'}]
    later = opening + [{'role': 'assistant', 'content': 'answer'}, {'role': 'user', 'content': 'followup'}]
    first = placement.acquire(opening)
    placement.release(first)
    assert placement.acquire(later) == first == 0
    placement.release(first)
    # A different marker on the same opening is the same session; new sessions spread over the ranks.
    reminted = [{'role': 'system', 'content': 'coding'}, {'role': 'user', 'content': '[rid:ffffff]\n\nsession-a'}]
    assert placement.acquire(reminted) == first
    placement.release(first)
    others = [placement.acquire([{'role': 'user', 'content': f'[rid:{i:06x}]\n\nsession-{i}'}]) for i in range(ranks)]
    assert others == [(i + 1) % ranks for i in range(ranks)]
    replay, _, _ = _inputs(tmp_path)
    first = _plan(replay=replay)
    changed = replace(first, replay=replace(replay, ramp_seconds_per_session=2))
    assert marginal_workload_digest(first) != marginal_workload_digest(changed)


def test_replay_needs_no_dummy_timed_batches_and_binds_conditioning_geometry(tmp_path):
    replay, _, _ = _inputs(tmp_path)
    plan = _plan(replay=replay)
    warmup = replace(plan, prompt_batches=plan.prompt_batches[:plan.warmup_count],
                     batch_max_new_tokens=(1,) * plan.warmup_count,
                     batch_expected_prompt_tokens=(5,) * plan.warmup_count)
    assert len(warmup.prompt_batches) == warmup.warmup_count
    changed = replace(warmup, batch_max_new_tokens=(2,) * plan.warmup_count)
    assert marginal_workload_digest(warmup) != marginal_workload_digest(changed)
    with pytest.raises(OuterSessionInfrastructureError, match="warmup and measured work"):
        replace(warmup, replay=None)


@pytest.mark.parametrize('input_tokens,output_tokens', [(8192, 1024), (65536, 4096)])
def test_declared_context_survives_cell_projection_and_must_fit(input_tokens, output_tokens):
    cell = WorkloadCell('coding', input_tokens, output_tokens, 2, 1)
    template = _plan().engine_config
    declared = replace(template, engine_kwargs={**template.engine_kwargs, 'context_length': 262144})
    assert engine_config(declared, cell, disable_cuda_graph=False).engine_kwargs['context_length'] == 262144
    too_short = replace(declared, engine_kwargs={**declared.engine_kwargs, 'context_length': input_tokens})
    with pytest.raises(B300DeploymentError, match='does not fit'):
        engine_config(too_short, cell, disable_cuda_graph=False)


def test_window_is_one_sealed_load_flushed_before_its_read_and_retained(tmp_path, monkeypatch):
    import asyncio
    from cacheon.eval import agent_replay
    from cacheon.eval.continuation_codec import ContinuationCodec
    from cacheon.eval.service_capacity import LoadRead, TurnRecord

    manifest = _write_slice(tmp_path, [_session(i, k=1, inner=0) for i in (1, 2)])
    with pytest.raises(ValueError, match="exactly one sealed load"):
        AgentReplayPlan(manifest, (1, 2), Path('/bin/aiperf'), Path('/model'),
                        tmp_path / 'window', ServiceContract(20, 2, .8), 'candidate', 1, 'B', 2.)
    plan = AgentReplayPlan(manifest, (2,), Path('/bin/aiperf'), Path('/model'),
                           tmp_path / 'window', ServiceContract(20, 2, .8), 'candidate', 1, 'B', 2.)
    events = []
    session = SimpleNamespace(replay_reads=[])
    async def flush(_session):
        events.append('flush')
    monkeypatch.setattr(agent_replay, 'flush_cache', flush)

    async def read(_session, read_plan, *, tokenizer):
        (load,) = read_plan.loads
        events.append(load)
        read_plan.output_directory.mkdir(parents=True)
        records = tuple(TurnRecord(root, 'main', 0, 1_000_000_000, 1_000_000_000,
                                  1_100_000_000, 1_200_000_000, 64, 8, 'ok')
                        for root in read_plan.slice.expected_work(load))
        return LoadRead(read_plan.arm, read_plan.window, read_plan.lane, load, records)

    monkeypatch.setattr(agent_replay, '_run_load_read', read)
    result = asyncio.run(agent_replay.run_replay(session, plan, tokenizer=object()))
    assert events == ['flush', 2]
    assert tuple(session.replay_reads) == result
    artifact = json.loads((plan.output_directory / 'window.json').read_text())
    assert set(artifact) == {'workload', 'reads'} and len(artifact['reads']) == 1
    codec = ContinuationCodec((LoadRead,))
    assert codec.decode(codec.encode(result[0])) == result[0]
