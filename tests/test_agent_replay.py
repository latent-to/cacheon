"""Bind AIPerf source/credit metadata to real controller evidence before scoring."""

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from cacheon.eval.agent_replay import AgentReplayPlan, collect_read, _Placement, _Rounds, _chat_input_ids
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
    rows, stamps, metadata = {}, {}, []
    for i, kind in enumerate(('main', 'inner')):
        request_id = f'{i + 1:032x}'
        stamps[request_id] = 1_000_000_000 + i * 1_000_000_000
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
            # AIPerf's own credit is issued before the bridge holds the request; the
            # scored credit is the bridge's round release stamp.
            'credit_issued_ns': 900_000_000 + i * 1_000_000_000,
            'benchmark_phase': 'profiling',
        }})
    path = export / 'profile_export.jsonl'
    path.write_text('\n'.join(json.dumps(m) for m in reversed(metadata)))
    return plan, SimpleNamespace(rows=rows, stamps=stamps, _ns=lambda t: round(t * 1e9)), path


def test_join_keeps_source_kind_clock_and_exact_work(tmp_path):
    plan, bridge, _ = _inputs(tmp_path)
    read = collect_read(plan, bridge)
    assert [(r.kind, r.ordinal) for r in read.records] == [('main', 0), ('inner', 0)]
    assert read.records[0].ttft_s == pytest.approx(.2)
    assert read.records[0].request_end_ns == 1_500_000_000
    # The credit is the round stamp: two rounds of 0.5 s each, not the 1.5 s makespan.
    assert [r.credit_issued_ns for r in read.records] == [1_000_000_000, 2_000_000_000]
    summary = json.loads((plan.output_directory / 'read.json').read_text())
    assert summary['turns'] == 2 and summary['elapsed_s'] == 1.0
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
        bridge.stamps[rows[0]['metadata']['x_request_id']] *= 1000
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
    changed = replace(first, replay=replace(replay, windows=2))
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
                        tmp_path / 'window', ServiceContract(20, 2, .8), 'candidate', 1, 'B')
    with pytest.raises(ValueError, match="at least one sealed window"):
        AgentReplayPlan(manifest, (2,), Path('/bin/aiperf'), Path('/model'),
                        tmp_path / 'window', ServiceContract(20, 2, .8), 'candidate', 1, 'B', windows=0)
    plan = AgentReplayPlan(manifest, (2,), Path('/bin/aiperf'), Path('/model'),
                           tmp_path / 'window', ServiceContract(20, 2, .8), 'candidate', 1, 'B', windows=2)
    events = []
    session = SimpleNamespace(replay_reads=[])

    async def flush(_session):
        events.append('flush')
    monkeypatch.setattr(agent_replay, 'flush_cache', flush)

    async def ready(load, window):
        events.append(('ready', load, window))

    async def read(_session, read_plan, *, tokenizer):
        (load,) = read_plan.loads
        assert read_plan.output_directory == plan.output_directory / f'window{read_plan.window}'
        events.append(('read', load, read_plan.window))
        read_plan.output_directory.mkdir(parents=True)
        records = tuple(TurnRecord(root, 'main', 0, 1_000_000_000, 1_000_000_000,
                                  1_100_000_000, 1_200_000_000, 64, 8, 'ok')
                        for root in read_plan.slice.expected_work(load))
        return LoadRead(read_plan.arm, read_plan.window, read_plan.lane, load, records)

    monkeypatch.setattr(agent_replay, '_run_load_read', read)
    result = asyncio.run(agent_replay.run_replay(session, plan, tokenizer=object(), before_read=ready))
    # Every sealed window is flushed, synchronized with the peer lane, then read as a fresh window.
    assert events == ['flush', ('ready', 2, 1), ('read', 2, 1), 'flush', ('ready', 2, 2), ('read', 2, 2)]
    assert tuple(session.replay_reads) == result and [r.window for r in result] == [1, 2]
    artifact = json.loads((plan.output_directory / 'window.json').read_text())
    assert set(artifact) == {'workload', 'reads'} and len(artifact['reads']) == 2
    assert (artifact['workload']['windows'], artifact['workload']['arrival']) == (2, 'lockstep-rounds')
    codec = ContinuationCodec((LoadRead,))
    assert codec.decode(codec.encode(result[0])) == result[0]


def test_lockstep_rounds_release_together_in_session_order_and_fail_a_starved_first_round():
    import asyncio

    async def scenario():
        loop = asyncio.get_running_loop()
        rounds = _Rounds(2, loop.time, settle_s=0.02, first_wait_s=0.5)
        order = []

        async def conversation(request_id, key):
            stamp, ordinal = await rounds.hold(request_id, key)
            order.append(key)
            return stamp, ordinal

        # The first round waits for exactly the sealed openings, then releases them with one stamp in key
        # order, numbering dispatch in that order (the engine requires batch indices in dispatch order).
        b = asyncio.create_task(conversation('r-b', 'session-b'))
        await asyncio.sleep(0.05)
        assert not b.done() and not rounds.first_released
        a = asyncio.create_task(conversation('r-a', 'session-a'))
        (stamp_a, ordinal_a), (stamp_b, ordinal_b) = await asyncio.gather(a, b)
        assert stamp_a == stamp_b and (ordinal_a, ordinal_b) == (0, 1) and order == ['session-a', 'session-b']
        # A request arriving while the round is in flight waits for the engine to drain, then for the
        # settle time with no new arrival, and carries the later round's stamp and the next ordinal.
        c = asyncio.create_task(conversation('r-c', 'session-c'))
        rounds.done('r-a')
        await asyncio.sleep(0.05)
        assert not c.done()
        rounds.done('r-b')
        stamp_c, ordinal_c = await c
        assert stamp_c > stamp_a and ordinal_c == 2
        # A first round the client never fills fails every held request instead of idling to the deadline.
        starved = _Rounds(3, loop.time, settle_s=0.02, first_wait_s=0.05)
        with pytest.raises(RuntimeError, match="1 of 3 conversations"):
            await starved.hold('r-x', 'session-x')

    asyncio.run(scenario())
