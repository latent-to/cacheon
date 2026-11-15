"""Select replay turns and retain a small incumbent control for the existing pristine reference."""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path
from uuid import UUID
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cacheon.eval.qualification import QualificationProfile
    from cacheon.eval.reference_quality import RawRolloutEvidence

from cacheon.eval.qualification import QualificationError, SelectionReceipt
from cacheon.eval.reference_protocol import ReferencePromptInput, ReferenceRoleInput
from cacheon.eval.scoring import marginal_workload_digest
from cacheon.stack_identity import canonical_digest


def source_digest(replay, root: str, outer: int, inner: int | None) -> str:
    """Name a dataset turn independently of its dispatch or completion order."""
    if (type(root) is not str or type(outer) is not int or outer < 0
        or (inner is not None and (type(inner) is not int or inner < 0))):
        raise QualificationError("replay source coordinate is malformed")
    return canonical_digest("cacheon.qualification.replay-prompt.v1", {
        "slice": replay.slice.digest, "root": root, "outer": outer, "inner": inner,
    })


def prompt_pool(replay) -> dict[str, int]:
    """Enumerate the windows a replay serves (the whole slice under closed loop); never choose or truncate another."""
    result = {}
    for index, session in enumerate(replay.slice.sessions[:replay.sessions]):
        path = replay.slice.directory / f"{index:03d}_{session.id}.json"
        trace = json.loads(path.read_text())
        counts = [0, 0]
        for outer, request in enumerate(trace["requests"]):
            nested = request.get("requests")
            rows = [(None, request)] if nested is None else enumerate(nested)
            for inner, row in rows:
                count = row["out"]
                if type(count) is not int or count < 1:
                    raise QualificationError("replay quality requires a positive recorded output budget")
                result[source_digest(replay, session.id, outer, inner)] = count
                counts[inner is not None] += 1
        if tuple(counts) != (session.main_turns, session.inner_requests):
            raise QualificationError("replay prompt pool differs from the sealed turn counts")
    return result


def _frame(prompt):
    if any(prompt.top_logprobs):
        raise QualificationError("replay quality requires teacher-NLL rollout evidence")
    return {"output_ids": list(prompt.output_ids), "top_logprobs": [[] for _ in prompt.output_ids]}


def trajectory_digest(workload: str, pairs, baseline_batches, candidate_batches) -> str:
    """Bind both completed timed trajectories before the selected control exists."""
    return canonical_digest("cacheon.qualification.replay-trajectories.v1", {
        "workload_digest": workload,
        "prompts": [[key, _frame(baseline_batches[b].evidence.prompts[0]),
                     _frame(candidate_batches[c].evidence.prompts[0])]
                    for key, b, c in pairs],
    })


def selected_frames(lifecycle):
    """Project the retained reference inputs onto the existing quality scorer's three roles."""
    return tuple((item.prompt_digest, [{"output_ids": list(role.output_ids),
                   "top_logprobs": [[] for _ in role.output_ids]} for role in item.roles])
                 for item in lifecycle.crossover.reference_inputs)


def _source_rows(replay, directory: Path, batches):
    rows = {row.request_id: row for row in batches}
    result = {}
    # Every window replays the same prompts; the first is the one every sequential read retains.
    path = directory / "window1" / "aiperf" / "profile_export.jsonl"
    for line in path.read_text().splitlines():
        meta = json.loads(line)["metadata"]
        if meta["benchmark_phase"] != "profiling":
            continue
        key = source_digest(replay, meta["source_trace_id"], meta["source_outer_idx"], meta.get("source_inner_idx"))
        request_id = UUID(meta["x_request_id"]).hex
        if key in result or request_id not in rows:
            raise QualificationError("replay source-to-request join is not one-to-one")
        result[key] = rows[request_id]
    if set(result) != set(prompt_pool(replay)):
        raise QualificationError("replay quality sources differ from the committed prompt pool")
    return result


def _stock_pair(executions, count):
    # Since 2026-11-05 no stock engine boots after a PASS. Four-boot policies use the first lifetime
    # from each fresh process. A single-pair policy uses its two separately booted stock engines.
    return executions[0], executions[count // 2]


def stock_rollouts(evidence):
    """The stock rollout and control lifetimes: a retained bracket for both, else the two stored ones."""
    bracket = evidence.baseline_execution
    return (bracket, bracket) if bracket is not None else _stock_pair(evidence.prior_executions, evidence.policy.reference_boots)


def _timed_rows(execution, arm, replay):
    count = replay.slice.turns(replay.sessions)
    first = arm.session_plan.warmup_count + arm.session_plan.replay.warm_windows * count
    return execution.session.batches[first:first + count]


def _matched(rows, compared, pool):
    """Join one stored stock window to the candidate's sources by input digest and output budget, one to one."""
    free = {}
    for row in rows:
        free.setdefault((row.input_ids_sha256, len(row.evidence.prompts[0].output_ids)), []).append(row)
    try:
        return {key: free[compared[key].input_ids_sha256, pool[key]].pop(0) for key in sorted(compared)}
    except (KeyError, IndexError):
        raise QualificationError("stored stock window lacks a timed request for a replay source") from None


def create_controls(value, reference, candidate, candidate_directory, candidate_quiescence, entropy_provider):
    """Select stored stock controls after the candidate's timed reads and teardown; no engine runs."""
    from cacheon.eval.agent_replay import _PROMPTS
    from cacheon.eval.goodput_runtime import _reference_arm

    plan = value.resident_speed_plan
    replay = plan.baseline.session_plan.replay
    pool = prompt_pool(replay)
    arm = _reference_arm(plan, getattr(reference, "origin", ""))
    compared = _source_rows(replay, candidate_directory, candidate.session.batches)
    lifetimes = _stock_pair(reference, plan.policy.reference_boots)
    stock, control = (_matched(_timed_rows(row, arm, replay), compared, pool) for row in lifetimes)
    pairs = tuple((key, stock[key].batch_index, compared[key].batch_index) for key in sorted(compared))
    entropy = entropy_provider(value.commitment, candidate_quiescence)
    selection = SelectionReceipt.reveal(
        value.commitment, secret=value.selection_secret, entropy=entropy,
        sealed_cohort_trajectory_digest=trajectory_digest(
            marginal_workload_digest(value.prepared.baseline_session_plan), pairs,
            lifetimes[0].session.batches, candidate.session.batches,
        ),
    )
    by_request = {compared[key].request_id: key for key in selection.selected_prompt_digests}
    # bridge.jsonl names each input by digest; its IDs are the ones this evaluator process primed and sent.
    primed = {digest: ids for ids, digest in _PROMPTS.values()}
    inputs = {}
    with (candidate_directory / "window1" / "bridge.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            key = by_request.get(UUID(row["x_request_id"]).hex)
            if key is not None and row["input_sha256"] in primed:
                inputs[key] = primed[row["input_sha256"]]
    if set(inputs) != set(selection.selected_prompt_digests):
        raise QualificationError("selected replay canonical inputs are incomplete")
    result = []
    for key in selection.selected_prompt_digests:
        ids = inputs[key]
        if (hashlib.sha256(struct.pack(f">{len(ids)}I", *ids)).hexdigest(),) != stock[key].input_ids_sha256:
            raise QualificationError("selected replay input bytes changed after execution")
        outputs = (tuple(rows[key].evidence.prompts[0].output_ids) for rows in (stock, compared, control))
        result.append(ReferencePromptInput(key, "", tuple(
            ReferenceRoleInput(output, ((),) * len(output)) for output in outputs), ids))
    return pairs, tuple(result), entropy


def trajectory_rows(lifecycle: object):
    from cacheon.eval.crossover_runtime import ResidentMarginalLifecycleEvidence

    if type(lifecycle) is not ResidentMarginalLifecycleEvidence:
        raise QualificationError("trajectory lifecycle is not typed")
    return marginal_workload_digest(lifecycle.prepared.baseline_session_plan), selected_frames(lifecycle)


def validate_controls(evidence, plan, stock_arm):
    """Reopen source joins and all three selected rollouts from retained host evidence."""
    replay = plan.baseline.session_plan.replay
    pool = prompt_pool(replay)
    pairs = evidence.prompt_pairs
    if tuple(key for key, _, _ in pairs) != tuple(sorted(pool)):
        raise QualificationError("replay quality prompt pairs differ from the committed source pool")
    stock, control = stock_rollouts(evidence)
    rows = (stock.session.batches, evidence.candidate_execution.session.batches)
    for role, arm in enumerate((stock_arm, plan.candidate)):
        first = arm.session_plan.warmup_count + arm.session_plan.replay.warm_windows * replay.slice.turns(replay.sessions)
        if {pair[role + 1] for pair in pairs} != set(range(first, first + len(pool))):
            raise QualificationError("replay source pairs do not cover the timed requests exactly")
    joined = {key: (rows[0][b], rows[1][c]) for key, b, c in pairs}
    for key, (baseline, candidate) in joined.items():
        if (not baseline.input_ids_sha256 or baseline.input_ids_sha256 != candidate.input_ids_sha256
            or any(len(row.evidence.prompts[0].output_ids) != pool[key] for row in (baseline, candidate))):
            raise QualificationError("replay source inputs or output budgets differ")
    keys = tuple(row.prompt_digest for row in evidence.reference_inputs)
    if keys != tuple(sorted(set(keys))) or not set(keys) <= pool.keys():
        raise QualificationError("selected replay controls are not unique source prompts")
    # A stored control is a timed answer of the second stock pair; a bracket's are checked with its session.
    stored = None if evidence.baseline_execution is not None else {
        (row.input_ids_sha256, tuple(row.evidence.prompts[0].output_ids))
        for row in _timed_rows(control, stock_arm, replay)}
    for prompt in evidence.reference_inputs:
        baseline, candidate = joined[prompt.prompt_digest]
        if (prompt.roles[0].output_ids != baseline.evidence.prompts[0].output_ids
            or prompt.roles[1].output_ids != candidate.evidence.prompts[0].output_ids
            or (hashlib.sha256(prompt.input_bytes).hexdigest(),) != baseline.input_ids_sha256
            or any(any(role.supports) for role in prompt.roles)
            or (stored is not None
                and (baseline.input_ids_sha256, tuple(prompt.roles[2].output_ids)) not in stored)):
            raise QualificationError("selected reference inputs differ from their timed replay requests")


def _rollout(
    *,
    profile: QualificationProfile,
    prompt_digest: str,
    frame: dict[str, object],
    role_input: ReferenceRoleInput,
    role_evidence: object,
) -> RawRolloutEvidence:
    """Retain teacher NLL evidence: replay quality seals no top-k width and no hidden tasks, so no
    distribution or judge evidence exists to project."""
    from cacheon.eval import qualification_runner as owner

    if profile.topk_width or profile.hidden_tasks_per_prompt or any(role_input.supports):
        raise owner.QualificationRunnerError("replay rollouts carry no top-k supports or hidden tasks")
    tokens = []
    for position, (output_id, teacher, raw_position) in enumerate(zip(
        role_input.output_ids, tuple(getattr(role_evidence, "tokens")), frame["top_logprobs"], strict=True,
    )):
        if raw_position:
            raise owner.QualificationRunnerError("rollout support differs from its T request")
        tokens.append(owner.RawTokenEvidence(
            position, output_id, owner.target_nll_from_f32(teacher.target_logprob), None, None,
        ))
    return owner.RawRolloutEvidence(tuple(tokens), ())
