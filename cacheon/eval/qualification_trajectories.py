"""Select replay turns and retain a small incumbent control for the existing pristine reference."""

from __future__ import annotations

import asyncio
import hashlib
import json
import struct
from pathlib import Path
from uuid import UUID
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cacheon.eval.qualification import QualificationProfile
    from cacheon.eval.qualification_runner import HiddenJudge
    from cacheon.eval.reference_quality import RawRolloutEvidence

from cacheon.eval.oci_outer_session import _fresh_id
from cacheon.eval.oci_session_protocol import BatchRequest
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
    """Enumerate the manifest's selected windows; never choose or truncate another window."""
    (load,) = replay.loads
    result = {}
    for index, session in enumerate(replay.slice.sessions[:load]):
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


def create_controls(value, controller, candidate, baseline_directory, candidate_directory,
                    candidate_quiescence, entropy_provider):
    """Generate selected incumbent controls after both timed reads and candidate teardown."""
    replay = value.resident_speed_plan.baseline.session_plan.replay
    baseline = _source_rows(replay, baseline_directory, controller.batch_rows)
    compared = _source_rows(replay, candidate_directory, candidate.session.batches)
    pairs = tuple((key, baseline[key].batch_index, compared[key].batch_index) for key in sorted(baseline))
    for key in baseline:
        b, c = baseline[key], compared[key]
        if (not b.input_ids_sha256 or b.input_ids_sha256 != c.input_ids_sha256
            or len(b.evidence.prompts[0].output_ids) != len(c.evidence.prompts[0].output_ids)):
            raise QualificationError("paired replay inputs or output budgets differ")
    entropy = entropy_provider(value.commitment, candidate_quiescence)
    selection = SelectionReceipt.reveal(
        value.commitment, secret=value.selection_secret, entropy=entropy,
        sealed_cohort_trajectory_digest=trajectory_digest(
            marginal_workload_digest(value.prepared.baseline_session_plan), pairs,
            controller.batch_rows, candidate.session.batches,
        ),
    )
    by_request = {baseline[key].request_id: key for key in selection.selected_prompt_digests}
    inputs = {}
    raw_path = baseline_directory / "window1" / "bridge.jsonl"
    with raw_path.open() as stream:
        for line in stream:
            row = json.loads(line)
            key = by_request.get(UUID(row["x_request_id"]).hex)
            if key is not None:
                inputs[key] = tuple(row["input_ids"])
    if set(inputs) != set(selection.selected_prompt_digests):
        raise QualificationError("selected replay canonical inputs are incomplete")

    async def generate():
        result = []
        async with controller.exchange() as channel:
            for key in selection.selected_prompt_digests:
                ids = inputs[key]
                b, c = baseline[key].evidence.prompts[0], compared[key].evidence.prompts[0]
                roles = tuple(ReferenceRoleInput(tuple(p.output_ids), ((),) * len(p.output_ids)) for p in (b, c))
                if (hashlib.sha256(struct.pack(f">{len(ids)}I", *ids)).hexdigest(),) != baseline[key].input_ids_sha256:
                    raise QualificationError("selected replay input bytes changed after execution")
                request = BatchRequest(
                    controller.session_id, controller.plan.launch_digest,
                    _fresh_id(controller.seen), _fresh_id(controller.seen), controller.next_batch_index,
                    (), len(b.output_ids), 0, controller.plan.temperature,
                    len(ids), False, (ids,),
                )
                row = await channel.execute(request, deadline=min(
                    controller.deadline, controller.clock() + controller.batch_timeout_s,
                ))
                controller.batch_rows.append(row)
                controller.last_host_time = row.response_completed_at
                control = row.evidence.prompts[0]
                result.append(ReferencePromptInput(key, "", (*roles, ReferenceRoleInput(
                    tuple(control.output_ids), ((),) * len(control.output_ids),
                )), ids))
        return tuple(result)
    return pairs, asyncio.run(generate()), entropy


def trajectory_rows(lifecycle: object):
    from cacheon.eval.crossover_runtime import ResidentMarginalLifecycleEvidence
    from cacheon.eval.oci_session_protocol import PromptEvidence
    from cacheon.eval.scoring import marginal_workload_digest

    if type(lifecycle) is not ResidentMarginalLifecycleEvidence:
        raise QualificationError("trajectory lifecycle is not typed")
    plan = lifecycle.prepared.baseline_session_plan
    if lifecycle.crossover.goodput is not None:
        return marginal_workload_digest(plan), selected_frames(lifecycle)
    batch_sets = tuple(
        lifecycle.role_batches(role) for role in lifecycle.role_names
    )
    workload = marginal_workload_digest(plan)
    from cacheon.eval.scoring import planned_prompt_texts
    from cacheon.eval.qualification import _validated_topk_position
    occurrences = iter(planned_prompt_texts(plan))
    rows = []
    for batch_index, prompts in enumerate(plan.prompt_batches):
        expected_tokens = plan.request_geometry(batch_index)[0]
        for prompt_index, prompt in enumerate(prompts):
            occurrence = next(occurrences)
            frames = []
            for batches in batch_sets:
                evidence = batches[batch_index].evidence.prompts[prompt_index]
                if (
                    type(evidence) is not PromptEvidence
                    or len(evidence.output_ids) != expected_tokens
                    or len(evidence.top_logprobs) != expected_tokens
                    or any(type(token) is not int or token < 0 for token in evidence.output_ids)
                    or any(len(position) != plan.top_logprobs_num for position in evidence.top_logprobs)
                ):
                    raise QualificationError("trajectory token/top-k coverage differs from workload")
                # A width-0 plan retains one empty support row per token; the
                # coverage check above already pins every row to length zero,
                # and the digest must seal that absence rather than reject it.
                if plan.top_logprobs_num:
                    topk = [
                        _validated_topk_position(position)
                        for position in evidence.top_logprobs
                    ]
                else:
                    topk = [[] for _ in evidence.top_logprobs]
                frames.append({"output_ids": list(evidence.output_ids), "top_logprobs": topk})
            rows.append((occurrence, frames))
    return workload, tuple(rows)



def validate_controls(evidence, plan):
    """Reopen source joins and all three selected rollouts from retained host evidence."""
    replay = plan.baseline.session_plan.replay
    pool = prompt_pool(replay)
    pairs = evidence.prompt_pairs
    if tuple(key for key, _, _ in pairs) != tuple(sorted(pool)):
        raise QualificationError("replay quality prompt pairs differ from the committed source pool")
    rows = (evidence.baseline_execution.session.batches, evidence.candidate_execution.session.batches)
    for role, arm in enumerate((plan.baseline, plan.candidate)):
        first = arm.session_plan.warmup_count
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
    for prompt in evidence.reference_inputs:
        baseline, candidate = joined[prompt.prompt_digest]
        if (prompt.roles[0].output_ids != baseline.evidence.prompts[0].output_ids
            or prompt.roles[1].output_ids != candidate.evidence.prompts[0].output_ids
            or (hashlib.sha256(prompt.input_bytes).hexdigest(),) != baseline.input_ids_sha256
            or any(any(role.supports) for role in prompt.roles)):
            raise QualificationError("selected reference inputs differ from their timed replay requests")


def _rollout(
    *,
    profile: QualificationProfile,
    prompt_digest: str,
    frame: dict[str, object],
    role_input: ReferenceRoleInput,
    role_evidence: object,
    hidden_judge: HiddenJudge,
) -> RawRolloutEvidence:
    """Retain teacher evidence and grade only hidden tasks declared by the profile."""
    from cacheon.eval import qualification_runner as owner

    tokens = []
    evidence_tokens = tuple(getattr(role_evidence, "tokens"))
    for position, (output_id, support, teacher, raw_position) in enumerate(zip(
        role_input.output_ids,
        role_input.supports,
        evidence_tokens,
        frame["top_logprobs"],
        strict=True,
    )):
        by_token = {row[1]: float(row[0]) for row in raw_position}
        if tuple(sorted(by_token)) != support:
            raise owner.QualificationRunnerError("rollout support differs from its T request")
        if support:
            rollout = owner.distribution_from_f32_logprobs(
                support,
                tuple(by_token[token] for token in support),
                true_argmax_token_id=raw_position[0][1],
            )
            teacher_distribution = owner.distribution_from_f32_logprobs(
                support,
                teacher.support_logprobs,
                true_argmax_token_id=teacher.true_argmax_token_id,
            )
        else:
            # Teacher-NLL-only mode (topk_width 0): the retained frames carry
            # no top-k and no distribution evidence exists to project.
            rollout = teacher_distribution = None
        tokens.append(owner.RawTokenEvidence(
            position,
            output_id,
            owner.target_nll_from_f32(teacher.target_logprob),
            teacher_distribution,
            rollout,
        ))
    tasks = owner._task_digests(profile, prompt_digest)
    if not tasks:
        return owner.RawRolloutEvidence(tuple(tokens), ())
    receipt = hidden_judge(
        prompt_digest=prompt_digest,
        output_ids=role_input.output_ids,
        task_digests=tasks,
    )
    if (
        type(receipt) is not owner.HiddenJudgeReceipt
        or receipt.binding_digest != owner._hidden_judge_binding(profile).digest
        or receipt.prompt_digest != prompt_digest
        or receipt.output_ids_digest
        != owner.hidden_judge_output_digest(prompt_digest, role_input.output_ids)
        or receipt.task_digests != tasks
    ):
        raise owner.QualificationRunnerError("hidden judge receipt differs from the sealed rollout")
    return owner.RawRolloutEvidence(
        tuple(tokens),
        tuple(
            owner.RawHiddenTaskResult(task, passed)
            for task, passed in zip(tasks, receipt.passed, strict=True)
        ),
    )
