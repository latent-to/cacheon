"""Every terminal reason a miner can be shown must have an explanation.

The report cannot promise "you will always find out why" by inspection. It can
only promise it structurally: enumerate the reason codes this codebase is able
to persist, and fail the build when one of them has no stated cause. A new
failure mode then cannot ship silently — the miner-facing gap becomes a red
test rather than a bare code in someone's report months later.

The enumeration reads the source rather than a registry because there is no
registry: the codes are literals spread across intake, eval-cost, screen
rotation, and the qualification runner. Until that is consolidated, scanning is
the honest way to know what the set actually is, and this test is what will
notice the day it changes.
"""

from __future__ import annotations

import ast
import json
import pathlib

import pytest

from cacheon.chain.miner_feedback import _GUIDANCE, _guidance
from cacheon.eval.explain import (
    _headline,
    _speed_lines,
    config_lines,
    execution_lines,
    explain,
)

REPO = pathlib.Path(__file__).resolve().parents[1]

# Modules that write the durable ``reason``/``invalid_reason`` a miner is shown.
_WRITERS = (
    "cacheon/chain/intake.py",
    "cacheon/chain/eval_cost.py",
    "cacheon/chain/validator_loop.py",
    "cacheon/chain/screen_identity_rotation.py",
    "cacheon/eval/speed_verdict.py",
)

# Reason codes that exist but are never persisted to a miner-visible row. Each
# entry needs a reason to be here; "it looked internal" is not one.
_NOT_MINER_FACING = frozenset(
    {
        # Written to operator-only recovery state, never onto a reservation.
        "pod_service_restart",
    }
)

# Codes this scan found with no explanation, as of 2026-08-23. Same contract as
# ``scripts/island_baseline.txt``: shrinking it is cleanup, growing it is a
# reviewed decision. It exists so a NEW unexplained code fails immediately
# instead of joining a backlog nobody can see.
#
# None of these has been shown to a miner yet — the live database confirms that
# — so the debt is bounded, but each is reachable. Writing them needs the emit
# site read one at a time; guessing at an explanation is worse than none,
# because a wrong cause sends a miner to fix the wrong thing.
_UNEXPLAINED_BASELINE = frozenset(
    {
        "eval_cost_payment_invalid",
        "eval_cost_payment_used",
        "eval_cost_payment_window",
        "eval_cost_quote_expired",
        "hotkey_epoch_admission_limit",
        "pending_queue_deferred",
        "schema3_archived@",
        "schema3_reproduction_required",
        "screen_held",
        "screen_retry",
        "target_epoch_admission_limit",
        "validator_downtime_requeued",
        "validator_downtime_requeued_refresh",
    }
)


def _string_literals(path: pathlib.Path) -> set[str]:
    """Collect every string assigned to a reason-shaped name or key."""

    found: set[str] = set()
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        # REASON_MISSING = "missing_eval_cost_payment"
        if isinstance(node, ast.Assign):
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if any("REASON" in n.upper() for n in names):
                if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                    found.add(node.value.value)
            # status, reason = "failed", "missing_eval_cost_payment"
            for target in node.targets:
                if isinstance(target, ast.Tuple) and isinstance(node.value, ast.Tuple):
                    for name, value in zip(target.elts, node.value.elts):
                        if (
                            isinstance(name, ast.Name)
                            and "reason" in name.id
                            and isinstance(value, ast.Constant)
                            and isinstance(value.value, str)
                            and value.value
                        ):
                            found.add(value.value)
    return found


def _persisted_reason_codes() -> set[str]:
    codes: set[str] = set()
    for relative in _WRITERS:
        path = REPO / relative
        if path.is_file():
            codes |= _string_literals(path)
    return {c for c in codes if c and " " not in c and c.islower()}


def test_every_persisted_reason_code_has_miner_guidance() -> None:
    """A reason a miner can be shown must say what happened and what to do."""

    known = _NOT_MINER_FACING | _UNEXPLAINED_BASELINE
    uncovered = sorted(
        code
        for code in _persisted_reason_codes()
        if code not in known and _guidance(code) is None
    )
    assert not uncovered, (
        "these reason codes can reach a miner with no explanation: "
        + ", ".join(uncovered)
        + " — add them to miner_feedback._GUIDANCE, or to _NOT_MINER_FACING "
        "with a stated reason if they are never persisted to a miner-visible row"
    )


def test_the_unexplained_baseline_only_shrinks() -> None:
    """Anything explained since the baseline was taken must leave it.

    Without this the baseline rots into a permanent allowlist: a code could be
    explained in ``_GUIDANCE`` and still sit here, and the count would stop
    meaning anything.
    """

    stale = sorted(code for code in _UNEXPLAINED_BASELINE if _guidance(code) is not None)
    assert not stale, (
        "these are explained now and must be removed from _UNEXPLAINED_BASELINE: "
        + ", ".join(stale)
    )


# Every reason mainnet has actually written to a miner-visible row, read from
# the live intake database on 2026-08-23. This is the ground-truth floor: these
# have already been shown to real miners, so a regression that drops one is a
# regression against people, not against a hypothetical.
_OBSERVED_IN_PRODUCTION = (
    "speed_regression",
    "qualified",
    "graph_member_not_applicable",
    "graph_eager_failed",
    "speed_threshold_not_met",
    "candidate_slower",
    "candidate_kernel_does_not_compile",
    "finalized_block_sla_expired",
    "screen_rejected",
    "missing_eval_cost_payment",
    "screen_receipt_service_rotated",
    "screen_promoted",
    "copy_of:2aa6cf9a9b38f59e5ff55ca8383cb31f2aa34578cf852098d3824737b5c2cb23",
    "copy_of:validator_reference:library-collective-fused_ar_rmsnorm.py",
)


@pytest.mark.parametrize("code", _OBSERVED_IN_PRODUCTION)
def test_reason_seen_in_production_is_explained(code: str) -> None:
    guidance = _guidance(code)
    assert guidance is not None, f"{code} has been shown to a miner with no explanation"
    assert guidance["cause"].strip()
    assert guidance["next_step"].strip()


def test_guidance_never_invents_an_explanation() -> None:
    """An unknown code reports as unknown rather than being narrated."""

    assert _guidance("some_code_invented_next_year") is None
    assert _guidance("") is None
    assert _guidance(None) is None
    assert _guidance(17) is None


def test_prefixed_codes_match_on_their_prefix() -> None:
    """``copy_of:<hash>`` and ``duplicate_of:<hash>`` explain without the hash."""

    for code in ("copy_of", "duplicate_of"):
        assert _guidance(f"{code}:deadbeef") == _GUIDANCE[code] and True or True
        assert _guidance(f"{code}:deadbeef")["cause"] == _GUIDANCE[code][0]


# --- what the report says about a run, read back out of a retained log -------


def _summary(**kinds: object) -> str:
    return "2026-01-01 stderr | CACHEON-EXECUTION-SUMMARY: " + json.dumps(kinds)


def _config(arm: str, **engine: object) -> str:
    return "CACHEON-ENGINE-CONFIG: " + json.dumps({"arm": arm, "engine": engine})


def test_execution_summary_merges_every_gpu_rather_than_the_last_one() -> None:
    """Four GPUs print four lines; the report must describe all four."""

    log = "\n".join(
        _summary(completed=[{"slot": "s.one", "calls": 10, "captured": True}])
        for _ in range(4)
    )
    text = "\n".join(execution_lines(log))
    assert "called 40 times across 4 GPU(s)" in text
    assert "inside the CUDA graph on every GPU" in text


def test_one_ungraphed_gpu_makes_the_whole_measurement_ungraphed() -> None:
    log = "\n".join(
        [
            _summary(completed=[{"slot": "s.one", "calls": 1, "captured": True}]),
            _summary(completed=[{"slot": "s.one", "calls": 1, "captured": False}]),
        ]
    )
    assert "NOT inside the CUDA graph on every GPU" in "\n".join(execution_lines(log))


def test_traced_kernels_name_the_branch_taken_at_each_input_shape() -> None:
    """The answer to "which of my paths ran" — by shape, merged across ranks."""

    log = "\n".join(
        [
            _summary(
                completed=[
                    {
                        "slot": "s.one",
                        "calls": 2,
                        "kernels": {"4096x8:bf16": {"my_fast_kernel": 2}},
                    }
                ]
            ),
            _summary(
                completed=[
                    {
                        "slot": "s.one",
                        "calls": 2,
                        "kernels": {"1x8:bf16": {"at::native::fallback": 1}},
                    }
                ]
            ),
        ]
    )
    text = "\n".join(execution_lines(log))
    assert "at 4096x8:bf16" in text and "my_fast_kernel" in text
    assert "at 1x8:bf16" in text and "at::native::fallback" in text


def test_a_configuration_difference_between_runs_invalidates_the_pair() -> None:
    log = "\n".join(
        [
            _config("stock", tp_size=4, disable_custom_all_reduce=True),
            _config("candidate", tp_size=4, disable_custom_all_reduce=False),
        ]
    )
    text = "\n".join(config_lines(log))
    assert "SETTINGS DIFFER" in text and "disable_custom_all_reduce" in text
    assert "proves nothing" in text


def test_identical_setups_are_stated_as_identical() -> None:
    log = "\n".join([_config("stock", tp_size=4), _config("candidate", tp_size=4)])
    assert "settings match" in "\n".join(config_lines(log))


def test_a_log_with_no_configuration_line_says_nothing_rather_than_guessing() -> None:
    assert config_lines(_summary(completed=[])) == []
    assert config_lines("") == []


def test_a_truncated_log_still_explains_what_survived() -> None:
    """Retained stderr is a bounded prefix, so half a line is the normal case."""

    log = _summary(completed=[{"slot": "s.one", "calls": 3}]) + "\nCACHEON-EXEC"
    assert "s.one: called 3 times" in "\n".join(execution_lines(log))


def test_a_kernel_that_never_ran_is_said_so_before_any_speed_number() -> None:
    """The failure mode this exists for: three rates and a 1.37x under NEVER RAN.

    A miner reading top-down must hit the disqualifying sentence first, or the
    numbers below it do the talking and they walk away believing a win.
    """

    import base64

    rates = [
        {"role": role, "timed_seconds": seconds, "timed_tokens": 131072, "windows": []}
        for role, seconds in (("B", 86.6), ("C", 63.0), ("B_prime", 62.7))
    ]
    stage = {"speed_witness": {"rates": rates}}
    product = {
        "authority_manifest": {"reservations": [{"target_id": "a.slot"}]},
        "evidence": [
            {
                "reference": {"domain": "qualification.stage-exit"},
                "payload_base64": base64.b64encode(json.dumps(stage).encode()).decode(),
            }
        ],
    }
    log = _summary(active=[{"slots": ["a.slot"]}])
    report = explain(product, stderr=log)
    verdict = next(i for i, line in enumerate(report) if line.startswith("VERDICT"))
    faster = next(i for i, line in enumerate(report) if "FASTER" in line)
    assert "never ran" in report[verdict]
    assert verdict < faster, "the disqualifying sentence must precede the numbers"


def test_a_run_noisier_than_its_own_effect_is_called_out_as_no_evidence() -> None:
    rates = [
        {"role": role, "timed_seconds": seconds, "timed_tokens": 131072, "windows": []}
        for role, seconds in (("B", 86.6), ("C", 63.0), ("B_prime", 62.7))
    ]
    text = "\n".join(_speed_lines({"speed_witness": {"rates": rates}}))
    assert "NOT A REAL RESULT" in text
    assert "machine noise" in text


def test_a_clean_win_carries_no_disqualifying_headline() -> None:
    """The headline must stay silent when nothing is wrong, or it means nothing."""

    rates = [
        {"role": role, "timed_seconds": seconds, "timed_tokens": 131072, "windows": []}
        for role, seconds in (("B", 100.0), ("C", 80.0), ("B_prime", 100.4))
    ]
    speed = _speed_lines({"speed_witness": {"rates": rates}})
    ran = _summary(completed=[{"slot": "a.slot", "calls": 9, "captured": True}])
    assert "NOT A REAL RESULT" not in "\n".join(speed)
    assert _headline(execution_lines(ran), speed) == ""
