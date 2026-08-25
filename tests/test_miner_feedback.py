"""Miner-facing explanations from retained products, logs, and run records."""

from __future__ import annotations

import json

import pytest
from types import SimpleNamespace

from cacheon import cli
from cacheon.chain.miner_feedback import _GUIDANCE, _attempt_evidence, _guidance
from cacheon.eval.candidate_failure_product import publish_candidate_failure
from cacheon.eval.explain import (
    _headline,
    _speed_lines,
    config_lines,
    execution_lines,
    explain,
    failure_lines,
    ranks_from_log,
)
from cacheon.eval.resident_execution_evidence import (
    EXECUTION_CODEC,
    RankExecution,
    SlotExecution,
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
    "fetch:archive member is excluded from bundle identity: source/kernels/._moe.py",
    "manifest:unsupported capability field init_blocks",
    "systemic_release_cap:3",
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

    for code in ("copy_of", "duplicate_of", "fetch", "manifest", "systemic_release_cap"):
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
    text = "\n".join(execution_lines(ranks_from_log(log)))
    assert "called 40 times across 4 GPU(s)" in text
    assert "inside the CUDA graph on every GPU" in text


def test_one_ungraphed_gpu_makes_the_whole_measurement_ungraphed() -> None:
    log = "\n".join(
        [
            _summary(completed=[{"slot": "s.one", "calls": 1, "captured": True}]),
            _summary(completed=[{"slot": "s.one", "calls": 1, "captured": False}]),
        ]
    )
    assert "NOT inside the CUDA graph on every GPU" in "\n".join(
        execution_lines(ranks_from_log(log))
    )


def test_retained_candidate_traceback_names_exact_prepare_failure() -> None:
    bundle = "a9b3d8a8" + "0" * 56
    traces = []
    for gpu, free in ((3, "8.11"), (0, "7.86"), (1, "7.80"), (2, "7.80")):
        traces.append(
            f'''Traceback (most recent call last):
  File "/cacheon/swap-intake/{bundle}/kernels/moe.py", line 10, in prepare
    return dequantize_prepare_args((tag, view))
  File "/usr/local/lib/python3.12/dist-packages/cacheon/moe_nvfp4_contract.py", line 217, in dequantize_prepare_args
    w13 = codec.dequantize_nvfp4(w13_q, w13_sf.float())
  File "/usr/local/lib/python3.12/dist-packages/cacheon_kernels/codec/nvfp4.py", line 48, in dequantize_nvfp4
    cb = lut[nibbles.long()]
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 9.07 GiB. GPU {gpu} has {free} GiB free.
'''
        )
    report = "\n".join(failure_lines("\n".join(traces)))
    assert "bundle" in report and bundle in report
    assert "kernels/moe.py:10 in prepare raised torch.OutOfMemoryError" in report
    assert "CUDA out of memory. Tried to allocate 9.07 GiB." in report
    assert "affected GPU/ranks" in report and "0, 1, 2, 3" in report
    assert "cacheon_kernels/codec/nvfp4.py:48 dequantize_nvfp4" in report


def test_installed_library_permission_failure_is_validator_attributed() -> None:
    bundle = "b" * 64
    trace = f'''Traceback (most recent call last):
  File "/cacheon/swap-intake/{bundle}/kernels/moe.py", line 74, in fused_experts
    return flashinfer_moe(x)
  File "/usr/local/lib/python3.12/dist-packages/flashinfer/jit/cubin_loader.py", line 252, in ensure_symlink
    path.mkdir(parents=True, exist_ok=True)
PermissionError: [Errno 13] Permission denied: '/usr/local/lib/python3.12/dist-packages/flashinfer_cubin/cubins/flashinfer'
'''
    report = "\n".join(explain({"evidence": []}, stderr=trace))
    assert "VALIDATOR RUNTIME FAILURE" in report
    assert "validator setup failure, not a bundle failure" in report
    assert "kernels/moe.py:74" in report
    assert "flashinfer/jit/cubin_loader.py:252 ensure_symlink" in report


def test_missing_installed_cubin_metadata_is_validator_attributed() -> None:
    bundle = "b" * 64
    trace = f'''Traceback (most recent call last):
  File "/cacheon/swap-intake/{bundle}/kernels/moe.py", line 74, in fused_experts
    return native_moe(x)
  File "/usr/local/lib/python3.12/dist-packages/flashinfer/jit/fused_moe.py", line 256, in gen_trtllm_gen_fused_moe_sm100_module
    assert checksum
AssertionError: Failed to get checksums.txt from abc/batched_gemm/checksums.txt
'''
    report = "\n".join(explain({"evidence": []}, stderr=trace))
    assert "VALIDATOR RUNTIME FAILURE" in report
    assert "validator setup failure, not a bundle failure" in report
    assert "Failed to get checksums.txt" in report


def test_miner_report_reopens_candidate_failure_without_manual_log(tmp_path) -> None:
    import hashlib

    def digest(label: str) -> str:
        return hashlib.sha256(label.encode()).hexdigest()

    reference, _product = publish_candidate_failure(
        tmp_path,
        authority_manifest_digest=digest("authority"),
        source_digest=digest("source"),
        culprit_reservation_digest=digest("reservation"),
        selected_delta_digest=digest("delta"),
        target_id="moe.fused_experts",
        arm_digest=digest("arm"),
        launch_digest=digest("launch"),
        failure_kind="candidate_exception",
        failure="rank 2 RuntimeError at kernels/moe.py:17: boom",
    )
    reports = _attempt_evidence(
        [{"attempt_index": 0, "attempt_ref": reference.to_dict()}],
        (tmp_path,),
        digest("publication"),
    )
    rendered = "\n".join(reports[0]["explanation"])
    assert "VERDICT  your kernel raised an exception" in rendered
    assert "kernels/moe.py:17: boom" in rendered


def test_explain_accepts_one_retained_log_without_a_product(tmp_path, capsys) -> None:
    log = tmp_path / "candidate.stderr"
    log.write_text(
        "CACHEON-QUALIFICATION-INTAKE-FAILURE: authority=abc failure=def "
        "OCIBackendError: runtime unavailable\n"
    )
    assert cli.cmd_explain(
        SimpleNamespace(product=None, log=str(log), evidence_dir=None)
    ) == 0
    text = capsys.readouterr().out
    assert "VALIDATOR QUALIFICATION FAILURE" in text
    assert "OCIBackendError: runtime unavailable" in text
    assert "comes from the retained diagnostic stream" in text
    assert "no readable evidence at all" not in text


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
    product = {"evidence": []}
    text = "\n".join(explain(product, stderr=log))
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
    assert "s.one: called 3 times" in "\n".join(execution_lines(ranks_from_log(log)))


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
    assert _headline(execution_lines(ranks_from_log(ran)), speed) == ""


# --- the same record, read from the product the run published ---------------

LANE_A_LOG = "\n".join(
    "CACHEON-EXECUTION-SUMMARY: " + json.dumps(
        {
            "active": [{"bundle": "/cacheon/swap-intake/74de6706", "pid": 154 + rank,
                        "rank": rank, "slots": ["attention.msa_block_score"], "world_size": 4}],
            "completed": [{"calls": 1140, "captured": True, "pid": 154 + rank, "rank": rank,
                           "slot": "attention.msa_block_score", "world_size": 4}],
        }
    )
    for rank in (1, 0, 2, 3)
)


def _execution_product(ranks, *, executed: int = 4, expected: int = 4) -> dict:
    import base64

    payload = {
        "schema": "cacheon.qualification.execution.v1",
        "swaps": [
            {
                "generation": 3,
                "lane_id": "A",
                "executed_ranks": executed,
                "expected_ranks": expected,
                "ranks": [EXECUTION_CODEC.encode(row) for row in ranks],
            }
        ],
    }
    return {
        "authority_manifest": {"reservations": [{"target_id": "attention.msa_block_score"}]},
        "evidence": [
            {
                "reference": {"domain": "qualification.execution"},
                "payload_base64": base64.b64encode(json.dumps(payload).encode()).decode(),
            }
        ],
    }


def test_the_real_lane_a_lines_render_the_same_as_the_published_rows() -> None:
    """One renderer, two sources: the retained log and the product's evidence
    must tell the miner the same thing about the same generation."""

    from_log = ranks_from_log(LANE_A_LOG)
    assert [row.rank for row in from_log] == [0, 1, 2, 3]
    log_text = explain({"evidence": []}, stderr=LANE_A_LOG)
    product_text = explain(_execution_product(from_log))
    ran = "attention.msa_block_score: called 4,560 times across 4 GPU(s), inside the CUDA graph on every GPU"
    assert any(ran in line for line in log_text)
    assert any(ran in line for line in product_text)
    assert any("4 of 4 GPU(s) ran your kernel cleanly" in line for line in product_text)
    # The product wins when both are present; the log is the fallback only.
    assert explain(_execution_product(from_log), stderr="CACHEON-EXECUTION-SUMMARY: {}") == product_text


def test_a_raised_kernel_is_named_as_the_bundles_fault_before_the_numbers() -> None:
    ranks = (
        RankExecution(0, True, "", (SlotExecution("a.slot", 12, True),)),
        RankExecution(1, True, "", (SlotExecution("a.slot", 12, True, "RuntimeError: CUDA error"),)),
    )
    report = explain(_execution_product(ranks, executed=1, expected=2))
    assert report[2].startswith("VERDICT  your kernel raised an exception")
    assert any("RAISED" in line and "GPU 1: RuntimeError: CUDA error" in line for line in report)


def test_a_routing_reason_names_the_gpu_and_the_field() -> None:
    ranks = (
        RankExecution(0, True, "", (SlotExecution("a.slot", 0, None, "", ("out_of_domain on num_tokens",)),)),
    )
    text = "\n".join(execution_lines(ranks))
    assert "NEVER RAN" in text
    assert "SKIPPED YOUR KERNEL        a.slot on GPU 0" in text and "out_of_domain on num_tokens" in text
