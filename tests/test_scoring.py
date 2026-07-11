"""Unit tests for the noise-robust speedup scorer (optima/eval/scoring.py).

The whole point of this module is to make a sub-10% real win resolvable on a box
whose clocks can't be locked, and to refuse to crown on measurement noise. These
tests pin both halves: a genuine win passes, and noise alone never does.
"""

from optima.eval.scoring import (
    EXTERNAL_QUALITY_GATE_V1,
    ExternalFidelityMetrics,
    relative_spread,
    score_external_quality,
    score_speedup,
    score_system_token_match,
)


def test_relative_spread_two_reads_is_range_over_mean():
    # The default bookend has exactly 2 baseline reads; spread is the honest gap.
    assert relative_spread([100.0, 110.0]) == (10.0 / 105.0)


def test_relative_spread_unmeasurable_below_two():
    assert relative_spread([100.0]) == float("inf")
    assert relative_spread([]) == float("inf")


def test_genuine_win_on_stable_box_passes():
    # Baselines agree (1% spread), candidate is a clean 12% faster -> real win.
    v = score_speedup([100.0, 101.0], 113.0, min_margin=0.02, k=2.0, max_noise=0.10)
    assert v.confident
    assert v.passed_speedup
    assert v.speedup > 1.11


def test_noise_alone_does_not_crown():
    # No real improvement (candidate ~= baseline mean) but candidate happens to read
    # a hair high; the noise-derived bar must reject it.
    v = score_speedup([100.0, 108.0], 106.0, min_margin=0.02, k=2.0, max_noise=0.10)
    # baseline spread = 8/104 ~= 7.7% -> required ~= 1 + 2*0.077 = 1.154; speedup ~1.019.
    assert not v.passed_speedup
    assert v.required > 1.15


def test_too_noisy_is_no_decision_not_a_pass():
    # Bracketing baselines disagree by >max_noise: untrustworthy round, never crown,
    # even if the raw ratio looks huge.
    v = score_speedup([100.0, 140.0], 150.0, max_noise=0.10)
    assert not v.confident
    assert not v.passed_speedup
    assert "NO-DECISION" in v.detail


def test_single_baseline_cannot_be_confident():
    # The legacy 2-launch shape (one baseline) can't measure noise -> not crownable.
    v = score_speedup([100.0], 130.0)
    assert not v.confident
    assert not v.passed_speedup
    assert "single baseline" in v.detail


def test_min_margin_floor_applies_on_a_perfectly_stable_box():
    # Zero measured noise still requires clearing the floor margin.
    v = score_speedup([100.0, 100.0], 101.0, min_margin=0.02, k=2.0)
    assert v.noise == 0.0
    assert v.required == 1.02
    assert not v.passed_speedup  # 1.01 < 1.02
    v2 = score_speedup([100.0, 100.0], 103.0, min_margin=0.02, k=2.0)
    assert v2.passed_speedup  # 1.03 >= 1.02 on a stable box


def test_a_real_loss_is_a_loss_not_no_decision():
    v = score_speedup([100.0, 101.0], 90.0, max_noise=0.10)
    assert v.confident  # the box was stable; we trust the verdict
    assert not v.passed_speedup
    assert v.speedup < 1.0


def _fidelity(*, positions=100, mean=0.001, argmax=0, coverage=0.001, dropped=0):
    return ExternalFidelityMetrics(
        num_positions=positions,
        mean_kl=mean,
        max_kl=max(mean, 0.002),
        p99_kl=max(mean, 0.0015),
        argmax_disagreements=argmax,
        mean_coverage_dev=coverage,
        dropped_positions=dropped,
    )


def test_external_quality_recomputes_candidate_against_stock_control():
    verdict = score_external_quality(
        _fidelity(mean=0.01), _fidelity(mean=0.005),
        gate=EXTERNAL_QUALITY_GATE_V1,
    )
    assert verdict.passed
    assert verdict.kl_limit == 0.0575


def test_external_quality_rejects_missing_positions_and_drift():
    control = _fidelity(positions=100, mean=0.001)
    assert not score_external_quality(
        _fidelity(positions=0), control, gate=EXTERNAL_QUALITY_GATE_V1
    ).passed
    assert not score_external_quality(
        _fidelity(mean=0.5), control, gate=EXTERNAL_QUALITY_GATE_V1
    ).passed


def test_external_quality_gate_name_is_versioned():
    try:
        score_external_quality(_fidelity(), _fidelity(), gate="miner-selected")
    except ValueError as exc:
        assert "unsupported external quality gate" in str(exc)
    else:
        raise AssertionError("unknown external quality policy was accepted")


def test_output_token_match_rejects_boolean_threshold():
    try:
        score_system_token_match(0, 100, threshold=False)
    except ValueError as exc:
        assert "invalid output token-match" in str(exc)
    else:
        raise AssertionError("boolean token-match policy was accepted as zero")


def test_system_token_match_uses_exact_stock_control_without_new_margin():
    # The arena floor remains load-bearing when stock clears it. If this exact
    # bracket's stock bookend only reproduces 96%, the candidate must meet that
    # measured control—not an invented control+epsilon envelope.
    passed, detail = score_system_token_match(
        96,
        100,
        threshold=0.99,
        stock_matched=96,
        stock_total=100,
    )
    assert passed
    assert "policy_floor=0.990000" in detail
    assert "stock=96/100 rate=0.960000" in detail
    assert "limit=0.960000" in detail

    failed, _ = score_system_token_match(
        95,
        100,
        threshold=0.99,
        stock_matched=96,
        stock_total=100,
    )
    assert not failed

    policy_failure, policy_detail = score_system_token_match(
        98,
        100,
        threshold=0.99,
        stock_matched=100,
        stock_total=100,
    )
    assert not policy_failure
    assert "limit=0.990000" in policy_detail


def test_system_token_match_rejects_partial_or_invalid_stock_control():
    for stock in ((96, None), (None, 100), (101, 100)):
        try:
            score_system_token_match(
                100,
                100,
                threshold=0.99,
                stock_matched=stock[0],
                stock_total=stock[1],
            )
        except ValueError as exc:
            assert "stock token-match evidence" in str(exc)
        else:
            raise AssertionError("invalid stock control was accepted")
