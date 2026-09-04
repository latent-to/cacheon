from decimal import Decimal

import pytest

from cacheon.chain.baseline_band import (
    BaselineBandError,
    RetainedHalfRates,
    baseline_band_verdict,
)


def _half(
    reservation_id: str, index: int, speedup: str, *reads: float, arena: str = "a" * 64
) -> RetainedHalfRates:
    return RetainedHalfRates(
        reservation_id, arena, index, Decimal(speedup),
        tuple(Decimal(str(read)) for read in reads),
    )


def test_credited_half_is_the_lower_speedup_and_its_reads_decide():
    # The 2026-09-04 arena: one pair drew the slow baseline state in both
    # halves, one drew it only in a B_prime read of the half that did not
    # set its credit, and one read normally throughout.
    halves = (
        _half("slow", 0, "1.1359", 2031.0),
        _half("slow", 1, "1.1321", 2021.8),
        _half("peer", 0, "1.0402", 2233.6, 2014.2),
        _half("peer", 1, "1.0190", 2289.4, 2274.9),
        _half("other", 0, "1.0291", 2245.6, 2235.8),
        _half("other", 1, "1.0225", 2253.2, 2248.6),
    )

    slow = baseline_band_verdict(halves, "slow")
    assert slow.credited_index == 1
    assert slow.credited_speedup == Decimal("1.1321")
    assert slow.baseline_reads == 10
    assert slow.baseline_median == Decimal("2240.7")
    assert slow.floor == Decimal("2128.7")
    assert slow.out_of_band
    assert "OUT OF BAND" in slow.describe()

    peer = baseline_band_verdict(halves, "peer")
    assert peer.credited_index == 1
    assert not peer.out_of_band
    assert "inside the band" in peer.describe()


def test_band_needs_both_halves_and_enough_arena_reads():
    with pytest.raises(BaselineBandError, match="both halves"):
        baseline_band_verdict((_half("x", 0, "1.05", 2000.0),), "x")
    thin = (
        _half("x", 0, "1.05", 2000.0),
        _half("x", 1, "1.04", 2000.0),
        _half("y", 0, "1.03", 2200.0, arena="b" * 64),
        _half("y", 1, "1.02", 2200.0, arena="b" * 64),
    )
    with pytest.raises(BaselineBandError, match="baseline reads"):
        baseline_band_verdict(thin, "x")
    with pytest.raises(BaselineBandError, match="both halves"):
        baseline_band_verdict(thin, "z")
