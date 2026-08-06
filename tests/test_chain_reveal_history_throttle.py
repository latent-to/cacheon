"""Historical-read patience: throttle-only retries, everything else fail-closed."""

import pytest

from cacheon import chain


class _EndpointError(Exception):
    pass


def test_throttled_read_retries_then_succeeds(monkeypatch):
    sleeps = []
    monkeypatch.setattr(chain.time, "sleep", sleeps.append)
    calls = {"n": 0}

    def read():
        calls["n"] += 1
        if calls["n"] < 3:
            raise _EndpointError("Historical work rate limit exceeded")
        return "ok"

    assert chain._with_historical_patience(read, what="unit") == "ok"
    assert calls["n"] == 3
    assert sleeps == [pytest.approx(5.0), pytest.approx(10.0)]


def test_backoff_is_bounded(monkeypatch):
    sleeps = []
    monkeypatch.setattr(chain.time, "sleep", sleeps.append)

    def read():
        raise _EndpointError("Historical work rate limit exceeded")

    with pytest.raises(_EndpointError):
        chain._with_historical_patience(read, what="unit")
    assert len(sleeps) == chain._HISTORICAL_THROTTLE_ATTEMPTS - 1
    assert max(sleeps) <= chain._HISTORICAL_THROTTLE_MAX_DELAY_S


def test_non_throttle_error_stays_fail_closed(monkeypatch):
    monkeypatch.setattr(
        chain.time, "sleep", lambda _s: pytest.fail("must not sleep")
    )

    def read():
        raise _EndpointError("connection reset by peer")

    with pytest.raises(_EndpointError):
        chain._with_historical_patience(read, what="unit")


def test_reveal_history_error_passes_through_without_retry(monkeypatch):
    monkeypatch.setattr(
        chain.time, "sleep", lambda _s: pytest.fail("must not sleep")
    )
    calls = {"n": 0}

    def read():
        calls["n"] += 1
        raise chain.ChainRevealHistoryError("chain reveal storage has a malformed row")

    with pytest.raises(chain.ChainRevealHistoryError):
        chain._with_historical_patience(read, what="unit")
    assert calls["n"] == 1
