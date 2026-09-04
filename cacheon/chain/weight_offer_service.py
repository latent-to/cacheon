"""Standalone eval-side weight-offer producer.

Offer production is the one stage that must survive an evaluation pause.
serve-weights hands followers a projection stamped with an effective block, and
follow-weights refuses a projection older than its refresh window. While offer
production lived only inside the standing supervisor, stopping the supervisor
for an evaluation-side repair also stopped re-minting: the served offer aged
past the window and the signer failed closed on a stale projection while the
chain vector silently froze (2026-09-01, offer 8972660 against tip 8973560).

This service composes exactly the supervisor's weights stage against the same
sealed screen and weights authorities, and nothing else. It never signs. It
projects from the intake store and HTTP-pushes to serve-weights; follow-weights
still owns every chain write.

The intake store is an exclusive single-writer authority. The stage holds it
only to build the projection -- the metagraph read and the HTTP push happen
outside that window -- so this service coexists with a live intake controller.
A lock collision means the controller owns the database for this tick and is a
skipped pass, not a failure.

Exactly one offer producer may run against a database. When this service is
armed, the standing supervisor's ``enable_weights`` must be false, otherwise two
producers race to push different effective blocks for the same policy digest.
"""

from __future__ import annotations

import argparse
import signal
import sys
import threading
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable

from cacheon.chain import sealed_config
from cacheon.chain.standing_cpu_supervisor import StandingCpuSupervisorError
from cacheon.chain.standing_weights_stage import (
    WeightsStageConfig,
    compose_weight_offer_push,
    load_weights_config,
)

CONFIG_SCHEMA = "cacheon-weight-offer-service-config-v1"
CONFIG_DOMAIN = "cacheon.chain.weight-offer-service-config.v1"
_CONFIG_FIELDS = frozenset(
    {
        "max_consecutive_failures",
        "poll_ms",
        "restart_initial_backoff_ms",
        "restart_max_backoff_ms",
        "schema",
        "screen_dispatcher_config",
        "weights_stage_config",
    }
)


class WeightOfferServiceError(StandingCpuSupervisorError):
    """Offer-service authority or composition failed closed."""


class WeightOfferBusyError(WeightOfferServiceError):
    """The intake controller owns the database for this tick.

    Raised from the store factory so it passes through the weights stage
    unwrapped: the stage re-raises ``StandingCpuSupervisorError`` untouched and
    only wraps foreign exception types.
    """


_absolute_path = partial(sealed_config.absolute_path, error=WeightOfferServiceError)
_authority_file = partial(sealed_config.authority_file, error=WeightOfferServiceError)
_positive_int = partial(sealed_config.positive_int, error=WeightOfferServiceError)


def _closed_config(value: object, fields: frozenset[str], label: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != fields:
        raise WeightOfferServiceError(f"{label} fields are not closed")
    return value


@dataclass(frozen=True)
class WeightOfferServiceConfig:
    """Closed weight-offer-producer composition authority."""

    raw: dict[str, Any]
    screen_dispatcher_config: Path
    weights_stage_config: Path
    weights_stage: WeightsStageConfig
    poll_s: float
    restart_initial_backoff_s: float
    restart_max_backoff_s: float
    max_consecutive_failures: int

    @property
    def digest(self) -> str:
        from cacheon.stack_identity import canonical_digest

        return canonical_digest(CONFIG_DOMAIN, self.raw)


def load_offer_service_config(path: str | Path) -> WeightOfferServiceConfig:
    """Strictly reopen one immutable weight-offer-service authority file."""

    from cacheon.chain.remote_worker_spool import load_json

    config_path = _absolute_path(str(path), "config path")
    _authority_file(config_path, "weight offer service config")
    try:
        raw = load_json(config_path)
    except Exception as exc:
        raise WeightOfferServiceError(
            f"weight offer service config cannot reopen: {exc}"
        ) from None
    row = _closed_config(raw, _CONFIG_FIELDS, "weight offer service config")
    if row["schema"] != CONFIG_SCHEMA:
        raise WeightOfferServiceError(
            "weight offer service config schema is unsupported"
        )

    screen_path = _absolute_path(
        row["screen_dispatcher_config"], "screen_dispatcher_config"
    )
    _authority_file(screen_path, "screen dispatcher config")
    weights_path = _absolute_path(row["weights_stage_config"], "weights_stage_config")
    # load_weights_config re-checks the file shape; the sealed push credentials
    # it names are validated there too.
    weights_stage = load_weights_config(weights_path)

    poll_ms = _positive_int(row["poll_ms"], "poll_ms", maximum=3_600_000)
    initial_ms = _positive_int(
        row["restart_initial_backoff_ms"],
        "restart_initial_backoff_ms",
        maximum=600_000,
    )
    max_ms = _positive_int(
        row["restart_max_backoff_ms"], "restart_max_backoff_ms", maximum=600_000
    )
    if initial_ms > max_ms:
        raise WeightOfferServiceError("restart initial backoff exceeds its maximum")
    failures = _positive_int(
        row["max_consecutive_failures"], "max_consecutive_failures", maximum=1_000
    )

    return WeightOfferServiceConfig(
        raw=dict(row),
        screen_dispatcher_config=screen_path,
        weights_stage_config=weights_path,
        weights_stage=weights_stage,
        poll_s=poll_ms / 1000.0,
        restart_initial_backoff_s=initial_ms / 1000.0,
        restart_max_backoff_s=max_ms / 1000.0,
        max_consecutive_failures=failures,
    )


def build_offer_publisher(
    config: WeightOfferServiceConfig,
    *,
    store_factory: Callable[..., Any] | None = None,
) -> Callable[[], Any]:
    """Compose the supervisor's weights stage against the sealed authorities."""

    from cacheon.chain.intake import IntakeError, is_lock_collision
    from cacheon.chain.mainnet_screen_dispatcher import load_config
    from cacheon.chain.recoverable_intake import RecoverableFinalizedIntakeStore

    if type(config) is not WeightOfferServiceConfig:
        raise WeightOfferServiceError("weight offer service config is not typed")

    screen_config = load_config(config.screen_dispatcher_config)
    resolved: Callable[..., Any] = (
        RecoverableFinalizedIntakeStore if store_factory is None else store_factory
    )
    if not callable(resolved):
        raise WeightOfferServiceError("store_factory is not callable")

    def open_store() -> Any:
        try:
            return resolved(
                screen_config.intake_db,
                screen_config.policy,
                scope=screen_config.scope,
            )
        except IntakeError as exc:
            if is_lock_collision(exc):
                raise WeightOfferBusyError(
                    "intake controller owns the database this tick"
                ) from None
            raise

    return compose_weight_offer_push(
        config.weights_stage,
        store_factory=open_store,
        scope=screen_config.scope,
    )


def run_forever(
    publish: Callable[[], Any],
    stop: threading.Event,
    *,
    poll_s: float,
    restart_initial_backoff_s: float = 1.0,
    restart_max_backoff_s: float = 60.0,
    max_consecutive_failures: int = 10,
    wait: Callable[[float], bool] | None = None,
    on_event: Callable[[str, str], None] | None = None,
) -> None:
    """Publish on a loop until stopped; a busy database is a skipped pass."""

    if not callable(publish) or not isinstance(stop, threading.Event):
        raise WeightOfferServiceError("run_forever authorities are not exactly typed")
    waiter = stop.wait if wait is None else wait
    report = (lambda _kind, _detail: None) if on_event is None else on_event
    failures = 0
    backoff = float(restart_initial_backoff_s)
    while not stop.is_set():
        try:
            result = publish()
        except WeightOfferBusyError as exc:
            # The controller holds the single-writer lock. Retrying on the
            # normal cadence is correct; counting it as a failure would trip
            # the circuit breaker during ordinary intake activity.
            report("busy", str(exc))
            if waiter(poll_s):
                return
            continue
        except Exception as exc:
            failures += 1
            report("failed", f"{type(exc).__name__}: {exc} ({failures} consecutive)")
            if failures >= max_consecutive_failures:
                raise
            if waiter(backoff):
                return
            backoff = min(backoff * 2.0, float(restart_max_backoff_s))
            continue
        failures = 0
        backoff = float(restart_initial_backoff_s)
        if result is None:
            report("idle", "refresh window not reached")
        else:
            digest = getattr(result, "request_id", None) or ""
            report("pushed", f"{getattr(result, 'disposition', '')} {digest}".strip())
        if waiter(poll_s):
            return


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        required=True,
        help=(
            "absolute path to a closed weight-offer-service config naming the "
            "sealed screen and weights-stage authorities to compose"
        ),
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="run exactly one publish pass and exit (operator verification)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Load the sealed offer-service config and run the publish loop."""

    args = build_parser().parse_args(argv)
    try:
        config = load_offer_service_config(args.config)
        publish = build_offer_publisher(config)
    except StandingCpuSupervisorError as exc:
        print(f"WEIGHT-OFFER-SERVICE-ERROR: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(
            "WEIGHT-OFFER-SERVICE-ERROR: sealed composition failed: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 2

    def report(kind: str, detail: str) -> None:
        print(f"weight-offer-service {kind} {detail}".rstrip(), flush=True)

    if args.once:
        try:
            result = publish()
        except WeightOfferBusyError as exc:
            report("busy", str(exc))
            return 3
        except Exception as exc:
            print(
                f"WEIGHT-OFFER-SERVICE-ERROR: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            return 2
        report(
            "idle" if result is None else "pushed",
            "" if result is None else str(getattr(result, "request_id", "")),
        )
        return 0

    stop = threading.Event()

    def _stop(_signum: int, _frame: object | None) -> None:
        stop.set()

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    try:
        run_forever(
            publish,
            stop,
            poll_s=config.poll_s,
            restart_initial_backoff_s=config.restart_initial_backoff_s,
            restart_max_backoff_s=config.restart_max_backoff_s,
            max_consecutive_failures=config.max_consecutive_failures,
            on_event=report,
        )
    except Exception as exc:
        print(
            f"WEIGHT-OFFER-SERVICE-ERROR: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - module entrypoint
    raise SystemExit(main())


__all__ = [
    "CONFIG_DOMAIN",
    "CONFIG_SCHEMA",
    "WeightOfferBusyError",
    "WeightOfferServiceConfig",
    "WeightOfferServiceError",
    "build_offer_publisher",
    "build_parser",
    "load_offer_service_config",
    "main",
    "run_forever",
]
