"""Main validator tick: chain -> challengers -> optional eval -> set weights.

This module ties together `chain` (Bittensor RPC), `state` (persisted
scores), and `challengers` (who still needs a run). It does **not**
manage Docker containers or GPU resources: actual evaluation is supplied
by ``eval_fn`` (see `docker_eval.make_eval_fn` for the production
implementation). That keeps imports here free of Docker/GPU code and
easy to test with a stub.

One pass through the loop:
  1. Fetch metagraph and revealed commitments from the chain.
  2. Parse commitments into ``{uid: CommitmentRecord}``.
  3. Select challengers not yet evaluated and not pre-rejected.
  4. Record any new precheck rejections in state.
  5. If there are no challengers: set weights to the current king, sleep, return.
  6. If there are challengers: call ``eval_fn(...)`` -> ``list[EvaluationRecord]``;
     merge results into state (king may change).
  7. Set weights to the current king.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Callable, Iterable, Protocol

from . import config as validator_config
from .challengers import (
    ChallengerSet,
    PrecheckFn,
    allow_all_precheck,
    select_challengers,
)
from .chain import (
    CommitmentRecord,
    build_commitments,
    fetch_metagraph,
    fetch_revealed_commitments,
    set_weights,
)
from .state import EvaluationRecord, ValidatorState, append_king_history

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Eval hook -- pluggable Docker evaluation (see `docker_eval`)
# --------------------------------------------------------------------------- #


class EvalFn(Protocol):
    """Evaluate one or more challengers and return per-challenger results.

    Production wiring uses ``docker_eval.make_eval_fn`` which pulls
    Docker images, starts containers, measures speed and correctness
    against the vLLM baseline, and returns ``EvaluationRecord`` rows.
    """

    def __call__(
        self,
        challengers: list[CommitmentRecord],
        *,
        current_block: int,
        block_hash: str | None,
    ) -> list[EvaluationRecord]: ...


def not_implemented_eval(
    challengers: list[CommitmentRecord],
    *,
    current_block: int,
    block_hash: str | None,
) -> list[EvaluationRecord]:
    """Default stub -- fails fast so the loop never records scores without a real ``eval_fn``."""
    raise NotImplementedError(
        f"eval_fn is not configured. "
        f"Got {len(challengers)} challenger(s) at block {current_block}. "
        f"Pass an `eval_fn=...` to `run_once(...)` / `run_forever(...)`."
    )


# --------------------------------------------------------------------------- #
# Per-tick orchestration
# --------------------------------------------------------------------------- #


@dataclass
class TickResult:
    current_block: int
    block_hash: str | None
    n_commitments: int
    challenger_set: ChallengerSet
    evaluations_recorded: list[EvaluationRecord]
    king_changed: bool
    weights_set: bool
    weights_set_error: str | None = None


def run_once(
    *,
    subtensor,
    wallet,
    state: ValidatorState,
    netuid: int = validator_config.NETUID,
    eval_fn: EvalFn = not_implemented_eval,
    precheck: PrecheckFn = allow_all_precheck,
    state_dir=validator_config.STATE_DIR,
    dry_run: bool = validator_config.DRY_RUN,
    version_key: int = validator_config.VERSION_KEY,
    chain_attempts: int = validator_config.CHAIN_RETRY_ATTEMPTS,
    chain_delay_s: int = validator_config.CHAIN_RETRY_DELAY_S,
) -> TickResult:
    """Run one tick of the validator loop and persist state afterwards.

    The caller owns `subtensor`, `wallet`, and `state` — this function
    doesn't construct them so tests can wire plain mocks.
    """
    metagraph, current_block, block_hash = fetch_metagraph(
        subtensor,
        netuid,
        attempts=chain_attempts,
        delay_s=chain_delay_s,
    )
    revealed = fetch_revealed_commitments(
        subtensor,
        netuid,
        attempts=chain_attempts,
        delay_s=chain_delay_s,
    )
    commitments = build_commitments(metagraph, revealed)
    state.last_scan_block = current_block

    logger.info(
        "Scan @ block %d: %d hotkey(s), %d valid commitment(s)",
        current_block,
        len(metagraph.hotkeys),
        len(commitments),
    )

    # UID slots can be recycled: if the king's hotkey deregistered, this UID
    # may now belong to someone else — don't set_weights without a hotkey match.
    if state.king is not None:
        live_hotkey = (
            metagraph.hotkeys[state.king.uid]
            if state.king.uid < len(metagraph.hotkeys)
            else None
        )
        if live_hotkey != state.king.hotkey:
            logger.warning(
                "King UID %d hotkey changed on chain (expected %s…, got %s). "
                "King was deregistered or UID recycled — clearing throne.",
                state.king.uid,
                state.king.hotkey[:16],
                (live_hotkey or "<out-of-range>")[:16],
            )
            state.king = None

    challenger_set = select_challengers(
        state,
        commitments.values(),
        precheck=precheck,
    )

    for com, reason in challenger_set.newly_rejected:
        state.record_precheck_failure(com.hotkey, com.commit_block, reason)

    if challenger_set.deferred:
        logger.warning(
            "Challenger selection: %d deferred — will retry next tick",
            len(challenger_set.deferred),
        )

    logger.info(
        "Challenger selection: %d new, %d pre-rejected, %d deferred, %d already known",
        len(challenger_set.challengers),
        len(challenger_set.newly_rejected),
        len(challenger_set.deferred),
        len(challenger_set.already_known),
    )
    for com in challenger_set.challengers:
        logger.info("  🆕 UID %d  %s  %s", com.uid, com.hotkey, com.image)

    evaluations_recorded: list[EvaluationRecord] = []
    king_changed = False

    if challenger_set.challengers:
        if dry_run:
            logger.info(
                "[dry-run] Would evaluate %d challenger(s): uid=%s",
                len(challenger_set.challengers),
                [c.uid for c in challenger_set.challengers],
            )
        else:
            try:
                results = eval_fn(
                    challenger_set.challengers,
                    current_block=current_block,
                    block_hash=block_hash,
                )
            except NotImplementedError:
                raise
            except Exception as exc:
                logger.exception("eval_fn raised: %s", exc)
                results = []

            for ev in results:
                prev_king = state.king
                outcome = state.record_evaluation(ev, current_block=current_block)
                evaluations_recorded.append(outcome.stored)
                _icon = "❌" if outcome.stored.disqualify_reason else "📊"
                logger.info(
                    "%s UID %d (hotkey %s…) score=%.4f threshold=%.4f "
                    "(dq=%s, dethroned=%s)",
                    _icon,
                    outcome.stored.uid,
                    outcome.stored.hotkey[:16],
                    outcome.stored.score,
                    outcome.dethrone_threshold,
                    outcome.stored.disqualify_reason or "no",
                    outcome.dethroned,
                )
                if outcome.dethroned:
                    king_changed = True
                    logger.info(
                        "👑 New king: UID %d (hotkey %s…, score=%.4f, "
                        "beat threshold=%.4f)",
                        outcome.stored.uid,
                        outcome.stored.hotkey[:16],
                        outcome.stored.score,
                        outcome.dethrone_threshold,
                    )
                    append_king_history(
                        state_dir,
                        outcome.stored,
                        prev_king,
                        current_block,
                        outcome.dethrone_threshold,
                    )

    state.save(state_dir)

    weights_set = False
    weights_error: str | None = None
    if state.king is None:
        logger.info(
            "No king yet — skipping set_weights (waiting for first "
            "non-DQ'd evaluation)."
        )
    elif dry_run:
        logger.info(
            "[dry-run] would set_weights(winner_uid=%d, version_key=%d) @ block %d",
            state.king.uid,
            version_key,
            current_block,
        )
        weights_set = True
    else:
        try:
            set_weights(
                subtensor,
                wallet,
                netuid,
                n_uids=len(metagraph.hotkeys),
                winner_uid=state.king.uid,
                version_key=version_key,
                attempts=chain_attempts,
                delay_s=chain_delay_s,
            )
            state.last_weights_set_block = current_block
            state.save(state_dir)
            weights_set = True
        except Exception as exc:
            weights_error = str(exc)
            logger.error("set_weights failed this tick: %s", exc)

    return TickResult(
        current_block=current_block,
        block_hash=block_hash,
        n_commitments=len(commitments),
        challenger_set=challenger_set,
        evaluations_recorded=evaluations_recorded,
        king_changed=king_changed,
        weights_set=weights_set,
        weights_set_error=weights_error,
    )


# --------------------------------------------------------------------------- #
# Long-running loop
# --------------------------------------------------------------------------- #


StopSignal = Callable[[], bool]


def run_forever(
    *,
    subtensor,
    wallet,
    state: ValidatorState,
    netuid: int = validator_config.NETUID,
    eval_fn: EvalFn = not_implemented_eval,
    precheck: PrecheckFn = allow_all_precheck,
    state_dir=validator_config.STATE_DIR,
    poll_interval_s: int = validator_config.POLL_INTERVAL_S,
    dry_run: bool = validator_config.DRY_RUN,
    version_key: int = validator_config.VERSION_KEY,
    stop: StopSignal | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> None:
    """Run forever (or until `stop()` returns True).

    On a fatal tick error (chain timeout after retries, unhandled
    exception) we log and sleep rather than crash — validator uptime
    matters more than surfacing the error.
    """
    if stop is None:
        stop = lambda: False

    logger.info(
        "Validator loop starting: netuid=%d, poll_interval_s=%d, "
        "state_dir=%s, dry_run=%s, version_key=%d",
        netuid,
        poll_interval_s,
        state_dir,
        dry_run,
        version_key,
    )

    while not stop():
        tick_started = time.time()
        try:
            result = run_once(
                subtensor=subtensor,
                wallet=wallet,
                state=state,
                netuid=netuid,
                eval_fn=eval_fn,
                precheck=precheck,
                state_dir=state_dir,
                dry_run=dry_run,
                version_key=version_key,
            )
            logger.info(
                "Tick OK @ block %d in %.1fs (king_changed=%s, new_evals=%d)",
                result.current_block,
                time.time() - tick_started,
                result.king_changed,
                len(result.evaluations_recorded),
            )
        except NotImplementedError:
            raise
        except Exception as exc:
            logger.exception(
                "Tick failed after %.1fs: %s — sleeping then retrying",
                time.time() - tick_started,
                exc,
            )

        if stop():
            break
        sleep_fn(poll_interval_s)
