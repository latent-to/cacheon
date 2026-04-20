"""Phase 5 Part A — Validator loop.

Wires state + chain + challenger selection together. The GPU-side eval
(Phase 5 Part B) is injected as a callable so this module stays testable
without SSH, torch, or a GPU.

Flow per tick:
  1. Fetch metagraph + revealed commitments from chain
  2. Parse commitments → `{uid: CommitmentRecord}`
  3. Select new challengers (filter out already-evaluated + pre-rejected)
  4. Record any new AST-sandbox rejections in state
  5. If no challengers: set_weights(king), sleep, return
  6. If challengers: call `eval_fn(...)` (Phase 5 Part B) to get
     `list[EvaluationRecord]`; record each, possibly dethroning the king
  7. Set weights to current king
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
from .state import EvaluationRecord, ValidatorState

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Eval hook — Phase 5 Part B will implement this over SSH
# --------------------------------------------------------------------------- #


class EvalFn(Protocol):
    """Evaluate one or more challengers and return per-challenger results.

    The Phase 5 Part B implementation ships `prompts.json` + policy
    pointers to the GPU pod over SSH, runs the harness + scoring, and
    pulls `results.json` back.
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
    """Default stub — explodes loudly so Part A never silently crowns
    a challenger without real eval behind it."""
    raise NotImplementedError(
        f"GPU eval is not wired (Phase 5 Part B). "
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
    state_dir = validator_config.STATE_DIR,
    dry_run: bool = validator_config.DRY_RUN,
    chain_attempts: int = validator_config.CHAIN_RETRY_ATTEMPTS,
    chain_delay_s: int = validator_config.CHAIN_RETRY_DELAY_S,
) -> TickResult:
    """Run one tick of the validator loop and persist state afterwards.

    The caller owns `subtensor`, `wallet`, and `state` — this function
    doesn't construct them so tests can wire plain mocks.
    """
    metagraph, current_block, block_hash = fetch_metagraph(
        subtensor, netuid,
        attempts=chain_attempts, delay_s=chain_delay_s,
    )
    revealed = fetch_revealed_commitments(
        subtensor, netuid,
        attempts=chain_attempts, delay_s=chain_delay_s,
    )
    commitments = build_commitments(metagraph, revealed)
    state.last_scan_block = current_block

    logger.info(
        "Scan @ block %d: %d hotkey(s), %d valid commitment(s)",
        current_block, len(metagraph.hotkeys), len(commitments),
    )

    challenger_set = select_challengers(
        state, commitments.values(), precheck=precheck,
    )

    for com, reason in challenger_set.newly_rejected:
        state.record_precheck_failure(com.hotkey, com.commit_block, reason)

    logger.info(
        "Challenger selection: %d new, %d pre-rejected, %d already known",
        len(challenger_set.challengers),
        len(challenger_set.newly_rejected),
        len(challenger_set.already_known),
    )

    evaluations_recorded: list[EvaluationRecord] = []
    king_changed = False

    if challenger_set.challengers:
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
            dethroned = state.record_evaluation(ev)
            evaluations_recorded.append(ev)
            if dethroned:
                king_changed = True
                logger.info(
                    "👑 New king: UID %d (hotkey %s…, score=%.4f)",
                    ev.uid, ev.hotkey[:16], ev.score,
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
            "[dry-run] would set_weights(winner_uid=%d) @ block %d",
            state.king.uid, current_block,
        )
        weights_set = True
    else:
        try:
            set_weights(
                subtensor, wallet, netuid,
                n_uids=len(metagraph.hotkeys),
                winner_uid=state.king.uid,
                attempts=chain_attempts, delay_s=chain_delay_s,
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
    state_dir = validator_config.STATE_DIR,
    poll_interval_s: int = validator_config.POLL_INTERVAL_S,
    dry_run: bool = validator_config.DRY_RUN,
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
        "state_dir=%s, dry_run=%s",
        netuid, poll_interval_s, state_dir, dry_run,
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
            )
            logger.info(
                "Tick OK @ block %d in %.1fs (king_changed=%s, "
                "new_evals=%d)",
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
                time.time() - tick_started, exc,
            )

        if stop():
            break
        sleep_fn(poll_interval_s)
