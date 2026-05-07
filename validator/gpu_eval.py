"""GPU eval entrypoint: S3 download -> eval all challengers -> S3 upload.

Zero bittensor dependency. Reads ``eval_job.json`` (written by the CPU
orchestrator) from the state directory, runs baseline + challenger
evaluations, writes results into ``state.json``, then uploads everything
back to Hippius S3.

Usage (inside Docker):
    python -m validator.gpu_eval

Env vars:
    CACHEON_STATE_DIR          (default: /app/state)
    CACHEON_MODEL_VOLUME       (default: /models)
    CACHEON_BASELINE_IMAGE     (default: vllm/vllm-openai:latest)
    CACHEON_BASELINE_DIGEST    (required)
    CACHEON_GPU_COUNT          (default: 4)
    HIPPIUS_ACCESS_KEY         (required for S3)
    HIPPIUS_SECRET_KEY         (required for S3)
    CACHEON_S3_BUCKET          (default: cacheon-validator)
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from . import config as validator_config
from .chain import CommitmentRecord
from .docker_eval import (
    _dq_record,
    _max_model_len,
    evaluate_challenger,
    run_baseline_if_needed,
)
from .eval_schema import EvalJob
from .state import ValidatorState, append_king_history

logger = logging.getLogger(__name__)


def _configure_logging(state_dir: str) -> None:
    logs_dir = Path(state_dir) / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"gpu_eval_{ts}.log"

    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        force=True,
    )
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logging.getLogger().addHandler(fh)
    logger.info("Logging to %s", log_path)


def main() -> int:
    state_dir = str(validator_config.STATE_DIR)
    _configure_logging(state_dir)

    model_volume = validator_config.MODEL_VOLUME
    baseline_image = validator_config.BASELINE_IMAGE
    baseline_digest = validator_config.BASELINE_DIGEST
    gpu_count = validator_config.GPU_COUNT

    logger.info(
        "GPU eval starting: state_dir=%s, model_volume=%s, "
        "baseline_image=%s, gpu_count=%d",
        state_dir,
        model_volume,
        baseline_image,
        gpu_count,
    )

    if not baseline_digest:
        import subprocess

        try:
            result = subprocess.run(
                [
                    "docker",
                    "image",
                    "inspect",
                    baseline_image,
                    "--format",
                    "{{index .RepoDigests 0}}",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0 and "@" in result.stdout:
                baseline_digest = result.stdout.strip().split("@", 1)[1]
                logger.info("Auto-detected baseline digest: %s", baseline_digest)
        except Exception as exc:
            logger.debug("Could not auto-detect baseline digest: %s", exc)

        if not baseline_digest:
            logger.error(
                "CACHEON_BASELINE_DIGEST is not set and could not be "
                "auto-detected. Pull the baseline image first."
            )
            return 1

    # S3 download
    try:
        from .sync import download

        download(state_dir)
    except Exception as exc:
        logger.error("S3 download failed: %s", exc)
        return 2

    # Load state and eval job
    state = ValidatorState.load(state_dir)
    eval_job = EvalJob.load(state_dir)
    if eval_job is None or not eval_job.challengers:
        logger.info("No challengers in eval job, nothing to do")
        return 0

    block = eval_job.block
    block_hash = eval_job.block_hash
    logger.info(
        "Eval job: block=%d, block_hash=%s, %d challenger(s)",
        block,
        block_hash[:16] if block_hash else "None",
        len(eval_job.challengers),
    )

    if gpu_count <= 0:
        from .docker_eval import _detect_gpu_count

        gpu_count = _detect_gpu_count()
        if gpu_count <= 0:
            logger.error("Could not detect GPU count via nvidia-smi")
            return 3
        logger.info("Auto-detected %d GPU(s)", gpu_count)

    # Generate prompts
    from .prompts import sample_prompts

    mml = _max_model_len(gpu_count)
    prompts = sample_prompts(block_hash, n=10, max_context_tokens=mml)
    logger.info("Generated %d prompts (max_model_len=%d)", len(prompts), mml)

    # Run baseline
    cache_dir = Path(state_dir) / "baseline_cache"
    try:
        baseline = run_baseline_if_needed(
            prompts,
            baseline_image=baseline_image,
            baseline_digest=baseline_digest,
            model_volume=model_volume,
            gpu_count=gpu_count,
            cache_dir=cache_dir,
            block_hash=block_hash,
            state_dir=state_dir,
        )
    except Exception as exc:
        logger.exception("Baseline failed: %s", exc)
        for ci in eval_job.challengers:
            com = CommitmentRecord(
                uid=ci.uid,
                hotkey=ci.hotkey,
                commit_block=ci.commit_block,
                image=ci.image,
                digest=ci.digest,
                raw="",
            )
            record = _dq_record(com, block, f"baseline_failed: {exc}")
            state.record_evaluation(record, current_block=block)
        state.save(state_dir)
        _upload_state(state_dir)
        return 4

    # Evaluate challengers
    for ci in eval_job.challengers:
        com = CommitmentRecord(
            uid=ci.uid,
            hotkey=ci.hotkey,
            commit_block=ci.commit_block,
            image=ci.image,
            digest=ci.digest,
            raw="",
        )
        logger.info(
            "⚔️  Evaluating challenger UID %d (%s) image=%s",
            com.uid,
            com.hotkey[:16],
            com.image,
        )
        record = evaluate_challenger(
            com,
            prompts,
            baseline,
            model_volume=model_volume,
            startup_timeout_s=600,
            per_prompt_timeout_s=120,
            n_warmup=2,
            current_block=block,
            state_dir=state_dir,
        )
        prev_king = state.king
        outcome = state.record_evaluation(record, current_block=block)
        icon = "❌" if outcome.stored.disqualify_reason else "📊"
        logger.info(
            "%s UID %d score=%.4f (dq=%s, dethroned=%s)",
            icon,
            outcome.stored.uid,
            outcome.stored.score,
            outcome.stored.disqualify_reason or "no",
            outcome.dethroned,
        )
        if outcome.dethroned:
            logger.info(
                "👑 New king: UID %d (score=%.4f)",
                outcome.stored.uid,
                outcome.stored.score,
            )
            append_king_history(
                state_dir,
                outcome.stored,
                prev_king,
                block,
                outcome.dethrone_threshold,
            )

    state.save(state_dir)
    logger.info(
        "State saved. King: %s", f"UID {state.king.uid}" if state.king else "none"
    )

    # S3 upload
    _upload_state(state_dir)
    logger.info("GPU eval complete")
    return 0


def _upload_state(state_dir: str) -> None:
    try:
        from .sync import upload

        upload(state_dir)
    except Exception as exc:
        logger.error("S3 upload failed: %s", exc)


if __name__ == "__main__":
    raise SystemExit(main())
