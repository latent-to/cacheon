"""The validator loop: chain commitments → fetch → evaluate → settle → weights.

One ``run_pass`` is the whole referee cycle against live chain state; ``run_validator``
repeats it forever with per-pass fault isolation (one bad submission or one RPC
hiccup must never kill the loop — reject, record, continue).

Trust boundaries, in order:
- the PAYLOAD is untrusted (fail-quiet decode, ``optima.chain.payload``);
- the ARTIFACT is untrusted (size-capped fetch, hostile-archive extraction, and the
  extracted tree must re-hash to the committed content hash, ``optima.chain.fetch``);
- the BUNDLE is untrusted (evaluated out-of-process via ``python -m optima.cli`` —
  this module never imports miner code, same discipline as ``cmd_verify``);
- weight POLICY is not this module's business: it consumes
  ``PerTargetSettleResult.weights`` from the Ledger so the emission scheme (currently
  per-target king-of-the-hill; NOT winner-take-all-forever) swaps without touching
  chain I/O.

Every processed submission is recorded in the Ledger (scores for settlement,
EvalRecords for the audit trail + retry suppression), so restarts re-derive state
from the ledger file instead of replaying work — the "re-derive, don't replay"
pattern from SUBNET_BLUEPRINT §2.
"""

from __future__ import annotations

import fcntl
import hashlib
import errno
import json
import logging
import math
import os
import re
import stat
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from optima import chain
from optima.arenas import ARENAS, ArenaProfile, derive_prompt_seed, get_arena
from optima.chain.fetch import (
    FetchError,
    FetchTransientError,
    fetch_bundle,
    fetch_bundle_from_local_file_for_testing,
)
from optima.chain.payload import (
    SubmissionRef,
    decode_payload,
    decode_payload_for_testing,
)

logger = logging.getLogger("optima.chain.validator")

# A "round" is one settlement window; by default it advances with the subnet tempo.
DEFAULT_ROUND_BLOCKS = 360
# Re-assert weights at least this often even when unchanged (activity cutoff prunes
# validators that go quiet; the subnet's cutoff is thousands of blocks — one tempo
# of headroom is comfortable).
DEFAULT_WEIGHTS_REFRESH_BLOCKS = 360

EVAL_TIMEOUT_S = 3600.0


class WeightSafetyError(RuntimeError):
    """Previously published emissions cannot be silently left stale."""

    validator_fault = True
    retryable = False


class QualificationAuthorityError(RuntimeError):
    """Validator-owned crown evidence is internally incomplete or inconsistent."""

    validator_fault = True
    retryable = False


class LedgerLockError(RuntimeError):
    """Another validator process already owns this ledger's whole referee pass."""

    validator_fault = True
    retryable = False


class SubmissionPolicyError(ValueError):
    """A miner-controlled manifest/source product violates declared intake policy."""


@dataclass
class EvalOutcome:
    """What an evaluator says about one fetched bundle."""
    passed: bool
    score: float
    kl_mean: float = 0.0
    target: str = ""
    mode: str = ""
    member_slots: tuple[str, ...] = ()
    detail: str = ""
    crownable: bool = False
    # A quality-passing but statistically unconfident bracket is a NO-DECISION,
    # not a terminal failed submission. Infrastructure failures use the same
    # deferred path and persistent backoff rather than poisoning dedup forever.
    confident: bool = True
    passed_speedup: bool = False
    passed_timed_quality: bool = False
    passed_warmup_quality: bool = False
    retryable: bool = False
    arena_name: str = ""
    arena_fingerprint: str = ""
    arena_bracket: str = ""
    regime: str = ""
    bundle_hash: str = ""
    sglang_version: str = ""
    validator_image: str = ""
    referee_source_digest: str = ""
    referee_tree_digest: str = ""
    model_revision: str = ""
    model_manifest_digest: str = ""
    model_content_digest: str = ""
    host_attestation_sha256: str = ""
    chain_scope: str = ""
    validator_hotkey: str = ""
    evaluation_id: str = ""
    miner_hotkey: str = ""
    settlement_round_id: int = 0
    evaluation_block: int = 0
    qualification_evidence_sha256: str = ""
    prompt_seed: int = 0
    prompt_engine_version: str = ""
    prompt_seed_scheme: str = ""
    seed_round_id: int = 0
    seed_block: int = 0
    seed_block_hash: str = ""
    quality_evidence: str = ""


@dataclass(frozen=True)
class EvaluationContext:
    """Validator-owned facts supplied to, then checked against, the evaluator."""

    arena: ArenaProfile
    bundle_hash: str
    round_id: int
    block: int
    block_hash: str
    prompt_seed: int
    chain_scope: str
    validator_hotkey: str
    evaluation_id: str
    miner_hotkey: str
    settlement_round_id: int
    evaluation_block: int


# An Evaluator takes the fetched bundle directory and returns an EvalOutcome.
# It must never raise for a *bad bundle* — that's a failed outcome; raising is
# reserved for validator-side faults (which fail the pass, not the submission).
Evaluator = Callable[[Path, EvaluationContext], EvalOutcome]


@dataclass
class PassResult:
    block: int
    round_id: int
    seen: int = 0
    new: list[str] = field(default_factory=list)        # content hashes processed
    rejected: dict[str, str] = field(default_factory=dict)  # hash/hotkey -> reason
    deferred: dict[str, str] = field(default_factory=dict)  # retry/backoff/hold
    held: dict[str, str] = field(default_factory=dict)  # operator release required
    copies: list[str] = field(default_factory=list)
    evaluated: dict[str, bool] = field(default_factory=dict)  # hash -> passed
    weights: dict[str, float] = field(default_factory=dict)
    # Weight publication is a state machine, not one boolean. ``submitted`` means
    # the SDK accepted a new extrinsic; ``pending`` means its vector has not yet
    # been proven by an authoritative chain read; ``confirmed`` means the live
    # sparse row and last-update block prove the desired vector is applied.
    weights_submitted: bool = False
    weights_pending: bool = False
    weights_held: bool = False
    weights_confirmed: bool = False
    # Compatibility headline: true only when a submission made in this pass was
    # also confirmed in this pass. An accepted-but-pending CR commit is not pushed.
    weights_pushed: bool = False


# --------------------------------------------------------------------------- #
# Evaluators — all out-of-process
# --------------------------------------------------------------------------- #


def _finalized_block_number(subtensor) -> int:
    """Return the canonical finalized height, or fail the validator pass."""
    direct = getattr(subtensor, "get_finalized_block_number", None)
    if callable(direct):
        value = direct()
    else:
        substrate = getattr(subtensor, "substrate", None)
        if substrate is None:
            raise RuntimeError("subtensor exposes no finalized-chain API")
        head = substrate.get_chain_finalised_head()
        if not head:
            raise RuntimeError("chain returned no finalized head")
        value = substrate.get_block_number(head)
    if type(value) is not int or value < 0:
        raise RuntimeError(f"invalid finalized block number: {value!r}")
    return value


def _canonical_block_hash(subtensor, block: int) -> str:
    value = subtensor.get_block_hash(int(block))
    if isinstance(value, bytes):
        value = "0x" + value.hex()
    text = str(value or "")
    if re.fullmatch(r"0x[0-9a-fA-F]{64}", text) is None:
        raise RuntimeError(f"chain returned an invalid block hash at {block}: {value!r}")
    return text.lower()


def _expected_submission_exception(exc: BaseException) -> bool:
    """Whether ``exc`` is defined miner-input rejection rather than controller fault."""

    from optima.competition import CompetitionError
    from optima.device_component import DeviceComponentError
    from optima.manifest import ManifestError
    from optima.system_patch import SystemPatchError

    expected: tuple[type[BaseException], ...] = (
        ManifestError,
        CompetitionError,
        DeviceComponentError,
        SystemPatchError,
        FileNotFoundError,
        IsADirectoryError,
        UnicodeError,
        SyntaxError,
        json.JSONDecodeError,
    )
    try:
        import tomllib

        expected += (tomllib.TOMLDecodeError,)
    except (ImportError, AttributeError):  # pragma: no cover - Python <3.11
        pass
    return isinstance(exc, expected)


def _resolve_bundle_competition(bundle_dir: Path):
    """Independently derive settlement identity from the fetched manifest."""
    from optima.competition import resolve_competition
    from optima.manifest import load_manifest

    try:
        return resolve_competition(
            load_manifest(bundle_dir),
            for_settlement=True,
            warn_legacy=False,
        )
    except BaseException as exc:
        if _expected_submission_exception(exc):
            raise SubmissionPolicyError(str(exc)) from exc
        raise


def _context_identity(
    context: EvaluationContext,
    *,
    qualification_evidence_sha256: str = "",
) -> dict[str, object]:
    """Static evaluation authority plus the optional post-run result digest.

    The qualification digest cannot exist until after B/C/B' evidence is graded,
    so this deliberately does not call ``host_attestation_context``. The prepared
    QualificationReport builds that final, exact sidecar context instead.
    """

    arena = context.arena
    return {
        "arena_name": arena.name,
        "arena_fingerprint": arena.fingerprint,
        "arena_bracket": arena.bracket,
        "regime": arena.workload.regime,
        "bundle_hash": context.bundle_hash,
        "sglang_version": arena.sglang_version,
        "validator_image": arena.validator_image,
        "referee_source_digest": arena.referee_source_digest,
        "referee_tree_digest": arena.referee_tree_digest,
        "model_revision": arena.model_revision,
        "model_manifest_digest": arena.model_manifest_digest,
        "model_content_digest": arena.model_content_digest,
        "chain_scope": context.chain_scope,
        "validator_hotkey": context.validator_hotkey,
        "evaluation_id": context.evaluation_id,
        "miner_hotkey": context.miner_hotkey,
        "settlement_round_id": context.settlement_round_id,
        "evaluation_block": context.evaluation_block,
        "qualification_evidence_sha256": qualification_evidence_sha256,
        "prompt_seed": context.prompt_seed,
        "prompt_engine_version": arena.workload.prompt_engine_version,
        "prompt_seed_scheme": arena.workload.prompt_seed_scheme,
        "seed_round_id": context.round_id,
        "seed_block": context.block,
        "seed_block_hash": context.block_hash,
    }


def verify_evaluator(device: str = "cpu", dtype: str = "float32",
                     timeout_s: float = EVAL_TIMEOUT_S) -> Evaluator:
    """PLUMBING-ONLY evaluator: runs ``optima verify`` and scores pass/fail as
    1.0/0.0. It proves the loop end-to-end (fetch → gates → settle → weights)
    without a GPU; the "score" is NOT a throughput measurement and must never be
    used for real emissions — production wires ``command_evaluator`` to the full
    ``optima evaluate`` gate chain on the GPU box."""
    def _run(bundle_dir: Path, context: EvaluationContext) -> EvalOutcome:
        try:
            competition = _resolve_bundle_competition(bundle_dir)
        except SubmissionPolicyError as exc:
            return EvalOutcome(
                False,
                0.0,
                detail=f"invalid competition target: {exc}",
            )
        identity = dict(
            target=competition.target or "",
            mode=competition.mode or "",
            member_slots=competition.members,
            **_context_identity(context),
        )
        cmd = [sys.executable, "-m", "optima.cli", "verify", str(bundle_dir),
               "--device", device, "--dtype", dtype]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
        except subprocess.TimeoutExpired:
            return EvalOutcome(
                False,
                0.0,
                detail=f"verify timed out after {timeout_s}s",
                **identity,
            )
        tail = (proc.stdout + proc.stderr)[-2000:]
        return EvalOutcome(proc.returncode == 0, 1.0 if proc.returncode == 0 else 0.0,
                           detail=tail,
                           quality_evidence="verify-only plumbing; non-crownable",
                           **identity)
    return _run


def command_evaluator(template: str, *, arena: ArenaProfile,
                      timeout_s: float = EVAL_TIMEOUT_S) -> Evaluator:
    """Development-only arbitrary command evaluator.

    ``template`` is shell text with
    ``{bundle}`` and ``{report}`` placeholders. A zero exit is necessary but not
    sufficient: the command MUST atomically write the complete typed qualification
    report emitted by ``optima evaluate --report {report}``. Missing, stale,
    malformed, incomplete, or inconsistent reports fail closed.  Even a valid
    report is deliberately non-crownable: a shell child controls the report path
    and is not the validator-owned authenticated OCI controller.
    """
    from optima.eval.qualification import QualificationReport, QualificationReportError

    expected_arena = get_arena(arena.name)
    if expected_arena.fingerprint != arena.fingerprint:
        raise ValueError("command evaluator arena is not the registered profile")

    def _run(bundle_dir: Path, context: EvaluationContext) -> EvalOutcome:
        if context.arena.fingerprint != expected_arena.fingerprint:
            return EvalOutcome(False, 0.0, detail="evaluator arena/context mismatch")
        report = bundle_dir.parent / f".{bundle_dir.name}.report.json"
        # A crashed retry must not inherit a valid report from an earlier process.
        report.unlink(missing_ok=True)
        cmd = template.format(
            bundle=str(bundle_dir),
            report=str(report),
            arena=expected_arena.name,
            prompt_seed=context.prompt_seed,
            round_id=context.round_id,
            block=context.block,
            block_hash=context.block_hash,
            bundle_hash=context.bundle_hash,
            chain_scope=context.chain_scope,
            validator_hotkey=context.validator_hotkey,
            evaluation_id=context.evaluation_id,
            miner_hotkey=context.miner_hotkey,
            settlement_round_id=context.settlement_round_id,
            evaluation_block=context.evaluation_block,
        )
        try:
            proc = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                                  timeout=timeout_s)
        except subprocess.TimeoutExpired:
            return EvalOutcome(False, 0.0, detail=f"eval timed out after {timeout_s}s")
        output = (proc.stdout + proc.stderr)[-2000:]
        if proc.returncode != 0:
            return EvalOutcome(
                False, 0.0,
                detail=f"eval command exited {proc.returncode}\n{output}"[-2000:],
            )
        if not report.exists():
            return EvalOutcome(
                False, 0.0,
                detail=f"eval command succeeded but wrote no qualification report\n{output}"[-2000:],
            )
        try:
            qualification = QualificationReport.read(report)
        except QualificationReportError as exc:
            return EvalOutcome(
                False, 0.0,
                detail=f"invalid qualification report: {exc}\n{output}"[-2000:],
            )
        reported_scope = (
            qualification.arena_name,
            qualification.arena_fingerprint,
            qualification.arena_bracket,
            qualification.regime,
            qualification.bundle_hash,
            qualification.sglang_version,
            qualification.validator_image,
            qualification.referee_source_digest,
            qualification.referee_tree_digest,
            qualification.model_revision,
            qualification.model_manifest_digest,
            qualification.model_content_digest,
            qualification.chain_scope,
            qualification.validator_hotkey,
            qualification.evaluation_id,
            qualification.miner_hotkey,
            qualification.settlement_round_id,
            qualification.evaluation_block,
            qualification.prompt_seed,
            qualification.prompt_engine_version,
            qualification.prompt_seed_scheme,
            qualification.seed_round_id,
            qualification.seed_block,
            qualification.seed_block_hash,
        )
        expected_scope_values = _context_identity(context)
        expected_scope = tuple(expected_scope_values[key] for key in (
            "arena_name", "arena_fingerprint", "arena_bracket", "regime",
            "bundle_hash", "sglang_version", "validator_image",
            "referee_source_digest", "referee_tree_digest", "model_revision",
            "model_manifest_digest",
            "model_content_digest",
            "chain_scope", "validator_hotkey", "evaluation_id",
            "miner_hotkey", "settlement_round_id", "evaluation_block",
            "prompt_seed", "prompt_engine_version", "prompt_seed_scheme",
            "seed_round_id", "seed_block", "seed_block_hash",
        ))
        if reported_scope != expected_scope:
            return EvalOutcome(
                False,
                0.0,
                detail=("qualification arena/bundle/seed mismatch: "
                        f"report={reported_scope!r} expected={expected_scope!r}"),
            )
        return EvalOutcome(
            qualification.passed_quality,
            0.0,
            kl_mean=qualification.kl_mean,
            target=qualification.target,
            mode=qualification.mode,
            member_slots=qualification.member_slots,
            detail=output,
            crownable=False,
            confident=qualification.confident,
            passed_speedup=qualification.passed_speedup,
            passed_timed_quality=qualification.passed_timed_quality,
            passed_warmup_quality=qualification.passed_warmup_quality,
            host_attestation_sha256=(
                qualification.host_attestation_sha256
            ),
            quality_evidence=(
                "development command report; non-authoritative for settlement: "
                + qualification.quality_evidence
            )[:4096],
            **_context_identity(
                context,
                qualification_evidence_sha256=(
                    qualification.qualification_evidence_sha256
                ),
            ),
        )

    # ``run_pass`` also enforces this for direct API users, not just the CLI.
    setattr(_run, "requires_positive_margin", True)
    return _run


def oci_evaluator(
    *,
    arena: ArenaProfile,
    source_dir: str | Path,
    model_dir: str | Path,
    artifact_root: str | Path,
    scratch_root: str | Path,
    gpu_devices: tuple[int, ...],
    timeout_s: float | None = None,
) -> Evaluator:
    """Validator-owned production evaluator: prebuild, then fresh OCI B/C/B'.

    No miner process writes a settlement report.  The trusted controller owns the
    arena config and prompt receipt, authenticates every worker result, recomputes
    qualification, and returns the typed outcome directly to the ledger loop.
    """
    from optima.eval.oci_backend import (
        OCIBackendError,
        OCICandidateArtifactError,
        OCIInfrastructureError,
        OCILauncher,
        profile_for_arena,
    )
    from optima.eval.oci_outer_session import (
        OuterSessionCandidateError,
        OuterSessionInfrastructureError,
    )
    from optima.eval.qualification import (
        QualificationReport,
        QualificationReportError,
    )

    expected_arena = get_arena(arena.name)
    if expected_arena.fingerprint != arena.fingerprint:
        raise ValueError("OCI evaluator arena is not the registered profile")
    source = Path(source_dir).resolve(strict=True)
    model = Path(model_dir).resolve(strict=True)
    artifact_base = Path(artifact_root).expanduser().resolve()
    scratch = Path(scratch_root).expanduser().resolve()
    artifact_base.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)
    if len(gpu_devices) != arena.tp_size or len(set(gpu_devices)) != len(gpu_devices):
        raise ValueError(
            f"OCI evaluator requires exactly {arena.tp_size} distinct GPU IDs"
        )

    def _run(bundle_dir: Path, context: EvaluationContext) -> EvalOutcome:
        try:
            competition = _resolve_bundle_competition(bundle_dir)
        except SubmissionPolicyError as exc:
            return EvalOutcome(False, 0.0, detail=f"invalid competition target: {exc}")
        identity = dict(
            target=competition.target or "",
            mode=competition.mode or "",
            member_slots=competition.members,
            **_context_identity(context),
        )
        if context.arena.fingerprint != expected_arena.fingerprint:
            return EvalOutcome(
                False, 0.0, detail="OCI evaluator arena/context mismatch", **identity
            )
        candidate_artifacts = (
            artifact_base / expected_arena.fingerprint / context.bundle_hash
        )
        candidate_artifacts.mkdir(parents=True, exist_ok=True)
        try:
            profile = profile_for_arena(
                expected_arena,
                source_dir=source,
                model_dir=model,
                artifact_dir=candidate_artifacts,
                scratch_root=scratch,
                gpu_devices=gpu_devices,
                bundle_dir=bundle_dir,
                competition_target=competition.target,
            )
            launcher = OCILauncher(profile)
            # Establish the whole evaluation deadline once. The prebuild method
            # applies its own smaller arena phase cap without resetting/tightening
            # the B/C/B' budget to the prebuild duration.
            launcher.begin_evaluation(timeout_s=timeout_s)
            launcher.prebuild_candidate_artifacts()

            from optima.eval.throughput_kl import EvalConfig, evaluate

            cfg_kwargs = expected_arena.eval_config_kwargs()
            cfg_kwargs["prompt_seed"] = context.prompt_seed
            # System products are arbitrary scheduler implementations within a
            # bounded source/manifest lane.  Their only load-bearing quality gate
            # is the controller-observed B/C/B' output, never in-engine receipts.
            if competition.mode == "system":
                cfg_kwargs["framework_mode"] = True
            cfg = EvalConfig(**cfg_kwargs)
            report = evaluate(
                cfg,
                str(bundle_dir),
                oci_launcher=launcher,
            )
            from optima.eval.host_attestation import publish_host_attestation

            runtime_receipt = launcher.runtime_preflight_receipt
            if runtime_receipt is None:
                raise OCIBackendError(
                    "crownable launcher lost its stock runtime preflight receipt"
                )
            prepared = QualificationReport.prepare_evidence(
                report,
                competition=competition,
                arena=expected_arena,
                bundle_hash=context.bundle_hash,
                prompt_seed=context.prompt_seed,
                seed_round_id=context.round_id,
                seed_block=context.block,
                seed_block_hash=context.block_hash,
                chain_scope=context.chain_scope,
                validator_hotkey=context.validator_hotkey,
                evaluation_id=context.evaluation_id,
                miner_hotkey=context.miner_hotkey,
                settlement_round_id=context.settlement_round_id,
                evaluation_block=context.evaluation_block,
            )
            host_reference = publish_host_attestation(
                artifact_base,
                context=prepared.attestation_context(),
                runtime_preflight=runtime_receipt.canonical_payload(),
                device_receipts=launcher.attestation_receipts,
                qualification_evidence=prepared.evidence_dict(),
            )
            qualification = prepared.bind_host_attestation(host_reference.sha256)
        except (OCICandidateArtifactError, OuterSessionCandidateError) as exc:
            return EvalOutcome(
                False,
                0.0,
                detail=f"terminal candidate OCI failure: {exc}"[-2000:],
                quality_evidence=(
                    "validator-owned OCI qualification rejected candidate build/protocol"
                ),
                retryable=False,
                **identity,
            )
        except (OCIInfrastructureError, OuterSessionInfrastructureError) as exc:
            return EvalOutcome(
                False,
                0.0,
                detail=f"transient OCI infrastructure failure: {exc}"[-2000:],
                quality_evidence="validator-owned OCI infrastructure did not complete",
                retryable=True,
                **identity,
            )
        except QualificationReportError as exc:
            raise OCIBackendError(
                f"trusted qualification report construction failed: {exc}"
            ) from None
        except OCIBackendError:
            # Generic backend/profile/source/model errors are validator faults. They
            # abort the pass and must never be turned into a miner retry/DQ.
            raise
        return EvalOutcome(
            qualification.passed_quality,
            qualification.score if qualification.crownable else 0.0,
            kl_mean=qualification.kl_mean,
            target=qualification.target,
            mode=qualification.mode,
            member_slots=qualification.member_slots,
            detail=(
                f"authenticated OCI launches={len(launcher.attestation_receipts)}"
            ),
            crownable=qualification.crownable,
            confident=qualification.confident,
            passed_speedup=qualification.passed_speedup,
            passed_timed_quality=qualification.passed_timed_quality,
            passed_warmup_quality=qualification.passed_warmup_quality,
            host_attestation_sha256=qualification.host_attestation_sha256,
            quality_evidence=qualification.quality_evidence,
            **_context_identity(
                context,
                qualification_evidence_sha256=(
                    qualification.qualification_evidence_sha256
                ),
            ),
        )

    setattr(_run, "requires_positive_margin", True)
    setattr(_run, "validator_owned_oci", True)
    from optima.eval.host_attestation import verify_host_attestation

    setattr(
        _run,
        "host_attestation_verifier",
        lambda reference, expected_context: verify_host_attestation(
            artifact_base,
            reference,
            expected_context=expected_context,
        ),
    )
    return _run


# --------------------------------------------------------------------------- #
# One referee pass
# --------------------------------------------------------------------------- #

def _load_weights_state(path: Path) -> dict:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(path, flags)
    except FileNotFoundError:
        return {}
    except OSError as exc:
        raise WeightSafetyError(
            f"cannot open weight-publication state safely: {exc}"
        ) from None

    def reject_duplicate_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        info = os.fstat(fd)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
            or info.st_uid != os.geteuid()
            or stat.S_IMODE(info.st_mode) & 0o077
            or info.st_size > 64 * 1024
        ):
            raise WeightSafetyError(
                "weight-publication state has unsafe type/owner/mode/link/size"
            )
        payload = b""
        while len(payload) <= 64 * 1024:
            chunk = os.read(fd, min(16 * 1024, 64 * 1024 + 1 - len(payload)))
            if not chunk:
                break
            payload += chunk
        if not payload or len(payload) > 64 * 1024:
            raise WeightSafetyError("weight-publication state is empty or oversized")
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=reject_duplicate_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"invalid JSON constant {token}")
            ),
        )
        if not isinstance(value, dict):
            raise ValueError("top-level state must be an object")
        return value
    except WeightSafetyError:
        raise
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise WeightSafetyError(
            f"weight-publication state is corrupt: {exc}"
        ) from None
    finally:
        os.close(fd)


def _atomic_write_weights_state(path: Path, data: dict) -> None:
    payload = json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    fd, raw_tmp = tempfile.mkstemp(
        prefix=f".{path.name}.tmp.{os.getpid()}.", dir=path.parent
    )
    tmp = Path(raw_tmp)
    try:
        with os.fdopen(fd, "wb", closefd=True) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
        parent_fd = os.open(
            path.parent,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0),
        )
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    finally:
        tmp.unlink(missing_ok=True)


def _weights_close(left: dict[str, float], right: dict[str, float]) -> bool:
    if set(left) != set(right):
        return False
    return all(
        math.isclose(float(left[key]), float(right[key]), rel_tol=2e-5, abs_tol=2e-5)
        for key in left
    )


WEIGHT_PUBLICATION_SCHEMA = "optima.weight-publication-state.v1"
WEIGHT_STATUS_INTENT = "intent"
WEIGHT_STATUS_PENDING = "pending"
WEIGHT_STATUS_HELD = "held"
WEIGHT_STATUS_CONFIRMED = "confirmed"
_WEIGHT_PUBLICATION_KEYS = frozenset({
    "schema",
    "chain_scope",
    "arena_set_sha256",
    "emission_policy",
    "expected_weights",
    "status",
    "submit_block",
    "reveal_round",
    "retry_after_block",
    "confirmed_block",
    "confirmed_last_update",
})


def _canonical_weight_map(weights: dict[str, float]) -> dict[str, float]:
    return {key: float(weights[key]) for key in sorted(weights)}


def _weight_publication_state(
    *,
    chain_scope: str,
    arena_set_sha256: str,
    emission_policy: str,
    expected_weights: dict[str, float],
    status: str,
    submit_block: int = 0,
    reveal_round: int = 0,
    retry_after_block: int = 0,
    confirmed_block: int = 0,
    confirmed_last_update: int = 0,
) -> dict[str, object]:
    if status not in {
        WEIGHT_STATUS_INTENT,
        WEIGHT_STATUS_PENDING,
        WEIGHT_STATUS_HELD,
        WEIGHT_STATUS_CONFIRMED,
    }:
        raise ValueError("invalid weight-publication status")
    return {
        "schema": WEIGHT_PUBLICATION_SCHEMA,
        "chain_scope": chain_scope,
        "arena_set_sha256": arena_set_sha256,
        "emission_policy": emission_policy,
        "expected_weights": _canonical_weight_map(expected_weights),
        "status": status,
        "submit_block": int(submit_block),
        "reveal_round": int(reveal_round),
        "retry_after_block": int(retry_after_block),
        "confirmed_block": int(confirmed_block),
        "confirmed_last_update": int(confirmed_last_update),
    }


def _submitted_reveal_round(result: dict) -> int:
    response = result.get("result")
    data = getattr(response, "data", None)
    if data is None and isinstance(response, dict):
        data = response.get("data")
    value = data.get("reveal_round") if isinstance(data, dict) else None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return 0
    return value


def _validate_weight_publication_state(
    state: dict,
    *,
    chain_scope: str,
    arena_set_sha256: str,
    emission_policy: str,
) -> None:
    """Validate the load-bearing v1 journal before it can suppress a submit."""

    if not state or "schema" not in state:  # empty or readback-checked legacy cache
        return
    if state.get("schema") != WEIGHT_PUBLICATION_SCHEMA:
        raise WeightSafetyError("unknown weight-publication state schema")
    if set(state) != _WEIGHT_PUBLICATION_KEYS:
        raise WeightSafetyError(
            "weight-publication state fields differ from the exact v1 schema"
        )
    if (
        state.get("chain_scope") != chain_scope
        or state.get("arena_set_sha256") != arena_set_sha256
        or state.get("emission_policy") != emission_policy
    ):
        raise WeightSafetyError(
            "weight-publication state context differs from active authority"
        )
    if (
        re.fullmatch(
            r"[0-9A-Za-z._-]{1,128}:sha256:[0-9a-f]{64}",
            state["chain_scope"],
        ) is None
        or re.fullmatch(r"sha256:[0-9a-f]{64}", state["arena_set_sha256"])
        is None
        or not isinstance(state["emission_policy"], str)
        or not state["emission_policy"]
        or len(state["emission_policy"]) > 256
    ):
        raise WeightSafetyError("weight-publication context fields are malformed")
    if state.get("status") not in {
        WEIGHT_STATUS_INTENT,
        WEIGHT_STATUS_PENDING,
        WEIGHT_STATUS_HELD,
        WEIGHT_STATUS_CONFIRMED,
    }:
        raise WeightSafetyError("weight-publication state has invalid status")
    expected = state.get("expected_weights")
    if not isinstance(expected, dict) or not expected:
        raise WeightSafetyError("weight-publication state lacks its expected vector")
    total = 0.0
    for hotkey, raw in expected.items():
        if (
            not isinstance(hotkey, str)
            or not hotkey
            or isinstance(raw, bool)
            or not isinstance(raw, (int, float))
            or not math.isfinite(float(raw))
            or float(raw) <= 0
        ):
            raise WeightSafetyError("weight-publication expected vector is malformed")
        total += float(raw)
    if not math.isclose(total, 1.0, rel_tol=2e-5, abs_tol=2e-5):
        raise WeightSafetyError("weight-publication expected vector is not normalized")
    numeric = (
        "submit_block",
        "reveal_round",
        "retry_after_block",
        "confirmed_block",
        "confirmed_last_update",
    )
    if any(
        isinstance(state.get(field), bool)
        or not isinstance(state.get(field), int)
        or state[field] < 0
        for field in numeric
    ):
        raise WeightSafetyError("weight-publication state has invalid block fields")
    if (
        state["status"] in {
            WEIGHT_STATUS_INTENT,
            WEIGHT_STATUS_PENDING,
            WEIGHT_STATUS_HELD,
        }
        and (
            state["submit_block"] <= 0
            or state["retry_after_block"] < state["submit_block"]
        )
    ):
        raise WeightSafetyError("weight-publication in-flight bounds are invalid")


def _global_arena_set_sha256(arenas) -> str:
    rows = [
        {
            "name": arena.name,
            "fingerprint": arena.fingerprint,
            "bracket": arena.bracket,
            "emission_policy": arena.settlement.emission_policy,
            "chain_scope_scheme": arena.settlement.chain_scope_scheme,
        }
        for arena in sorted(tuple(arenas), key=lambda value: value.name)
    ]
    return "sha256:" + hashlib.sha256(
        json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _weight_state_path(ledger_path: str | Path, chain_scope: str) -> Path:
    return Path(
        str(ledger_path)
        + f".weights_state.{chain_scope.rsplit(':', 1)[-1][:12]}.global.json"
    )


def _archive_released_weight_hold(
    path: Path, *, release_block: int, reason: str
) -> Path:
    """Durably archive one held journal before removing its active marker."""

    if (
        type(release_block) is not int
        or release_block < 0
        or not isinstance(reason, str)
        or not reason.strip()
        or len(reason) > 1_000
    ):
        raise WeightSafetyError("weight-publication release requires a bounded reason")
    state = _load_weights_state(path)
    if state.get("schema") != WEIGHT_PUBLICATION_SCHEMA:
        raise WeightSafetyError("only a v1 weight-publication hold can be released")
    if state.get("status") != WEIGHT_STATUS_HELD:
        raise WeightSafetyError("only a held weight publication can be released")
    released = dict(state)
    released["operator_release"] = {
        "block": release_block,
        "reason": reason.strip(),
    }
    encoded = json.dumps(
        released, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    archive = path.with_name(path.name + f".released.{digest}")
    _atomic_write_weights_state(archive, released)
    try:
        path.unlink()
    except OSError as exc:
        raise WeightSafetyError(
            f"cannot clear released weight-publication hold: {exc}"
        ) from None
    parent_fd = os.open(
        path.parent,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)
    return archive


def _intake_disposition_id(
    *,
    chain_scope: str,
    arena_bracket: str,
    validator_hotkey: str,
    hotkey: str,
    bundle_hash: str,
    stage: str,
) -> str:
    material = json.dumps(
        {
            "chain_scope": chain_scope,
            "arena_bracket": arena_bracket,
            "validator_hotkey": validator_hotkey,
            "hotkey": hotkey,
            "bundle_hash": bundle_hash,
            "stage": stage,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def _intake_fingerprints(
    bundle_dir: Path, *, target: str, mode: str
) -> dict[str, object]:
    """Build copy signals without crossing competition namespaces.

    Component products keep per-slot identities.  System products deliberately
    clear every component-shaped signal and emit only target-keyed product
    fingerprints, selecting the implementation from the manifest product shape:
    exact SGLang source patches and op-declared whole-serving host bundles have
    different normalizers but the same non-component ledger contract.
    """

    from optima.copy_fingerprint import (
        bundle_fingerprint,
        bundle_slot_file_fingerprints,
        bundle_slot_fingerprints,
        bundle_structural_fingerprint,
    )

    if mode != "system":
        return {
            "fingerprint": bundle_fingerprint(bundle_dir),
            "structural_fingerprint": bundle_structural_fingerprint(bundle_dir),
            "slot_fingerprints": bundle_slot_fingerprints(bundle_dir),
            "slot_file_fingerprints": bundle_slot_file_fingerprints(bundle_dir),
            "product_fingerprints": {},
        }

    from optima.manifest import load_manifest

    manifest = load_manifest(bundle_dir)
    if manifest.system is not None:
        from optima.system_patch import system_patch_fingerprints

        product = system_patch_fingerprints(bundle_dir)
    else:
        from optima.device_component import untrusted_host_product_fingerprints

        product = untrusted_host_product_fingerprints(bundle_dir)
    if not product:
        raise SubmissionPolicyError(
            f"system product {target!r} produced no copy fingerprints"
        )
    return {
        # Never let a system product match, populate, or later settle through a
        # component slot namespace—even when its bundle happens to contain [[ops]].
        "fingerprint": "",
        "structural_fingerprint": "",
        "slot_fingerprints": {},
        "slot_file_fingerprints": {},
        "product_fingerprints": {target: list(product)},
    }


def _recover_pending_settlements(
    ledger,
    *,
    ledger_path: str,
    arena: ArenaProfile,
    margin: float,
    host_attestation_verifier,
    validator_hotkey: str,
) -> None:
    """Settle durable qualifications in causal reveal order without GPU replay.

    Each reveal has two persistence boundaries. First the champion/disposition is
    saved while the exact pending rows remain present. Only then are those rows
    removed and the cleared state saved. A crash or fsync failure at either point
    therefore reloads either a pending candidate or a complete champion—never a
    forgotten score. Re-running settlement against an already-installed champion
    is idempotent because that same score cannot dethrone itself.
    """

    rows = ledger.pending_settlements_for(
        arena_bracket=arena.bracket,
        chain_scope=ledger.chain_scope,
    )
    # Apply the dethrone margin once per canonical chain reveal. Settling an
    # entire pass as one tournament made the result depend on validator uptime:
    # A=1.05 then B=1.06 in separate passes left A, while seeing both together
    # crowned B. ``pending_settlements_for`` is already ordered by settlement
    # round and ledger commit sequence, so singleton dispositions make both
    # arrival patterns identical.
    for pending in rows:
        batch = (pending,)
        candidate_evidence = ledger.verify_pending_settlements(
            batch,
            arena=arena,
            host_attestation_verifier=host_attestation_verifier,
            validator_hotkey=validator_hotkey,
        )
        ledger.settle_per_target(
            pending.round_id,
            margin=margin,
            current_sglang_version=arena.sglang_version,
            arena=arena,
            host_attestation_verifier=host_attestation_verifier,
            candidate_evidence_sha256=candidate_evidence,
            validator_hotkey=validator_hotkey,
        )
        # Champion + still-pending disposition become durable together. If the
        # process dies before/during this save, the prior durable marker drives
        # the exact batch through standalone verification again.
        ledger.save(ledger_path)
        ledger.clear_pending_settlements(batch)
        # Clearing is a second atomic commit. A crash before it leaves pending;
        # a crash after it leaves the champion and completion inseparable.
        ledger.save(ledger_path)


@contextmanager
def _exclusive_ledger_pass(ledger_path: str | Path):
    """Hold one no-replace OS lock across load, GPU work, settlement and weights."""

    ledger = Path(ledger_path).expanduser()
    ledger.parent.mkdir(parents=True, exist_ok=True)
    try:
        parent_info = ledger.parent.lstat()
    except OSError as exc:
        raise LedgerLockError(
            f"cannot inspect ledger parent {ledger.parent}: {exc}"
        ) from None
    if (
        not stat.S_ISDIR(parent_info.st_mode)
        or stat.S_ISLNK(parent_info.st_mode)
        or parent_info.st_uid != os.geteuid()
        or stat.S_IMODE(parent_info.st_mode) & 0o022
    ):
        raise LedgerLockError(
            "ledger parent must be an owner-controlled non-writable directory"
        )
    lock_path = ledger.with_name(ledger.name + ".pass.lock")
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(lock_path, flags, 0o600)
    except OSError as exc:
        raise LedgerLockError(f"cannot open ledger pass lock {lock_path}: {exc}") from None
    try:
        info = os.fstat(fd)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
            or info.st_uid != os.geteuid()
            or stat.S_IMODE(info.st_mode) & 0o077
        ):
            raise LedgerLockError("ledger pass lock has unsafe type/ownership/link count")
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno in (errno.EACCES, errno.EAGAIN):
                raise LedgerLockError(
                    f"another validator process owns the whole pass for {ledger}"
                ) from None
            raise LedgerLockError(f"cannot acquire ledger pass lock: {exc}") from None
        os.ftruncate(fd, 0)
        os.write(fd, f"pid={os.getpid()}\n".encode("ascii"))
        os.fsync(fd)
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def _run_pass_unlocked(subtensor, wallet, netuid: int, *, ledger_path: str, bundles_dir: str,
             evaluator: Evaluator, arena: ArenaProfile, margin: float | None = None,
             round_blocks: int | None = None,
             weights_refresh_blocks: int | None = None,
             dry_run_weights: bool = False,
             host_attestation_verifier=None,
             validator_hotkey: str | None = None,
             test_only_allow_local_file_urls: bool = False) -> PassResult:
    """One full referee cycle. Per-submission failures are recorded and contained;
    an exception out of this function is a validator-side fault (RPC, disk)."""
    from optima.commit_reveal import (
        EvalRecord,
        Ledger,
        RETRY_KIND_INFRASTRUCTURE,
        RETRY_KIND_NO_DECISION,
        RETRY_STATE_HELD,
        RETRY_STATE_IN_PROGRESS,
        RevealError,
        make_chain_scope,
        make_commitment,
    )
    from optima.compat import PINNED_SGLANG

    registered_arena = get_arena(arena.name)
    if registered_arena.fingerprint != arena.fingerprint:
        raise ValueError("run_pass arena does not match registered profile")
    if registered_arena.sglang_version != PINNED_SGLANG:
        raise ValueError(
            "run_pass arena pin disagrees with validator PINNED_SGLANG"
        )
    arena = registered_arena
    policy = arena.settlement
    margin = policy.dethrone_margin if margin is None else float(margin)
    round_blocks = policy.round_blocks if round_blocks is None else int(round_blocks)
    weights_refresh_blocks = (
        policy.weights_refresh_blocks
        if weights_refresh_blocks is None else int(weights_refresh_blocks)
    )
    real_evaluator = bool(getattr(evaluator, "requires_positive_margin", False))
    validator_owned_oci = bool(
        getattr(evaluator, "validator_owned_oci", False)
    )
    host_attestation_verifier = (
        host_attestation_verifier
        if host_attestation_verifier is not None
        else getattr(evaluator, "host_attestation_verifier", None)
    )
    if (not math.isfinite(margin) or margin < 0
            or (real_evaluator and margin != policy.dethrone_margin)
            or (not real_evaluator and margin not in (0.0, policy.dethrone_margin))):
        raise ValueError(
            "settlement margin must equal the immutable arena policy "
            f"({policy.dethrone_margin:g}); margin=0 is reserved for verify-only plumbing"
        )
    if round_blocks != policy.round_blocks:
        raise ValueError("round_blocks disagrees with immutable arena settlement policy")
    if weights_refresh_blocks != policy.weights_refresh_blocks:
        raise ValueError(
            "weights_refresh_blocks disagrees with immutable arena settlement policy"
        )
    wallet_hotkey = None
    if wallet is not None:
        try:
            wallet_hotkey = wallet.hotkey.ss58_address
        except (AttributeError, TypeError):
            wallet_hotkey = None
        if not isinstance(wallet_hotkey, str) or not wallet_hotkey:
            raise chain.ChainWeightStateError(
                "weight-signing wallet does not expose an exact hotkey address"
            )
    elif not dry_run_weights:
        raise chain.ChainWeightStateError(
            "non-dry-run validator pass requires its weight-signing wallet"
        )
    if validator_hotkey is None:
        validator_hotkey = wallet_hotkey
    if not isinstance(validator_hotkey, str) or not validator_hotkey:
        raise chain.ChainWeightStateError(
            "weight-capable validator pass requires its exact hotkey so current "
            "on-chain emissions can be reconciled"
        )
    if wallet_hotkey is not None and wallet_hotkey != validator_hotkey:
        raise chain.ChainWeightStateError(
            "weight-signing wallet hotkey differs from reconciliation hotkey"
        )
    if type(test_only_allow_local_file_urls) is not bool:
        raise ValueError("test_only_allow_local_file_urls must be boolean")
    block = int(subtensor.get_current_block())
    round_id = block // round_blocks
    res = PassResult(block=block, round_id=round_id)

    finalized_block = _finalized_block_number(subtensor)
    if finalized_block > block:
        raise RuntimeError(
            f"finalized chain height {finalized_block} exceeds current block {block}"
        )

    # Read the exact finalized state, not a head snapshot filtered by block number
    # afterward. A reorg between a head read and finality query could otherwise pair
    # fork-A payload bytes with fork-B's finalized height at the same block number.
    revealed = chain.read_reveal_history(
        subtensor, netuid, block=finalized_block
    )
    decoder = (
        decode_payload_for_testing
        if test_only_allow_local_file_urls
        else decode_payload
    )
    refs: list[SubmissionRef] = []
    for rc in revealed:
        ref = decoder(rc.hotkey, rc.block, rc.data)
        if ref is not None:
            refs.append(ref)
    # Chain order = anti-copy priority: replay into the ledger sorted by reveal block.
    refs.sort(key=lambda r: (r.block, r.hotkey, r.content_hash, r.url))
    res.seen = len(refs)
    finalized_refs: list[SubmissionRef] = []
    for ref in refs:
        if ref.block > finalized_block:
            reason = (
                f"reveal block {ref.block} is not finalized "
                f"(finalized={finalized_block}); evaluation deferred"
            )
            res.deferred[ref.content_hash] = reason
            continue
        finalized_refs.append(ref)
    refs = finalized_refs

    led = Ledger.load(ledger_path)
    chain_scope = make_chain_scope(
        genesis_hash=_canonical_block_hash(subtensor, 0),
        netuid=int(netuid),
        scheme=arena.settlement.chain_scope_scheme,
    )
    led.bind_chain_scope(chain_scope)
    led.bind_validator_hotkey(validator_hotkey)
    # Recovery precedes intake/evaluation. An authoritative result from a prior
    # pass or round is settled from retained host evidence and can never trigger
    # another model-sized evaluator launch.
    _recover_pending_settlements(
        led,
        ledger_path=ledger_path,
        arena=arena,
        margin=margin,
        host_attestation_verifier=host_attestation_verifier,
        validator_hotkey=validator_hotkey,
    )
    known_reveals = {(r.hotkey, r.content_hash): r for r in led.reveals}
    eval_arena_fields = {
        "arena_name": arena.name,
        "arena_fingerprint": arena.fingerprint,
        "arena_bracket": arena.bracket,
        "regime": arena.workload.regime,
        "sglang_version": arena.sglang_version,
        "validator_image": arena.validator_image,
        "referee_source_digest": arena.referee_source_digest,
        "referee_tree_digest": arena.referee_tree_digest,
        "model_revision": arena.model_revision,
        "model_manifest_digest": arena.model_manifest_digest,
        "model_content_digest": arena.model_content_digest,
        "chain_scope": chain_scope,
    }

    def terminal_authority(ref: SubmissionRef, stage: str) -> dict[str, object]:
        return {
            "validator_hotkey": validator_hotkey,
            "evaluation_id": _intake_disposition_id(
                chain_scope=chain_scope,
                arena_bracket=arena.bracket,
                validator_hotkey=validator_hotkey,
                hotkey=ref.hotkey,
                bundle_hash=ref.content_hash,
                stage=stage,
            ),
            "miner_hotkey": ref.hotkey,
            "settlement_round_id": round_id,
            "evaluation_block": block,
            "development_only": not validator_owned_oci,
        }

    for ref in refs:
        key = (ref.hotkey, ref.content_hash)
        if led.is_known(
            ref.hotkey,
            ref.content_hash,
            arena_bracket=arena.bracket,
            require_authoritative=validator_owned_oci,
            arena=arena,
        ):
            # A terminal eval row is authoritative. Clean up retry debris from an
            # interrupted older build instead of persisting contradictory state.
            led.clear_retry(
                ref.hotkey, ref.content_hash, arena_bracket=arena.bracket
            )
            continue  # already processed in THIS arena — re-derive, don't replay
        validator_fault = led.validator_fault_for(
            ref.hotkey, ref.content_hash, arena_bracket=arena.bracket
        )
        if validator_fault is not None:
            reason = (
                "validator-fault hold requires trusted release: "
                + validator_fault.reason
            )
            res.held[ref.content_hash] = reason
            res.deferred[ref.content_hash] = reason
            continue
        retry = led.retry_for(
            ref.hotkey, ref.content_hash, arena_bracket=arena.bracket
        )
        if retry is not None and retry.state == RETRY_STATE_IN_PROGRESS:
            # A lease present at pass startup means the previous validator process
            # died after durably counting the attempt and before classifying it.
            # Process death is ambiguous controller/infrastructure state. Do not
            # charge it to a miner or replay model-sized work automatically.
            reason = (
                "recovered abandoned in-progress evaluation lease "
                f"{retry.lease_id[:16]}…"
            )
            hold = led.hold_validator_fault(
                hotkey=ref.hotkey,
                bundle_hash=ref.content_hash,
                arena_bracket=arena.bracket,
                lease_id=retry.lease_id,
                current_block=block,
                reason=reason,
            )
            led.save(ledger_path)
            detail = f"validator-fault hold: {hold.reason}"
            res.held[ref.content_hash] = detail
            res.deferred[ref.content_hash] = detail
            continue
        if retry is not None and retry.state == RETRY_STATE_HELD:
            reason = (
                f"operator-held infrastructure retry after {retry.attempts} "
                f"failed attempts: {retry.last_reason}"
            )
            res.held[ref.content_hash] = reason
            res.deferred[ref.content_hash] = reason
            continue
        if retry is not None and block < retry.next_block:
            reason = (
                f"{retry.kind} retry backoff attempt {retry.attempts}; eligible at block "
                f"{retry.next_block}: {retry.last_reason}"
            )
            res.deferred[ref.content_hash] = reason
            continue
        res.new.append(ref.content_hash)

        try:
            fetcher = (
                fetch_bundle_from_local_file_for_testing
                if test_only_allow_local_file_urls and ref.url.startswith("file://")
                else fetch_bundle
            )
            bundle_dir = fetcher(ref.url, ref.content_hash, bundles_dir)
        except FetchTransientError as exc:
            reason = f"bundle transport infrastructure: {exc}"
            retry = led.record_retry(
                hotkey=ref.hotkey,
                bundle_hash=ref.content_hash,
                arena_bracket=arena.bracket,
                kind=RETRY_KIND_INFRASTRUCTURE,
                current_block=block,
                reason=reason,
                base_backoff_blocks=policy.retry_backoff_blocks,
                max_backoff_blocks=policy.retry_max_backoff_blocks,
                max_automatic_infrastructure_attempts=(
                    policy.retry_max_automatic_infrastructure_attempts
                ),
                max_automatic_no_decision_attempts=(
                    policy.retry_max_automatic_no_decision_attempts
                ),
                max_total_attempts=policy.retry_max_total_attempts,
            )
            led.save(ledger_path)
            if retry.state == RETRY_STATE_HELD:
                detail = (
                    f"{reason}; attempt={retry.attempts}, operator hold entered; "
                    "trusted release required"
                )
                res.held[ref.content_hash] = detail
                res.deferred[ref.content_hash] = detail
            else:
                res.deferred[ref.content_hash] = (
                    f"{reason}; attempt={retry.attempts}, "
                    f"next_block={retry.next_block}"
                )
            logger.warning(
                "submission %s… by %s deferred after transient fetch failure: %s",
                ref.content_hash[:16],
                ref.hotkey,
                exc,
            )
            continue
        except FetchError as e:
            logger.warning("submission %s… by %s rejected: %s",
                           ref.content_hash[:16], ref.hotkey, e)
            res.rejected[ref.content_hash] = str(e)
            led.clear_retry(
                ref.hotkey, ref.content_hash, arena_bracket=arena.bracket
            )
            led.record_eval(EvalRecord(ref.hotkey, ref.content_hash, slot="",
                                       round_id=round_id, score=0.0, passed=False,
                                       dq_reason=f"fetch: {e}", **eval_arena_fields,
                                       **terminal_authority(ref, "fetch")))
            led.save(ledger_path)
            continue

        try:
            competition = _resolve_bundle_competition(bundle_dir)
        except SubmissionPolicyError as exc:
            reason = f"competition target: {exc}"
            logger.warning("submission %s… by %s rejected: %s",
                           ref.content_hash[:16], ref.hotkey, reason)
            res.rejected[ref.content_hash] = reason
            led.clear_retry(
                ref.hotkey, ref.content_hash, arena_bracket=arena.bracket
            )
            led.record_eval(EvalRecord(
                ref.hotkey,
                ref.content_hash,
                slot="",
                round_id=round_id,
                score=0.0,
                passed=False,
                dq_reason=reason,
                **eval_arena_fields,
                **terminal_authority(ref, "competition"),
            ))
            led.save(ledger_path)
            continue
        target = competition.target or ""  # require_crownable guarantees non-empty
        mode = competition.mode or ""
        members = competition.members
        legacy_slot = target if mode == "slot" else ""
        identity = dict(
            slot=legacy_slot,
            target=target,
            mode=mode,
            member_slots=members,
            **eval_arena_fields,
        )
        try:
            fingerprints = _intake_fingerprints(
                bundle_dir, target=target, mode=mode
            )
        except BaseException as e:
            if not (
                isinstance(e, SubmissionPolicyError)
                or _expected_submission_exception(e)
            ):
                raise
            logger.warning("submission %s… by %s rejected: unfingerprintable: %s",
                           ref.content_hash[:16], ref.hotkey, e)
            res.rejected[ref.content_hash] = f"unfingerprintable: {e}"
            led.clear_retry(
                ref.hotkey, ref.content_hash, arena_bracket=arena.bracket
            )
            led.record_eval(EvalRecord(ref.hotkey, ref.content_hash, **identity,
                                       round_id=round_id, score=0.0, passed=False,
                                       dq_reason=f"unfingerprintable: {e}",
                                       **terminal_authority(ref, "fingerprint")))
            led.save(ledger_path)
            continue
        rev = known_reveals.get(key)
        if rev is None:
            salt = f"chain:{ref.block}"
            reveal_round_id = int(ref.block) // round_blocks
            led.commit(
                ref.hotkey,
                make_commitment(ref.content_hash, ref.hotkey, salt),
                reveal_round_id,
            )
            try:
                rev = led.reveal(
                    ref.hotkey,
                    ref.content_hash,
                    salt,
                    reveal_round_id,
                    **fingerprints,
                )
            except RevealError as e:  # commit was just derived by trusted code
                raise QualificationAuthorityError(
                    f"trusted chain reveal replay failed: {e}"
                ) from e
            known_reveals[key] = rev
        if not rev.original:
            logger.info("submission %s… by %s is a COPY of an earlier commit; skipping eval",
                        ref.content_hash[:16], ref.hotkey)
            res.copies.append(ref.content_hash)
            led.clear_retry(
                ref.hotkey, ref.content_hash, arena_bracket=arena.bracket
            )
            led.record_eval(EvalRecord(ref.hotkey, ref.content_hash, **identity,
                                       round_id=round_id, score=0.0, passed=False,
                                       dq_reason="copy",
                                       **terminal_authority(ref, "copy")))
            led.save(ledger_path)
            continue

        # Use the reveal block, not "whatever block this validator happened to
        # evaluate at": it is post-commit entropy yet identical for every validator.
        seed_round_id = int(ref.block) // round_blocks
        seed_block_hash = _canonical_block_hash(subtensor, int(ref.block))
        prompt_seed = derive_prompt_seed(
            arena,
            bundle_hash=ref.content_hash,
            round_id=seed_round_id,
            block_hash=seed_block_hash,
        )
        lease = led.begin_retry_attempt(
            hotkey=ref.hotkey,
            bundle_hash=ref.content_hash,
            arena_bracket=arena.bracket,
            current_block=block,
            reason="evaluation lease acquired before GPU work",
            max_automatic_infrastructure_attempts=(
                policy.retry_max_automatic_infrastructure_attempts
            ),
            max_automatic_no_decision_attempts=(
                policy.retry_max_automatic_no_decision_attempts
            ),
            max_total_attempts=policy.retry_max_total_attempts,
        )
        if lease.state == RETRY_STATE_HELD:
            led.save(ledger_path)
            reason = (
                f"automatic retry budget exhausted before evaluation; "
                f"attempts={lease.attempts}: {lease.last_reason}"
            )
            res.held[ref.content_hash] = reason
            res.deferred[ref.content_hash] = reason
            continue
        # This write is the crash boundary: no model-sized process may start until
        # the exact attempt and lease identifier are durable on disk.
        led.save(ledger_path)
        context = EvaluationContext(
            arena=arena,
            bundle_hash=ref.content_hash,
            round_id=seed_round_id,
            block=int(ref.block),
            block_hash=seed_block_hash,
            prompt_seed=prompt_seed,
            chain_scope=chain_scope,
            validator_hotkey=validator_hotkey,
            evaluation_id=lease.lease_id,
            miner_hotkey=ref.hotkey,
            settlement_round_id=round_id,
            evaluation_block=block,
        )

        def raise_authority_fault(reason: str) -> None:
            led.hold_validator_fault(
                hotkey=ref.hotkey,
                bundle_hash=ref.content_hash,
                arena_bracket=arena.bracket,
                lease_id=lease.lease_id,
                current_block=block,
                reason=reason,
            )
            led.save(ledger_path)
            raise QualificationAuthorityError(reason)

        try:
            outcome = evaluator(bundle_dir, context)
        except Exception as exc:
            # Candidate/build/protocol/infrastructure failures are required to be
            # returned as typed EvalOutcomes. Any exception escaping the evaluator
            # is therefore controller code/configuration failure, never a miner
            # attempt. Preserve a durable validator-fault circuit breaker before
            # surfacing it, so a supervisor restart cannot replay GPU work.
            led.hold_validator_fault(
                hotkey=ref.hotkey,
                bundle_hash=ref.content_hash,
                arena_bracket=arena.bracket,
                lease_id=lease.lease_id,
                current_block=block,
                reason=(
                    f"evaluator/controller {type(exc).__name__}: {exc}"
                ),
            )
            led.save(ledger_path)
            if bool(getattr(exc, "validator_fault", False)):
                raise
            raise QualificationAuthorityError(
                f"evaluator escaped an untyped {type(exc).__name__}: {exc}"
            ) from exc
        if type(outcome) is not EvalOutcome:
            raise_authority_fault(
                "evaluator returned a non-EvalOutcome controller result"
            )
        decision_values = (
            outcome.passed,
            outcome.passed_timed_quality,
            outcome.passed_warmup_quality,
            outcome.passed_speedup,
            outcome.confident,
            outcome.crownable,
            outcome.retryable,
        )
        if any(type(value) is not bool for value in decision_values):
            raise_authority_fault("evaluator decisions are not exact booleans")
        if (
            isinstance(outcome.score, bool)
            or not isinstance(outcome.score, (int, float))
            or not math.isfinite(float(outcome.score))
            or isinstance(outcome.kl_mean, bool)
            or not isinstance(outcome.kl_mean, (int, float))
            or not math.isfinite(float(outcome.kl_mean))
            or float(outcome.kl_mean) < 0
            or not isinstance(outcome.target, str)
            or not isinstance(outcome.mode, str)
            or not isinstance(outcome.member_slots, tuple)
            or any(
                not isinstance(member, str) or not member
                for member in outcome.member_slots
            )
            or not isinstance(outcome.detail, str)
            or len(outcome.detail) > 16_384
            or not isinstance(outcome.quality_evidence, str)
            or len(outcome.quality_evidence) > 4_096
        ):
            raise_authority_fault(
                "evaluator returned malformed numeric, competition, or evidence fields"
            )
        reported_identity = (
            outcome.target,
            outcome.mode,
            outcome.member_slots,
        )
        trusted_identity = (target, mode, members)
        if reported_identity != trusted_identity:
            reason = (
                "qualification competition identity mismatch: "
                f"report={reported_identity!r} manifest={trusted_identity!r}"
            )
            if outcome.detail:
                reason += f"; evaluator={outcome.detail[-500:]}"
            if validator_owned_oci:
                raise_authority_fault(reason)
            logger.warning("submission %s… by %s rejected: %s",
                           ref.content_hash[:16], ref.hotkey, reason)
            res.rejected[ref.content_hash] = reason
            res.evaluated[ref.content_hash] = False
            led.record_eval(EvalRecord(
                ref.hotkey,
                ref.content_hash,
                **identity,
                round_id=round_id,
                score=0.0,
                passed=False,
                mean_kl=outcome.kl_mean,
                development_only=not validator_owned_oci,
                dq_reason=reason,
            ))
            led.complete_retry_terminal(
                hotkey=ref.hotkey,
                bundle_hash=ref.content_hash,
                arena_bracket=arena.bracket,
                lease_id=lease.lease_id,
            )
            led.save(ledger_path)
            continue
        reported_scope = (
            outcome.arena_name,
            outcome.arena_fingerprint,
            outcome.arena_bracket,
            outcome.regime,
            outcome.bundle_hash,
            outcome.sglang_version,
            outcome.validator_image,
            outcome.referee_source_digest,
            outcome.referee_tree_digest,
            outcome.model_revision,
            outcome.model_manifest_digest,
            outcome.model_content_digest,
            outcome.chain_scope,
            outcome.validator_hotkey,
            outcome.evaluation_id,
            outcome.miner_hotkey,
            outcome.settlement_round_id,
            outcome.evaluation_block,
            outcome.prompt_seed,
            outcome.prompt_engine_version,
            outcome.prompt_seed_scheme,
            outcome.seed_round_id,
            outcome.seed_block,
            outcome.seed_block_hash,
        )
        expected_scope_values = _context_identity(context)
        expected_scope = tuple(expected_scope_values[key] for key in (
            "arena_name", "arena_fingerprint", "arena_bracket", "regime",
            "bundle_hash", "sglang_version", "validator_image",
            "referee_source_digest", "referee_tree_digest", "model_revision",
            "model_manifest_digest",
            "model_content_digest",
            "chain_scope", "validator_hotkey", "evaluation_id",
            "miner_hotkey", "settlement_round_id", "evaluation_block",
            "prompt_seed", "prompt_engine_version", "prompt_seed_scheme",
            "seed_round_id", "seed_block", "seed_block_hash",
        ))
        if reported_scope != expected_scope:
            reason = (
                "qualification arena/bundle/seed mismatch: "
                f"report={reported_scope!r} expected={expected_scope!r}"
            )
            if validator_owned_oci:
                raise_authority_fault(reason)
            logger.warning("submission %s… by %s rejected: %s",
                           ref.content_hash[:16], ref.hotkey, reason)
            res.rejected[ref.content_hash] = reason
            res.evaluated[ref.content_hash] = False
            led.record_eval(EvalRecord(
                ref.hotkey, ref.content_hash, **identity,
                round_id=round_id, score=0.0, passed=False,
                mean_kl=outcome.kl_mean,
                development_only=not validator_owned_oci,
                dq_reason=reason,
            ))
            led.complete_retry_terminal(
                hotkey=ref.hotkey,
                bundle_hash=ref.content_hash,
                arena_bracket=arena.bracket,
                lease_id=lease.lease_id,
            )
            led.save(ledger_path)
            continue
        if validator_owned_oci:
            decisions = (
                outcome.passed,
                outcome.passed_timed_quality,
                outcome.passed_warmup_quality,
                outcome.passed_speedup,
                outcome.confident,
                outcome.crownable,
            )
            if any(type(value) is not bool for value in decisions):
                raise_authority_fault(
                    "qualification decisions are not exact booleans"
                )
            if outcome.passed != (
                outcome.passed_timed_quality
                and outcome.passed_warmup_quality
            ):
                raise_authority_fault(
                    "qualification passed_quality disagrees with timed/warmup phases"
                )
            if outcome.crownable != (
                outcome.passed
                and outcome.passed_speedup
                and outcome.confident
            ):
                raise_authority_fault(
                    "qualification crownable decision disagrees with its gates"
                )
            if (
                (outcome.crownable and float(outcome.score) <= 1.0)
                or (not outcome.crownable and float(outcome.score) != 0.0)
            ):
                raise_authority_fault(
                    "qualification score disagrees with its crownable decision"
                )
        if outcome.retryable or not outcome.confident:
            kind = (
                RETRY_KIND_INFRASTRUCTURE
                if outcome.retryable
                else RETRY_KIND_NO_DECISION
            )
            category = "infrastructure" if outcome.retryable else "no-decision"
            reason = f"{category}: {outcome.detail or 'evaluation must be retried'}"
            retry = led.complete_retry_attempt(
                hotkey=ref.hotkey,
                bundle_hash=ref.content_hash,
                arena_bracket=arena.bracket,
                lease_id=lease.lease_id,
                kind=kind,
                current_block=block,
                reason=reason,
                base_backoff_blocks=policy.retry_backoff_blocks,
                max_backoff_blocks=policy.retry_max_backoff_blocks,
                max_automatic_infrastructure_attempts=(
                    policy.retry_max_automatic_infrastructure_attempts
                ),
                max_automatic_no_decision_attempts=(
                    policy.retry_max_automatic_no_decision_attempts
                ),
                max_total_attempts=policy.retry_max_total_attempts,
            )
            led.save(ledger_path)
            if retry.state == RETRY_STATE_HELD:
                held_reason = (
                    f"{reason}; attempt={retry.attempts}, operator hold entered; "
                    "trusted release required"
                )
                res.held[ref.content_hash] = held_reason
                res.deferred[ref.content_hash] = held_reason
                logger.warning(
                    "submission %s… entered operator hold after %d total attempts "
                    "(%d infrastructure, %d no-decision)",
                    ref.content_hash[:16], retry.attempts,
                    retry.infrastructure_attempts, retry.no_decision_attempts,
                )
            else:
                res.deferred[ref.content_hash] = (
                    f"{reason}; attempt={retry.attempts}, "
                    f"next_block={retry.next_block}"
                )
                logger.info(
                    "submission %s… deferred after %s (attempt=%d next_block=%d)",
                    ref.content_hash[:16], category, retry.attempts, retry.next_block,
                )
            continue
        if outcome.crownable and not validator_owned_oci:
            reason = (
                "crownable outcome did not come from a validator-owned OCI "
                "qualification controller"
            )
            raise_authority_fault(reason)
        if outcome.crownable and not callable(host_attestation_verifier):
            raise_authority_fault(
                "validator-owned OCI outcome lacks retained-host verification"
            )
        if outcome.crownable and (
            not isinstance(outcome.quality_evidence, str)
            or not outcome.quality_evidence
            or len(outcome.quality_evidence) > 4096
            or re.fullmatch(
                r"sha256:[0-9a-f]{64}", outcome.host_attestation_sha256
            ) is None
            or re.fullmatch(
                r"sha256:[0-9a-f]{64}",
                outcome.qualification_evidence_sha256,
            ) is None
        ):
            raise_authority_fault(
                "validator-owned crown outcome has missing/oversized quality or "
                "retained host/qualification evidence"
            )
        crownable = bool(outcome.passed and outcome.crownable and outcome.score > 1.0)
        score_row = None
        if crownable:
            score_row = led.record_score(
                ref.hotkey, ref.content_hash, round_id, outcome.score,
                outcome.kl_mean, True,
                sglang_version=arena.sglang_version,
                slot=legacy_slot, target=target, mode=mode,
                member_slots=members, arena=arena,
                prompt_seed=prompt_seed,
                prompt_engine_version=arena.workload.prompt_engine_version,
                prompt_seed_scheme=arena.workload.prompt_seed_scheme,
                seed_round_id=seed_round_id,
                seed_block=int(ref.block),
                seed_block_hash=seed_block_hash,
                host_attestation_sha256=outcome.host_attestation_sha256,
                quality_evidence=outcome.quality_evidence,
                validator_hotkey=outcome.validator_hotkey,
                evaluation_id=outcome.evaluation_id,
                miner_hotkey=outcome.miner_hotkey,
                settlement_round_id=outcome.settlement_round_id,
                evaluation_block=outcome.evaluation_block,
                passed_quality=outcome.passed,
                passed_timed_quality=outcome.passed_timed_quality,
                passed_warmup_quality=outcome.passed_warmup_quality,
                passed_speedup=outcome.passed_speedup,
                confident=outcome.confident,
                crownable=outcome.crownable,
                qualification_evidence_sha256=(
                    outcome.qualification_evidence_sha256
                ),
            )
        led.record_eval(EvalRecord(ref.hotkey, ref.content_hash, **identity,
                                   round_id=round_id, score=outcome.score,
                                   passed=outcome.passed, mean_kl=outcome.kl_mean,
                                   prompt_seed=prompt_seed,
                                   prompt_engine_version=arena.workload.prompt_engine_version,
                                   prompt_seed_scheme=arena.workload.prompt_seed_scheme,
                                   seed_round_id=seed_round_id,
                                   seed_block=int(ref.block),
                                   seed_block_hash=seed_block_hash,
                                   host_attestation_sha256=(
                                       outcome.host_attestation_sha256
                                   ),
                                   quality_evidence=outcome.quality_evidence,
                                   validator_hotkey=outcome.validator_hotkey,
                                   evaluation_id=outcome.evaluation_id,
                                   miner_hotkey=outcome.miner_hotkey,
                                   settlement_round_id=outcome.settlement_round_id,
                                   evaluation_block=outcome.evaluation_block,
                                   passed_quality=outcome.passed,
                                   passed_timed_quality=(
                                       outcome.passed_timed_quality
                                   ),
                                   passed_warmup_quality=(
                                       outcome.passed_warmup_quality
                                   ),
                                   passed_speedup=outcome.passed_speedup,
                                   confident=outcome.confident,
                                   crownable=outcome.crownable,
                                   qualification_evidence_sha256=(
                                       outcome.qualification_evidence_sha256
                                   ),
                                   development_only=not validator_owned_oci,
                                   dq_reason="" if outcome.passed else "failed gates"))
        if score_row is not None:
            # Score, production EvalRecord, and its exact pending disposition are
            # committed by the single save below before settlement is attempted.
            led.mark_pending_settlement(score_row)
        led.complete_retry_terminal(
            hotkey=ref.hotkey,
            bundle_hash=ref.content_hash,
            arena_bracket=arena.bracket,
            lease_id=lease.lease_id,
        )
        led.save(ledger_path)
        res.evaluated[ref.content_hash] = outcome.passed
        logger.info("evaluated %s… by %s: passed=%s score=%.4f target=%s mode=%s members=%s",
                    ref.content_hash[:16], ref.hotkey, outcome.passed, outcome.score,
                    target, mode, members)

    _recover_pending_settlements(
        led,
        ledger_path=ledger_path,
        arena=arena,
        margin=margin,
        host_attestation_verifier=host_attestation_verifier,
        validator_hotkey=validator_hotkey,
    )
    registered_arenas = tuple(ARENAS[name] for name in sorted(ARENAS))
    arena_set_sha256 = _global_arena_set_sha256(registered_arenas)
    res.weights = led.current_weights_across_arenas(
        registered_arenas,
        host_attestation_verifier=host_attestation_verifier,
        validator_hotkey=validator_hotkey,
    )
    # Preserve durability for non-settlement mutations such as retry-debris
    # cleanup on an already-known authoritative evaluation.
    led.save(ledger_path)

    # Push weights when they changed, or on the refresh cadence (stay "active").
    state_path = _weight_state_path(ledger_path, chain_scope)
    state = _load_weights_state(state_path)
    if state.get("schema") == WEIGHT_PUBLICATION_SCHEMA:
        _validate_weight_publication_state(
            state,
            chain_scope=chain_scope,
            arena_set_sha256=arena_set_sha256,
            emission_policy=arena.settlement.emission_policy,
        )
    elif (
        state.get("chain_scope") != chain_scope
        or state.get("arena_set_sha256") != arena_set_sha256
    ):
        # Pre-v1 cache bytes were never dedup authority. They may be ignored and
        # later upgraded only after the live sparse row independently confirms
        # the desired vector.
        state = {}
    else:
        _validate_weight_publication_state(
            state,
            chain_scope=chain_scope,
            arena_set_sha256=arena_set_sha256,
            emission_policy=arena.settlement.emission_policy,
        )
    desired_weights = _canonical_weight_map(res.weights)
    snapshot = chain.read_validator_weight_snapshot(
        subtensor, netuid, validator_hotkey
    )
    if snapshot.last_update_block > block:
        raise chain.ChainWeightStateError(
            "validator weight last-update block exceeds current chain height"
        )
    on_chain_weights = snapshot.weights
    if on_chain_weights and not desired_weights:
        # The chain API has no honest empty/all-zero weight vector: its normalizer
        # drops those entries. Quietly doing nothing would leave the old, now
        # unauthenticated emissions live. Stop loudly for explicit operator
        # neutralization/requalification instead of recording a successful pass.
        raise WeightSafetyError(
            "all retained champions became invalid while prior on-chain weights "
            "remain active; refusing to silently preserve stale emissions"
        )
    if not desired_weights:
        return res

    on_chain_matches = _weights_close(desired_weights, on_chain_weights)
    res.weights_confirmed = on_chain_matches

    # Upgrade the old best-effort cache only after the live chain independently
    # confirms it. A legacy local file is never publication authority.
    if (
        state.get("schema") != WEIGHT_PUBLICATION_SCHEMA
        and on_chain_matches
    ):
        legacy_submit = state.get("block", 0)
        if isinstance(legacy_submit, bool) or not isinstance(legacy_submit, int):
            legacy_submit = 0
        state = _weight_publication_state(
            chain_scope=chain_scope,
            arena_set_sha256=arena_set_sha256,
            emission_policy=arena.settlement.emission_policy,
            expected_weights=desired_weights,
            status=WEIGHT_STATUS_CONFIRMED,
            submit_block=max(0, legacy_submit),
            confirmed_block=block,
            confirmed_last_update=snapshot.last_update_block,
        )
        _atomic_write_weights_state(state_path, state)

    expected_state_weights = state.get("expected_weights")
    try:
        state_matches_desired = (
            isinstance(expected_state_weights, dict)
            and _weights_close(desired_weights, expected_state_weights)
        )
    except (TypeError, ValueError, OverflowError):
        raise WeightSafetyError("weight-publication expected vector is malformed") from None
    inflight_status = state.get("status") in {
        WEIGHT_STATUS_INTENT,
        WEIGHT_STATUS_PENDING,
        WEIGHT_STATUS_HELD,
    }
    if (
        state.get("schema") == WEIGHT_PUBLICATION_SCHEMA
        and inflight_status
        and not state_matches_desired
    ):
        raise WeightSafetyError(
            "unresolved weight publication belongs to a different expected vector; "
            "operator release is required"
        )
    pending_same_vector = bool(
        state.get("schema") == WEIGHT_PUBLICATION_SCHEMA
        and inflight_status
        and state_matches_desired
    )
    if pending_same_vector:
        submit_block = state.get("submit_block")
        retry_after = state.get("retry_after_block")
        if (
            isinstance(submit_block, bool)
            or not isinstance(submit_block, int)
            or submit_block < 0
            or isinstance(retry_after, bool)
            or not isinstance(retry_after, int)
            or retry_after < submit_block
        ):
            raise WeightSafetyError("weight-publication pending state is malformed")
        if on_chain_matches and snapshot.last_update_block >= submit_block:
            state = _weight_publication_state(
                chain_scope=chain_scope,
                arena_set_sha256=arena_set_sha256,
                emission_policy=arena.settlement.emission_policy,
                expected_weights=desired_weights,
                status=WEIGHT_STATUS_CONFIRMED,
                submit_block=submit_block,
                reveal_round=int(state.get("reveal_round", 0) or 0),
                confirmed_block=block,
                confirmed_last_update=snapshot.last_update_block,
            )
            _atomic_write_weights_state(state_path, state)
            pending_same_vector = False
        elif state.get("status") == WEIGHT_STATUS_HELD:
            res.weights_pending = True
            res.weights_held = True
            return res
        elif block < retry_after:
            res.weights_pending = True
            return res
        else:
            held_state = dict(state)
            held_state["status"] = WEIGHT_STATUS_HELD
            _validate_weight_publication_state(
                held_state,
                chain_scope=chain_scope,
                arena_set_sha256=arena_set_sha256,
                emission_policy=arena.settlement.emission_policy,
            )
            _atomic_write_weights_state(state_path, held_state)
            res.weights_pending = True
            res.weights_held = True
            return res

    # The live last-update block, rather than a local cache timestamp, owns the
    # refresh cadence. This also lets a restart confirm a previously submitted
    # CR vector without resubmitting it.
    stale = (
        on_chain_matches
        and block - snapshot.last_update_block >= weights_refresh_blocks
    )
    if on_chain_matches and not stale and not pending_same_vector:
        if (
            state.get("schema") != WEIGHT_PUBLICATION_SCHEMA
            or state.get("status") != WEIGHT_STATUS_CONFIRMED
            or not state_matches_desired
            or state.get("confirmed_last_update") != snapshot.last_update_block
        ):
            state = _weight_publication_state(
                chain_scope=chain_scope,
                arena_set_sha256=arena_set_sha256,
                emission_policy=arena.settlement.emission_policy,
                expected_weights=desired_weights,
                status=WEIGHT_STATUS_CONFIRMED,
                submit_block=int(state.get("submit_block", 0) or 0),
                confirmed_block=block,
                confirmed_last_update=snapshot.last_update_block,
            )
            _atomic_write_weights_state(state_path, state)
        return res

    if dry_run_weights:
        # A dry run is pure with respect to publication state. Existing pending
        # authority was handled above; a hypothetical submission creates no intent.
        chain.set_weights(
            subtensor,
            wallet,
            netuid,
            desired_weights,
            dry_run=True,
        )
        return res

    # Persist the exact intent before the external signing/submission effect.
    # A controller death anywhere inside the SDK call then leaves a restart-safe
    # ambiguity marker instead of silently duplicating a timelocked commit.
    intent_state = _weight_publication_state(
        chain_scope=chain_scope,
        arena_set_sha256=arena_set_sha256,
        emission_policy=arena.settlement.emission_policy,
        expected_weights=desired_weights,
        status=WEIGHT_STATUS_INTENT,
        submit_block=block,
        retry_after_block=block + weights_refresh_blocks,
        confirmed_block=(block if on_chain_matches else 0),
        confirmed_last_update=(
            snapshot.last_update_block if on_chain_matches else 0
        ),
    )
    _atomic_write_weights_state(state_path, intent_state)
    pushed = chain.set_weights(
        subtensor,
        wallet,
        netuid,
        desired_weights,
        dry_run=False,
    )
    res.weights_submitted = bool(pushed.get("submitted"))
    if not res.weights_submitted:
        # SDK failure is not proof that no external effect occurred. Preserve the
        # pre-submission intent until authoritative readback or its retry bound.
        res.weights_pending = True
        return res

    pending_state = _weight_publication_state(
        chain_scope=chain_scope,
        arena_set_sha256=arena_set_sha256,
        emission_policy=arena.settlement.emission_policy,
        expected_weights=desired_weights,
        status=WEIGHT_STATUS_PENDING,
        submit_block=block,
        reveal_round=_submitted_reveal_round(pushed),
        retry_after_block=block + weights_refresh_blocks,
        confirmed_block=(block if on_chain_matches else 0),
        confirmed_last_update=(
            snapshot.last_update_block if on_chain_matches else 0
        ),
    )
    _atomic_write_weights_state(state_path, pending_state)
    res.weights_pending = True

    # Non-CR subnets (and synchronous test doubles) may apply immediately. Only
    # a second authoritative sparse-row/last-update read may upgrade pending to
    # confirmed; SDK success by itself is never enough.
    post_submit = chain.read_validator_weight_snapshot(
        subtensor, netuid, validator_hotkey
    )
    post_observation_block = int(subtensor.get_current_block())
    if post_observation_block < block:
        raise chain.ChainWeightStateError(
            "post-submit chain height moved backward"
        )
    if post_submit.last_update_block > post_observation_block:
        raise chain.ChainWeightStateError(
            "post-submit weight update exceeds current chain height"
        )
    if (
        post_submit.last_update_block >= block
        and _weights_close(desired_weights, post_submit.weights)
    ):
        confirmed_state = _weight_publication_state(
            chain_scope=chain_scope,
            arena_set_sha256=arena_set_sha256,
            emission_policy=arena.settlement.emission_policy,
            expected_weights=desired_weights,
            status=WEIGHT_STATUS_CONFIRMED,
            submit_block=block,
            reveal_round=int(pending_state["reveal_round"]),
            confirmed_block=post_observation_block,
            confirmed_last_update=post_submit.last_update_block,
        )
        _atomic_write_weights_state(state_path, confirmed_state)
        res.weights_pending = False
        res.weights_confirmed = True
        res.weights_pushed = True
    return res


def run_pass(subtensor, wallet, netuid: int, *, ledger_path: str, bundles_dir: str,
             evaluator: Evaluator, arena: ArenaProfile, margin: float | None = None,
             round_blocks: int | None = None,
             weights_refresh_blocks: int | None = None,
             dry_run_weights: bool = False,
             host_attestation_verifier=None,
             validator_hotkey: str | None = None,
             test_only_allow_local_file_urls: bool = False) -> PassResult:
    """Execute one serialized chain-scoped referee pass.

    The persistent lock covers every mutation and external effect, including GPU
    evaluation and the final on-chain weight write. The test-only local transport is
    intentionally absent from ``run_validator``/CLI configuration.
    """

    with _exclusive_ledger_pass(ledger_path):
        return _run_pass_unlocked(
            subtensor,
            wallet,
            netuid,
            ledger_path=ledger_path,
            bundles_dir=bundles_dir,
            evaluator=evaluator,
            arena=arena,
            margin=margin,
            round_blocks=round_blocks,
            weights_refresh_blocks=weights_refresh_blocks,
            dry_run_weights=dry_run_weights,
            host_attestation_verifier=host_attestation_verifier,
            validator_hotkey=validator_hotkey,
            test_only_allow_local_file_urls=test_only_allow_local_file_urls,
        )


def run_validator(subtensor, wallet, netuid: int, *, ledger_path: str, bundles_dir: str,
                  evaluator: Evaluator, arena: ArenaProfile, margin: float | None = None,
                  interval_s: float = 60.0,
                  once: bool = False, dry_run_weights: bool = False,
                  round_blocks: int | None = None,
                  max_consecutive_failures: int = 10,
                  host_attestation_verifier=None,
                  validator_hotkey: str | None = None) -> Optional[PassResult]:
    """The daemon: run passes forever (or ``once``). A failing pass is logged and
    retried with linear backoff; ``max_consecutive_failures`` in a row exits nonzero
    so a supervisor restarts us with fresh connections (crash-only discipline)."""
    failures = 0
    last: Optional[PassResult] = None
    while True:
        try:
            last = run_pass(subtensor, wallet, netuid, ledger_path=ledger_path,
                            bundles_dir=bundles_dir, evaluator=evaluator, arena=arena,
                            margin=margin,
                            round_blocks=round_blocks, dry_run_weights=dry_run_weights,
                            host_attestation_verifier=host_attestation_verifier,
                            validator_hotkey=validator_hotkey)
            failures = 0
            logger.info("pass @block %d: seen=%d new=%d copies=%d rejected=%d held=%d "
                        "weights=%s publication(submitted=%s pending=%s held=%s "
                        "confirmed=%s)",
                        last.block, last.seen, len(last.new), len(last.copies),
                        len(last.rejected), len(last.held), last.weights,
                        last.weights_submitted, last.weights_pending,
                        last.weights_held, last.weights_confirmed)
        except Exception as exc:  # noqa: BLE001 — validator-side fault; contain and retry
            failures += 1
            logger.exception("validator pass failed (%d consecutive)", failures)
            if once or bool(getattr(exc, "validator_fault", False)):
                # One-shot mode is used by operators/automation as a health check.
                # Returning ``None`` with exit zero would turn a validator fault into
                # an apparently successful pass.
                raise
            if failures >= max_consecutive_failures:
                raise
        if once:
            return last
        time.sleep(interval_s * (1 + min(failures, 5)))
