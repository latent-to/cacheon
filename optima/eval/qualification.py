"""Typed, fail-closed report exchanged by GPU evaluation and the chain loop.

The subprocess exit code only says whether the evaluator process completed its
CLI path. Boolean/score headlines are not evidence either: the consumer recomputes
them from bounded raw B/C/B' throughput samples and B-vs-C/B-vs-B' external fidelity
metrics under the registered arena policy. Scheduler audit text remains diagnostic.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import statistics
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

from optima.arenas import ArenaProfile, ArenaPolicyError, get_arena
from optima.competition import ATOMIC_MODE, SLOT_MODE, SYSTEM_MODE, ResolvedCompetition
from optima.eval.external_quality import (
    ExternalQualityError,
    QUALITY_NO_DECISION,
    QUALITY_PASS,
    TeacherForcedExternalQualityEvidence,
    score_teacher_forced_quality,
)
from optima.eval.scoring import score_speedup


QUALIFICATION_SCHEMA_VERSION = 12
QUALIFICATION_KIND = "optima.evaluate"
HOST_ATTESTATION_PLACEHOLDER = "sha256:" + "0" * 64
_EVIDENCE_DIGEST_PLACEHOLDER = "sha256:" + "0" * 64
_MAX_THROUGHPUT_SAMPLE = 1e12
_MAX_REPORT_BYTES = 256 * 1024


class QualificationReportError(ValueError):
    """The evaluator report is incomplete or internally inconsistent."""


def _required_bool(data: Mapping[str, Any], key: str) -> bool:
    value = data.get(key)
    if type(value) is not bool:  # deliberately stricter than truthiness
        raise QualificationReportError(f"{key!r} must be a boolean")
    return value


def _required_number(data: Mapping[str, Any], key: str) -> float:
    value = data.get(key)
    # bool subclasses int, but accepting true as score=1 would be fail-open.
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise QualificationReportError(f"{key!r} must be a finite number")
    return float(value)


def _required_string(data: Mapping[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise QualificationReportError(f"{key!r} must be a non-empty string")
    return value


def _required_text(data: Mapping[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str):
        raise QualificationReportError(f"{key!r} must be a string")
    return value


def _required_int(data: Mapping[str, Any], key: str, *, minimum: int = 0) -> int:
    value = data.get(key)
    if type(value) is not int or value < minimum:
        raise QualificationReportError(
            f"{key!r} must be an integer >= {minimum}"
        )
    return value


def _exact_keys(data: Any, expected: set[str], *, path: str) -> Mapping[str, Any]:
    if not isinstance(data, Mapping):
        raise QualificationReportError(f"{path} must be a JSON object")
    actual = set(data)
    if actual != expected:
        raise QualificationReportError(
            f"{path} fields mismatch: missing={sorted(expected - actual)!r} "
            f"extra={sorted(actual - expected)!r}"
        )
    return data


def _same_number(name: str, actual: float, expected: float) -> None:
    if not math.isfinite(actual) or not math.isclose(
        actual, expected, rel_tol=1e-12, abs_tol=1e-12
    ):
        raise QualificationReportError(
            f"{name} headline disagrees with raw evidence: {actual!r} != {expected!r}"
        )


def _teacher_evidence_from_dict(raw: Any) -> TeacherForcedExternalQualityEvidence:
    try:
        return TeacherForcedExternalQualityEvidence.from_dict(raw)
    except ExternalQualityError as exc:
        raise QualificationReportError(str(exc)) from exc


@dataclass(frozen=True)
class ThroughputEvidence:
    """Raw B/C/B' samples plus host-timed conditioning-tail floors.

    Every warmup is quality-graded. The initial setup warmups (the total count
    minus ``conditioning_iters``) are throughput-free; the charged tail starts at
    completion of the last such response, or at the trusted ready frame when
    there are none. It then spans the final conditioning warmups, every
    inter-request gap and readiness sample, and the first timed response.
    The arm settles at the lower of its timed median and the minimum constituent
    or aggregate tail rate, so work or cooldown inside the charged tail cannot be
    hidden.
    """

    baseline_samples: tuple[float, ...]
    candidate_samples: tuple[float, ...]
    bookend_samples: tuple[float, ...]
    baseline_conditioning_tok_per_s: float
    candidate_conditioning_tok_per_s: float
    bookend_conditioning_tok_per_s: float

    @classmethod
    def from_evaluation(cls, report: Any) -> "ThroughputEvidence":
        if getattr(report, "baseline2", None) is None:
            raise QualificationReportError(
                "qualification requires a trailing B' throughput bookend"
            )

        def conditioning(result: Any, *, arm: str) -> float:
            value = getattr(result, "conditioning_tok_per_s", None)
            if type(value) not in (int, float) or not math.isfinite(float(value)):
                raise QualificationReportError(
                    f"qualification requires trusted {arm} conditioning-tail throughput"
                )
            return float(value)

        return cls(
            baseline_samples=tuple(report.baseline.tok_per_s_samples),
            candidate_samples=tuple(report.candidate.tok_per_s_samples),
            bookend_samples=tuple(report.baseline2.tok_per_s_samples),
            baseline_conditioning_tok_per_s=conditioning(
                report.baseline, arm="baseline"
            ),
            candidate_conditioning_tok_per_s=conditioning(
                report.candidate, arm="candidate"
            ),
            bookend_conditioning_tok_per_s=conditioning(
                report.baseline2, arm="bookend"
            ),
        )

    @classmethod
    def from_dict(cls, raw: Any) -> "ThroughputEvidence":
        data = _exact_keys(
            raw,
            {
                "baseline_samples", "candidate_samples", "bookend_samples",
                "baseline_conditioning_tok_per_s",
                "candidate_conditioning_tok_per_s",
                "bookend_conditioning_tok_per_s",
            },
            path="throughput_evidence",
        )

        def samples(name: str) -> tuple[float, ...]:
            values = data[name]
            if not isinstance(values, list):
                raise QualificationReportError(
                    f"throughput_evidence.{name} must be a JSON array"
                )
            return tuple(
                _required_number({"value": value}, "value") for value in values
            )

        return cls(
            baseline_samples=samples("baseline_samples"),
            candidate_samples=samples("candidate_samples"),
            bookend_samples=samples("bookend_samples"),
            baseline_conditioning_tok_per_s=_required_number(
                data, "baseline_conditioning_tok_per_s"
            ),
            candidate_conditioning_tok_per_s=_required_number(
                data, "candidate_conditioning_tok_per_s"
            ),
            bookend_conditioning_tok_per_s=_required_number(
                data, "bookend_conditioning_tok_per_s"
            ),
        )

    def validated(self, arena: ArenaProfile) -> "ThroughputEvidence":
        expected = arena.scoring.timed_iters
        for name, values in (
            ("baseline_samples", self.baseline_samples),
            ("candidate_samples", self.candidate_samples),
            ("bookend_samples", self.bookend_samples),
        ):
            if not isinstance(values, tuple) or len(values) != expected:
                raise QualificationReportError(
                    f"throughput_evidence.{name} must contain exactly "
                    f"arena timed_iters={expected} samples"
                )
            if any(
                type(value) not in (int, float)
                or not math.isfinite(float(value))
                or not (0.0 < float(value) <= _MAX_THROUGHPUT_SAMPLE)
                for value in values
            ):
                raise QualificationReportError(
                    f"throughput_evidence.{name} contains an invalid sample"
                )
        for name in (
            "baseline_conditioning_tok_per_s",
            "candidate_conditioning_tok_per_s",
            "bookend_conditioning_tok_per_s",
        ):
            value = getattr(self, name)
            if (
                type(value) not in (int, float)
                or not math.isfinite(float(value))
                or not 0.0 < float(value) <= _MAX_THROUGHPUT_SAMPLE
            ):
                raise QualificationReportError(
                    f"throughput_evidence.{name} is invalid"
                )
        for conditioning_name, samples_name in (
            ("baseline_conditioning_tok_per_s", "baseline_samples"),
            ("candidate_conditioning_tok_per_s", "candidate_samples"),
            ("bookend_conditioning_tok_per_s", "bookend_samples"),
        ):
            if getattr(self, conditioning_name) > getattr(self, samples_name)[0]:
                raise QualificationReportError(
                    f"throughput_evidence.{conditioning_name} cannot exceed "
                    "the first timed sample"
                )
        return self

    def point_estimates(self) -> tuple[float, float, float]:
        return (
            min(
                statistics.median(self.baseline_samples),
                self.baseline_conditioning_tok_per_s,
            ),
            min(
                statistics.median(self.candidate_samples),
                self.candidate_conditioning_tok_per_s,
            ),
            min(
                statistics.median(self.bookend_samples),
                self.bookend_conditioning_tok_per_s,
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_samples": list(self.baseline_samples),
            "candidate_samples": list(self.candidate_samples),
            "bookend_samples": list(self.bookend_samples),
            "baseline_conditioning_tok_per_s": (
                self.baseline_conditioning_tok_per_s
            ),
            "candidate_conditioning_tok_per_s": (
                self.candidate_conditioning_tok_per_s
            ),
            "bookend_conditioning_tok_per_s": (
                self.bookend_conditioning_tok_per_s
            ),
        }


@dataclass(frozen=True)
class QualificationReport:
    """The complete settlement-facing result of one ``optima evaluate`` run."""

    passed_quality: bool
    passed_timed_quality: bool
    passed_warmup_quality: bool
    quality_decision: str
    timed_quality_decision: str
    warmup_quality_decision: str
    passed_speedup: bool
    confident: bool
    crownable: bool
    score: float
    speedup: float
    noise: float
    required_speedup: float
    teacher_forced_mean_nll: float
    throughput_evidence: ThroughputEvidence
    external_quality_evidence: TeacherForcedExternalQualityEvidence
    target: str
    mode: str
    member_slots: tuple[str, ...]
    arena_name: str
    arena_fingerprint: str
    arena_bracket: str
    regime: str
    bundle_hash: str
    sglang_version: str
    validator_image: str
    referee_source_digest: str
    referee_tree_digest: str
    model_revision: str
    model_manifest_digest: str
    model_content_digest: str
    host_attestation_sha256: str
    chain_scope: str
    validator_hotkey: str
    evaluation_id: str
    miner_hotkey: str
    settlement_round_id: int
    evaluation_block: int
    qualification_evidence_sha256: str
    prompt_seed: int
    prompt_engine_version: str
    prompt_seed_scheme: str
    seed_round_id: int
    seed_block: int
    seed_block_hash: str
    # Human-readable summaries only. Decisions are recomputed from the typed fields.
    quality_evidence: str
    audit_evidence: str = ""
    completed: bool = True

    @property
    def kl_mean(self) -> float:
        """Legacy ledger spelling for the scalar quality loss.

        The serialized v12 field is explicit: this is teacher-forced target NLL,
        not rollout KL.  Keeping the read-only property avoids widening this change
        into settlement storage before that generic column is renamed.
        """

        return self.teacher_forced_mean_nll

    def attestation_context(self) -> dict[str, object]:
        """Canonical context expected in the retained trusted-host sidecar."""

        from optima.eval.host_attestation import host_attestation_context

        return host_attestation_context(
            get_arena(self.arena_name),
            bundle_hash=self.bundle_hash,
            prompt_seed=self.prompt_seed,
            seed_round_id=self.seed_round_id,
            seed_block=self.seed_block,
            seed_block_hash=self.seed_block_hash,
            chain_scope=self.chain_scope,
            validator_hotkey=self.validator_hotkey,
            evaluation_id=self.evaluation_id,
            miner_hotkey=self.miner_hotkey,
            settlement_round_id=self.settlement_round_id,
            evaluation_block=self.evaluation_block,
            target=self.target,
            mode=self.mode,
            member_slots=self.member_slots,
            score=self.score,
            passed_quality=self.passed_quality,
            passed_timed_quality=self.passed_timed_quality,
            passed_warmup_quality=self.passed_warmup_quality,
            passed_speedup=self.passed_speedup,
            confident=self.confident,
            crownable=self.crownable,
            quality_evidence=self.quality_evidence,
            qualification_evidence_sha256=self.qualification_evidence_sha256,
        )

    @classmethod
    def prepare_evidence(
        cls,
        report: Any,
        *,
        competition: ResolvedCompetition,
        arena: ArenaProfile,
        bundle_hash: str,
        prompt_seed: int,
        seed_round_id: int,
        seed_block: int,
        seed_block_hash: str,
        chain_scope: str,
        validator_hotkey: str,
        evaluation_id: str,
        miner_hotkey: str,
        settlement_round_id: int,
        evaluation_block: int,
    ) -> "QualificationReport":
        """Build and hash the non-circular evidence before sidecar publication.

        The producer must pass the result of ``resolve_competition(...,
        for_settlement=True)``.  In particular, it must never derive settlement
        identity from manifest order or copy a miner-provided target directly. The
        returned object deliberately carries ``HOST_ATTESTATION_PLACEHOLDER`` and
        cannot be serialized as a final report until ``bind_host_attestation``.
        """
        competition.require_crownable()
        if not arena.competable:
            raise QualificationReportError(
                f"arena {arena.name!r} is not crownable"
            )
        if competition.target is None or competition.mode is None:
            raise QualificationReportError(
                "crownable competition is missing target or mode"
            )
        throughput_evidence = ThroughputEvidence.from_evaluation(report)
        external_quality_evidence = getattr(report, "external_quality_evidence", None)
        if type(external_quality_evidence) is not TeacherForcedExternalQualityEvidence:
            raise QualificationReportError(
                "qualification requires typed post-hoc teacher-forced evidence"
            )
        try:
            quality_verdict = score_teacher_forced_quality(
                external_quality_evidence, arena=arena
            )
        except ExternalQualityError as exc:
            raise QualificationReportError(str(exc)) from exc
        passed_quality = quality_verdict.decision == QUALITY_PASS
        passed_timed_quality = quality_verdict.timed_decision == QUALITY_PASS
        passed_warmup_quality = quality_verdict.warmup_decision == QUALITY_PASS
        crownable = bool(
            passed_quality and report.passed_speedup and report.confident
        )
        provisional = cls(
            passed_quality=passed_quality,
            passed_timed_quality=passed_timed_quality,
            passed_warmup_quality=passed_warmup_quality,
            quality_decision=quality_verdict.decision,
            timed_quality_decision=quality_verdict.timed_decision,
            warmup_quality_decision=quality_verdict.warmup_decision,
            passed_speedup=bool(report.passed_speedup),
            confident=bool(report.confident),
            crownable=crownable,
            score=float(report.speedup) if crownable else 0.0,
            speedup=float(report.speedup),
            noise=float(report.noise),
            required_speedup=float(report.required_speedup),
            teacher_forced_mean_nll=quality_verdict.candidate_mean_nll,
            throughput_evidence=throughput_evidence,
            external_quality_evidence=external_quality_evidence,
            target=competition.target,
            mode=competition.mode,
            member_slots=tuple(competition.members),
            arena_name=arena.name,
            arena_fingerprint=arena.fingerprint,
            arena_bracket=arena.bracket,
            regime=arena.workload.regime,
            bundle_hash=bundle_hash,
            sglang_version=arena.sglang_version,
            validator_image=arena.validator_image,
            referee_source_digest=arena.referee_source_digest,
            referee_tree_digest=arena.referee_tree_digest,
            model_revision=arena.model_revision,
            model_manifest_digest=arena.model_manifest_digest,
            model_content_digest=arena.model_content_digest,
            host_attestation_sha256=HOST_ATTESTATION_PLACEHOLDER,
            chain_scope=chain_scope,
            validator_hotkey=validator_hotkey,
            evaluation_id=evaluation_id,
            miner_hotkey=miner_hotkey,
            settlement_round_id=settlement_round_id,
            evaluation_block=evaluation_block,
            qualification_evidence_sha256=_EVIDENCE_DIGEST_PLACEHOLDER,
            prompt_seed=int(prompt_seed),
            prompt_engine_version=arena.workload.prompt_engine_version,
            prompt_seed_scheme=arena.workload.prompt_seed_scheme,
            seed_round_id=int(seed_round_id),
            seed_block=int(seed_block),
            seed_block_hash=str(seed_block_hash),
            quality_evidence=quality_verdict.detail[:4096],
            audit_evidence=(
                str(getattr(report, "audit_desc", ""))[:4096]
                if getattr(report, "fidelity_mode", "") == "audit"
                else ""
            ),
        )
        prepared = replace(
            provisional,
            qualification_evidence_sha256=(
                provisional._computed_evidence_sha256()
            ),
        )
        return prepared.validated(allow_placeholder_host=True)

    @classmethod
    def from_evaluation(
        cls,
        report: Any,
        *,
        competition: ResolvedCompetition,
        arena: ArenaProfile,
        bundle_hash: str,
        prompt_seed: int,
        seed_round_id: int,
        seed_block: int,
        seed_block_hash: str,
        chain_scope: str,
        validator_hotkey: str,
        evaluation_id: str,
        miner_hotkey: str,
        settlement_round_id: int,
        evaluation_block: int,
        host_attestation_sha256: str,
    ) -> "QualificationReport":
        """Build evidence and bind an already-published host sidecar digest."""

        return cls.prepare_evidence(
            report,
            competition=competition,
            arena=arena,
            bundle_hash=bundle_hash,
            prompt_seed=prompt_seed,
            seed_round_id=seed_round_id,
            seed_block=seed_block,
            seed_block_hash=seed_block_hash,
            chain_scope=chain_scope,
            validator_hotkey=validator_hotkey,
            evaluation_id=evaluation_id,
            miner_hotkey=miner_hotkey,
            settlement_round_id=settlement_round_id,
            evaluation_block=evaluation_block,
        ).bind_host_attestation(host_attestation_sha256)

    def bind_host_attestation(self, digest: str) -> "QualificationReport":
        """Finalize prepared evidence without changing its evidence hash."""

        if self.host_attestation_sha256 != HOST_ATTESTATION_PLACEHOLDER:
            raise QualificationReportError(
                "qualification evidence is already bound to a host attestation"
            )
        bound = replace(self, host_attestation_sha256=str(digest))
        if bound._computed_evidence_sha256() != self.qualification_evidence_sha256:
            raise QualificationReportError(
                "host attestation binding changed non-circular qualification evidence"
            )
        return bound.validated()

    @classmethod
    def from_dict(
        cls, raw: Any, *, allow_placeholder_host: bool = False
    ) -> "QualificationReport":
        if not isinstance(raw, Mapping):
            raise QualificationReportError("report root must be a JSON object")
        if (type(raw.get("schema_version")) is not int
                or raw.get("schema_version") != QUALIFICATION_SCHEMA_VERSION):
            raise QualificationReportError(
                f"unsupported or missing schema_version (expected "
                f"{QUALIFICATION_SCHEMA_VERSION})"
            )
        if raw.get("kind") != QUALIFICATION_KIND:
            raise QualificationReportError(
                f"unsupported or missing report kind (expected {QUALIFICATION_KIND!r})"
            )
        _exact_keys(
            raw,
            {
                "schema_version", "kind", "completed", "passed_quality",
                "passed_timed_quality", "passed_warmup_quality",
                "quality_decision", "timed_quality_decision",
                "warmup_quality_decision",
                "passed_speedup", "confident", "crownable", "score", "speedup",
                "noise", "required_speedup", "teacher_forced_mean_nll",
                "throughput_evidence",
                "external_quality_evidence", "target", "mode", "member_slots",
                "arena_name", "arena_fingerprint", "arena_bracket", "regime",
                "bundle_hash", "sglang_version", "validator_image",
                "referee_source_digest", "referee_tree_digest", "model_revision",
                "model_manifest_digest",
                "model_content_digest", "host_attestation_sha256", "chain_scope",
                "validator_hotkey", "evaluation_id", "miner_hotkey",
                "settlement_round_id", "evaluation_block",
                "qualification_evidence_sha256", "prompt_seed", "prompt_engine_version",
                "prompt_seed_scheme", "seed_round_id", "seed_block",
                "seed_block_hash", "quality_evidence", "audit_evidence",
            },
            path="report",
        )
        target = raw.get("target")
        if not isinstance(target, str) or not target.strip():
            raise QualificationReportError("'target' must be a non-empty string")
        mode = raw.get("mode")
        if not isinstance(mode, str):
            raise QualificationReportError("'mode' must be a string")
        raw_members = raw.get("member_slots")
        if not isinstance(raw_members, list):
            raise QualificationReportError("'member_slots' must be a JSON array")
        if any(
            not isinstance(member, str) or not member.strip()
            for member in raw_members
        ):
            raise QualificationReportError(
                "'member_slots' must contain non-empty strings"
            )
        return cls(
            completed=_required_bool(raw, "completed"),
            passed_quality=_required_bool(raw, "passed_quality"),
            passed_timed_quality=_required_bool(raw, "passed_timed_quality"),
            passed_warmup_quality=_required_bool(raw, "passed_warmup_quality"),
            quality_decision=_required_string(raw, "quality_decision"),
            timed_quality_decision=_required_string(raw, "timed_quality_decision"),
            warmup_quality_decision=_required_string(
                raw, "warmup_quality_decision"
            ),
            passed_speedup=_required_bool(raw, "passed_speedup"),
            confident=_required_bool(raw, "confident"),
            crownable=_required_bool(raw, "crownable"),
            score=_required_number(raw, "score"),
            speedup=_required_number(raw, "speedup"),
            noise=_required_number(raw, "noise"),
            required_speedup=_required_number(raw, "required_speedup"),
            teacher_forced_mean_nll=_required_number(
                raw, "teacher_forced_mean_nll"
            ),
            throughput_evidence=ThroughputEvidence.from_dict(
                raw.get("throughput_evidence")
            ),
            external_quality_evidence=_teacher_evidence_from_dict(
                raw.get("external_quality_evidence")
            ),
            target=target,
            mode=mode,
            member_slots=tuple(raw_members),
            arena_name=_required_string(raw, "arena_name"),
            arena_fingerprint=_required_string(raw, "arena_fingerprint"),
            arena_bracket=_required_string(raw, "arena_bracket"),
            regime=_required_string(raw, "regime"),
            bundle_hash=_required_string(raw, "bundle_hash"),
            sglang_version=_required_string(raw, "sglang_version"),
            validator_image=_required_string(raw, "validator_image"),
            referee_source_digest=_required_string(raw, "referee_source_digest"),
            referee_tree_digest=_required_string(raw, "referee_tree_digest"),
            model_revision=_required_string(raw, "model_revision"),
            model_manifest_digest=_required_string(raw, "model_manifest_digest"),
            model_content_digest=_required_string(raw, "model_content_digest"),
            host_attestation_sha256=_required_string(
                raw, "host_attestation_sha256"
            ),
            chain_scope=_required_string(raw, "chain_scope"),
            validator_hotkey=_required_string(raw, "validator_hotkey"),
            evaluation_id=_required_string(raw, "evaluation_id"),
            miner_hotkey=_required_string(raw, "miner_hotkey"),
            settlement_round_id=raw.get("settlement_round_id"),
            evaluation_block=raw.get("evaluation_block"),
            qualification_evidence_sha256=_required_string(
                raw, "qualification_evidence_sha256"
            ),
            prompt_seed=raw.get("prompt_seed"),
            prompt_engine_version=_required_string(raw, "prompt_engine_version"),
            prompt_seed_scheme=_required_string(raw, "prompt_seed_scheme"),
            seed_round_id=raw.get("seed_round_id"),
            seed_block=raw.get("seed_block"),
            seed_block_hash=_required_string(raw, "seed_block_hash"),
            quality_evidence=_required_string(raw, "quality_evidence"),
            audit_evidence=_required_text(raw, "audit_evidence"),
        ).validated(allow_placeholder_host=allow_placeholder_host)

    @classmethod
    def from_evidence_dict(
        cls,
        evidence: Mapping[str, Any],
        *,
        qualification_evidence_sha256: str,
    ) -> "QualificationReport":
        """Reconstitute prepared evidence retained inside the host sidecar."""

        if not isinstance(evidence, Mapping):
            raise QualificationReportError(
                "qualification evidence must be a JSON object"
            )
        if (
            "host_attestation_sha256" in evidence
            or "qualification_evidence_sha256" in evidence
        ):
            raise QualificationReportError(
                "qualification evidence contains circular digest fields"
            )
        raw = dict(evidence)
        raw["host_attestation_sha256"] = HOST_ATTESTATION_PLACEHOLDER
        raw["qualification_evidence_sha256"] = str(
            qualification_evidence_sha256
        )
        return cls.from_dict(raw, allow_placeholder_host=True)

    @classmethod
    def read(cls, path: str | Path) -> "QualificationReport":
        report_path = Path(path)
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            fd = os.open(report_path, flags)
        except OSError as exc:
            raise QualificationReportError(f"cannot read report: {exc}") from exc
        try:
            before = os.fstat(fd)
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_nlink != 1
                or before.st_size < 0
                or before.st_size > _MAX_REPORT_BYTES
            ):
                raise QualificationReportError(
                    "qualification report must be one bounded regular file"
                )
            chunks: list[bytes] = []
            remaining = before.st_size
            while remaining:
                chunk = os.read(fd, min(64 * 1024, remaining))
                if not chunk:
                    raise QualificationReportError(
                        "qualification report was truncated while reading"
                    )
                chunks.append(chunk)
                remaining -= len(chunk)
            if os.read(fd, 1):
                raise QualificationReportError(
                    f"qualification report exceeds {_MAX_REPORT_BYTES} bytes"
                )
            after = os.fstat(fd)
            stable = (
                "st_dev", "st_ino", "st_mode", "st_nlink", "st_size",
                "st_mtime_ns", "st_ctime_ns",
            )
            if any(getattr(before, name) != getattr(after, name) for name in stable):
                raise QualificationReportError(
                    "qualification report changed while reading"
                )
            def object_pairs(pairs):
                result = {}
                for key, value in pairs:
                    if key in result:
                        raise ValueError(f"duplicate JSON key {key!r}")
                    result[key] = value
                return result

            raw = json.loads(
                b"".join(chunks),
                object_pairs_hook=object_pairs,
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError(f"invalid JSON constant {value}")
                ),
            )
        except (
            OSError,
            json.JSONDecodeError,
            UnicodeDecodeError,
            RecursionError,
            ValueError,
        ) as exc:
            raise QualificationReportError(f"malformed JSON report: {exc}") from exc
        finally:
            os.close(fd)
        return cls.from_dict(raw)

    def validated(
        self, *, allow_placeholder_host: bool = False
    ) -> "QualificationReport":
        if (
            not isinstance(self.target, str)
            or not self.target.strip()
            or len(self.target) > 256
        ):
            raise QualificationReportError("target must be a non-empty string")
        if self.mode not in (SLOT_MODE, ATOMIC_MODE, SYSTEM_MODE):
            raise QualificationReportError("mode must be 'slot', 'atomic', or 'system'")
        if not isinstance(self.member_slots, tuple):
            raise QualificationReportError("member_slots must be a tuple")
        if len(self.member_slots) > 64 or any(
            not isinstance(member, str) or not member.strip() or len(member) > 256
            for member in self.member_slots
        ):
            raise QualificationReportError(
                "member_slots must contain non-empty strings"
            )
        if len(set(self.member_slots)) != len(self.member_slots):
            raise QualificationReportError("member_slots must not contain duplicates")
        if self.mode == SLOT_MODE and self.member_slots != (self.target,):
            raise QualificationReportError(
                "slot-mode report must name exactly its target as the sole member"
            )
        if self.mode == ATOMIC_MODE and len(self.member_slots) < 2:
            raise QualificationReportError(
                "atomic report must contain at least two member slots"
            )
        if self.mode == SYSTEM_MODE and self.member_slots:
            raise QualificationReportError(
                "system report must not manufacture component member slots"
            )
        if not self.arena_name:
            raise QualificationReportError("arena_name must be non-empty")
        try:
            arena = get_arena(self.arena_name)
        except ArenaPolicyError as exc:
            raise QualificationReportError(str(exc)) from exc
        expected_arena = {
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
            "prompt_engine_version": arena.workload.prompt_engine_version,
            "prompt_seed_scheme": arena.workload.prompt_seed_scheme,
        }
        for field_name, expected in expected_arena.items():
            actual = getattr(self, field_name)
            if actual != expected:
                raise QualificationReportError(
                    f"{field_name} disagrees with registered arena "
                    f"{self.arena_name!r}: {actual!r} != {expected!r}"
                )
        if not re.fullmatch(r"[0-9a-f]{64}", self.bundle_hash):
            raise QualificationReportError(
                "bundle_hash must be a lowercase SHA-256 content hash"
            )
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", self.referee_source_digest):
            raise QualificationReportError(
                "referee_source_digest must be a SHA-256 content identity"
            )
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", self.referee_tree_digest):
            raise QualificationReportError(
                "referee_tree_digest must be a SHA-256 content identity"
            )
        if not re.fullmatch(r"[0-9a-f]{40,64}", self.model_revision):
            raise QualificationReportError(
                "model_revision must be an immutable hex revision"
            )
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", self.model_manifest_digest):
            raise QualificationReportError(
                "model_manifest_digest must be a SHA-256 content identity"
            )
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", self.model_content_digest):
            raise QualificationReportError(
                "model_content_digest must be a SHA-256 content identity"
            )
        if re.fullmatch(r"sha256:[0-9a-f]{64}", self.host_attestation_sha256) is None:
            raise QualificationReportError(
                "host_attestation_sha256 must bind retained trusted-host evidence"
            )
        if (
            self.host_attestation_sha256 == HOST_ATTESTATION_PLACEHOLDER
            and not allow_placeholder_host
        ):
            raise QualificationReportError(
                "final qualification report still has the host attestation placeholder"
            )
        expected_chain_prefix = (
            arena.settlement.chain_scope_scheme + ":sha256:"
        )
        if (
            not isinstance(self.chain_scope, str)
            or len(self.chain_scope) > 256
            or not self.chain_scope.startswith(expected_chain_prefix)
            or re.fullmatch(
                r"[A-Za-z0-9_.-]{1,128}:sha256:[0-9a-f]{64}",
                self.chain_scope,
            ) is None
        ):
            raise QualificationReportError(
                "chain_scope must be a canonical registered-chain identity"
            )
        if (
            not isinstance(self.validator_hotkey, str)
            or not self.validator_hotkey
            or self.validator_hotkey.strip() != self.validator_hotkey
            or len(self.validator_hotkey) > 256
            or any(char in self.validator_hotkey for char in "\x00\r\n")
        ):
            raise QualificationReportError(
                "validator_hotkey must be a bounded non-empty identity"
            )
        if re.fullmatch(r"[0-9a-f]{64}", self.evaluation_id) is None:
            raise QualificationReportError(
                "evaluation_id must be the exact lowercase 64-hex evaluation lease ID"
            )
        if (
            not isinstance(self.miner_hotkey, str)
            or not self.miner_hotkey
            or self.miner_hotkey.strip() != self.miner_hotkey
            or len(self.miner_hotkey) > 256
            or any(char in self.miner_hotkey for char in "\x00\r\n")
        ):
            raise QualificationReportError(
                "miner_hotkey must be the exact bounded submission owner"
            )
        if (
            type(self.settlement_round_id) is not int
            or type(self.evaluation_block) is not int
            or self.settlement_round_id < 0
            or self.evaluation_block < 0
            or self.settlement_round_id
            != self.evaluation_block // arena.settlement.round_blocks
        ):
            raise QualificationReportError(
                "settlement round/evaluation block provenance is inconsistent"
            )
        if re.fullmatch(
            r"sha256:[0-9a-f]{64}", self.qualification_evidence_sha256
        ) is None:
            raise QualificationReportError(
                "qualification_evidence_sha256 must be a SHA-256 identity"
            )
        if type(self.prompt_seed) is not int or self.prompt_seed <= 0:
            raise QualificationReportError(
                "prompt_seed must be a positive post-commit-derived integer"
            )
        if (type(self.seed_round_id) is not int or type(self.seed_block) is not int
                or self.seed_round_id < 0 or self.seed_block < 0
                or self.seed_round_id
                != self.seed_block // arena.settlement.round_blocks):
            raise QualificationReportError(
                "seed round/block provenance disagrees with arena settlement policy"
            )
        if self.evaluation_block < self.seed_block:
            raise QualificationReportError(
                "evaluation block cannot predate the finalized reveal seed block"
            )
        if re.fullmatch(r"0x[0-9a-f]{64}", self.seed_block_hash) is None:
            raise QualificationReportError(
                "seed_block_hash must be a canonical finalized block hash"
            )
        from optima.arenas import derive_prompt_seed

        expected_prompt_seed = derive_prompt_seed(
            arena,
            bundle_hash=self.bundle_hash,
            round_id=self.seed_round_id,
            block_hash=self.seed_block_hash,
        )
        if self.prompt_seed != expected_prompt_seed:
            raise QualificationReportError(
                "prompt_seed does not match its block-hash derivation receipt"
            )
        self.throughput_evidence.validated(arena)
        baseline, candidate, bookend = self.throughput_evidence.point_estimates()
        speed_verdict = score_speedup(
            [baseline, bookend],
            candidate,
            min_margin=arena.scoring.speedup_margin,
            k=arena.scoring.score_k,
            max_noise=arena.scoring.max_noise,
        )
        _same_number("speedup", self.speedup, speed_verdict.speedup)
        _same_number("noise", self.noise, speed_verdict.noise)
        _same_number(
            "required_speedup", self.required_speedup, speed_verdict.required
        )
        if type(self.confident) is not bool or self.confident != speed_verdict.confident:
            raise QualificationReportError(
                "confident headline disagrees with raw throughput evidence"
            )
        if (
            type(self.passed_speedup) is not bool
            or self.passed_speedup != speed_verdict.passed_speedup
        ):
            raise QualificationReportError(
                "passed_speedup headline disagrees with raw throughput evidence"
            )

        try:
            quality_verdict = score_teacher_forced_quality(
                self.external_quality_evidence, arena=arena
            )
        except ExternalQualityError as exc:
            raise QualificationReportError(str(exc)) from exc
        expected_timed_quality = quality_verdict.timed_decision == QUALITY_PASS
        expected_warmup_quality = quality_verdict.warmup_decision == QUALITY_PASS
        expected_quality = quality_verdict.decision == QUALITY_PASS
        for field_name, actual, expected in (
            ("passed_timed_quality", self.passed_timed_quality, expected_timed_quality),
            ("passed_warmup_quality", self.passed_warmup_quality, expected_warmup_quality),
            ("passed_quality", self.passed_quality, expected_quality),
        ):
            if type(actual) is not bool or actual != expected:
                raise QualificationReportError(
                    f"{field_name} headline disagrees with teacher-forced evidence"
                )
        for field_name, actual, expected in (
            ("quality_decision", self.quality_decision, quality_verdict.decision),
            (
                "timed_quality_decision",
                self.timed_quality_decision,
                quality_verdict.timed_decision,
            ),
            (
                "warmup_quality_decision",
                self.warmup_quality_decision,
                quality_verdict.warmup_decision,
            ),
        ):
            if actual != expected:
                raise QualificationReportError(
                    f"{field_name} headline disagrees with teacher-forced evidence"
                )
        _same_number(
            "teacher_forced_mean_nll",
            self.teacher_forced_mean_nll,
            quality_verdict.candidate_mean_nll,
        )
        if self.quality_evidence != quality_verdict.detail[:4096]:
            raise QualificationReportError(
                "quality_evidence summary disagrees with raw external fidelity evidence"
            )
        if not isinstance(self.audit_evidence, str) or len(self.audit_evidence) > 4096:
            raise QualificationReportError(
                "audit_evidence must be a bounded diagnostic string"
            )
        if not self.completed:
            raise QualificationReportError("report does not mark evaluation completed")
        expected_crownable = expected_quality and speed_verdict.passed_speedup
        if type(self.crownable) is not bool or self.crownable != expected_crownable:
            raise QualificationReportError(
                "crownable headline disagrees with recomputed quality/speed evidence"
            )
        expected_score = speed_verdict.speedup if expected_crownable else 0.0
        _same_number("score", self.score, expected_score)
        computed_evidence = self._computed_evidence_sha256()
        if self.qualification_evidence_sha256 != computed_evidence:
            raise QualificationReportError(
                "qualification_evidence_sha256 disagrees with canonical raw evidence"
            )
        return self

    def _raw_dict(self) -> dict[str, Any]:
        return {
            "schema_version": QUALIFICATION_SCHEMA_VERSION,
            "kind": QUALIFICATION_KIND,
            "completed": self.completed,
            "passed_quality": self.passed_quality,
            "passed_timed_quality": self.passed_timed_quality,
            "passed_warmup_quality": self.passed_warmup_quality,
            "quality_decision": self.quality_decision,
            "timed_quality_decision": self.timed_quality_decision,
            "warmup_quality_decision": self.warmup_quality_decision,
            "passed_speedup": self.passed_speedup,
            "confident": self.confident,
            "crownable": self.crownable,
            "score": self.score,
            "speedup": self.speedup,
            "noise": self.noise,
            "required_speedup": self.required_speedup,
            "teacher_forced_mean_nll": self.teacher_forced_mean_nll,
            "throughput_evidence": self.throughput_evidence.to_dict(),
            "external_quality_evidence": self.external_quality_evidence.to_dict(),
            "target": self.target,
            "mode": self.mode,
            "member_slots": list(self.member_slots),
            "arena_name": self.arena_name,
            "arena_fingerprint": self.arena_fingerprint,
            "arena_bracket": self.arena_bracket,
            "regime": self.regime,
            "bundle_hash": self.bundle_hash,
            "sglang_version": self.sglang_version,
            "validator_image": self.validator_image,
            "referee_source_digest": self.referee_source_digest,
            "referee_tree_digest": self.referee_tree_digest,
            "model_revision": self.model_revision,
            "model_manifest_digest": self.model_manifest_digest,
            "model_content_digest": self.model_content_digest,
            "host_attestation_sha256": self.host_attestation_sha256,
            "chain_scope": self.chain_scope,
            "validator_hotkey": self.validator_hotkey,
            "evaluation_id": self.evaluation_id,
            "miner_hotkey": self.miner_hotkey,
            "settlement_round_id": self.settlement_round_id,
            "evaluation_block": self.evaluation_block,
            "qualification_evidence_sha256": self.qualification_evidence_sha256,
            "prompt_seed": self.prompt_seed,
            "prompt_engine_version": self.prompt_engine_version,
            "prompt_seed_scheme": self.prompt_seed_scheme,
            "seed_round_id": self.seed_round_id,
            "seed_block": self.seed_block,
            "seed_block_hash": self.seed_block_hash,
            "quality_evidence": self.quality_evidence,
            "audit_evidence": self.audit_evidence,
        }

    def _evidence_dict_unchecked(self) -> dict[str, Any]:
        evidence = self._raw_dict()
        del evidence["host_attestation_sha256"]
        del evidence["qualification_evidence_sha256"]
        return evidence

    def _computed_evidence_sha256(self) -> str:
        try:
            raw = json.dumps(
                self._evidence_dict_unchecked(),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("ascii")
        except (TypeError, ValueError, UnicodeError, RecursionError) as exc:
            raise QualificationReportError(
                f"qualification evidence is not canonical JSON: {exc}"
            ) from None
        if not raw or len(raw) > _MAX_REPORT_BYTES:
            raise QualificationReportError(
                f"qualification evidence exceeds {_MAX_REPORT_BYTES} bytes"
            )
        return "sha256:" + hashlib.sha256(raw).hexdigest()

    def evidence_dict(self) -> dict[str, Any]:
        """Canonical non-circular evidence retained by the host sidecar."""

        self.validated(
            allow_placeholder_host=(
                self.host_attestation_sha256 == HOST_ATTESTATION_PLACEHOLDER
            )
        )
        return self._evidence_dict_unchecked()

    def evidence_bytes(self) -> bytes:
        raw = json.dumps(
            self.evidence_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        if not raw or len(raw) > _MAX_REPORT_BYTES:  # pragma: no cover - rechecked
            raise QualificationReportError(
                f"qualification evidence exceeds {_MAX_REPORT_BYTES} bytes"
            )
        return raw

    def to_dict(self) -> dict[str, Any]:
        self.validated()
        return self._raw_dict()

    def write(self, path: str | Path) -> None:
        """Durably publish a complete report; never leave partial JSON."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = (json.dumps(self.to_dict(), sort_keys=True) + "\n").encode(
            "utf-8"
        )
        if len(payload) > _MAX_REPORT_BYTES:
            raise QualificationReportError(
                f"qualification report exceeds {_MAX_REPORT_BYTES} bytes"
            )
        fd, raw_tmp = tempfile.mkstemp(
            prefix=f".{target.name}.tmp.{os.getpid()}.", dir=target.parent
        )
        tmp = Path(raw_tmp)
        try:
            with os.fdopen(fd, "wb", closefd=True) as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(tmp, target)
            parent_fd = os.open(
                target.parent,
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
