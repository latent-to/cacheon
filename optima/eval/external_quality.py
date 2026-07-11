"""Controller-only post-hoc quality evidence for nondeterministic serving arenas.

The candidate process is gone before this protocol starts.  A surviving stock B'
engine teacher-forces the sealed B, C, and B' token trajectories under each
trajectory's *own* prefix.  This makes exact B/C rollout equality diagnostic only:
the crown decision comes from independently recomputed target likelihood and trusted
top-k evidence, paired by prompt cluster and calibrated against B versus B'.

Only bounded prompt summaries and digests enter ``QualificationReport``.  The raw
token/logprob frames remain controller artifacts addressed by ``raw_evidence_sha256``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import statistics
import tempfile
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Sequence, TYPE_CHECKING

from optima.eval.kl import kl_over_positions

if TYPE_CHECKING:  # pragma: no cover
    from optima.arenas import ArenaProfile, TeacherForcedQualityPolicy


TEACHER_FORCED_QUALITY_PROTOCOL_V2 = "controller-posthoc-teacher-forced-v2"
QUALITY_PASS = "PASS"
QUALITY_FAIL = "FAIL"
QUALITY_NO_DECISION = "NO_DECISION"
QUALITY_DECISIONS = frozenset({QUALITY_PASS, QUALITY_FAIL, QUALITY_NO_DECISION})
MAX_NLL = 1_000.0
MAX_RAW_QUALITY_BYTES = 128 * 1024 * 1024


class ExternalQualityError(ValueError):
    """The post-hoc evidence is malformed or cannot be graded safely."""


def _finite(value: Any, *, name: str, low: float = 0.0, high: float = MAX_NLL) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise ExternalQualityError(f"{name} must be finite")
    result = float(value)
    if not low <= result <= high:
        raise ExternalQualityError(f"{name} is out of bounds")
    return result


def _integer(value: Any, *, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ExternalQualityError(f"{name} must be an integer >= {minimum}")
    return value


def _sha256_id(value: Any, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 71
        or not value.startswith("sha256:")
        or any(char not in "0123456789abcdef" for char in value[7:])
    ):
        raise ExternalQualityError(f"{name} must be a canonical SHA-256 identity")
    return value


@dataclass(frozen=True)
class TeacherForcedTrace:
    """Raw stock-B' scores for one sealed rollout (controller artifact only)."""

    target_logprobs: tuple[float, ...]
    trusted_topk: tuple[tuple[tuple[float, int, None], ...], ...]

    def validated(self, *, expected_tokens: int, topk_num: int) -> "TeacherForcedTrace":
        if len(self.target_logprobs) != expected_tokens:
            raise ExternalQualityError("teacher target-logprob length mismatch")
        if len(self.trusted_topk) != expected_tokens:
            raise ExternalQualityError("teacher top-k length mismatch")
        for logprob in self.target_logprobs:
            _finite(logprob, name="teacher target logprob", low=-MAX_NLL, high=0.0001)
        for position in self.trusted_topk:
            if not isinstance(position, tuple) or len(position) != topk_num:
                raise ExternalQualityError("teacher top-k width mismatch")
            seen: set[int] = set()
            previous = math.inf
            mass = 0.0
            for entry in position:
                if not isinstance(entry, tuple) or len(entry) != 3 or entry[2] is not None:
                    raise ExternalQualityError("teacher top-k entry is malformed")
                logprob = _finite(
                    entry[0], name="teacher top-k logprob", low=-MAX_NLL, high=0.0001
                )
                token_id = _integer(entry[1], name="teacher top-k token")
                if token_id > 2_147_483_647 or token_id in seen or logprob > previous:
                    raise ExternalQualityError("teacher top-k ordering/token identity is invalid")
                seen.add(token_id)
                previous = logprob
                mass += math.exp(logprob)
            if mass > 1.0001:
                raise ExternalQualityError("teacher top-k probability mass exceeds one")
        return self


@dataclass(frozen=True)
class TeacherForcedPromptTrace:
    """Raw B' teacher output for the three rollouts of one prompt cluster."""

    prompt_token_count: int
    prompt_token_sha256: str
    baseline: TeacherForcedTrace
    candidate: TeacherForcedTrace
    stock_control: TeacherForcedTrace


@dataclass(frozen=True)
class PosthocPromptPlan:
    prompt_index: int
    prompt: str
    baseline: tuple[tuple[int, ...], tuple[tuple[tuple[float, int, None], ...], ...]]
    candidate: tuple[tuple[int, ...], tuple[tuple[tuple[float, int, None], ...], ...]]


@dataclass(frozen=True)
class PosthocBatchPlan:
    phase: str
    batch_index: int
    prompts: tuple[PosthocPromptPlan, ...]


@dataclass(frozen=True)
class PosthocReferencePlan:
    sealed_rollout_sha256: str
    timed_batches: tuple[PosthocBatchPlan, ...]
    warmup_batches: tuple[PosthocBatchPlan, ...]
    sealed_rollout_bytes: bytes = field(repr=False, compare=False)


def _clean_run(run: Any, *, expected_tokens: int, topk_num: int) -> tuple[
    tuple[int, ...], tuple[tuple[tuple[float, int, None], ...], ...]
]:
    if not isinstance(run, (list, tuple)) or len(run) != 2:
        raise ExternalQualityError("sealed rollout is malformed")
    raw_ids, raw_topk = run
    if (
        not isinstance(raw_ids, (list, tuple))
        or len(raw_ids) != expected_tokens
        or not isinstance(raw_topk, (list, tuple))
        or len(raw_topk) != expected_tokens
    ):
        raise ExternalQualityError("sealed rollout lacks its fixed token/top-k coverage")
    ids: list[int] = []
    topk: list[tuple[tuple[float, int, None], ...]] = []
    for token in raw_ids:
        if type(token) is not int or not 0 <= token <= 2_147_483_647:
            raise ExternalQualityError("sealed rollout contains an invalid token ID")
        ids.append(token)
    for position in raw_topk:
        if not isinstance(position, (list, tuple)) or len(position) != topk_num:
            raise ExternalQualityError("sealed rollout top-k width mismatch")
        clean_position: list[tuple[float, int, None]] = []
        seen: set[int] = set()
        previous = math.inf
        mass = 0.0
        for entry in position:
            if not isinstance(entry, (list, tuple)) or len(entry) < 2:
                raise ExternalQualityError("sealed rollout top-k entry is malformed")
            logprob = _finite(
                entry[0], name="sealed top-k logprob", low=-MAX_NLL, high=0.0001
            )
            token_id = _integer(entry[1], name="sealed top-k token")
            if token_id > 2_147_483_647 or token_id in seen or logprob > previous:
                raise ExternalQualityError("sealed rollout top-k ordering is invalid")
            seen.add(token_id)
            previous = logprob
            mass += math.exp(logprob)
            clean_position.append((logprob, token_id, None))
        if mass > 1.0001:
            raise ExternalQualityError("sealed rollout top-k mass exceeds one")
        topk.append(tuple(clean_position))
    return tuple(ids), tuple(topk)


def seal_posthoc_reference_plan(
    prompt_batches: Sequence[Sequence[str]],
    *,
    baseline_batches: Sequence[Sequence[Any]],
    candidate_batches: Sequence[Sequence[Any]],
    warmup_iters: int,
    clusters_per_batch: int,
    expected_tokens: int,
    topk_num: int,
    selection_secret: bytes,
) -> PosthocReferencePlan:
    """Seal every B/C result, then choose bounded post-C prompt clusters.

    ``selection_secret`` is created by the trusted controller only after C has been
    destroyed.  Its value never crosses the worker boundary; only the selected
    ordinary prompts and the digest of the complete B/C evidence do.
    """

    if not isinstance(selection_secret, bytes) or len(selection_secret) < 32:
        raise ExternalQualityError("post-hoc cluster selection needs 256 bits of entropy")
    if not (
        len(prompt_batches) == len(baseline_batches) == len(candidate_batches)
        and 0 <= warmup_iters <= len(prompt_batches)
    ):
        raise ExternalQualityError("B/C prompt batch coverage differs before sealing")
    raw_batches: list[dict[str, Any]] = []
    selected: list[PosthocBatchPlan] = []
    for global_index, (prompts, baseline, candidate) in enumerate(
        zip(prompt_batches, baseline_batches, candidate_batches, strict=True)
    ):
        if not (
            isinstance(prompts, Sequence)
            and len(prompts) == len(baseline) == len(candidate)
            and clusters_per_batch <= len(prompts)
        ):
            raise ExternalQualityError("one B/C batch lacks prompt coverage")
        rows: list[dict[str, Any]] = []
        clean: list[tuple[str, Any, Any]] = []
        for prompt, baseline_run, candidate_run in zip(
            prompts, baseline, candidate, strict=True
        ):
            if not isinstance(prompt, str) or not prompt:
                raise ExternalQualityError("post-hoc plan contains an invalid prompt")
            b = _clean_run(
                baseline_run, expected_tokens=expected_tokens, topk_num=topk_num
            )
            c = _clean_run(
                candidate_run, expected_tokens=expected_tokens, topk_num=topk_num
            )
            rows.append({
                "prompt": prompt,
                "baseline_ids": list(b[0]),
                "baseline_topk": [[[lp, tid] for lp, tid, _ in pos] for pos in b[1]],
                "candidate_ids": list(c[0]),
                "candidate_topk": [[[lp, tid] for lp, tid, _ in pos] for pos in c[1]],
            })
            clean.append((prompt, b, c))
        raw_batches.append({"batch_index": global_index, "rows": rows})
        ranked = sorted(
            range(len(clean)),
            key=lambda prompt_index: hashlib.sha256(
                selection_secret
                + global_index.to_bytes(4, "big")
                + prompt_index.to_bytes(4, "big")
                + clean[prompt_index][0].encode("utf-8")
            ).digest(),
        )[:clusters_per_batch]
        phase = "warmup" if global_index < warmup_iters else "timed"
        phase_index = global_index if phase == "warmup" else global_index - warmup_iters
        selected.append(PosthocBatchPlan(
            phase=phase,
            batch_index=phase_index,
            prompts=tuple(
                PosthocPromptPlan(
                    prompt_index=index,
                    prompt=clean[index][0],
                    baseline=clean[index][1],
                    candidate=clean[index][2],
                )
                for index in ranked
            ),
        ))
    sealed_payload = {
        "protocol": TEACHER_FORCED_QUALITY_PROTOCOL_V2,
        "batches": raw_batches,
    }
    sealed_bytes = canonical_evidence_bytes(sealed_payload)
    seal = "sha256:" + hashlib.sha256(sealed_bytes).hexdigest()
    return PosthocReferencePlan(
        sealed_rollout_sha256=seal,
        timed_batches=tuple(batch for batch in selected if batch.phase == "timed"),
        warmup_batches=tuple(batch for batch in selected if batch.phase == "warmup"),
        sealed_rollout_bytes=sealed_bytes,
    )


@dataclass(frozen=True)
class RolloutQualitySummary:
    """Prompt-level summary of one rollout under stock B' teacher forcing."""

    token_count: int
    target_nll_sum: float
    target_nll_max: float
    target_tail_count: int
    topk_positions: int
    topk_mean_kl: float
    topk_max_kl: float
    topk_p99_kl: float
    topk_argmax_disagreements: int
    topk_mean_coverage_dev: float
    hidden_score: float
    hidden_total: int

    @property
    def mean_nll(self) -> float:
        return self.target_nll_sum / self.token_count if self.token_count else MAX_NLL

    @property
    def tail_rate(self) -> float:
        return self.target_tail_count / self.token_count if self.token_count else 1.0

    @property
    def argmax_rate(self) -> float:
        return (
            self.topk_argmax_disagreements / self.topk_positions
            if self.topk_positions else 1.0
        )

    @property
    def hidden_rate(self) -> float | None:
        return self.hidden_score / self.hidden_total if self.hidden_total else None

    def validated(self, *, expected_tokens: int) -> "RolloutQualitySummary":
        if self.token_count != expected_tokens or self.topk_positions != expected_tokens:
            raise ExternalQualityError("teacher summary does not cover the fixed rollout")
        _finite(
            self.target_nll_sum,
            name="target_nll_sum",
            high=MAX_NLL * expected_tokens,
        )
        _finite(self.target_nll_max, name="target_nll_max")
        _integer(self.target_tail_count, name="target_tail_count")
        _integer(
            self.topk_argmax_disagreements,
            name="topk_argmax_disagreements",
        )
        if self.target_tail_count > expected_tokens:
            raise ExternalQualityError("target tail count exceeds rollout length")
        if self.topk_argmax_disagreements > expected_tokens:
            raise ExternalQualityError("top-k disagreement count exceeds rollout length")
        for name in ("topk_mean_kl", "topk_max_kl", "topk_p99_kl"):
            _finite(getattr(self, name), name=name)
        if self.topk_mean_kl > self.topk_max_kl or self.topk_p99_kl > self.topk_max_kl:
            raise ExternalQualityError("top-k KL aggregates are inconsistent")
        _finite(
            self.topk_mean_coverage_dev,
            name="topk_mean_coverage_dev",
            high=1.0,
        )
        _integer(self.hidden_total, name="hidden_total")
        _finite(
            self.hidden_score,
            name="hidden_score",
            high=float(max(1, self.hidden_total)),
        )
        if self.hidden_score > self.hidden_total:
            raise ExternalQualityError("hidden score exceeds its denominator")
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "token_count": self.token_count,
            "target_nll_sum": self.target_nll_sum,
            "target_nll_max": self.target_nll_max,
            "target_tail_count": self.target_tail_count,
            "topk_positions": self.topk_positions,
            "topk_mean_kl": self.topk_mean_kl,
            "topk_max_kl": self.topk_max_kl,
            "topk_p99_kl": self.topk_p99_kl,
            "topk_argmax_disagreements": self.topk_argmax_disagreements,
            "topk_mean_coverage_dev": self.topk_mean_coverage_dev,
            "hidden_score": self.hidden_score,
            "hidden_total": self.hidden_total,
        }

    @classmethod
    def from_dict(cls, raw: Any, *, path: str) -> "RolloutQualitySummary":
        fields = {
            "token_count", "target_nll_sum", "target_nll_max", "target_tail_count",
            "topk_positions", "topk_mean_kl", "topk_max_kl", "topk_p99_kl",
            "topk_argmax_disagreements", "topk_mean_coverage_dev",
            "hidden_score", "hidden_total",
        }
        if not isinstance(raw, Mapping) or set(raw) != fields:
            raise ExternalQualityError(f"{path} fields do not match the schema")
        return cls(
            token_count=_integer(raw["token_count"], name=f"{path}.token_count"),
            target_nll_sum=_finite(
                raw["target_nll_sum"], name=f"{path}.target_nll_sum", high=MAX_NLL * 1_000_000
            ),
            target_nll_max=_finite(raw["target_nll_max"], name=f"{path}.target_nll_max"),
            target_tail_count=_integer(
                raw["target_tail_count"], name=f"{path}.target_tail_count"
            ),
            topk_positions=_integer(raw["topk_positions"], name=f"{path}.topk_positions"),
            topk_mean_kl=_finite(raw["topk_mean_kl"], name=f"{path}.topk_mean_kl"),
            topk_max_kl=_finite(raw["topk_max_kl"], name=f"{path}.topk_max_kl"),
            topk_p99_kl=_finite(raw["topk_p99_kl"], name=f"{path}.topk_p99_kl"),
            topk_argmax_disagreements=_integer(
                raw["topk_argmax_disagreements"],
                name=f"{path}.topk_argmax_disagreements",
            ),
            topk_mean_coverage_dev=_finite(
                raw["topk_mean_coverage_dev"],
                name=f"{path}.topk_mean_coverage_dev",
                high=1.0,
            ),
            hidden_score=_finite(
                raw["hidden_score"], name=f"{path}.hidden_score", high=1_000_000
            ),
            hidden_total=_integer(raw["hidden_total"], name=f"{path}.hidden_total"),
        )


@dataclass(frozen=True)
class PromptClusterEvidence:
    baseline: RolloutQualitySummary
    candidate: RolloutQualitySummary
    stock_control: RolloutQualitySummary
    exact_token_matches: int
    exact_token_total: int

    def validated(self, *, expected_tokens: int) -> "PromptClusterEvidence":
        self.baseline.validated(expected_tokens=expected_tokens)
        self.candidate.validated(expected_tokens=expected_tokens)
        self.stock_control.validated(expected_tokens=expected_tokens)
        if self.exact_token_total != expected_tokens:
            raise ExternalQualityError("exact-token diagnostic length mismatch")
        if not 0 <= self.exact_token_matches <= self.exact_token_total:
            raise ExternalQualityError("exact-token diagnostic count is invalid")
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline": self.baseline.to_dict(),
            "candidate": self.candidate.to_dict(),
            "stock_control": self.stock_control.to_dict(),
            "exact_token_matches": self.exact_token_matches,
            "exact_token_total": self.exact_token_total,
        }

    @classmethod
    def from_dict(cls, raw: Any, *, path: str) -> "PromptClusterEvidence":
        fields = {
            "baseline", "candidate", "stock_control",
            "exact_token_matches", "exact_token_total",
        }
        if not isinstance(raw, Mapping) or set(raw) != fields:
            raise ExternalQualityError(f"{path} fields do not match the schema")
        return cls(
            baseline=RolloutQualitySummary.from_dict(raw["baseline"], path=f"{path}.baseline"),
            candidate=RolloutQualitySummary.from_dict(
                raw["candidate"], path=f"{path}.candidate"
            ),
            stock_control=RolloutQualitySummary.from_dict(
                raw["stock_control"], path=f"{path}.stock_control"
            ),
            exact_token_matches=_integer(
                raw["exact_token_matches"], name=f"{path}.exact_token_matches"
            ),
            exact_token_total=_integer(
                raw["exact_token_total"], name=f"{path}.exact_token_total", minimum=1
            ),
        )


@dataclass(frozen=True)
class TeacherForcedBatchEvidence:
    clusters: tuple[PromptClusterEvidence, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"clusters": [cluster.to_dict() for cluster in self.clusters]}

    @classmethod
    def from_dict(cls, raw: Any, *, path: str) -> "TeacherForcedBatchEvidence":
        if not isinstance(raw, Mapping) or set(raw) != {"clusters"}:
            raise ExternalQualityError(f"{path} fields do not match the schema")
        values = raw["clusters"]
        if not isinstance(values, list):
            raise ExternalQualityError(f"{path}.clusters must be an array")
        return cls(tuple(
            PromptClusterEvidence.from_dict(value, path=f"{path}.clusters[{index}]")
            for index, value in enumerate(values)
        ))


@dataclass(frozen=True)
class TeacherForcedExternalQualityEvidence:
    protocol: str
    sealed_rollout_sha256: str
    raw_evidence_sha256: str
    raw_evidence_size: int
    raw_artifact_published: bool
    hidden_tasks_present: bool
    timed_batches: tuple[TeacherForcedBatchEvidence, ...]
    warmup_batches: tuple[TeacherForcedBatchEvidence, ...]
    raw_evidence_bytes: bytes | None = field(
        default=None, repr=False, compare=False
    )

    def validated(
        self, arena: "ArenaProfile", *, require_published: bool = True
    ) -> "TeacherForcedExternalQualityEvidence":
        policy = arena.fidelity.teacher_forced_policy
        if self.protocol != TEACHER_FORCED_QUALITY_PROTOCOL_V2:
            raise ExternalQualityError("external quality protocol is unsupported")
        if self.protocol != arena.fidelity.external_quality_gate:
            raise ExternalQualityError("external quality protocol disagrees with arena")
        _sha256_id(self.sealed_rollout_sha256, name="sealed_rollout_sha256")
        _sha256_id(self.raw_evidence_sha256, name="raw_evidence_sha256")
        if (
            type(self.raw_evidence_size) is not int
            or not 0 < self.raw_evidence_size <= MAX_RAW_QUALITY_BYTES
        ):
            raise ExternalQualityError("raw_evidence_size is out of bounds")
        if type(self.raw_artifact_published) is not bool:
            raise ExternalQualityError("raw_artifact_published must be boolean")
        if require_published and not self.raw_artifact_published:
            raise ExternalQualityError(
                "raw teacher evidence was not content-addressed before qualification"
            )
        if self.raw_evidence_bytes is not None:
            if (
                not isinstance(self.raw_evidence_bytes, bytes)
                or len(self.raw_evidence_bytes) != self.raw_evidence_size
                or "sha256:" + hashlib.sha256(self.raw_evidence_bytes).hexdigest()
                != self.raw_evidence_sha256
            ):
                raise ExternalQualityError("retained raw teacher evidence digest/size mismatch")
        if type(self.hidden_tasks_present) is not bool:
            raise ExternalQualityError("hidden_tasks_present must be boolean")
        expected = {
            "timed_batches": arena.scoring.timed_iters,
            "warmup_batches": arena.scoring.warmup_iters,
        }
        for phase, count in expected.items():
            batches = getattr(self, phase)
            if not isinstance(batches, tuple) or len(batches) != count:
                raise ExternalQualityError(
                    f"{phase} must contain exactly arena {phase.split('_')[0]} batches"
                )
            for batch_index, batch in enumerate(batches):
                if type(batch) is not TeacherForcedBatchEvidence:
                    raise ExternalQualityError(f"{phase}[{batch_index}] is not typed evidence")
                if len(batch.clusters) != policy.clusters_per_batch:
                    raise ExternalQualityError(
                        f"{phase}[{batch_index}] must contain exactly "
                        f"{policy.clusters_per_batch} prompt clusters"
                    )
                for cluster in batch.clusters:
                    cluster.validated(expected_tokens=arena.workload.max_new_tokens)
                    hidden_totals = (
                        cluster.baseline.hidden_total,
                        cluster.candidate.hidden_total,
                        cluster.stock_control.hidden_total,
                    )
                    if self.hidden_tasks_present:
                        if any(total <= 0 for total in hidden_totals):
                            raise ExternalQualityError(
                                "declared hidden-task evidence is incomplete"
                            )
                    elif any(total != 0 for total in hidden_totals):
                        raise ExternalQualityError(
                            "hidden-task scores exist without a declared hidden judge"
                        )
        if (
            policy.calibration_state != "uncalibrated"
            and policy.require_hidden_tasks
            and not self.hidden_tasks_present
        ):
            raise ExternalQualityError(
                "a calibrated policy requiring hidden tasks has no hidden judge evidence"
            )
        return self

    @property
    def candidate_mean_nll(self) -> float:
        values = [
            cluster.candidate.mean_nll
            for batch in self.timed_batches
            for cluster in batch.clusters
        ]
        return statistics.fmean(values) if values else MAX_NLL

    @property
    def diagnostic_exact_match(self) -> tuple[int, int]:
        clusters = [
            cluster
            for batches in (self.timed_batches, self.warmup_batches)
            for batch in batches
            for cluster in batch.clusters
        ]
        return (
            sum(cluster.exact_token_matches for cluster in clusters),
            sum(cluster.exact_token_total for cluster in clusters),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol": self.protocol,
            "sealed_rollout_sha256": self.sealed_rollout_sha256,
            "raw_evidence_sha256": self.raw_evidence_sha256,
            "raw_evidence_size": self.raw_evidence_size,
            "raw_artifact_published": self.raw_artifact_published,
            "hidden_tasks_present": self.hidden_tasks_present,
            "timed_batches": [batch.to_dict() for batch in self.timed_batches],
            "warmup_batches": [batch.to_dict() for batch in self.warmup_batches],
        }

    @classmethod
    def from_dict(cls, raw: Any) -> "TeacherForcedExternalQualityEvidence":
        fields = {
            "protocol", "sealed_rollout_sha256", "raw_evidence_sha256",
            "raw_evidence_size", "raw_artifact_published", "hidden_tasks_present",
            "timed_batches", "warmup_batches",
        }
        if not isinstance(raw, Mapping) or set(raw) != fields:
            raise ExternalQualityError("external_quality_evidence fields do not match schema")

        def batches(name: str) -> tuple[TeacherForcedBatchEvidence, ...]:
            values = raw[name]
            if not isinstance(values, list):
                raise ExternalQualityError(f"external_quality_evidence.{name} must be an array")
            return tuple(
                TeacherForcedBatchEvidence.from_dict(
                    value, path=f"external_quality_evidence.{name}[{index}]"
                )
                for index, value in enumerate(values)
            )

        return cls(
            protocol=str(raw["protocol"]),
            sealed_rollout_sha256=_sha256_id(
                raw["sealed_rollout_sha256"], name="sealed_rollout_sha256"
            ),
            raw_evidence_sha256=_sha256_id(
                raw["raw_evidence_sha256"], name="raw_evidence_sha256"
            ),
            raw_evidence_size=_integer(
                raw["raw_evidence_size"], name="raw_evidence_size", minimum=1
            ),
            raw_artifact_published=raw["raw_artifact_published"],
            hidden_tasks_present=raw["hidden_tasks_present"],
            timed_batches=batches("timed_batches"),
            warmup_batches=batches("warmup_batches"),
        )


@dataclass(frozen=True)
class TeacherForcedQualityVerdict:
    decision: str
    timed_decision: str
    warmup_decision: str
    candidate_mean_nll: float
    detail: str

    @property
    def passed(self) -> bool:
        return self.decision == QUALITY_PASS


def _bounds(values: Sequence[float], z: float) -> tuple[float, float, float]:
    if not values:
        raise ExternalQualityError("quality statistic has no prompt clusters")
    mean = statistics.fmean(values)
    if len(values) == 1:
        return mean, -math.inf, math.inf
    standard_error = statistics.stdev(values) / math.sqrt(len(values))
    return mean, mean - z * standard_error, mean + z * standard_error


def _metric_values(cluster: PromptClusterEvidence, name: str) -> tuple[float, float, float]:
    def value(summary: RolloutQualitySummary) -> float:
        if name == "mean_nll":
            return summary.mean_nll
        if name == "worst_nll":
            return summary.target_nll_max
        if name == "tail_rate":
            return summary.tail_rate
        if name == "topk_kl":
            return summary.topk_mean_kl
        if name == "argmax_rate":
            return summary.argmax_rate
        if name == "coverage_dev":
            return summary.topk_mean_coverage_dev
        if name == "hidden_regression":
            rates = tuple(
                item.hidden_rate
                for item in (cluster.baseline, cluster.candidate, cluster.stock_control)
            )
            if any(rate is None for rate in rates):
                raise ExternalQualityError("hidden-task evidence is required but absent")
            # Higher is better; orient as positive regression.
            return float(rates[0])
        raise ExternalQualityError(f"unknown quality metric {name!r}")

    if name == "hidden_regression":
        b = cluster.baseline.hidden_rate
        c = cluster.candidate.hidden_rate
        bp = cluster.stock_control.hidden_rate
        if b is None or c is None or bp is None:
            raise ExternalQualityError("hidden-task evidence is required but absent")
        return float(b), float(c), float(bp)
    return value(cluster.baseline), value(cluster.candidate), value(cluster.stock_control)


def _score_batch(
    batch: TeacherForcedBatchEvidence,
    policy: "TeacherForcedQualityPolicy",
) -> tuple[str, str]:
    metric_limits = {
        "mean_nll": (policy.stock_mean_nll_envelope, policy.mean_nll_delta),
        "worst_nll": (policy.stock_worst_nll_envelope, policy.worst_nll_delta),
        "tail_rate": (policy.stock_tail_rate_envelope, policy.tail_rate_delta),
        "topk_kl": (policy.stock_topk_kl_envelope, policy.topk_kl_delta),
        "argmax_rate": (policy.stock_argmax_rate_envelope, policy.argmax_rate_delta),
        "coverage_dev": (policy.stock_coverage_envelope, policy.coverage_delta),
    }
    if policy.require_hidden_tasks:
        metric_limits["hidden_regression"] = (
            policy.stock_hidden_score_envelope,
            policy.hidden_score_delta,
        )

    diagnostics: list[str] = []
    candidate_overlap = False
    candidate_failure = False
    for name, (stock_limit, candidate_limit) in metric_limits.items():
        triplets = [_metric_values(cluster, name) for cluster in batch.clusters]
        if name == "hidden_regression":
            stock_diffs = [abs(b - bp) for b, _c, bp in triplets]
            regressions = [b - c for b, c, _bp in triplets]
        else:
            stock_diffs = [abs(b - bp) for b, _c, bp in triplets]
            regressions = [c - b for b, c, _bp in triplets]
        stock_mean, stock_lcb, stock_ucb = _bounds(stock_diffs, policy.familywise_z)
        regression_mean, regression_lcb, regression_ucb = _bounds(
            regressions, policy.familywise_z
        )
        diagnostics.append(
            f"{name}:stock={stock_mean:.6g}[{stock_lcb:.6g},{stock_ucb:.6g}]"
            f"/{stock_limit:.6g},cand={regression_mean:.6g}"
            f"[{regression_lcb:.6g},{regression_ucb:.6g}]/{candidate_limit:.6g}"
        )
        # A bracket whose stock control is not clearly inside its frozen null
        # envelope is infrastructure drift, never evidence against the candidate.
        if stock_ucb > stock_limit:
            return QUALITY_NO_DECISION, "; ".join(diagnostics)
        if regression_lcb > candidate_limit:
            candidate_failure = True
        elif regression_ucb > candidate_limit:
            candidate_overlap = True

    if policy.require_hidden_tasks:
        candidate_hidden = [
            cluster.candidate.hidden_rate for cluster in batch.clusters
        ]
        assert all(value is not None for value in candidate_hidden)
        hidden_mean, hidden_lcb, hidden_ucb = _bounds(
            [float(value) for value in candidate_hidden], policy.familywise_z
        )
        diagnostics.append(
            f"hidden_abs={hidden_mean:.6g}[{hidden_lcb:.6g},{hidden_ucb:.6g}]"
            f"/{policy.hidden_score_floor:.6g}"
        )
        if hidden_ucb < policy.hidden_score_floor:
            candidate_failure = True
        elif hidden_lcb < policy.hidden_score_floor:
            candidate_overlap = True

    if candidate_failure:
        return QUALITY_FAIL, "; ".join(diagnostics)
    if candidate_overlap:
        return QUALITY_NO_DECISION, "; ".join(diagnostics)
    return QUALITY_PASS, "; ".join(diagnostics)


def score_teacher_forced_quality(
    evidence: TeacherForcedExternalQualityEvidence,
    *,
    arena: "ArenaProfile",
) -> TeacherForcedQualityVerdict:
    evidence.validated(arena)
    policy = arena.fidelity.teacher_forced_policy
    if policy.calibration_state == "uncalibrated":
        exact, total = evidence.diagnostic_exact_match
        return TeacherForcedQualityVerdict(
            decision=QUALITY_NO_DECISION,
            timed_decision=QUALITY_NO_DECISION,
            warmup_decision=QUALITY_NO_DECISION,
            candidate_mean_nll=evidence.candidate_mean_nll,
            detail=(
                "teacher-forced policy is explicitly uncalibrated; crown refused; "
                f"hidden_tasks_present={int(evidence.hidden_tasks_present)}; "
                f"exact-token diagnostic={exact}/{total}"
            ),
        )

    def phase(name: str, batches: tuple[TeacherForcedBatchEvidence, ...]) -> tuple[str, str]:
        # Batch boundaries are a coverage/schema invariant, not independent null
        # tests. Repeating an all-or-nothing decision per batch makes a faithful
        # candidate fail with probability that grows with timed_iters. Grade one
        # registered familywise test over all prompt clusters in this phase.
        flattened = TeacherForcedBatchEvidence(tuple(
            cluster for batch in batches for cluster in batch.clusters
        ))
        decision, detail = _score_batch(flattened, policy)
        return decision, f"{name}[phase:{decision}:{detail}]"

    timed, timed_detail = phase("timed", evidence.timed_batches)
    warmup, warmup_detail = phase("warmup", evidence.warmup_batches)
    if QUALITY_FAIL in (timed, warmup):
        overall = QUALITY_FAIL
    elif QUALITY_NO_DECISION in (timed, warmup):
        overall = QUALITY_NO_DECISION
    else:
        overall = QUALITY_PASS
    exact, total = evidence.diagnostic_exact_match
    return TeacherForcedQualityVerdict(
        decision=overall,
        timed_decision=timed,
        warmup_decision=warmup,
        candidate_mean_nll=evidence.candidate_mean_nll,
        detail=(
            f"{timed_detail}; {warmup_detail}; exact-token diagnostic="
            f"{exact}/{total} (non-authoritative)"
        ),
    )


def summarize_rollout(
    reported_topk: Sequence[Sequence[tuple]],
    trace: TeacherForcedTrace,
    *,
    tail_nll_threshold: float,
    nll_clip: float,
    topk_num: int,
    hidden_score: float = 0.0,
    hidden_total: int = 0,
) -> RolloutQualitySummary:
    """Build one bounded prompt summary from the sealed and stock-teacher frames."""

    expected = len(reported_topk)
    trace.validated(expected_tokens=expected, topk_num=topk_num)
    if expected <= 0:
        raise ExternalQualityError("teacher summary requires a non-empty rollout")
    nlls = [min(float(nll_clip), max(0.0, -float(value))) for value in trace.target_logprobs]
    kl = kl_over_positions(trace.trusted_topk, reported_topk)
    return RolloutQualitySummary(
        token_count=expected,
        target_nll_sum=sum(nlls),
        target_nll_max=max(nlls),
        target_tail_count=sum(value > tail_nll_threshold for value in nlls),
        topk_positions=kl.num_positions,
        topk_mean_kl=kl.mean_kl,
        topk_max_kl=kl.max_kl,
        topk_p99_kl=kl.p99_kl,
        topk_argmax_disagreements=kl.argmax_disagreements,
        topk_mean_coverage_dev=kl.mean_coverage_dev,
        hidden_score=float(hidden_score),
        hidden_total=int(hidden_total),
    )


def canonical_evidence_bytes(value: Any) -> bytes:
    """Encode a controller-validated raw evidence structure without float NaNs."""
    try:
        raw = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ExternalQualityError(f"raw quality evidence is not canonical JSON: {exc}") from None
    if not raw or len(raw) > MAX_RAW_QUALITY_BYTES:
        raise ExternalQualityError("raw quality evidence exceeds its hard byte bound")
    return raw


def canonical_evidence_digest(value: Any) -> str:
    """Digest a controller-validated raw evidence structure without float NaNs."""

    raw = canonical_evidence_bytes(value)
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _strict_raw_json(raw: bytes) -> Mapping[str, Any]:
    def pairs(values):
        result = {}
        for key, value in values:
            if key in result:
                raise ExternalQualityError(f"duplicate raw quality key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw,
            object_pairs_hook=pairs,
            parse_constant=lambda item: (_ for _ in ()).throw(
                ExternalQualityError(f"non-finite raw quality constant {item}")
            ),
        )
    except ExternalQualityError:
        raise
    except (json.JSONDecodeError, UnicodeDecodeError, RecursionError) as exc:
        raise ExternalQualityError(f"raw quality artifact is malformed JSON: {exc}") from None
    if not isinstance(value, Mapping) or canonical_evidence_bytes(value) != raw:
        raise ExternalQualityError("raw quality artifact is not canonical exact-schema JSON")
    return value


def regrade_raw_quality_artifact(
    raw: bytes,
    evidence: TeacherForcedExternalQualityEvidence,
    *,
    arena: "ArenaProfile",
) -> None:
    """Rebuild every v12 summary/verdict from the retained raw controller frames."""

    root = _strict_raw_json(raw)
    expected_root = {
        "protocol", "sealed_rollout_sha256", "sealed_rollouts",
        "selected_batches", "teacher_traces",
    }
    if set(root) != expected_root:
        raise ExternalQualityError("raw quality artifact root fields do not match v2")
    if (
        root["protocol"] != TEACHER_FORCED_QUALITY_PROTOCOL_V2
        or root["sealed_rollout_sha256"] != evidence.sealed_rollout_sha256
    ):
        raise ExternalQualityError("raw quality protocol/seal binding mismatch")
    sealed = root["sealed_rollouts"]
    if not isinstance(sealed, Mapping) or set(sealed) != {"protocol", "batches"}:
        raise ExternalQualityError("raw sealed B/C evidence fields do not match v2")
    if sealed["protocol"] != TEACHER_FORCED_QUALITY_PROTOCOL_V2:
        raise ExternalQualityError("raw sealed B/C protocol mismatch")
    sealed_bytes = canonical_evidence_bytes(sealed)
    if "sha256:" + hashlib.sha256(sealed_bytes).hexdigest() != evidence.sealed_rollout_sha256:
        raise ExternalQualityError("raw full B/C evidence does not recompute its seal")
    batches = sealed["batches"]
    expected_batches = arena.scoring.warmup_iters + arena.scoring.timed_iters
    if not isinstance(batches, list) or len(batches) != expected_batches:
        raise ExternalQualityError("raw full B/C batch coverage mismatch")
    clean_batches: list[list[tuple[str, Any, Any]]] = []
    for batch_index, batch in enumerate(batches):
        if (
            not isinstance(batch, Mapping)
            or set(batch) != {"batch_index", "rows"}
            or batch["batch_index"] != batch_index
            or not isinstance(batch["rows"], list)
            or len(batch["rows"]) != arena.workload.num_prompts
        ):
            raise ExternalQualityError("raw full B/C batch ordering/schema mismatch")
        rows = []
        for row in batch["rows"]:
            if not isinstance(row, Mapping) or set(row) != {
                "prompt", "baseline_ids", "baseline_topk",
                "candidate_ids", "candidate_topk",
            }:
                raise ExternalQualityError("raw full B/C prompt schema mismatch")
            if not isinstance(row["prompt"], str) or not row["prompt"]:
                raise ExternalQualityError("raw full B/C prompt is invalid")
            baseline = _clean_run(
                (row["baseline_ids"], row["baseline_topk"]),
                expected_tokens=arena.workload.max_new_tokens,
                topk_num=arena.workload.top_logprobs,
            )
            candidate = _clean_run(
                (row["candidate_ids"], row["candidate_topk"]),
                expected_tokens=arena.workload.max_new_tokens,
                topk_num=arena.workload.top_logprobs,
            )
            rows.append((row["prompt"], baseline, candidate))
        clean_batches.append(rows)

    selected = root["selected_batches"]
    traces_raw = root["teacher_traces"]
    if (
        not isinstance(selected, list)
        or not isinstance(traces_raw, list)
        or len(selected) != expected_batches
        or len(traces_raw) != expected_batches
    ):
        raise ExternalQualityError("raw selected/teacher batch coverage mismatch")
    plan_batches: list[PosthocBatchPlan] = []
    traces: dict[tuple[str, int], tuple[TeacherForcedPromptTrace, ...]] = {}
    stock_batches: list[list[Any]] = [
        [rows[0][1] for _ in range(arena.workload.num_prompts)]
        for rows in clean_batches
    ]
    hidden_values: dict[tuple[str, int, int, str], tuple[float, int]] = {}

    def raw_trace(value: Any, *, path: str) -> TeacherForcedTrace:
        if not isinstance(value, Mapping) or set(value) != {
            "target_logprobs", "trusted_topk",
        }:
            raise ExternalQualityError(f"{path} trace fields do not match v2")
        if not isinstance(value["target_logprobs"], list) or not isinstance(
            value["trusted_topk"], list
        ):
            raise ExternalQualityError(f"{path} trace arrays are invalid")
        trace = TeacherForcedTrace(
            tuple(value["target_logprobs"]),
            tuple(
                tuple((entry[0], entry[1], None) for entry in position)
                for position in value["trusted_topk"]
            ),
        )
        return trace.validated(
            expected_tokens=arena.workload.max_new_tokens,
            topk_num=arena.workload.top_logprobs,
        )

    for order, (selection, teacher) in enumerate(zip(selected, traces_raw, strict=True)):
        expected_phase = (
            "warmup" if order < arena.scoring.warmup_iters else "timed"
        )
        expected_index = (
            order if expected_phase == "warmup" else order - arena.scoring.warmup_iters
        )
        if (
            not isinstance(selection, Mapping)
            or set(selection) != {"phase", "batch_index", "prompt_indices"}
            or selection["phase"] != expected_phase
            or selection["batch_index"] != expected_index
            or not isinstance(selection["prompt_indices"], list)
            or len(selection["prompt_indices"])
            != arena.fidelity.teacher_forced_policy.clusters_per_batch
            or len(set(selection["prompt_indices"]))
            != len(selection["prompt_indices"])
            or any(
                type(index) is not int or not 0 <= index < arena.workload.num_prompts
                for index in selection["prompt_indices"]
            )
        ):
            raise ExternalQualityError("raw selected prompt mapping is invalid")
        if (
            not isinstance(teacher, Mapping)
            or set(teacher) != {"phase", "batch_index", "prompts"}
            or teacher["phase"] != expected_phase
            or teacher["batch_index"] != expected_index
            or not isinstance(teacher["prompts"], list)
            or len(teacher["prompts"]) != len(selection["prompt_indices"])
        ):
            raise ExternalQualityError("raw teacher batch mapping is invalid")
        global_index = order
        prompt_plans: list[PosthocPromptPlan] = []
        prompt_traces: list[TeacherForcedPromptTrace] = []
        for expected_prompt_index, prompt in zip(
            selection["prompt_indices"], teacher["prompts"], strict=True
        ):
            expected_prompt_fields = {
                "prompt_index", "prompt_token_count", "prompt_token_sha256",
                "stock_control_ids", "stock_control_reported_topk", "hidden",
                "baseline", "candidate", "stock_control",
            }
            if (
                not isinstance(prompt, Mapping)
                or set(prompt) != expected_prompt_fields
                or prompt["prompt_index"] != expected_prompt_index
            ):
                raise ExternalQualityError("raw teacher prompt mapping/schema mismatch")
            source_row = clean_batches[global_index][expected_prompt_index]
            prompt_plans.append(PosthocPromptPlan(
                prompt_index=expected_prompt_index,
                prompt=source_row[0],
                baseline=source_row[1],
                candidate=source_row[2],
            ))
            control = _clean_run(
                (prompt["stock_control_ids"], prompt["stock_control_reported_topk"]),
                expected_tokens=arena.workload.max_new_tokens,
                topk_num=arena.workload.top_logprobs,
            )
            stock_batches[global_index][expected_prompt_index] = control
            hidden = prompt["hidden"]
            if not isinstance(hidden, Mapping) or set(hidden) != {
                "baseline", "candidate", "stock_control",
            }:
                raise ExternalQualityError("raw hidden-task fields do not match v2")
            for source, value in hidden.items():
                if (
                    not isinstance(value, list)
                    or len(value) != 2
                    or type(value[1]) is not int
                ):
                    raise ExternalQualityError("raw hidden-task score is invalid")
                hidden_values[(
                    expected_phase, expected_index, expected_prompt_index, source
                )] = (float(value[0]), value[1])
            prompt_traces.append(TeacherForcedPromptTrace(
                prompt_token_count=prompt["prompt_token_count"],
                prompt_token_sha256=prompt["prompt_token_sha256"],
                baseline=raw_trace(prompt["baseline"], path="baseline"),
                candidate=raw_trace(prompt["candidate"], path="candidate"),
                stock_control=raw_trace(prompt["stock_control"], path="stock_control"),
            ))
        plan_batches.append(PosthocBatchPlan(
            expected_phase, expected_index, tuple(prompt_plans)
        ))
        traces[(expected_phase, expected_index)] = tuple(prompt_traces)
    plan = PosthocReferencePlan(
        sealed_rollout_sha256=evidence.sealed_rollout_sha256,
        timed_batches=tuple(batch for batch in plan_batches if batch.phase == "timed"),
        warmup_batches=tuple(batch for batch in plan_batches if batch.phase == "warmup"),
        sealed_rollout_bytes=sealed_bytes,
    )

    def hidden_judge(phase, batch_index, prompt_index, source, _ids):
        return hidden_values[(phase, batch_index, prompt_index, source)]

    rebuilt = build_teacher_forced_evidence(
        plan,
        stock_control_batches=stock_batches,
        warmup_iters=arena.scoring.warmup_iters,
        traces=traces,
        arena=arena,
        hidden_judge=hidden_judge if evidence.hidden_tasks_present else None,
    )
    if rebuilt.raw_evidence_bytes != raw:
        raise ExternalQualityError("raw quality artifact does not canonically rebuild")
    rebuilt = replace(
        rebuilt,
        raw_artifact_published=evidence.raw_artifact_published,
        raw_evidence_bytes=evidence.raw_evidence_bytes,
    )
    if rebuilt.to_dict() != evidence.to_dict():
        raise ExternalQualityError(
            "raw quality artifact regrade differs from v12 prompt summaries"
        )


def raw_quality_artifact_path(
    root: str | os.PathLike[str], digest: str
) -> Path:
    identity = _sha256_id(digest, name="raw_evidence_sha256")[7:]
    return Path(root) / "external-quality-v2" / f"sha256-{identity}" / "evidence.json"


def _validated_raw_root(root: str | os.PathLike[str]) -> Path:
    path = Path(root)
    if not path.is_absolute() or ".." in path.parts or any(
        char in str(path) for char in "\x00\r\n"
    ):
        raise ExternalQualityError("raw quality root must be absolute and normalized")
    try:
        info = path.lstat()
    except OSError as exc:
        raise ExternalQualityError(f"raw quality root is unavailable: {exc}") from None
    if (
        stat.S_ISLNK(info.st_mode)
        or not stat.S_ISDIR(info.st_mode)
        or info.st_uid != os.geteuid()
        or stat.S_IMODE(info.st_mode) & 0o022
    ):
        raise ExternalQualityError(
            "raw quality root must be controller-owned and not group/world writable"
        )
    return path


def reopen_raw_quality_artifact(
    root: str | os.PathLike[str],
    evidence: TeacherForcedExternalQualityEvidence,
    *,
    arena: "ArenaProfile",
) -> bytes:
    """Reopen one bounded content-addressed raw artifact without following links."""

    root_path = _validated_raw_root(root)
    path = raw_quality_artifact_path(root_path, evidence.raw_evidence_sha256)
    try:
        parent_info = path.parent.lstat()
    except OSError as exc:
        raise ExternalQualityError(f"raw teacher artifact parent is unavailable: {exc}") from None
    if (
        stat.S_ISLNK(parent_info.st_mode)
        or not stat.S_ISDIR(parent_info.st_mode)
        or parent_info.st_uid != os.geteuid()
        or stat.S_IMODE(parent_info.st_mode) != 0o700
    ):
        raise ExternalQualityError("raw teacher artifact parent is not private/controller-owned")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise ExternalQualityError(f"cannot reopen raw teacher artifact: {exc}") from None
    try:
        before = os.fstat(fd)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_uid != os.geteuid()
            or stat.S_IMODE(before.st_mode) != 0o600
            or before.st_size != evidence.raw_evidence_size
            or before.st_size > MAX_RAW_QUALITY_BYTES
        ):
            raise ExternalQualityError("raw teacher artifact is not one bounded regular file")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(fd, min(1 << 20, remaining))
            if not chunk:
                raise ExternalQualityError("raw teacher artifact was truncated")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(fd, 1):
            raise ExternalQualityError("raw teacher artifact exceeds its receipt")
        after = os.fstat(fd)
        stable = (
            "st_dev", "st_ino", "st_mode", "st_nlink", "st_size",
            "st_mtime_ns", "st_ctime_ns",
        )
        if any(getattr(before, name) != getattr(after, name) for name in stable):
            raise ExternalQualityError("raw teacher artifact changed while reopening")
    finally:
        os.close(fd)
    raw = b"".join(chunks)
    if "sha256:" + hashlib.sha256(raw).hexdigest() != evidence.raw_evidence_sha256:
        raise ExternalQualityError("raw teacher artifact digest differs from qualification")
    regrade_raw_quality_artifact(raw, evidence, arena=arena)
    return raw


def publish_raw_quality_artifact(
    root: str | os.PathLike[str],
    evidence: TeacherForcedExternalQualityEvidence,
    *,
    arena: "ArenaProfile",
) -> TeacherForcedExternalQualityEvidence:
    """Durably publish raw frames under their digest before qualification exists."""

    raw = evidence.raw_evidence_bytes
    if raw is None:
        raise ExternalQualityError("raw teacher evidence bytes are unavailable for publication")
    if (
        len(raw) != evidence.raw_evidence_size
        or "sha256:" + hashlib.sha256(raw).hexdigest() != evidence.raw_evidence_sha256
    ):
        raise ExternalQualityError("raw teacher evidence does not match its receipt")
    root_path = _validated_raw_root(root)
    namespace = root_path / "external-quality-v2"
    namespace.mkdir(mode=0o700, exist_ok=True)
    os.chmod(namespace, 0o700)
    path = raw_quality_artifact_path(root_path, evidence.raw_evidence_sha256)
    path.parent.mkdir(mode=0o700, exist_ok=True)
    os.chmod(path.parent, 0o700)
    for directory in (namespace, path.parent):
        info = directory.lstat()
        if (
            stat.S_ISLNK(info.st_mode)
            or not stat.S_ISDIR(info.st_mode)
            or info.st_uid != os.geteuid()
            or stat.S_IMODE(info.st_mode) != 0o700
        ):
            raise ExternalQualityError(
                "raw teacher artifact directory is not private/controller-owned"
            )
    if not path.exists():
        fd, temporary = tempfile.mkstemp(prefix=".evidence-", dir=path.parent)
        try:
            os.fchmod(fd, 0o600)
            view = memoryview(raw)
            offset = 0
            while offset < len(view):
                written = os.write(fd, view[offset:])
                if written <= 0:
                    raise ExternalQualityError("raw teacher artifact write made no progress")
                offset += written
            os.fsync(fd)
            os.close(fd)
            fd = -1
            try:
                os.link(temporary, path)
            except FileExistsError:
                pass
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            if fd >= 0:
                os.close(fd)
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
    published = replace(evidence, raw_artifact_published=True)
    reopened = reopen_raw_quality_artifact(root_path, published, arena=arena)
    if reopened != raw:
        raise ExternalQualityError("published raw teacher artifact changed")
    return published


def build_teacher_forced_evidence(
    plan: PosthocReferencePlan,
    *,
    stock_control_batches: Sequence[Sequence[Any]],
    warmup_iters: int,
    traces: Mapping[tuple[str, int], Sequence[TeacherForcedPromptTrace]],
    arena: "ArenaProfile",
    hidden_judge=None,
) -> TeacherForcedExternalQualityEvidence:
    """Reduce validated raw B' frames to settlement-sized prompt summaries."""

    policy = arena.fidelity.teacher_forced_policy
    all_plan_batches = (*plan.warmup_batches, *plan.timed_batches)
    if len(stock_control_batches) != arena.scoring.warmup_iters + arena.scoring.timed_iters:
        raise ExternalQualityError("B' batch coverage differs from the sealed plan")
    raw_trace_payload: list[dict[str, Any]] = []
    built: dict[tuple[str, int], TeacherForcedBatchEvidence] = {}
    for batch_plan in all_plan_batches:
        key = (batch_plan.phase, batch_plan.batch_index)
        batch_traces = traces.get(key)
        if not isinstance(batch_traces, Sequence) or len(batch_traces) != len(batch_plan.prompts):
            raise ExternalQualityError("post-hoc B' trace coverage differs from the plan")
        global_index = (
            batch_plan.batch_index
            if batch_plan.phase == "warmup"
            else warmup_iters + batch_plan.batch_index
        )
        control_batch = stock_control_batches[global_index]
        clusters: list[PromptClusterEvidence] = []
        raw_prompts: list[dict[str, Any]] = []
        for prompt_plan, trace in zip(batch_plan.prompts, batch_traces, strict=True):
            if type(trace) is not TeacherForcedPromptTrace:
                raise ExternalQualityError("post-hoc B' returned an untyped prompt trace")
            if (
                type(trace.prompt_token_count) is not int
                or trace.prompt_token_count <= 0
                or not isinstance(trace.prompt_token_sha256, str)
                or len(trace.prompt_token_sha256) != 64
                or any(char not in "0123456789abcdef" for char in trace.prompt_token_sha256)
            ):
                raise ExternalQualityError("B' canonical prompt-token receipt is invalid")
            if not 0 <= prompt_plan.prompt_index < len(control_batch):
                raise ExternalQualityError("selected B' prompt index is out of bounds")
            control = _clean_run(
                control_batch[prompt_plan.prompt_index],
                expected_tokens=arena.workload.max_new_tokens,
                topk_num=arena.workload.top_logprobs,
            )

            def hidden(source: str, ids: tuple[int, ...]) -> tuple[float, int]:
                if hidden_judge is None:
                    return 0.0, 0
                value = hidden_judge(
                    batch_plan.phase,
                    batch_plan.batch_index,
                    prompt_plan.prompt_index,
                    source,
                    ids,
                )
                if (
                    not isinstance(value, tuple)
                    or len(value) != 2
                    or type(value[1]) is not int
                ):
                    raise ExternalQualityError("hidden-task judge returned invalid evidence")
                return float(value[0]), value[1]

            b_hidden = hidden("baseline", prompt_plan.baseline[0])
            c_hidden = hidden("candidate", prompt_plan.candidate[0])
            bp_hidden = hidden("stock_control", control[0])
            baseline_summary = summarize_rollout(
                prompt_plan.baseline[1],
                trace.baseline,
                tail_nll_threshold=policy.tail_nll_threshold,
                nll_clip=policy.nll_clip,
                topk_num=arena.workload.top_logprobs,
                hidden_score=b_hidden[0],
                hidden_total=b_hidden[1],
            )
            candidate_summary = summarize_rollout(
                prompt_plan.candidate[1],
                trace.candidate,
                tail_nll_threshold=policy.tail_nll_threshold,
                nll_clip=policy.nll_clip,
                topk_num=arena.workload.top_logprobs,
                hidden_score=c_hidden[0],
                hidden_total=c_hidden[1],
            )
            control_summary = summarize_rollout(
                control[1],
                trace.stock_control,
                tail_nll_threshold=policy.tail_nll_threshold,
                nll_clip=policy.nll_clip,
                topk_num=arena.workload.top_logprobs,
                hidden_score=bp_hidden[0],
                hidden_total=bp_hidden[1],
            )
            exact = sum(
                left == right
                for left, right in zip(
                    prompt_plan.baseline[0], prompt_plan.candidate[0], strict=True
                )
            )
            clusters.append(PromptClusterEvidence(
                baseline=baseline_summary,
                candidate=candidate_summary,
                stock_control=control_summary,
                exact_token_matches=exact,
                exact_token_total=arena.workload.max_new_tokens,
            ))

            def trace_payload(item: TeacherForcedTrace) -> dict[str, Any]:
                return {
                    "target_logprobs": list(item.target_logprobs),
                    "trusted_topk": [
                        [[lp, token_id] for lp, token_id, _none in position]
                        for position in item.trusted_topk
                    ],
                }

            raw_prompts.append({
                "prompt_index": prompt_plan.prompt_index,
                "prompt_token_count": trace.prompt_token_count,
                "prompt_token_sha256": trace.prompt_token_sha256,
                "stock_control_ids": list(control[0]),
                "stock_control_reported_topk": [
                    [[lp, token_id] for lp, token_id, _none in position]
                    for position in control[1]
                ],
                "hidden": {
                    "baseline": [b_hidden[0], b_hidden[1]],
                    "candidate": [c_hidden[0], c_hidden[1]],
                    "stock_control": [bp_hidden[0], bp_hidden[1]],
                },
                "baseline": trace_payload(trace.baseline),
                "candidate": trace_payload(trace.candidate),
                "stock_control": trace_payload(trace.stock_control),
            })
        built[key] = TeacherForcedBatchEvidence(tuple(clusters))
        raw_trace_payload.append({
            "phase": batch_plan.phase,
            "batch_index": batch_plan.batch_index,
            "prompts": raw_prompts,
        })
    selected_payload = [
        {
            "phase": batch.phase,
            "batch_index": batch.batch_index,
            "prompt_indices": [prompt.prompt_index for prompt in batch.prompts],
        }
        for batch in (*plan.warmup_batches, *plan.timed_batches)
    ]
    try:
        sealed_payload = json.loads(plan.sealed_rollout_bytes)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:  # pragma: no cover
        raise ExternalQualityError(f"internal sealed rollout is malformed: {exc}") from None
    raw_payload = {
        "protocol": TEACHER_FORCED_QUALITY_PROTOCOL_V2,
        "sealed_rollout_sha256": plan.sealed_rollout_sha256,
        "sealed_rollouts": sealed_payload,
        "selected_batches": selected_payload,
        "teacher_traces": raw_trace_payload,
    }
    raw_bytes = canonical_evidence_bytes(raw_payload)
    raw_digest = "sha256:" + hashlib.sha256(raw_bytes).hexdigest()
    evidence = TeacherForcedExternalQualityEvidence(
        protocol=TEACHER_FORCED_QUALITY_PROTOCOL_V2,
        sealed_rollout_sha256=plan.sealed_rollout_sha256,
        raw_evidence_sha256=raw_digest,
        raw_evidence_size=len(raw_bytes),
        raw_artifact_published=False,
        hidden_tasks_present=hidden_judge is not None,
        timed_batches=tuple(
            built[("timed", index)] for index in range(arena.scoring.timed_iters)
        ),
        warmup_batches=tuple(
            built[("warmup", index)] for index in range(arena.scoring.warmup_iters)
        ),
        raw_evidence_bytes=raw_bytes,
    )
    return evidence.validated(arena, require_published=False)
