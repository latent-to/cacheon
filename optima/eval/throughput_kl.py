"""End-to-end evaluation: throughput + output-distribution (KL) fidelity.

Two launches of the same model (identical weights/seed/sampler), differing only
by whether the miner kernel is enabled, isolate the kernel's effect: the
throughput delta is the kernel's, and the per-position KL between the two runs is
how much it perturbed the output. A faithful kernel yields KL ~ 0 and (hopefully)
speedup > 1.

Robustness measures (vs the first MVP):

* tamper-resistant timing — the driver process calls ``seam.mark_driver()`` so it
  never imports the miner module; the kernel runs only in the spawned scheduler,
  which the driver times over IPC. A malicious kernel cannot reach the clock.
* median-of-K — each launch retains the timed median plus spread, then caps the
  load-bearing point by its charged conditioning-tail floor, so neither a single
  noisy sample nor discarded cooldown can swing the score.
* larger, seeded prompt set — sampled per epoch from a corpus so a kernel can't
  special-case a fixed handful of prompts, and more positions stabilize the KL.

GPU-only; imports sglang lazily.
"""

from __future__ import annotations

import logging
import secrets
import statistics
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Optional

import torch

from optima.eval._launch import call_in_subprocess

logger = logging.getLogger("optima.eval")


def effective_fidelity_mode(cfg) -> str:
    """Return the configured fidelity lane for this evaluation.

    The controller-side stock-control comparison is always load-bearing. ``audit``
    additionally collects process-local diagnostics; it is not qualification authority.
    Framework/setup submissions are likewise judged from externally observed evidence
    and may not disguise their broader execution class as a component audit lane.
    """
    if bool(getattr(cfg, "framework_mode", False)):
        return "framework"
    return "audit" if getattr(cfg, "fidelity_mode", "kl") == "audit" else "kl"


_TRANSIENT_LAUNCH_MARKERS = (
    "cuda out of memory",
    "kv cache pool",
    "memory pool",
    "address already in use",
    "connection reset",
    "connection refused",
    "scheduler process terminated",
    "engine process terminated",
    "zmqerror",
)


def is_transient_launch_failure(exc: BaseException) -> bool:
    """Conservatively classify infrastructure failures eligible for one retry.

    Candidate errors, receipt/coverage failures, graph failures, and watchdog timeouts
    are deterministic until code/config changes and must surface immediately instead
    of paying for another cold model load.
    """
    typed_retryable = getattr(exc, "retryable", None)
    if type(typed_retryable) is bool:
        return typed_retryable
    message = str(exc).lower()
    if any(marker in message for marker in (
        "receipt", "execution coverage", "kernel raised", "graph", "timed out",
        "qualification",
    )):
        return False
    return any(marker in message for marker in _TRANSIENT_LAUNCH_MARKERS)
from optima.eval.kl import KLReport, aligned_kl, extract_per_prompt, kl_gate_ok, token_match_rate
from optima.eval.prompts import sample_prompt_batches
from optima.eval.scoring import (
    EXTERNAL_QUALITY_GATE_V1,
    ExternalFidelityMetrics,
    ExternalQualityBatch,
    score_external_quality_batches,
    score_output_token_match_batches,
    score_speedup,
)
from optima.eval.external_quality import (
    QUALITY_FAIL,
    QUALITY_NO_DECISION,
    QUALITY_PASS,
    TeacherForcedExternalQualityEvidence,
    build_teacher_forced_evidence,
    publish_raw_quality_artifact,
    score_teacher_forced_quality,
    seal_posthoc_reference_plan,
)


@dataclass
class EvalConfig:
    model_path: str
    dtype: str = "bfloat16"
    max_new_tokens: int = 64
    num_prompts: int = 32
    timed_iters: int = 3  # median-of-K timed passes per launch
    top_logprobs_num: int = 20
    temperature: float = 0.0  # greedy -> deterministic alignment
    # Fixed token budget by default: with greedy decode this forces baseline AND
    # candidate to emit EXACTLY max_new_tokens, so throughput is a pure per-token
    # latency comparison and a kernel can't inflate tok/s by nudging EOS timing
    # (the self-reported token count is no longer a lever). Turn off only for a
    # natural-length probe, never for scoring.
    ignore_eos: bool = True
    warmup_iters: int = 2  # >=2 full rounds: 1 leaves the documented ±17-32% clock-ramp in-window
    # The final N warmups form a continuous host-charged conditioning tail. Earlier
    # warmups may absorb stock request-lazy JIT/graph setup, but remain quality-graded.
    # Registered arenas pin N>=2 so one cold/boosted batch cannot define the score.
    conditioning_iters: int = 2
    deterministic: bool = False
    # None -> advisory (KL reported but not gated; for big MoE where the
    # nondeterminism floor exceeds any sane threshold and accuracy carries quality).
    kl_threshold: Optional[float] = 5e-3
    # Sparse-cheat guards alongside mean_kl (active only when kl_threshold is set):
    #   argmax_disagree_rate catches a kernel that flips a few tokens while keeping
    #   the mean low; p99 catches a catastrophic tail. Calibrate to the noise floor —
    #   in deterministic mode a faithful kernel sits at 0 flips (see README).
    argmax_disagree_rate_threshold: Optional[float] = 0.01
    p99_kl_threshold: Optional[float] = None  # opt-in (needs per-model calibration)
    # Tail-mass guard: top-k KL is blind to mass moved into the unreported tail, so a
    # flattened/diversity-collapsed candidate with a matching head passes it. mean
    # coverage deviation catches that. Loose default (faithful kernels sit ~0); tighten
    # per model. None -> off.
    coverage_dev_threshold: Optional[float] = 0.25
    # FRAMEWORK MODE: when the miner may patch the engine (a setup() callable), its
    # self-reported logprobs are NOT trustworthy, so the quality gate switches from
    # in-process KL to TOKEN-MATCH vs the trusted stock baseline — the candidate's
    # emitted tokens are only correct if it actually computed correctly. Full
    # cheat-resistance also needs no-egress isolation (see the threat model docs).
    framework_mode: bool = False
    token_match_threshold: float = 0.99  # min fraction of generated tokens matching baseline
    # No-egress isolation for EVERY candidate launch (the untrusted side): run it in a
    # fresh network namespace so miner code cannot fetch the reference output. Default
    # ON for both component and framework submissions; local pods without namespace
    # capability must opt into the explicit unsafe development escape hatch below.
    isolate: bool = True
    # Dev-only escape hatch for pods that cannot create a netns. Production scoring
    # must leave this False so failed isolation is a hard error.
    allow_unsafe_no_isolation: bool = False
    seed: int = 0  # model seed
    prompt_seed: int = 0  # per-epoch prompt sampling seed
    # Approximate tokens per prompt. None -> the short corpus (10-20 tok, a pure-decode
    # regime). Set for prefill-heavy arenas: without it a prefill-side win (e.g. the MSA
    # prefill indexer, ~30% of long-context serving prefill) is INVISIBLE to the scorer
    # — the workload never exercises the kernel. See optima/eval/prompts.py.
    input_len: Optional[int] = None
    # FLOOR on the required improvement (see optima/eval/scoring.py). The ACTUAL bar
    # is max(speedup_margin, score_k * measured_baseline_noise) — derived from the box,
    # not hand-picked. 0.5% floor (2026-07-07): real wins stack at 1-2%, and the
    # k*noise term — not this constant — is what guards a drifting box; a quiet box
    # resolves sub-1% deltas (locked-clock bracket spread 0.013%, 2026-06-15).
    speedup_margin: float = 0.005
    # Noise-robust scoring (we cannot lock GPU clocks on rented pods):
    #  * bookend_baseline: measure stock BEFORE and AFTER the candidate (B,C,B') so the
    #    candidate is bracketed; the two baseline reads bound the drift across it and
    #    give a per-round noise estimate. Off -> the old single-baseline 2-launch (cheap
    #    debug only; cannot be confident, so it never crowns).
    #  * score_k: how many measured-noise-widths above 1.0 a speedup must clear.
    #  * max_noise: if the bracketing baselines disagree by more than this, the round is
    #    untrustworthy -> NO-DECISION (never crowns), the subnet re-queues it.
    bookend_baseline: bool = True
    score_k: float = 2.0
    max_noise: float = 0.10
    # None -> sglang auto-picks the best backend for the hardware (fa3 on Hopper,
    # etc.). Don't hard-code a weak backend: a production-strong baseline is required,
    # or miners optimize against a slow reference. Override per-HW only if needed.
    attention_backend: Optional[str] = None
    # Graphs ON by default. Disabling CUDA graphs cripples the baseline (~6.5x slower
    # on 0.5B decode, measured on an H100), so a faithful kernel would "win" against a
    # weak reference. The seam is CUDA-graph-safe (validated). Set True only for quick
    # eager debugging, never for scoring.
    disable_cuda_graph: bool = False
    mem_fraction_static: float = 0.6
    log_level: str = "warning"
    # Serving regime: cap the concurrently-running requests so throughput is measured at a
    # production-like batch, not just whatever a single generate() call packs. The right
    # kernel is regime-dependent (low-batch=dispatch-bound, high-batch=memory-bound), so a
    # win must be measured at the serving operating point. None -> sglang default. PARTIAL
    # fix for the eval-vs-serving-distribution gap (report M2/#12): the knob exists; a full
    # per-epoch multi-regime sweep + worst-regime gate is still future work.
    max_running_requests: Optional[int] = None
    # multi-GPU knobs (TP size, MoE backend, custom-allreduce toggle for tensor-parallel
    # runs; see docs/DEV_ENVIRONMENT.md). Left unset by default so single-GPU runs are
    # byte-for-byte unchanged.
    tp_size: Optional[int] = None
    moe_runner_backend: Optional[str] = None
    disable_custom_all_reduce: bool = False
    candidate_attention_backend: Optional[str] = None
    candidate_moe_runner_backend: Optional[str] = None
    candidate_disable_custom_all_reduce: Optional[bool] = None
    extra_engine_kwargs: dict[str, Any] = field(default_factory=dict)
    candidate_extra_engine_kwargs: dict[str, Any] = field(default_factory=dict)
    # FIDELITY MODE (2026-07-07 finding, measured): on non-deterministic stacks
    # rollout-KL between two launches gates BATCHING/TACTICS, not fidelity — a
    # bit-stock candidate at 0.545x speed scored mean_kl 0.96, a single-prompt
    # bit-stock control still scored 0.81, and sglang's deterministic mode refuses
    # some arena backends (fa4). "audit" replaces the KL razor with the IN-ENGINE
    # AUDIT (optima/audit.py): an extra UNTIMED candidate diagnostic launch runs with
    # sampled per-call stock-baseline comparison under the slot's verify
    # tolerances. Scheduler-written audit evidence is not hostile-code authority and
    # never replaces the controller-side stock-control gate; KL is still reported
    # (advisory calibration data), and timed launches carry zero audit overhead.
    # "kl" additionally applies the arena's strict deterministic KL thresholds.
    fidelity_mode: str = "kl"  # "kl" | "audit"
    audit_rate: float = 0.05  # fraction of eligible dispatcher calls audited
    audit_min_calls: int = 32  # insufficient audit coverage is a FAIL, not a pass


@dataclass
class ModeResult:
    # Load-bearing point estimate: min(median timed rate, host-timed conditioning
    # tail). In OCI all warmups are quality-graded, but only the final
    # ``conditioning_iters`` are charged. The continuous tail starts at completion
    # of the last free setup response, or at ready when none are free, and spans
    # every charged warmup/gap through the first timed response without discarding
    # any charged constituent batch.
    tok_per_s: float
    tok_per_s_samples: list[float]
    tokens: int
    per_prompt: list[tuple[list[int], list]]  # (output_ids, per-position top-k)
    conditioning_tok_per_s: float = 0.0
    # Host-observed text for every TIMED response, in batch order.  It is not used
    # as a throughput numerator; it is retained for controller-side secret tasks.
    texts: list[str] = field(default_factory=list)
    # Phase-separated controller evidence. ``per_prompt``/``texts`` above are timed
    # only.  Batch boundaries remain explicit so neither warmup nor a clean timed
    # batch can subsidize a corrupt timed batch.
    per_prompt_batches: list[list[tuple[list[int], list]]] = field(default_factory=list)
    warmup_per_prompt: list[tuple[list[int], list]] = field(default_factory=list)
    warmup_texts: list[str] = field(default_factory=list)
    warmup_per_prompt_batches: list[list[tuple[list[int], list]]] = field(
        default_factory=list
    )

    @property
    def spread(self) -> tuple[float, float, float]:
        s = self.tok_per_s_samples
        if len(s) < 2:
            return (min(s, default=0.0), max(s, default=0.0), 0.0)
        return (min(s), max(s), statistics.pstdev(s))


@dataclass
class EvalReport:
    baseline: ModeResult
    candidate: ModeResult
    speedup: float  # informational: candidate / mean(bracketing baselines)
    kl: KLReport
    passed_quality: bool
    passed_speedup: bool  # NOISE-AWARE: cleared the measured bar AND the round was trustworthy
    score: float  # the crownable speedup (>=bar, confident) or 0.0 — what the ledger records
    token_match: float = 1.0  # fraction of tokens matching baseline (the framework-mode gate)
    noise: float = 0.0  # measured relative spread of the baseline reads
    required_speedup: float = 1.0  # the bar the speedup had to clear this round
    confident: bool = True  # False -> box too noisy this round; NO-DECISION, never crowns
    baseline2: Optional[ModeResult] = None  # the trailing bookend baseline (B'), if measured
    fidelity_mode: str = "kl"  # which quality gate produced passed_quality
    audit_desc: str = ""  # audit-mode: human-readable audit verdict (calls/violations)
    audit_receipts: list = field(default_factory=list)  # raw per-rank rolling audit stats
    # Controller-recomputable qualification evidence. ``kl`` is B-vs-C; this is
    # B-vs-B'.  Audit receipts stay diagnostic and are not substituted for it.
    control_kl: Optional[KLReport] = None
    external_quality_desc: str = ""
    token_matches: int = 0
    token_total: int = 0
    stock_token_matches: int = 0
    stock_token_total: int = 0
    warmup_kl: Optional[KLReport] = None
    warmup_control_kl: Optional[KLReport] = None
    warmup_token_matches: int = 0
    warmup_token_total: int = 0
    warmup_stock_token_matches: int = 0
    warmup_stock_token_total: int = 0
    timed_quality_batches: tuple[ExternalQualityBatch, ...] = ()
    warmup_quality_batches: tuple[ExternalQualityBatch, ...] = ()
    passed_timed_quality: bool = False
    passed_warmup_quality: bool = False
    external_quality_evidence: Optional[TeacherForcedExternalQualityEvidence] = None
    quality_decision: str = QUALITY_FAIL
    timed_quality_decision: str = QUALITY_FAIL
    warmup_quality_decision: str = QUALITY_FAIL


@contextmanager
def _env(**overrides: str):
    import os

    saved = {k: os.environ.get(k) for k in overrides}
    os.environ.update(overrides)
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _sampling_params(cfg: EvalConfig) -> dict:
    sp = {"temperature": cfg.temperature, "max_new_tokens": cfg.max_new_tokens}
    if cfg.ignore_eos:
        sp["ignore_eos"] = True
    return sp


def _timed_generate(engine, prompts: list[str], cfg: EvalConfig, *, with_logprobs: bool):
    sp = _sampling_params(cfg)
    kwargs: dict[str, Any] = {}
    if with_logprobs:
        kwargs = dict(return_logprob=True, logprob_start_len=-1, top_logprobs_num=cfg.top_logprobs_num)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    outputs = engine.generate(prompt=list(prompts), sampling_params=sp, **kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    if isinstance(outputs, dict):
        outputs = [outputs]
    tokens = _counted_tokens(outputs, prompts, cfg)
    return outputs, tokens, elapsed


def _counted_tokens(outputs, prompts, cfg) -> int:
    """The throughput numerator. The token COUNT is produced in the scheduler process
    where the miner kernel also runs, so it isn't trustworthy on its own. Under the
    scoring default (ignore_eos + a fixed max_new_tokens) the driver knows the count a
    PRIORI — ``len(prompts) * max_new_tokens`` — so we use that and never trust a
    scheduler-reported field. Only when natural-length generation is explicitly
    requested (--no-ignore-eos) do we fall back to the reported completion_tokens."""
    if getattr(cfg, "ignore_eos", False):
        return len(prompts) * int(cfg.max_new_tokens)
    return sum(int(o.get("meta_info", {}).get("completion_tokens", 0)) for o in outputs)


def _measure(engine, prompt_batches: list[list[str]], cfg: EvalConfig) -> ModeResult:
    """Measure disjoint batches and retain evidence from every timed response.

    Replaying one prompt list across warmup/timing lets a stateful candidate populate
    a prompt->result cache off-clock.  The plan is generated in the trusted controller;
    this function requires one fresh batch per warmup and timed iteration and requests
    top-k evidence on *every* timed call, so a candidate cannot identify a cheap timed
    call on which quality will not be checked.
    """
    required = max(0, cfg.warmup_iters) + max(1, cfg.timed_iters)
    if len(prompt_batches) != required:
        raise ValueError(
            f"prompt plan has {len(prompt_batches)} batches; expected exactly {required}"
        )
    if any(len(batch) != cfg.num_prompts for batch in prompt_batches):
        raise ValueError("every prompt batch must match cfg.num_prompts")

    # Engine construction and the initial ``warmup_iters - conditioning_iters``
    # warmups are deliberately free setup. Retain quality evidence for every
    # warmup, but charge each of the final conditioning warmups and the aggregate
    # continuous tail from the last setup response through the first timed
    # response. This prevents a charged warmup from being sacrificed for cooldown
    # while keeping all warmup quality separate from timed work.
    warmup_per_prompt: list[tuple[list[int], list]] = []
    warmup_texts: list[str] = []
    warmup_batches: list[list[tuple[list[int], list]]] = []
    conditioning_tok_per_s = 0.0
    if not 1 <= int(cfg.conditioning_iters) <= int(cfg.warmup_iters):
        raise ValueError("conditioning_iters must be in 1..warmup_iters")
    conditioning_start_index = int(cfg.warmup_iters) - int(cfg.conditioning_iters)
    conditioning_started_at = (
        time.monotonic() if conditioning_start_index == 0 else None
    )
    conditioning_tokens = 0
    conditioning_batch_rates: list[float] = []
    for warmup_index, prompts in enumerate(prompt_batches[:cfg.warmup_iters]):
        outputs, warmup_tokens, warmup_elapsed = _timed_generate(
            engine, prompts, cfg, with_logprobs=True
        )
        if warmup_index + 1 == conditioning_start_index:
            conditioning_started_at = time.monotonic()
        if warmup_index >= conditioning_start_index:
            conditioning_tokens += warmup_tokens
            if warmup_elapsed > 0:
                conditioning_batch_rates.append(warmup_tokens / warmup_elapsed)
        batch = extract_per_prompt(outputs or [])
        warmup_batches.append(batch)
        warmup_per_prompt.extend(batch)
        warmup_texts.extend(
            str(output.get("text", "")) for output in (outputs or [])
        )

    samples: list[float] = []
    all_per_prompt: list[tuple[list[int], list]] = []
    all_texts: list[str] = []
    timed_batches: list[list[tuple[list[int], list]]] = []
    total_tokens = 0
    for prompts in prompt_batches[cfg.warmup_iters:]:
        outputs, tokens, elapsed = _timed_generate(
            engine, prompts, cfg, with_logprobs=True
        )
        if elapsed > 0:
            rate = tokens / elapsed
            samples.append(rate)
            if len(samples) == 1 and cfg.warmup_iters > 0:
                assert conditioning_started_at is not None
                transition_elapsed = time.monotonic() - conditioning_started_at
                conditioning_tokens += tokens
                if (
                    transition_elapsed <= 0
                    or len(conditioning_batch_rates) != cfg.conditioning_iters
                ):
                    conditioning_tok_per_s = 0.0
                else:
                    conditioning_tok_per_s = min(
                        *conditioning_batch_rates,
                        rate,
                        conditioning_tokens / transition_elapsed,
                    )
        total_tokens += tokens
        batch = extract_per_prompt(outputs or [])
        timed_batches.append(batch)
        all_per_prompt.extend(batch)
        all_texts.extend(str(output.get("text", "")) for output in (outputs or []))

    timed_point = statistics.median(samples) if samples else 0.0
    effective_point = (
        min(timed_point, conditioning_tok_per_s)
        if cfg.warmup_iters > 0 else timed_point
    )
    return ModeResult(
        tok_per_s=effective_point,
        tok_per_s_samples=samples,
        tokens=total_tokens,
        per_prompt=all_per_prompt,
        conditioning_tok_per_s=conditioning_tok_per_s,
        texts=all_texts,
        per_prompt_batches=timed_batches,
        warmup_per_prompt=warmup_per_prompt,
        warmup_texts=warmup_texts,
        warmup_per_prompt_batches=warmup_batches,
    )


def _run_launch(cfg: EvalConfig, prompt_batches: list[list[str]], *, bundle_path: str,
                active: bool) -> ModeResult:
    # launched_engine marks THIS process as the timer/driver before importing sglang
    # (seam pass-through; the miner module never loads where wall-clock is measured)
    # and, for an ACTIVE launch, demands seam receipts — the candidate must PROVE the
    # bundle loaded and the impl was selected, else the run is stock-vs-stock and
    # scoring it would be a phantom pass (see optima/receipts.py).
    from optima.eval._launch import launched_engine

    with launched_engine(cfg, bundle_path=bundle_path, active=active) as engine:
        return _measure(engine, prompt_batches, cfg)


def _run_quality_launch(cfg: EvalConfig, prompt_batches: list[list[str]], *,
                        bundle_path: str) -> tuple[ModeResult, list, list]:
    """Audit-mode candidate QUALITY launch: UNTIMED (its tok/s is discarded), with
    the in-engine audit armed at cfg.audit_rate and logprobs captured for the
    advisory KL. Kept separate from the timed candidate launch so audited calls'
    clone+baseline overhead can never bias the throughput comparison.

    Runs EAGER regardless of the scoring config: calls replayed inside a captured
    CUDA graph never re-enter the Python dispatcher, so a graphs-on launch would
    audit ~nothing. The audit checks the kernel's FUNCTION (regime-independent);
    capture-conditional divergence is covered elsewhere — verify's capture-replay
    stress plus the graphs-on benchmark accuracy gate (a kernel that computes
    correctly eager but garbage under capture trashes its own benchmark run)."""
    import dataclasses

    from optima.eval._launch import launched_engine

    qcfg = dataclasses.replace(
        cfg,
        timed_iters=1,
        warmup_iters=1,
        conditioning_iters=1,
        disable_cuda_graph=True,
    )
    audit_out: list = []
    member_out: list = []
    if len(prompt_batches) != 2:
        raise ValueError("audit quality launch requires one warmup + one checked batch")
    with launched_engine(qcfg, bundle_path=bundle_path, active=True,
                         audit_rate=cfg.audit_rate, audit_out=audit_out,
                         member_out=member_out) as engine:
        result = _measure(engine, prompt_batches, qcfg)
    return result, audit_out, member_out


def _aligned_kl(baseline: ModeResult, candidate: ModeResult) -> KLReport:
    # Per-prompt alignment up to the first token divergence; see kl.aligned_kl.
    return aligned_kl(baseline.per_prompt, candidate.per_prompt)


def _phase_prompt_batches(
    result: ModeResult, *, warmup: bool
) -> tuple[list[tuple[list[int], list]], ...]:
    explicit = (
        result.warmup_per_prompt_batches
        if warmup else result.per_prompt_batches
    )
    if explicit:
        return tuple(explicit)
    flat = result.warmup_per_prompt if warmup else result.per_prompt
    # Compatibility for focused callers constructing a one-batch ModeResult. The
    # crownable path always supplies explicit batch boundaries and QualificationReport
    # enforces the arena's exact batch cardinality.
    return (list(flat),) if flat else ()


def _fidelity_metrics(report: KLReport) -> ExternalFidelityMetrics:
    return ExternalFidelityMetrics(
        num_positions=report.num_positions,
        mean_kl=report.mean_kl,
        max_kl=report.max_kl,
        p99_kl=report.p99_kl,
        argmax_disagreements=report.argmax_disagreements,
        mean_coverage_dev=report.mean_coverage_dev,
        dropped_positions=report.dropped_positions,
    )


def _external_quality_gate(
    baseline: ModeResult,
    candidate: ModeResult,
    baseline2: ModeResult | None,
) -> tuple[bool, str, KLReport]:
    """Trusted-controller paired quality gate for nondeterministic arenas.

    Scheduler audit/receipt files are useful diagnostics but are writable by the
    candidate process.  This gate instead compares candidate evidence against a
    baseline result the candidate namespace never receives, and calibrates the
    allowed launch-to-launch drift from the independent B' control.  A candidate can
    reproduce the hidden top-k trajectories only by doing inference (or an equivalent
    amount of useful model work) on each one-shot timed prompt.
    """
    passed, detail, cand, _batches = _external_quality_phase(
        baseline, candidate, baseline2, warmup=False
    )
    return passed, detail, cand


def _external_quality_phase(
    baseline: ModeResult,
    candidate: ModeResult,
    baseline2: ModeResult | None,
    *,
    warmup: bool,
) -> tuple[bool, str, KLReport, tuple[ExternalQualityBatch, ...]]:
    """Grade one phase batch-by-batch and retain its exact paired controls."""

    baseline_batches = _phase_prompt_batches(baseline, warmup=warmup)
    candidate_batches = _phase_prompt_batches(candidate, warmup=warmup)
    control_batches = (
        _phase_prompt_batches(baseline2, warmup=warmup)
        if baseline2 is not None else ()
    )
    baseline_flat = (
        baseline.warmup_per_prompt if warmup else baseline.per_prompt
    )
    candidate_flat = (
        candidate.warmup_per_prompt if warmup else candidate.per_prompt
    )
    cand = aligned_kl(baseline_flat, candidate_flat)
    if baseline2 is None:
        return False, "missing trusted B' quality control", cand, ()
    if (
        not baseline_batches
        or len(candidate_batches) != len(baseline_batches)
        or len(control_batches) != len(baseline_batches)
    ):
        return False, "phase batch coverage differs across B/C/B'", cand, ()
    evidence: list[ExternalQualityBatch] = []
    for index, (base, contender, control) in enumerate(
        zip(baseline_batches, candidate_batches, control_batches), start=1
    ):
        if not base or len(contender) != len(base) or len(control) != len(base):
            return (
                False,
                f"phase batch {index} prompt coverage differs across B/C/B'",
                cand,
                (),
            )
        candidate_kl = aligned_kl(base, contender)
        control_kl = aligned_kl(base, control)
        candidate_matches, candidate_total = token_match_rate(base, contender)
        control_matches, control_total = token_match_rate(base, control)
        evidence.append(ExternalQualityBatch(
            candidate=_fidelity_metrics(candidate_kl),
            stock_control=_fidelity_metrics(control_kl),
            token_matches=candidate_matches,
            token_total=candidate_total,
            stock_token_matches=control_matches,
            stock_token_total=control_total,
        ))
    phase = "warmup" if warmup else "timed"
    passed, detail = score_external_quality_batches(
        tuple(evidence),
        gate=EXTERNAL_QUALITY_GATE_V1,
        phase=phase,
    )
    return passed, detail, cand, tuple(evidence)


def evaluate(
    cfg: EvalConfig,
    bundle_path: str,
    prompts: Optional[list[str]] = None,
    *,
    oci_launcher=None,
) -> EvalReport:
    registered_arena = None
    if oci_launcher is not None:
        # One controller-owned wall deadline spans B, C, B', all batches and any
        # bounded retry. Individual session watchdogs may not reset the budget.
        oci_launcher.begin_evaluation()
        profile = getattr(oci_launcher, "profile", None)
        arena_name = getattr(profile, "arena_name", None)
        if arena_name:
            from optima.arenas import get_arena

            registered_arena = get_arena(arena_name)
            if getattr(profile, "arena_fingerprint", None) != registered_arena.fingerprint:
                raise RuntimeError("OCI launcher arena fingerprint changed before evaluation")
    batch_count = max(0, cfg.warmup_iters) + max(1, cfg.timed_iters)
    if prompts:
        base = list(prompts)
        if len(base) != cfg.num_prompts:
            raise ValueError("explicit prompts must contain exactly cfg.num_prompts entries")
        # Preserve caller-provided semantics while making every concrete request
        # prefix-disjoint across iterations.
        prompt_batches = [
            [f"[iteration {index} case {item}] {prompt}" for item, prompt in enumerate(base)]
            for index in range(batch_count)
        ]
    else:
        prompt_batches = sample_prompt_batches(
            batch_count, cfg.num_prompts, cfg.prompt_seed, input_len=cfg.input_len
        )
    audit_batches = sample_prompt_batches(
        2, cfg.num_prompts, cfg.prompt_seed ^ 0xA5A55A5AF00DFACE,
        input_len=cfg.input_len,
    )

    # Bookended A/B (we cannot lock GPU clocks on rented pods): measure stock BEFORE
    # and AFTER the candidate so the candidate is bracketed and the two baseline reads
    # bound the warmup/thermal drift across it. Each launch runs in its own fresh
    # process (call_in_subprocess) so the baseline's deterministic/CUDA global state
    # can't corrupt the candidate. See optima/eval/scoring.py.
    #
    # One retry per launch: engine startup can die on a TRANSIENT — this build's
    # KV-pool sizing snapshots free memory as a distributed MIN across ranks while
    # weight-shard load buffers may still be in flight on a straggler rank, so an
    # identical config can pass one launch and OOM the next (measured 2026-07-10).
    # The relaunch enters through the child's drain-wait (+ optional orphan sweep),
    # so a retry starts from clean GPUs. A launch that fails TWICE propagates.
    def _launch(label: str, fn, *args, oci_mode: str | None = None, **kwargs):
        posthoc_plan = kwargs.pop("_posthoc_plan", None)
        def invoke():
            if oci_launcher is not None:
                if oci_mode is None:
                    raise RuntimeError(f"OCI launch {label!r} has no fixed worker mode")
                # The OCI worker owns the only allowed dispatch table.  Do not send
                # ``fn`` or kwargs across the boundary.
                prompt_plan = args[1] if len(args) >= 2 else kwargs.get("prompt_batches")
                launch_kwargs = {"mode": oci_mode, "arm": label}
                if posthoc_plan is not None:
                    launch_kwargs["posthoc_plan"] = posthoc_plan
                return oci_launcher.run(cfg, prompt_plan, **launch_kwargs)
            if posthoc_plan is not None:
                raise RuntimeError(
                    "post-hoc teacher forcing requires the isolated OCI controller"
                )
            return call_in_subprocess(fn, *args, **kwargs)
        try:
            return invoke()
        except RuntimeError as exc:
            if not is_transient_launch_failure(exc):
                raise
            logger.warning("optima: %s launch failed (%s); retrying once", label, exc)
            time.sleep(30.0)
            return invoke()

    baseline = _launch(
        "baseline", _run_launch, cfg, prompt_batches, bundle_path="", active=False,
        oci_mode="baseline",
    )
    # Engine-wide setup() may alter code outside audited dispatcher call sites. Its
    # correctness gate must therefore be the externally observed candidate tokens,
    # never the in-engine eager audit. Framework fidelity takes precedence even when
    # an operator also requested ``--fidelity-mode audit``.
    quality_mode = effective_fidelity_mode(cfg)
    framework_mode = quality_mode == "framework"
    audit_mode = quality_mode == "audit"
    # Candidate-process audit receipts are diagnostics, never crown authority.
    # A production OCI bracket already pays for three fresh engines (B/C/B') and
    # must not add a fourth cold engine merely to collect forgeable evidence.
    # Keep the diagnostic launch only for the local development evaluator.
    run_audit_diagnostic = audit_mode and oci_launcher is None
    quality_result, audit_receipts, audit_members = (
        _launch(
            "quality", _run_quality_launch, cfg, audit_batches,
            bundle_path=bundle_path, oci_mode="candidate_audit",
        )
        if run_audit_diagnostic else (None, [], []))
    candidate = _launch(
        "candidate", _run_launch, cfg, prompt_batches,
        bundle_path=bundle_path, active=True, oci_mode="candidate",
    )
    posthoc_plan = None
    raw_teacher_traces = None
    external_quality_evidence = None
    if registered_arena is not None:
        if not cfg.bookend_baseline:
            raise RuntimeError("registered teacher-forced quality requires B/C/B' bookends")
        posthoc_plan = seal_posthoc_reference_plan(
            prompt_batches,
            baseline_batches=(
                *baseline.warmup_per_prompt_batches,
                *baseline.per_prompt_batches,
            ),
            candidate_batches=(
                *candidate.warmup_per_prompt_batches,
                *candidate.per_prompt_batches,
            ),
            warmup_iters=cfg.warmup_iters,
            clusters_per_batch=(
                registered_arena.fidelity.teacher_forced_policy.clusters_per_batch
            ),
            expected_tokens=cfg.max_new_tokens,
            topk_num=cfg.top_logprobs_num,
            # Generated only after the candidate session has been force-removed.
            selection_secret=secrets.token_bytes(32),
        )
        baseline2, raw_teacher_traces = _launch(
            "bookend",
            _run_launch,
            cfg,
            prompt_batches,
            bundle_path="",
            active=False,
            oci_mode="baseline",
            _posthoc_plan=posthoc_plan,
        )
        external_quality_evidence = build_teacher_forced_evidence(
            posthoc_plan,
            stock_control_batches=(
                *baseline2.warmup_per_prompt_batches,
                *baseline2.per_prompt_batches,
            ),
            warmup_iters=cfg.warmup_iters,
            traces=raw_teacher_traces,
            arena=registered_arena,
        )
        artifact_root = getattr(getattr(oci_launcher, "profile", None), "artifact_dir", None)
        if artifact_root is None:
            raise RuntimeError("registered OCI launcher has no controller artifact root")
        external_quality_evidence = publish_raw_quality_artifact(
            artifact_root, external_quality_evidence, arena=registered_arena
        )
    else:
        baseline2 = (
            _launch(
                "bookend", _run_launch, cfg, prompt_batches,
                bundle_path="", active=False, oci_mode="baseline",
            )
            if cfg.bookend_baseline else None
        )

    baseline_reads = [baseline.tok_per_s] + ([baseline2.tok_per_s] if baseline2 else [])
    verdict = score_speedup(
        baseline_reads, candidate.tok_per_s,
        min_margin=cfg.speedup_margin, k=cfg.score_k, max_noise=cfg.max_noise,
    )

    # Warmup and timed output evidence are disjoint authorities. Every batch is
    # graded independently against the corresponding B-vs-B' stock control; neither
    # phase nor a clean batch can dilute a failure in another.
    timed_external_ok, timed_external_desc, timed_kl, timed_quality_batches = (
        _external_quality_phase(
            baseline, candidate, baseline2, warmup=False
        )
    )
    warmup_external_ok, warmup_external_desc, warmup_kl, warmup_quality_batches = (
        _external_quality_phase(
            baseline, candidate, baseline2, warmup=True
        )
    )
    external_desc = timed_external_desc + "; " + warmup_external_desc
    control_kl = (
        _aligned_kl(baseline, baseline2) if baseline2 is not None else None
    )
    warmup_control_kl = (
        aligned_kl(baseline.warmup_per_prompt, baseline2.warmup_per_prompt)
        if baseline2 is not None else None
    )
    kl = timed_kl
    matched = sum(batch.token_matches for batch in timed_quality_batches)
    total = sum(batch.token_total for batch in timed_quality_batches)
    stock_matched = sum(
        batch.stock_token_matches for batch in timed_quality_batches
    )
    stock_total = sum(batch.stock_token_total for batch in timed_quality_batches)
    warmup_matched = sum(
        batch.token_matches for batch in warmup_quality_batches
    )
    warmup_total = sum(batch.token_total for batch in warmup_quality_batches)
    warmup_stock_matched = sum(
        batch.stock_token_matches for batch in warmup_quality_batches
    )
    warmup_stock_total = sum(
        batch.stock_token_total for batch in warmup_quality_batches
    )
    token_match = (matched / total) if total else 1.0
    # Model-consumed output IDs and raw top-k are independent products. Grade both
    # in every lane against the paired B-vs-B' stock control; restricting token
    # authority to whole-system submissions lets a component return corrupt tokens
    # while preserving an otherwise perfect raw distribution.
    timed_token_ok, timed_token_desc = score_output_token_match_batches(
        timed_quality_batches,
        threshold=cfg.token_match_threshold,
        phase="timed",
    )
    warmup_token_ok, warmup_token_desc = score_output_token_match_batches(
        warmup_quality_batches,
        threshold=cfg.token_match_threshold,
        phase="warmup",
    )
    external_desc += timed_token_desc + warmup_token_desc
    audit_desc = ""
    if audit_mode:
        if run_audit_diagnostic:
            # The in-engine audit gives useful slot-level diagnostics but is NOT
            # hostile-code authority: candidate code shares that scheduler.
            from optima import audit as _audit
            from optima.competition import resolve_competition
            from optima.manifest import load_manifest

            competition = resolve_competition(
                load_manifest(bundle_path), for_settlement=True, warn_legacy=False
            )
            audit_ok, audit_desc = _audit.gate(
                audit_receipts,
                min_calls=cfg.audit_min_calls,
                expected_slots=competition.members,
                member_receipts=audit_members,
                min_calls_per_member=cfg.audit_min_calls,
            )
        else:
            audit_ok = False
            audit_desc = "skipped in production OCI bracket"
        # Scheduler-written audit evidence is defense-in-depth only: candidate code
        # shares that process and can forge it.  Keep the result for diagnosis, but
        # only the controller-side paired gate is load-bearing for qualification.
        passed_timed_quality = timed_external_ok and timed_token_ok
        passed_warmup_quality = warmup_external_ok and warmup_token_ok
        audit_desc = (
            f"diagnostic audit {'PASS' if audit_ok else 'FAIL'}: {audit_desc}"
        )
    elif framework_mode:
        # The miner may have patched the engine (setup()), so its self-reported logprobs
        # are not trusted: gate on token-match vs the trusted stock baseline, not KL.
        passed_timed_quality = timed_external_ok and timed_token_ok
        passed_warmup_quality = warmup_external_ok and warmup_token_ok
        audit_desc = external_desc
    else:
        def strict_batch_ok(batch: ExternalQualityBatch) -> bool:
            metrics = batch.candidate
            return metrics.num_positions > 0 and kl_gate_ok(
                KLReport(
                    num_positions=metrics.num_positions,
                    mean_kl=metrics.mean_kl,
                    max_kl=metrics.max_kl,
                    p99_kl=metrics.p99_kl,
                    argmax_disagreements=metrics.argmax_disagreements,
                    mean_coverage_dev=metrics.mean_coverage_dev,
                    dropped_positions=metrics.dropped_positions,
                ),
                kl_threshold=cfg.kl_threshold,
                p99_kl_threshold=cfg.p99_kl_threshold,
                argmax_disagree_rate_threshold=cfg.argmax_disagree_rate_threshold,
                coverage_dev_threshold=cfg.coverage_dev_threshold,
            )

        passed_timed_quality = timed_external_ok and timed_token_ok and all(
            strict_batch_ok(batch) for batch in timed_quality_batches
        )
        passed_warmup_quality = warmup_external_ok and warmup_token_ok and all(
            strict_batch_ok(batch) for batch in warmup_quality_batches
        )
        audit_desc = external_desc
    quality_decision = QUALITY_PASS if (
        passed_timed_quality and passed_warmup_quality
    ) else QUALITY_FAIL
    timed_quality_decision = (
        QUALITY_PASS if passed_timed_quality else QUALITY_FAIL
    )
    warmup_quality_decision = (
        QUALITY_PASS if passed_warmup_quality else QUALITY_FAIL
    )
    if external_quality_evidence is not None:
        assert registered_arena is not None
        teacher_verdict = score_teacher_forced_quality(
            external_quality_evidence, arena=registered_arena
        )
        quality_decision = teacher_verdict.decision
        timed_quality_decision = teacher_verdict.timed_decision
        warmup_quality_decision = teacher_verdict.warmup_decision
        passed_timed_quality = timed_quality_decision == QUALITY_PASS
        passed_warmup_quality = warmup_quality_decision == QUALITY_PASS
        external_desc = teacher_verdict.detail
    passed_quality = quality_decision == QUALITY_PASS
    # Crownable only when quality holds AND the speedup is a noise-confident real win.
    # The ledger records the speedup only when crownable, else 0.0 — so a cheat (quality
    # fail), a faithful-but-not-faster kernel, OR a too-noisy round can never take the
    # title. The raw speedup is still reported for the human read.
    crownable = passed_quality and verdict.passed_speedup
    score = verdict.speedup if crownable else 0.0

    return EvalReport(
        baseline, candidate, verdict.speedup, kl, passed_quality, verdict.passed_speedup, score,
        token_match, noise=verdict.noise, required_speedup=verdict.required,
        confident=verdict.confident, baseline2=baseline2,
        fidelity_mode=quality_mode,
        audit_desc=audit_desc, audit_receipts=audit_receipts,
        control_kl=control_kl, external_quality_desc=external_desc,
        token_matches=matched, token_total=total,
        stock_token_matches=stock_matched, stock_token_total=stock_total,
        warmup_kl=warmup_kl, warmup_control_kl=warmup_control_kl,
        warmup_token_matches=warmup_matched,
        warmup_token_total=warmup_total,
        warmup_stock_token_matches=warmup_stock_matched,
        warmup_stock_token_total=warmup_stock_total,
        timed_quality_batches=timed_quality_batches,
        warmup_quality_batches=warmup_quality_batches,
        passed_timed_quality=passed_timed_quality,
        passed_warmup_quality=passed_warmup_quality,
        external_quality_evidence=external_quality_evidence,
        quality_decision=quality_decision,
        timed_quality_decision=timed_quality_decision,
        warmup_quality_decision=warmup_quality_decision,
    )
