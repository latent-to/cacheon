"""Op-correctness — the cheap gate before any end-to-end eval.

Given a slot and a miner ``entry`` callable, generate deterministic inputs over the
slot's standard shapes, run the miner kernel and the trusted *high-precision*
reference, and compare under the slot's ``Correctness`` policy:

* ``allclose`` — every element within ``atol + rtol*|e|`` (numerically-equivalent
  ops, e.g. a faster silu).
* ``matched_ratio`` — at least ``min_ratio`` of elements within that bound (kernels
  that legitimately differ from the reference: attention's reordered softmax, fp8,
  MLA weight absorption). The reference is always high-precision ground truth, never
  the stock kernel — so a faster *and slightly different* kernel can still pass.

Multi-output slots (blocks) are supported: the validator allocates one ``out`` per
declared output shape and the miner fills them.

This is the per-op analogue of a unit test: necessary but NOT sufficient — small
per-op errors that pass here can compound into large end-to-end KL, which is why the
pipeline still runs the end-to-end gate. To stop a kernel from special-casing the fixed
verification inputs, the input VALUES vary with ``seed`` and, when ``jitter_seed`` is
set (the CLI path does this per run), the COUNT dimensions (num_tokens / batch / ctx)
are perturbed too — so a kernel can't hard-code the exact verify shapes. Feature dims
(hidden / head_dim) are left intact since kernels legitimately specialize on them; the
end-to-end gate on fresh prompts is the backstop against shape-branching there.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Protocol

import torch

from optima.capabilities import CallDescriptor, msa_prefill_call_descriptor
from optima.registry import Eligibility
from optima.slots import SlotSpec
from optima.tensor_spec import allocate_output_spec


@dataclass
class ShapeResult:
    shape: dict
    dtype: str
    passed: bool
    max_abs_err: float
    max_rel_err: float
    pass_ratio: float = 1.0  # fraction within tol (matched_ratio) OR cosine (cosine mode); informative
    detail: str = ""
    metric: str = "ratio"  # label for pass_ratio: "ratio" | "cosine"
    # Number of successful CUDA-graph replays checked against the trusted
    # reference for this shape.  Zero means this was an eager-only verification
    # (including every CPU run), not that graph correctness was established.
    graph_replays: int = 0
    # False means the validator proved this catalog shape lies outside the
    # variant's declared domain and did not invoke miner code.
    applicable: bool = True


@dataclass
class VerifyResult:
    slot: str
    dtype: str
    passed: bool
    shape_results: list[ShapeResult]
    # ``passed`` remains the ordinary numerical verdict so CPU verification keeps
    # its historical meaning.  A crown/qualification path for a graph-safe bundle
    # must additionally require ``graph_verified`` whenever ``graph_required`` is
    # true; this prevents a CPU-only run from masquerading as graph proof.
    graph_required: bool = False
    graph_verified: bool = False
    # Zero denotes an unfiltered legacy run.  A positive value is the minimum
    # number of capability-matching shapes the variant had to execute.
    coverage_required: int = 0

    @property
    def num_failed(self) -> int:
        return sum(1 for r in self.shape_results if r.applicable and not r.passed)

    @property
    def num_applicable(self) -> int:
        return sum(1 for r in self.shape_results if r.applicable)

    @property
    def num_not_applicable(self) -> int:
        return len(self.shape_results) - self.num_applicable

    @property
    def coverage_sufficient(self) -> bool:
        return self.coverage_required == 0 or self.num_applicable >= self.coverage_required


def _as_list(x) -> list:
    """Normalize a slot's reference/out_shapes return to a list.

    Accepts a bare tensor or bare shape-tuple (single-output slots may return one
    directly) as well as an explicit sequence (multi-output blocks)."""
    if isinstance(x, (list, tuple)) and (len(x) == 0 or not isinstance(x[0], int)):
        return list(x)
    return [x]


def _compare(
    actual: torch.Tensor, expected: torch.Tensor, *, atol: float, rtol: float, correctness
) -> tuple[bool, float, float, float, str, str]:
    # Returns (passed, max_abs, max_rel, score, detail, metric_label).
    if actual.shape != expected.shape:
        return False, float("inf"), float("inf"), 0.0, f"shape mismatch {tuple(actual.shape)} vs {tuple(expected.shape)}", "ratio"
    a = actual.float()
    e = expected.float()
    if correctness.mode == "topk_overlap":
        # Selection metric: only WHICH top-k are picked matters (not the score values), so
        # this runs BEFORE the finite guard (masked-out positions are legitimately -inf). A
        # buggy kernel emitting NaN -> ranked last (nan_to_num), so it just loses overlap.
        k = correctness.top_k
        ta = torch.nan_to_num(a, nan=float("-inf")).topk(k, dim=-1).indices
        te = e.topk(k, dim=-1).indices
        overlap = (ta.unsqueeze(-1) == te.unsqueeze(-2)).any(dim=-1).float().mean(dim=-1)
        score = float(overlap.mean())
        passed = score >= correctness.min_overlap
        detail = "" if passed else f"topk_overlap {score:.4f} < min_overlap {correctness.min_overlap}"
        return passed, 0.0, 0.0, score, detail, "overlap"
    if not torch.isfinite(a).all():
        return False, float("inf"), float("inf"), 0.0, "actual has non-finite values", "ratio"
    abs_err = (a - e).abs()
    rel_err = abs_err / (e.abs() + 1e-12)
    mode = correctness.mode
    if mode == "cosine":
        # Low-bit fidelity: direction (and optionally energy) vs the HP reference.
        cos = float(torch.nn.functional.cosine_similarity(a.flatten(), e.flatten(), dim=0))
        ne = float(e.flatten().norm())
        rel_norm = abs(float(a.flatten().norm()) - ne) / (ne + 1e-12)
        ok_cos = cos >= correctness.min_cosine
        ok_norm = correctness.max_rel_norm_err <= 0 or rel_norm <= correctness.max_rel_norm_err
        passed = ok_cos and ok_norm
        if passed:
            detail = ""
        elif not ok_cos:
            detail = f"cosine {cos:.5f} < min_cosine {correctness.min_cosine}"
        else:
            detail = f"rel_norm_err {rel_norm:.3f} > {correctness.max_rel_norm_err}"
        return passed, float(abs_err.max()), float(rel_err.max()), cos, detail, "cosine"
    slack = atol + rtol * e.abs()  # allclose: |a-e| <= atol + rtol*|e|
    within = abs_err <= slack
    ratio = float(within.float().mean())
    if mode == "matched_ratio":
        passed = ratio >= correctness.min_ratio
        detail = "" if passed else f"matched {ratio:.4f} < min_ratio {correctness.min_ratio}"
    else:
        passed = bool(within.all())
        detail = ""
    return passed, float(abs_err.max()), float(rel_err.max()), ratio, detail, "ratio"


@dataclass
class _OutputCheck:
    passed: bool
    max_abs: float
    max_rel: float
    min_score: float
    detail: str
    metric: str


def _compare_outputs(outs: list[torch.Tensor], expected: list[torch.Tensor], *, tol,
                     correctness) -> _OutputCheck:
    """Compare every declared output and retain the worst result.

    Kept separate from ``verify_entry`` because CUDA-graph replay must apply the
    exact same comparator as eager verification on every replay.  A different or
    weaker graph comparator would recreate the very eager-vs-captured gap this gate
    is intended to close.
    """
    if len(outs) != len(expected):
        return _OutputCheck(
            False, float("inf"), float("inf"), 0.0,
            f"output count mismatch {len(outs)} vs {len(expected)}", "ratio",
        )

    passed = True
    max_abs = 0.0
    max_rel = 0.0
    min_score_seen = 1.0
    metric = "ratio"
    details: list[str] = []
    for j, (out, reference) in enumerate(zip(outs, expected)):
        p, ma, mr, score, detail, metric = _compare(
            out, reference, atol=tol.atol, rtol=tol.rtol, correctness=correctness
        )
        passed = passed and p
        max_abs = max(max_abs, ma)
        max_rel = max(max_rel, mr)
        min_score_seen = min(min_score_seen, score)
        if detail:
            details.append(f"out[{j}]: {detail}" if len(outs) > 1 else detail)
    return _OutputCheck(
        passed, max_abs, max_rel, min_score_seen, "; ".join(details), metric
    )


class _GraphBackend(Protocol):
    """Small adapter so graph orchestration can be unit-tested without a GPU."""

    def warmup(self, fn: Callable[[], None]) -> None: ...

    def capture(self, fn: Callable[[], None]): ...

    def replay(self, graph) -> None: ...

    def synchronize(self) -> None: ...


class _CudaGraphBackend:
    """Real PyTorch CUDA-graph capture backend.

    Warmup happens on a side stream, as required for graph-safe lazy/JIT kernels,
    before genuine ``torch.cuda.CUDAGraph`` capture.  Candidate Python runs during
    capture but not replay, which is load-bearing: a branch on
    ``torch.cuda.is_current_stream_capturing()`` is frozen into the graph and its
    captured behavior is what the replay comparisons grade.
    """

    def warmup(self, fn: Callable[[], None]) -> None:
        current = torch.cuda.current_stream()
        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(current)
        with torch.cuda.stream(warmup_stream):
            fn()
        current.wait_stream(warmup_stream)
        torch.cuda.synchronize()

    def capture(self, fn: Callable[[], None]):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fn()
        return graph

    def replay(self, graph) -> None:
        graph.replay()

    def synchronize(self) -> None:
        torch.cuda.synchronize()


_DEFAULT_GRAPH_REPLAYS = 3


def _poison_outputs(outs: list[torch.Tensor], replay: int) -> None:
    """Overwrite outputs before each replay so a partial/no-op graph cannot pass.

    The poison is intentionally changed per replay for integral outputs.  Floating
    outputs use NaN, which every comparator rejects unless the replay overwrites the
    relevant cells (``topk_overlap`` converts NaN to ``-inf`` and therefore still
    penalizes an untouched score sheet).
    """
    with torch.no_grad():
        for out in outs:
            if out.dtype.is_floating_point or out.dtype.is_complex:
                out.fill_(float("nan"))
            elif out.dtype == torch.bool:
                out.fill_(bool(replay % 2))
            else:
                info = torch.iinfo(out.dtype)
                out.fill_(info.max - (replay % 17))


@dataclass
class _GraphCheck:
    check: _OutputCheck
    replays: int


def _verify_graph_replays(
    slot: SlotSpec,
    entry: Callable[..., None],
    inputs: dict,
    outs: list[torch.Tensor],
    prepared,
    expected: list[torch.Tensor],
    *,
    tol,
    replay_count: int = _DEFAULT_GRAPH_REPLAYS,
    backend: Optional[_GraphBackend] = None,
) -> _GraphCheck:
    """Capture the candidate once and grade multiple genuine graph replays.

    ``backend`` is injectable solely to exercise the orchestration with CPU tensors
    in unit tests.  Production callers omit it and therefore always use
    ``torch.cuda.CUDAGraph``.
    """
    if replay_count < 2:
        raise ValueError("CUDA graph verification requires at least two replays")
    graph_backend = backend or _CudaGraphBackend()

    def invoke() -> None:
        slot.invoke_entry(entry, inputs, outs, prepared)

    try:
        graph_backend.warmup(invoke)
    except Exception as exc:  # noqa: BLE001 - candidate warmup failure is a verdict
        return _GraphCheck(
            _OutputCheck(False, float("inf"), float("inf"), 0.0,
                         f"cuda graph warmup raised: {type(exc).__name__}: {exc}", "ratio"),
            0,
        )
    try:
        graph = graph_backend.capture(invoke)
    except Exception as exc:  # noqa: BLE001 - a graph_safe claim must actually capture
        return _GraphCheck(
            _OutputCheck(False, float("inf"), float("inf"), 0.0,
                         f"cuda graph capture raised: {type(exc).__name__}: {exc}", "ratio"),
            0,
        )

    max_abs = 0.0
    max_rel = 0.0
    min_score = 1.0
    metric = "ratio"
    completed = 0
    for replay in range(replay_count):
        try:
            _poison_outputs(outs, replay)
            # Be explicit about the poison-before-replay happens-before edge.  This
            # is a correctness gate, not a benchmark; the synchronization is desired.
            graph_backend.synchronize()
            graph_backend.replay(graph)
            graph_backend.synchronize()
        except Exception as exc:  # noqa: BLE001 - replay failure is a failed claim
            return _GraphCheck(
                _OutputCheck(
                    False, float("inf"), float("inf"), 0.0,
                    f"cuda graph replay[{replay}] raised: {type(exc).__name__}: {exc}",
                    "ratio",
                ),
                completed,
            )

        completed = replay + 1
        current = _compare_outputs(outs, expected, tol=tol, correctness=slot.correctness)
        max_abs = max(max_abs, current.max_abs)
        max_rel = max(max_rel, current.max_rel)
        min_score = min(min_score, current.min_score)
        metric = current.metric
        if not current.passed:
            detail = current.detail or "output mismatch"
            return _GraphCheck(
                _OutputCheck(False, max_abs, max_rel, min_score,
                             f"cuda graph replay[{replay}]: {detail}", metric),
                completed,
            )

    return _GraphCheck(
        _OutputCheck(True, max_abs, max_rel, min_score, "", metric), completed
    )


# Count-like shape keys safe to jitter (varying these doesn't break a kernel that
# legitimately specializes on the feature dims like hidden / head_dim / inter).
_JITTER_KEYS = ("num_tokens", "batch", "ctx", "q_len", "prefix_blocks")


def _jitter_shapes(shapes: list[dict], seed: int) -> list[dict]:
    """Perturb the count dimensions of each shape deterministically from ``seed`` so the
    verify shapes vary per run — a kernel can't hard-code the exact verification token
    counts. Feature dims are untouched; counts stay >= 1."""
    import random

    rng = random.Random(seed)
    out: list[dict] = []
    for sh in shapes:
        s = dict(sh)
        for k in _JITTER_KEYS:
            if k in s and isinstance(s[k], int):
                s[k] = max(1, s[k] + rng.randint(-1, 3) + (s[k] // 3) * rng.randint(0, 1))
        out.append(s)
    return out


def verify_entry(
    slot: SlotSpec,
    entry: Callable[..., None],
    *,
    prepare: Optional[Callable] = None,
    dtype: torch.dtype = torch.bfloat16,
    device: Optional[str] = None,
    seed: int = 0,
    shapes: Optional[list[dict]] = None,
    jitter_seed: Optional[int] = None,
    graph_safe: Optional[bool] = None,
    graph_replays: int = _DEFAULT_GRAPH_REPLAYS,
    eligibility: Optional[Eligibility] = None,
    architecture: Optional[str] = None,
    tp_size: Optional[int] = None,
    world_size: Optional[int] = None,
    _graph_backend: Optional[_GraphBackend] = None,
) -> VerifyResult:
    """Verify a miner ``entry`` against the slot's reference.

    ``entry`` is called via ``slot.invoke_entry(entry, inputs, outs, prepared)`` and
    must write its result into the validator-allocated tensors in ``outs``. For a
    *(prepare, forward)* slot (``slot.prepare`` set, e.g. ``moe.fused_experts``) pass
    the miner's ``prepare`` callable too — it runs once on the raw weights and its
    result is handed to ``entry`` as ``prepared`` (otherwise ``prepared`` is None).

    On CUDA, op slots are graph-verified by default because their serving seam is
    always captured.  Block slots are graph-verified when the caller passes their
    declared ``graph_safe=True`` metadata.  CPU runs retain the eager numerical gate
    but return ``graph_required=True, graph_verified=False`` when graph proof was
    requested.  With ``eligibility``, validator code describes every generated
    call before invocation: off-domain shapes are reported N/A without entering
    miner code, and ``slot.min_capability_shapes`` must remain in-domain.
    ``_graph_backend`` is a private CPU-test hook; production must omit it.
    """
    if getattr(slot, "kind", None) == "collective":
        raise ValueError(
            f"slot {slot.name!r} is a collective slot — verify it distributed with "
            "optima.verify_collective.verify_collective, not the single-process verify_entry"
        )
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    graph_required = slot.kind == "op" if graph_safe is None else bool(graph_safe)
    graph_capable_run = str(device).startswith("cuda") or _graph_backend is not None
    tol = slot.tolerance_for(dtype)
    test_shapes = shapes if shapes is not None else list(slot.shapes)
    if jitter_seed is not None:
        test_shapes = _jitter_shapes(test_shapes, jitter_seed)

    results: list[ShapeResult] = []
    for i, shape in enumerate(test_shapes):
        inputs = slot.make_inputs(dtype=dtype, device=device, seed=seed + i, **shape)
        if eligibility is not None:
            descriptor = _verification_call_descriptor(
                slot,
                inputs,
                dtype=dtype,
                device=device,
                architecture=architecture,
                tp_size=tp_size,
                world_size=world_size,
            )
            match = eligibility.match(descriptor)
            if not match.accepted:
                reasons = "; ".join(
                    f"{m.field} {m.reason}: expected {m.expected}"
                    + ("" if m.actual is None else f", got {m.actual!r}")
                    for m in match.mismatches
                )
                results.append(ShapeResult(
                    shape=shape,
                    dtype=_name(dtype),
                    passed=True,
                    max_abs_err=0.0,
                    max_rel_err=0.0,
                    detail=f"validator N/A (outside declared capability domain): {reasons}",
                    metric="n/a",
                    applicable=False,
                ))
                continue
        expected = _as_list(slot.invoke_reference(inputs))
        # Allocate from the same typed contract used by the live arena binding.
        # Legacy slots resolve ``out_shapes`` to inherited-dtype contiguous tensors,
        # exactly preserving their historical behavior.
        allocation = allocate_output_spec(
            slot.output_contract(inputs),
            fallback_dtype=dtype,
            fallback_device=device,
            inputs=(v for v in inputs.values() if torch.is_tensor(v)),
        )
        outs = allocation.outputs
        try:
            prepared = None
            if slot.invoke_prepare is not None:
                if prepare is None:
                    raise RuntimeError(
                        f"slot {slot.name!r} is a (prepare, forward) slot but no 'prepare' callable was provided"
                    )
                prepared = slot.invoke_prepare(prepare, inputs)  # runs the miner's weight-prep
            slot.invoke_entry(entry, inputs, outs, prepared)
        except Exception as exc:  # noqa: BLE001 - report kernel failure as a fail
            results.append(
                ShapeResult(shape=shape, dtype=_name(dtype), passed=False,
                            max_abs_err=float("inf"), max_rel_err=float("inf"), pass_ratio=0.0,
                            detail=f"kernel raised: {type(exc).__name__}: {exc}")
            )
            continue

        eager = _compare_outputs(outs, expected, tol=tol, correctness=slot.correctness)
        passed = eager.passed
        max_abs = eager.max_abs
        max_rel = eager.max_rel
        min_score_seen = eager.min_score
        metric = eager.metric
        details = [eager.detail] if eager.detail else []
        checked_replays = 0

        # Do not attempt capture after an eager mismatch: it cannot rescue the
        # candidate, and some broken kernels leave state that only obscures the root
        # error.  Every eager-correct graph-required GPU shape must capture and replay.
        if passed and graph_required and graph_capable_run:
            graph = _verify_graph_replays(
                slot, entry, inputs, outs, prepared, expected, tol=tol,
                replay_count=graph_replays, backend=_graph_backend,
            )
            checked_replays = graph.replays
            passed = passed and graph.check.passed
            max_abs = max(max_abs, graph.check.max_abs)
            max_rel = max(max_rel, graph.check.max_rel)
            min_score_seen = min(min_score_seen, graph.check.min_score)
            metric = graph.check.metric
            if graph.check.detail:
                details.append(graph.check.detail)
        results.append(
            ShapeResult(shape=shape, dtype=_name(dtype), passed=passed,
                        max_abs_err=max_abs, max_rel_err=max_rel, pass_ratio=min_score_seen,
                        detail="; ".join(details), metric=metric,
                        graph_replays=checked_replays)
        )

    applicable = [result for result in results if result.applicable]
    coverage_required = (
        max(1, int(slot.min_capability_shapes)) if eligibility is not None else 0
    )
    coverage_sufficient = (
        len(applicable) >= coverage_required if coverage_required else bool(applicable)
    )
    graph_verified = bool(
        graph_required and graph_capable_run and applicable
        and all(r.passed and r.graph_replays == graph_replays for r in applicable)
    )
    return VerifyResult(
        slot=slot.name,
        dtype=_name(dtype),
        passed=coverage_sufficient and all(r.passed for r in applicable),
        shape_results=results,
        graph_required=graph_required,
        graph_verified=graph_verified,
        coverage_required=coverage_required,
    )


def _verification_call_descriptor(
    slot: SlotSpec,
    inputs: dict,
    *,
    dtype: torch.dtype,
    device: str,
    architecture: Optional[str],
    tp_size: Optional[int],
    world_size: Optional[int],
) -> CallDescriptor:
    """Build the same canonical call description as a live arena binding.

    Fields are semantic and therefore never guessed from vaguely similar tensor
    names.  A slot needs an explicit validator mapping before miners can constrain
    its richer dimensions; MSA prefill is the first migrated binding.
    """

    resolved_arch = architecture or _device_architecture(device)
    if slot.name != "attention.msa_prefill_block_score":
        return CallDescriptor(dtype=_name(dtype), architecture=resolved_arch)
    q = inputs["q"]
    index_k = inputs["index_k"]
    return msa_prefill_call_descriptor(
        dtype=_name(q.dtype),
        architecture=resolved_arch,
        head_dim=int(q.shape[-1]),
        block_size=int(inputs["block_size"]),
        q_len=int(q.shape[0]),
        kv_len=int(index_k.shape[0]),
        top_k=int(slot.correctness.top_k),
        num_kv_heads=1,
        tp_size=tp_size,
        world_size=world_size,
    )


def _device_architecture(device: str) -> Optional[str]:
    resolved = torch.device(device)
    if resolved.type != "cuda" or not torch.cuda.is_available():
        return None
    index = resolved.index
    if index is None:
        index = torch.cuda.current_device()
    major, minor = torch.cuda.get_device_capability(index)
    return f"sm{major}{minor}"


def verify_entry_from_source(
    slot_name: str,
    source_path: str,
    entry_name: str,
    *,
    prepare_name: Optional[str] = None,
    dtype_name: str = "bfloat16",
    device: Optional[str] = None,
    seed: int = 0,
    shapes: Optional[list[dict]] = None,
    jitter_seed: Optional[int] = None,
    model_key: Optional[str] = None,
    override_point: Optional[str] = None,
    graph_safe: Optional[bool] = None,
    graph_replays: int = _DEFAULT_GRAPH_REPLAYS,
    eligibility_metadata: Optional[dict] = None,
    manifest_dtypes: tuple[str, ...] = (),
    manifest_architectures: tuple[str, ...] = (),
) -> VerifyResult:
    """Load the miner module and verify it — module-level + picklable so the CLI can run
    it via ``call_in_subprocess`` in a FRESH process. This keeps the trusted validator/CLI
    process from ever importing miner code (import-time payloads + the kernel run only in the
    throwaway child). It is NOT a security boundary by itself — production still needs the
    child namespaced/no-egress — but it removes the in-process-RCE-in-the-CLI sink (#6).

    ``model_key`` (a validator/model fact, e.g. ``"MiniMax-M3"``) selects the per-model slot
    specialization (right activation reference + low-bit metric). None -> the generic slot.
    ``override_point`` (an override submission) composes the miner's epilogue into the
    validator-owned base kernel instead of loading a whole-kernel ``entry``."""
    from optima.sandbox import callable_from, load_module
    from optima.slots import slot_for_model

    slot = slot_for_model(slot_name, model_key)
    dtype = getattr(torch, dtype_name)
    # ONE module instance: entry/prepare (or an override's device fns) must share a
    # namespace — separate load_entry calls re-execute the body per callable.
    module = load_module(source_path)  # runs the miner module body — in THIS child
    if override_point is not None:
        from optima_kernels.override import build_override

        def _loader(name, _mod=module):
            fn = getattr(_mod, name, None)
            return fn if callable(fn) else None  # absent symbol (GPU-only device fn) -> None

        entry, prepare = build_override(slot_name, override_point, entry_name, _loader)
    else:
        entry = callable_from(module, entry_name)
        prepare = callable_from(module, prepare_name) if prepare_name else None
    eligibility = None
    if eligibility_metadata is not None:
        from optima.registry import eligibility_from_metadata

        eligibility = eligibility_from_metadata(
            eligibility_metadata, manifest_dtypes, manifest_architectures
        )
    return verify_entry(slot, entry, prepare=prepare, dtype=dtype, device=device, seed=seed,
                        shapes=shapes, jitter_seed=jitter_seed, graph_safe=graph_safe,
                        graph_replays=graph_replays, eligibility=eligibility)


def _name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def format_verify(result: VerifyResult) -> str:
    graph = " graph=not-required"
    if result.graph_required:
        graph = " graph=verified" if result.graph_verified else " graph=NOT_VERIFIED"
    coverage = ""
    if result.coverage_required or result.num_not_applicable:
        coverage = (
            f" coverage={result.num_applicable}/{result.coverage_required}"
            f" n/a={result.num_not_applicable}"
        )
    lines = [
        f"[{'PASS' if result.passed else 'FAIL'}] {result.slot} "
        f"dtype={result.dtype}{graph}{coverage}"
    ]
    for r in result.shape_results:
        status = "N/A" if not r.applicable else ("ok " if r.passed else "FAIL")
        if r.metric == "cosine":
            score = f" cos={r.pass_ratio:.5f}"
        elif r.metric == "overlap":
            score = f" overlap={r.pass_ratio:.4f}"
        elif r.metric == "n/a":
            score = ""
        else:
            score = "" if r.pass_ratio >= 1.0 else f" ratio={r.pass_ratio:.4f}"
        replay = f" graph_replays={r.graph_replays}" if r.graph_replays else ""
        lines.append(
            f"  {status} shape={r.shape} max_abs={r.max_abs_err:.3e} max_rel={r.max_rel_err:.3e}{score}"
            + replay + (f"  {r.detail}" if r.detail else "")
        )
    return "\n".join(lines)
