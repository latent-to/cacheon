from types import SimpleNamespace

from optima.eval.throughput_kl import (
    EvalConfig,
    ModeResult,
    _external_quality_gate,
    effective_fidelity_mode,
    evaluate,
    is_transient_launch_failure,
)


def test_framework_fidelity_overrides_requested_audit():
    cfg = SimpleNamespace(framework_mode=True, fidelity_mode="audit")

    assert effective_fidelity_mode(cfg) == "framework"


def test_audit_lane_remains_available_for_component_diagnostics():
    cfg = SimpleNamespace(framework_mode=False, fidelity_mode="audit")

    assert effective_fidelity_mode(cfg) == "audit"


def test_kl_is_default_component_fidelity():
    cfg = SimpleNamespace(framework_mode=False, fidelity_mode="kl")

    assert effective_fidelity_mode(cfg) == "kl"


def test_only_infrastructure_launch_failures_are_retried():
    assert is_transient_launch_failure(RuntimeError("CUDA out of memory in KV cache pool"))
    assert is_transient_launch_failure(RuntimeError("scheduler process terminated unexpectedly"))
    assert not is_transient_launch_failure(RuntimeError("candidate engine run failed execution coverage"))
    assert not is_transient_launch_failure(RuntimeError("launch subprocess timed out after 10s"))
    assert not is_transient_launch_failure(RuntimeError("kernel raised: wrong answer"))


def test_production_oci_audit_profile_runs_exactly_b_c_bprime():
    topk = [[(-0.01, 7, None), (-5.0, 8, None)]]
    result = ModeResult(
        tok_per_s=100.0,
        tok_per_s_samples=[100.0, 100.0],
        tokens=2,
        per_prompt=[([7], topk), ([7], topk)],
        texts=["ok", "ok"],
        per_prompt_batches=[[([7], topk)], [([7], topk)]],
        warmup_per_prompt=[([7], topk)],
        warmup_texts=["ok"],
        warmup_per_prompt_batches=[[([7], topk)]],
    )

    class Launcher:
        def __init__(self):
            self.modes = []
            self.result = result

        def begin_evaluation(self):
            pass

        def run(self, cfg, prompt_batches, *, mode, arm=None):
            self.modes.append(mode)
            return self.result

    launcher = Launcher()
    cfg = EvalConfig(
        model_path="m",
        num_prompts=1,
        max_new_tokens=1,
        warmup_iters=1,
        conditioning_iters=1,
        timed_iters=2,
        fidelity_mode="audit",
        prompt_seed=123,
    )
    report = evaluate(cfg, "unused-bundle", oci_launcher=launcher)

    assert launcher.modes == ["baseline", "candidate", "baseline"]
    assert report.passed_quality
    assert "skipped in production OCI bracket" in report.audit_desc


def test_production_system_profile_overrides_audit_and_gates_emitted_tokens():
    topk = [[(-0.01, 7, None), (-5.0, 8, None)]]
    baseline = ModeResult(
        tok_per_s=100.0,
        tok_per_s_samples=[100.0, 100.0],
        tokens=2,
        per_prompt=[([7], topk), ([7], topk)],
        texts=["stock", "stock"],
        per_prompt_batches=[[([7], topk)], [([7], topk)]],
        warmup_per_prompt=[([7], topk)],
        warmup_texts=["stock"],
        warmup_per_prompt_batches=[[([7], topk)]],
    )
    # Keep the reported top-k distributions identical so the controller-side
    # external KL envelope passes.  Only the model-consumed output IDs differ.
    candidate = ModeResult(
        tok_per_s=110.0,
        tok_per_s_samples=[110.0, 110.0],
        tokens=2,
        per_prompt=[([8], topk), ([8], topk)],
        texts=["candidate", "candidate"],
        per_prompt_batches=[[([8], topk)], [([8], topk)]],
        warmup_per_prompt=[([8], topk)],
        warmup_texts=["candidate"],
        warmup_per_prompt_batches=[[([8], topk)]],
    )

    class Launcher:
        def __init__(self):
            self.modes = []
            self.results = iter((baseline, candidate, baseline))

        def begin_evaluation(self):
            pass

        def run(self, cfg, prompt_batches, *, mode, arm=None):
            self.modes.append(mode)
            return next(self.results)

    launcher = Launcher()
    cfg = EvalConfig(
        model_path="m",
        num_prompts=1,
        max_new_tokens=1,
        warmup_iters=1,
        conditioning_iters=1,
        timed_iters=2,
        fidelity_mode="audit",
        framework_mode=True,
        token_match_threshold=1.0,
        prompt_seed=123,
    )
    report = evaluate(cfg, "unused-bundle", oci_launcher=launcher)

    assert launcher.modes == ["baseline", "candidate", "baseline"]
    assert report.fidelity_mode == "framework"
    assert not report.passed_quality
    assert report.score == 0.0
    assert report.external_quality_desc.count("output_token_match=0/1") == 3
    assert "timed_output_tokens" in report.external_quality_desc
    assert "warmup_output_tokens" in report.external_quality_desc


def test_framework_bracket_accepts_stock_sampled_token_raw_topk_argmax_mismatch():
    # Stock MiniMax-M3 may report raw output_top_logprobs before sampler
    # transforms, so its model-consumed token need not be that raw top-k argmax.
    topk = [[(-0.8, 8, None), (-1.0, 7, None)]]

    def mode(rate: float) -> ModeResult:
        run = ([7], topk)
        return ModeResult(
            tok_per_s=rate,
            tok_per_s_samples=[rate, rate],
            tokens=2,
            per_prompt=[run, run],
            conditioning_tok_per_s=rate,
            texts=["", ""],
            per_prompt_batches=[[run], [run]],
            warmup_per_prompt=[run],
            warmup_texts=[""],
            warmup_per_prompt_batches=[[run]],
        )

    class Launcher:
        def __init__(self):
            self.results = iter((mode(100.0), mode(110.0), mode(100.0)))

        def begin_evaluation(self):
            pass

        def run(self, cfg, prompt_batches, *, mode, arm=None):
            return next(self.results)

    report = evaluate(
        EvalConfig(
            model_path="m",
            num_prompts=1,
            max_new_tokens=1,
            warmup_iters=1,
            conditioning_iters=1,
            timed_iters=2,
            framework_mode=True,
            token_match_threshold=1.0,
            fidelity_mode="audit",
            prompt_seed=123,
        ),
        "unused-bundle",
        oci_launcher=Launcher(),
    )

    assert report.passed_timed_quality
    assert report.passed_warmup_quality
    assert report.passed_quality


def test_component_audit_cannot_crown_corrupt_tokens_with_identical_raw_topk():
    topk_position = [(-0.01, 7, None), (-5.0, 8, None)]

    def run(*, wrong: int = 0):
        ids = [8] * wrong + [7] * (100 - wrong)
        return ids, [list(topk_position) for _ in range(100)]

    def mode(*, wrong: int, rate: float) -> ModeResult:
        timed = [[run(wrong=wrong)] for _ in range(2)]
        warmup = [[run()]]
        return ModeResult(
            tok_per_s=rate,
            tok_per_s_samples=[rate, rate],
            tokens=200,
            per_prompt=[item for batch in timed for item in batch],
            conditioning_tok_per_s=rate,
            texts=["", ""],
            per_prompt_batches=timed,
            warmup_per_prompt=[item for batch in warmup for item in batch],
            warmup_texts=[""],
            warmup_per_prompt_batches=warmup,
        )

    class Launcher:
        def __init__(self):
            self.results = iter((
                mode(wrong=0, rate=100.0),
                mode(wrong=5, rate=110.0),
                mode(wrong=0, rate=100.0),
            ))

        def begin_evaluation(self):
            pass

        def run(self, cfg, prompt_batches, *, mode, arm=None):
            return next(self.results)

    report = evaluate(
        EvalConfig(
            model_path="m",
            num_prompts=1,
            max_new_tokens=100,
            warmup_iters=1,
            conditioning_iters=1,
            timed_iters=2,
            fidelity_mode="audit",
            token_match_threshold=0.99,
            prompt_seed=123,
        ),
        "unused-bundle",
        oci_launcher=Launcher(),
    )

    assert not report.passed_timed_quality
    assert not report.passed_quality
    assert report.score == 0.0
    assert "output_token_match=95/100" in report.external_quality_desc


def test_correct_warmups_cannot_subsidize_corrupt_timed_system_batches():
    topk_position = [(-0.01, 7, None), (-5.0, 8, None)]

    def prompt_run(*, wrong: int = 0):
        ids = [8] * wrong + [7] * (100 - wrong)
        return (ids, [list(topk_position) for _ in range(100)])

    def mode(*, timed_wrong: int):
        timed_batches = [[prompt_run(wrong=timed_wrong)] for _ in range(3)]
        warmup_batches = [[prompt_run()] for _ in range(3)]
        return ModeResult(
            tok_per_s=110.0 if timed_wrong else 100.0,
            tok_per_s_samples=[110.0 if timed_wrong else 100.0] * 3,
            tokens=300,
            per_prompt=[run for batch in timed_batches for run in batch],
            texts=[""] * 3,
            per_prompt_batches=timed_batches,
            warmup_per_prompt=[run for batch in warmup_batches for run in batch],
            warmup_texts=[""] * 3,
            warmup_per_prompt_batches=warmup_batches,
        )

    baseline = mode(timed_wrong=0)
    candidate = mode(timed_wrong=2)  # 98% in every timed batch; warmup is 100%.

    class Launcher:
        def __init__(self):
            self.results = iter((baseline, candidate, baseline))

        def begin_evaluation(self):
            pass

        def run(self, cfg, prompt_batches, *, mode, arm=None):
            return next(self.results)

    report = evaluate(
        EvalConfig(
            model_path="m",
            num_prompts=1,
            max_new_tokens=100,
            warmup_iters=3,
            timed_iters=3,
            fidelity_mode="audit",
            framework_mode=True,
            token_match_threshold=0.99,
            prompt_seed=123,
        ),
        "unused-bundle",
        oci_launcher=Launcher(),
    )

    # The old aggregate was exactly 99%: three clean warmups plus three 98% timed
    # batches. Timed evidence is now its own authority and every timed batch fails.
    assert report.passed_warmup_quality
    assert not report.passed_timed_quality
    assert not report.passed_quality and report.score == 0.0
    assert report.token_matches == 294 and report.token_total == 300
    assert report.warmup_token_matches == report.warmup_token_total == 300


def test_clean_batches_cannot_dilute_one_bad_timed_topk_batch():
    clean = [(-0.1053605, 7, None), (-2.3025851, 8, None)]  # 90% / 10%
    shifted = [(-0.3566749, 7, None), (-1.2039728, 8, None)]  # 70% / 30%

    def prompt_run(*, corrupt: bool = False):
        topk = shifted if corrupt else clean
        return [7] * 100, [list(topk) for _ in range(100)]

    def mode(*, bad_batch: bool, rate: float):
        timed_batches = [
            [prompt_run(corrupt=bad_batch and index == 0)]
            for index in range(3)
        ]
        warmup_batches = [[prompt_run()] for _ in range(3)]
        return ModeResult(
            tok_per_s=rate,
            tok_per_s_samples=[rate] * 3,
            tokens=300,
            per_prompt=[run for batch in timed_batches for run in batch],
            texts=[""] * 3,
            per_prompt_batches=timed_batches,
            warmup_per_prompt=[run for batch in warmup_batches for run in batch],
            warmup_texts=[""] * 3,
            warmup_per_prompt_batches=warmup_batches,
        )

    baseline = mode(bad_batch=False, rate=100.0)
    candidate = mode(bad_batch=True, rate=110.0)

    class Launcher:
        def __init__(self):
            self.results = iter((baseline, candidate, baseline))

        def begin_evaluation(self):
            pass

        def run(self, cfg, prompt_batches, *, mode, arm=None):
            return next(self.results)

    report = evaluate(
        EvalConfig(
            model_path="m",
            num_prompts=1,
            max_new_tokens=100,
            warmup_iters=3,
            timed_iters=3,
            fidelity_mode="audit",
            prompt_seed=123,
        ),
        "unused-bundle",
        oci_launcher=Launcher(),
    )

    # The shifted first batch has KL ~=0.116 and fails V1's 0.05 limit. Flattening
    # it with two exact batches yields ~=0.039 and passed the old aggregate gate.
    flat_baseline = ModeResult(100.0, [100.0] * 3, 300, baseline.per_prompt)
    flat_candidate = ModeResult(110.0, [110.0] * 3, 300, candidate.per_prompt)
    aggregate_ok, _, _ = _external_quality_gate(
        flat_baseline, flat_candidate, flat_baseline
    )
    assert aggregate_ok
    assert report.passed_warmup_quality
    assert not report.passed_timed_quality
    assert not report.passed_quality and report.score == 0.0
    assert "batch1" in report.external_quality_desc
