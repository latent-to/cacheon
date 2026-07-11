import statistics

from optima.eval.throughput_kl import EvalConfig, _external_quality_gate, _measure


def _topk(token: int):
    return [[(-0.01, token, None), (-5.0, token + 1, None)]]


class RecordingEngine:
    def __init__(self):
        self.calls = []

    def generate(self, *, prompt, sampling_params, **kwargs):
        prompts = list(prompt)
        self.calls.append((prompts, dict(kwargs)))
        return [
            {
                "text": f"answer:{value}",
                "output_ids": [index + 10],
                "meta_info": {
                    "completion_tokens": 1,
                    "output_top_logprobs": _topk(index + 10),
                },
            }
            for index, value in enumerate(prompts)
        ]


def test_measure_never_reuses_warmup_or_timed_prompt_and_checks_every_timed_call():
    cfg = EvalConfig(
        model_path="m",
        num_prompts=2,
        max_new_tokens=1,
        warmup_iters=2,
        timed_iters=3,
    )
    batches = [[f"batch-{i}-prompt-{j}" for j in range(2)] for i in range(5)]
    engine = RecordingEngine()

    result = _measure(engine, batches, cfg)

    assert [call[0] for call in engine.calls] == batches
    assert len({prompt for batch in batches for prompt in batch}) == 10
    assert all(kwargs.get("return_logprob") is True for _, kwargs in engine.calls)
    assert len(result.tok_per_s_samples) == 3
    assert result.conditioning_tok_per_s > 0
    assert result.tok_per_s == min(
        statistics.median(result.tok_per_s_samples),
        result.conditioning_tok_per_s,
    )
    assert len(result.per_prompt) == len(result.texts) == 6
    assert len(result.warmup_per_prompt) == len(result.warmup_texts) == 4
    assert len(result.per_prompt_batches) == 3
    assert len(result.warmup_per_prompt_batches) == 2


def test_measure_refuses_a_reused_single_batch_plan():
    cfg = EvalConfig(
        model_path="m",
        num_prompts=1,
        warmup_iters=1,
        conditioning_iters=1,
        timed_iters=2,
    )
    engine = RecordingEngine()
    try:
        _measure(engine, [["same"]], cfg)
    except ValueError as exc:
        assert "expected exactly 3" in str(exc)
    else:
        raise AssertionError("reused/undersized prompt plan was accepted")


def test_external_quality_is_calibrated_against_stock_control():
    from optima.eval.throughput_kl import ModeResult

    def mode(token: int, lp: float = -0.01):
        topk = [[(lp, token, None), (-5.0, token + 1, None)]]
        return ModeResult(1.0, [1.0, 1.0], 1, [([token], topk)])

    baseline = mode(7)
    stock_control = mode(7, -0.02)
    honest = mode(7, -0.03)
    broken = ModeResult(
        100.0,
        [100.0, 100.0],
        1,
        [([99], [[(-0.01, 99, None), (-0.02, 100, None)]])],
    )

    ok, _, _ = _external_quality_gate(baseline, honest, stock_control)
    bad, _, _ = _external_quality_gate(baseline, broken, stock_control)
    assert ok
    assert not bad
