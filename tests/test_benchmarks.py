"""Torch-free tests for the realistic-workload eval pieces.

Covers the answer extractors (numeric + multiple-choice), the GSM8K/MMLU
``check`` logic, the benchmark registry, and the shared KL helpers
(``extract_per_prompt`` / ``aligned_kl``). All pure-Python: no GPU, no torch, no
network (the HF ``load`` paths are exercised on the pod, not here).
"""

from __future__ import annotations

from optima.eval.benchmarks import (
    GSM8K,
    MMLU,
    Problem,
    extract_choice_letter,
    extract_final_number,
    get_benchmark,
    list_benchmarks,
)
from optima.eval.kl import aligned_kl, extract_per_prompt


# ---- numeric extraction (GSM8K / MATH-style) --------------------------------


def test_extract_final_number_answer_cue():
    assert extract_final_number("lots of words. The answer is 42.") == 42.0


def test_extract_final_number_handles_commas_and_dollar():
    assert extract_final_number("So the total is $1,234.50 in the end.") == 1234.5


def test_extract_final_number_falls_back_to_last():
    assert extract_final_number("we add 3 and 4 to get 7") == 7.0


def test_extract_final_number_none_when_no_digits():
    assert extract_final_number("no numbers anywhere here") is None


# ---- multiple-choice extraction (MMLU / GPQA-style) -------------------------


def test_extract_choice_letter_cue_parenthesized():
    assert extract_choice_letter("...therefore The answer is (C).") == "C"


def test_extract_choice_letter_cue_bare():
    assert extract_choice_letter("Answer: B") == "B"


def test_extract_choice_letter_paren_fallback():
    assert extract_choice_letter("I'll go with (D) here") == "D"


def test_extract_choice_letter_out_of_range_is_none():
    # 'E' is beyond a 4-option question, so it must not be returned.
    assert extract_choice_letter("The answer is (E).", num_choices=4) is None


def test_extract_choice_letter_none():
    assert extract_choice_letter("no idea, honestly") is None


# ---- benchmark check logic --------------------------------------------------


def test_gsm8k_check():
    g = GSM8K()
    p = Problem(id="x", prompt="", answer="18")
    assert g.check(p, "work work work. The answer is 18.")
    assert not g.check(p, "work work work. The answer is 17.")


def test_mmlu_format_and_check():
    m = MMLU()
    body = m._format_question("What is 2+2?", ["3", "4", "5", "6"])
    assert "(B) 4" in body and "(A) 3" in body
    p = Problem(id="x", prompt="", answer="B", meta={"num_choices": 4})
    assert m.check(p, "reasoning... The answer is (B).")
    assert not m.check(p, "reasoning... The answer is (A).")


def test_registry_has_gsm8k_and_mmlu():
    assert get_benchmark("gsm8k").name == "gsm8k"
    assert get_benchmark("mmlu").name == "mmlu"
    assert {"gsm8k", "mmlu"} <= set(list_benchmarks())


# ---- shared KL helpers ------------------------------------------------------


def _out(ids, topk):
    """A minimal sglang-shaped generate() output."""
    return {"output_ids": ids, "meta_info": {"output_top_logprobs": topk}}


def test_aligned_kl_zero_on_identical():
    pos = [[(-0.1, 5, None), (-2.0, 9, None)], [(-0.2, 7, None), (-1.0, 3, None)]]
    base = extract_per_prompt([_out([5, 7], pos)])
    rep = aligned_kl(base, base)
    assert rep.num_positions == 2
    assert rep.mean_kl == 0.0
    assert rep.argmax_disagreements == 0


def test_aligned_kl_stops_at_first_divergence_but_scores_position_zero():
    base_pos = [[(-0.1, 5, None), (-2.0, 9, None)], [(-0.2, 7, None), (-1.0, 3, None)]]
    base = extract_per_prompt([_out([5, 7], base_pos)])
    # Candidate flips the very first token (argmax 5 -> 6); later positions must be
    # dropped (different context) but position 0 is still scored with a real KL.
    cand_pos = [[(-2.0, 5, None), (-0.1, 6, None)], [(-0.2, 7, None), (-1.0, 3, None)]]
    cand = extract_per_prompt([_out([6, 7], cand_pos)])
    rep = aligned_kl(base, cand)
    assert rep.num_positions == 1
    assert rep.mean_kl > 0.0
    assert rep.argmax_disagreements == 1
