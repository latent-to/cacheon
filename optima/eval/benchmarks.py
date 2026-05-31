"""Capability benchmarks — real tasks that double as the eval distribution.

The quality gate is NOT just KL distance; it's "did the model's *task
performance* survive the kernel?" We run a small fixed sample of real benchmark
problems and check answers. A kernel that subtly degrades the model will drop
accuracy on these even if KL looks small, and the workload itself stresses the
model the way production does (math/reasoning/agentic), not "what's the date of
US independence".

Tractability tiers (you only need ~5 problems each per epoch):

* **Now (generate -> extract -> check, no execution):** GSM8K (math word
  problems), and the same interface fits AIME/MATH (numeric) and GPQA/MMLU
  (multiple choice). Small models have measurable signal here, so a broken
  kernel visibly collapses the score.
* **Later (need an execution sandbox + a capable model):** SWE-bench Verified,
  Terminal-Bench, LiveCodeBench, KernelBench, Tau-bench. These plug into the
  SAME ``Benchmark`` protocol — only ``check()`` changes (run tests / tools in a
  sandbox instead of regexing a number). That sandbox is also part of the
  isolation layer we need anyway.

This module implements GSM8K end to end and defines the protocol the rest hang
off.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class Problem:
    id: str
    prompt: str  # full text fed to the model (instructions + few-shot + question)
    answer: str  # gold answer, for checking
    meta: dict = field(default_factory=dict)


class Benchmark(Protocol):
    name: str

    def load(self, n: int, seed: int) -> list[Problem]:
        """Deterministically sample n problems for an epoch."""
        ...

    def check(self, problem: Problem, output_text: str) -> bool:
        """Return True if the model's output solves the problem."""
        ...

    @property
    def max_new_tokens(self) -> int:
        ...


# ---------------------------------------------------------------------------
# numeric answer extraction (shared by GSM8K / MATH / AIME-style benchmarks)
# ---------------------------------------------------------------------------

_NUM = re.compile(r"-?\$?\d[\d,]*\.?\d*")


def _to_float(s: str) -> float | None:
    s = s.replace(",", "").replace("$", "").rstrip(".")
    try:
        return float(s)
    except ValueError:
        return None


def extract_final_number(text: str) -> float | None:
    """Pull the model's final numeric answer.

    Prefer the number after an "answer is" cue; else the last number in the
    text. Robust to commas, $, and trailing punctuation.
    """
    m = list(re.finditer(r"(?:answer\s+is|answer:|####)\s*(-?\$?\d[\d,]*\.?\d*)", text, re.IGNORECASE))
    if m:
        return _to_float(m[-1].group(1))
    nums = _NUM.findall(text)
    if nums:
        return _to_float(nums[-1])
    return None


def numbers_equal(a: float | None, b: float | None, tol: float = 1e-4) -> bool:
    return a is not None and b is not None and abs(a - b) <= tol


# ---------------------------------------------------------------------------
# multiple-choice answer extraction (shared by MMLU / GPQA / ARC-style)
# ---------------------------------------------------------------------------


def extract_choice_letter(text: str, num_choices: int = 4) -> str | None:
    """Pull the selected option letter from a chain-of-thought answer.

    Prefer an explicit 'the answer is (X)' / 'answer: X' cue; else fall back to the
    last clearly-delimited '(X)'. Only letters in A..(A+num_choices-1) count, so a
    stray capital in the reasoning isn't misread as the answer.
    """
    hi = chr(ord("A") + max(1, num_choices) - 1)
    cue = re.compile(rf"(?:answer\s+is|answer:|####)\s*\(?([A-{hi}])\)?", re.IGNORECASE)
    m = list(cue.finditer(text))
    if m:
        return m[-1].group(1).upper()
    paren = re.findall(rf"\(([A-{hi}])\)", text)
    return paren[-1].upper() if paren else None


# ---------------------------------------------------------------------------
# GSM8K
# ---------------------------------------------------------------------------


class GSM8K:
    name = "gsm8k"
    _max_new_tokens = 256

    def __init__(self, num_fewshot: int = 4) -> None:
        self.num_fewshot = num_fewshot
        self._fewshot_prefix: str | None = None

    @property
    def max_new_tokens(self) -> int:
        return self._max_new_tokens

    def _gold(self, answer_field: str) -> str:
        # GSM8K gold answer is the number after "####".
        if "####" in answer_field:
            return answer_field.split("####")[-1].strip()
        return answer_field.strip()

    def _build_fewshot(self, train) -> str:
        # Deterministic few-shot from the train split, reformatted to end with
        # "The answer is N." (a cue our extractor keys on).
        rng = random.Random(12345)
        idxs = rng.sample(range(len(train)), self.num_fewshot)
        blocks = []
        for i in idxs:
            ex = train[i]
            cot = ex["answer"].split("####")[0].strip()
            gold = self._gold(ex["answer"])
            blocks.append(f"Question: {ex['question']}\nAnswer: {cot}\nThe answer is {gold}.")
        return "\n\n".join(blocks)

    def load(self, n: int, seed: int) -> list[Problem]:
        from datasets import load_dataset  # lazy; `uv pip install datasets`

        try:
            ds = load_dataset("openai/gsm8k", "main")
        except Exception:  # noqa: BLE001 - fall back to the legacy alias
            ds = load_dataset("gsm8k", "main")
        train, test = ds["train"], ds["test"]
        if self._fewshot_prefix is None:
            self._fewshot_prefix = self._build_fewshot(train)

        rng = random.Random(seed)
        idxs = rng.sample(range(len(test)), min(n, len(test)))
        problems: list[Problem] = []
        for i in idxs:
            ex = test[i]
            prompt = (
                "Solve the math problem. Show your reasoning, then end with "
                "'The answer is N.'\n\n"
                f"{self._fewshot_prefix}\n\n"
                f"Question: {ex['question']}\nAnswer:"
            )
            problems.append(Problem(id=f"gsm8k-{i}", prompt=prompt, answer=self._gold(ex["answer"])))
        return problems

    def check(self, problem: Problem, output_text: str) -> bool:
        return numbers_equal(extract_final_number(output_text), _to_float(problem.answer))


# ---------------------------------------------------------------------------
# MMLU  (knowledge, multiple choice) — prompted for chain-of-thought so it is a
# *long-generation* workload, not a 1-token logit rank. Adds domain diversity to
# the eval distribution next to GSM8K's math.
# ---------------------------------------------------------------------------


class MMLU:
    name = "mmlu"
    _max_new_tokens = 512

    def __init__(self, subject: str = "all") -> None:
        self.subject = subject

    @property
    def max_new_tokens(self) -> int:
        return self._max_new_tokens

    @staticmethod
    def _letter(i: int) -> str:
        return chr(ord("A") + int(i))

    @classmethod
    def _format_question(cls, question: str, choices) -> str:
        opts = "\n".join(f"({cls._letter(j)}) {c}" for j, c in enumerate(choices))
        return f"Question: {question}\n{opts}"

    def load(self, n: int, seed: int) -> list[Problem]:
        from datasets import load_dataset  # lazy; `uv pip install datasets`

        try:
            ds = load_dataset("cais/mmlu", self.subject)
        except Exception:  # noqa: BLE001 - fall back to the legacy alias
            ds = load_dataset("hendrycks_test", self.subject)
        test = ds["test"]

        rng = random.Random(seed)
        idxs = rng.sample(range(len(test)), min(n, len(test)))
        problems: list[Problem] = []
        for i in idxs:
            ex = test[i]
            choices = list(ex["choices"])
            prompt = (
                "Answer the multiple-choice question. Reason step by step, then end "
                "with 'The answer is (X).' where X is the correct option letter.\n\n"
                f"{self._format_question(ex['question'], choices)}\nAnswer:"
            )
            problems.append(
                Problem(
                    id=f"mmlu-{i}",
                    prompt=prompt,
                    answer=self._letter(int(ex["answer"])),
                    meta={"num_choices": len(choices)},
                )
            )
        return problems

    def check(self, problem: Problem, output_text: str) -> bool:
        n = int(problem.meta.get("num_choices", 4))
        got = extract_choice_letter(output_text, num_choices=n)
        return got is not None and got == problem.answer.strip().upper()


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------

BENCHMARKS: dict[str, Benchmark] = {
    "gsm8k": GSM8K(),
    "mmlu": MMLU(),
}


def get_benchmark(name: str) -> Benchmark:
    try:
        return BENCHMARKS[name]
    except KeyError:
        known = ", ".join(sorted(BENCHMARKS)) or "(none)"
        raise KeyError(f"unknown benchmark {name!r}; known: {known}") from None


def list_benchmarks() -> list[str]:
    return sorted(BENCHMARKS)
