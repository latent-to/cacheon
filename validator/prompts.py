"""Block-hash-seeded prompt sampling from PG19 with instruction templates.

Prompts are deterministic: same block hash always produces the same set.
Passages are sampled from PG19 (Project Gutenberg books), paired with
instruction templates, and wrapped as OpenAI chat messages.

PG19 must be pre-downloaded to the local HuggingFace cache. The validator
fails fast at startup if the dataset is missing rather than attempting an
11 GB download mid-evaluation.
"""

from __future__ import annotations

import hashlib
import logging
import random
from typing import Sequence

from .eval_schema import ChatMessage, Prompt

logger = logging.getLogger(__name__)

PROMPT_ENGINE_VERSION: int = 1

DATASET_NAME: str = "emozilla/pg19"
DATASET_SPLIT: str = "train"

MIN_CONTEXT_CHARS: int = 16_000
MAX_CONTEXT_CHARS: int = 131_072
MAX_SAMPLE_ATTEMPTS: int = 1_000

CHARS_PER_TOKEN: float = 3.2
"""Conservative chars/token for English prose with the Qwen2.5 tokenizer.
Real average is ~3.5-4.0; using 3.2 so we slightly underestimate chars per
token and stay safely within the context budget."""

OVERHEAD_TOKENS: int = 300
"""Tokens consumed by the chat template, instruction text, and special tokens.
The instruction templates are short (~30-50 tokens) but the chat template
wraps the message with role markers, BOS/EOS, etc."""

MAX_OUTPUT_TOKENS: int = 256
"""Output tokens requested per prompt (must match Prompt.max_tokens)."""

MIN_ALPHA_RATIO: float = 0.5
MAX_WHITESPACE_RATIO: float = 0.35

TEMPLATES: list[str] = [
    "Summarize the following passage in 5 concise bullet points:\n\n{context}",
    "List the main named entities in the passage, grouped by person, place, and organization:\n\n{context}",
    "Identify the central conflict or tension in this passage and explain it in 3 paragraphs:\n\n{context}",
    "Create a chronological timeline of the key events described in this text:\n\n{context}",
    "Describe the relationships between the main characters mentioned in this passage:\n\n{context}",
    "Extract five important facts from the passage. Use only information present in the text:\n\n{context}",
    "Analyze the writing style of this passage, focusing on tone, pacing, and point of view:\n\n{context}",
    "State the central theme of this passage and cite three short supporting phrases:\n\n{context}",
    "Generate ten comprehension questions and answers based only on this passage:\n\n{context}",
    "Explain the setting of this passage and how it influences the events described:\n\n{context}",
    "What are the main arguments or positions presented in the passage? List them:\n\n{context}",
    "Rewrite the key points of this passage as a structured outline with headings:\n\n{context}",
    "Identify the emotional tone of this passage and explain what textual evidence supports it:\n\n{context}",
    "Compare and contrast any two viewpoints or characters presented in this text:\n\n{context}",
    "Write a brief abstract for this passage, suitable for a library catalog entry:\n\n{context}",
    "What assumptions does the author make in this passage? List and explain each one:\n\n{context}",
    "Identify any cause-and-effect relationships described in this passage:\n\n{context}",
    "Explain what happens in this passage as if describing it to someone who has not read it:\n\n{context}",
    "List the most important vocabulary words in this passage and define each in context:\n\n{context}",
    "Based on the following text, answer: what is the author's purpose in writing this?\n\n{context}",
]


_pg19_cache: list[str] | None = None


def _load_pg19() -> list[str]:
    """Load PG19 train split from local HuggingFace cache.

    Raises RuntimeError with a helpful message if the dataset is not
    cached locally.
    """
    global _pg19_cache
    if _pg19_cache is not None:
        return _pg19_cache

    try:
        from datasets import load_dataset
    except ImportError:
        raise RuntimeError(
            "The 'datasets' package is required for prompt sampling. "
            "Install it with: pip install datasets"
        )

    try:
        ds = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load PG19. If the dataset is not cached locally, "
            f"pre-download it with:\n"
            f'  python -c "from datasets import load_dataset; '
            f"load_dataset('{DATASET_NAME}', split='{DATASET_SPLIT}')\"\n"
            f"Original error: {exc}"
        ) from exc

    rows = [row["text"] for row in ds if isinstance(row.get("text"), str)]
    if not rows:
        raise RuntimeError("PG19 loaded but contains no valid text rows.")

    _pg19_cache = rows
    logger.info("PG19 loaded: %d rows", len(rows))
    return rows


def _is_valid_passage(text: str) -> bool:
    """Cheap quality filter for PG19 passages."""
    if len(text) < MIN_CONTEXT_CHARS:
        return False
    alpha_count = sum(c.isalpha() for c in text)
    ws_count = sum(c.isspace() for c in text)
    total = len(text)
    if total == 0:
        return False
    if alpha_count / total < MIN_ALPHA_RATIO:
        return False
    if ws_count / total > MAX_WHITESPACE_RATIO:
        return False
    return True


def max_passage_chars(max_context_tokens: int | None = None) -> int:
    """Compute the safe passage character limit for a given context window.

    ``max_context_tokens`` is the vLLM ``--max-model-len`` value.  We
    subtract output tokens and overhead, then convert the remaining token
    budget to characters using the conservative CHARS_PER_TOKEN estimate.

    Falls back to MAX_CONTEXT_CHARS when no token limit is provided.
    """
    if max_context_tokens is None:
        return MAX_CONTEXT_CHARS
    passage_tokens = max_context_tokens - MAX_OUTPUT_TOKENS - OVERHEAD_TOKENS
    return max(MIN_CONTEXT_CHARS, int(passage_tokens * CHARS_PER_TOKEN))


def _sample_passage(
    rng: random.Random,
    rows: Sequence[str],
    max_chars: int = MAX_CONTEXT_CHARS,
) -> str:
    """Sample a single long passage from PG19 with quality filtering.

    Picks a random row, picks a random start offset, extracts a slice
    up to *max_chars* snapped to a word boundary. Retries on quality
    failures up to MAX_SAMPLE_ATTEMPTS times.
    """
    n_rows = len(rows)
    for _ in range(MAX_SAMPLE_ATTEMPTS):
        idx = rng.randrange(n_rows)
        row_text = rows[idx]
        if len(row_text) < MIN_CONTEXT_CHARS:
            continue

        max_start = max(0, len(row_text) - MIN_CONTEXT_CHARS)
        start = rng.randint(0, max_start) if max_start > 0 else 0
        end = min(start + max_chars, len(row_text))

        passage = row_text[start:end]

        space_idx = passage.rfind(" ", 0, len(passage))
        if space_idx > MIN_CONTEXT_CHARS:
            passage = passage[:space_idx]

        if _is_valid_passage(passage):
            return passage

    raise RuntimeError(
        f"Could not sample a valid PG19 passage after {MAX_SAMPLE_ATTEMPTS} "
        f"attempts (min_chars={MIN_CONTEXT_CHARS}). The dataset may be "
        f"too small or too noisy."
    )


def derive_seed(block_hash: str) -> int:
    """SHA-256 of block hash, first 8 bytes as int."""
    h = hashlib.sha256(block_hash.encode()).digest()
    return int.from_bytes(h[:8], "big")


def sample_prompts(
    block_hash: str,
    n: int = 10,
    *,
    max_context_tokens: int | None = None,
    _rows: list[str] | None = None,
) -> list[Prompt]:
    """Sample n deterministic prompts seeded by block hash.

    Each prompt pairs a random PG19 passage with a random instruction
    template, formatted as an OpenAI chat message.

    ``max_context_tokens`` is the vLLM ``--max-model-len``.  When
    provided, passage length is capped to fit inside the context window.
    Pass ``_rows`` to override the PG19 dataset (used in tests).
    """
    rows = _rows if _rows is not None else _load_pg19()
    seed = derive_seed(block_hash)
    rng = random.Random(seed)

    mc = max_passage_chars(max_context_tokens)

    prompts: list[Prompt] = []
    for _ in range(n):
        passage = _sample_passage(rng, rows, max_chars=mc)
        template = rng.choice(TEMPLATES)
        content = template.format(context=passage)
        prompts.append(
            Prompt(
                messages=[ChatMessage(role="user", content=content)],
                max_tokens=MAX_OUTPUT_TOKENS,
            )
        )
    return prompts
