"""Data shapes passed between the validator loop and the GPU evaluation function.

``EvaluationJob`` is what the loop builds and hands to ``eval_fn``.
``EvaluationResult`` is what ``eval_fn`` returns. Both are plain
dataclasses -- no Pydantic, no torch, no bittensor imports -- so they
can be tested and serialized cheaply.

``EvalJob`` is what the CPU writes to ``eval_job.json`` for the GPU
entrypoint to read. It bundles the block context and challenger list
so the GPU side needs zero chain access.

The ``eval_fn`` contract:
    eval_fn(job: EvaluationJob) -> EvaluationResult

The implementation lives elsewhere (Docker orchestration, GPU harness,
etc.) and is wired in by the CLI entry point.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChatMessage:
    """A single message in the OpenAI chat format."""

    role: str  # "system" | "user" | "assistant"
    content: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChatMessage:
        return cls(role=str(data["role"]), content=str(data["content"]))


@dataclass(frozen=True)
class Prompt:
    """One evaluation prompt: a list of chat messages plus generation config."""

    messages: list[ChatMessage]
    max_tokens: int = 256

    def to_dict(self) -> dict[str, Any]:
        return {
            "messages": [m.to_dict() for m in self.messages],
            "max_tokens": self.max_tokens,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Prompt:
        return cls(
            messages=[ChatMessage.from_dict(m) for m in data["messages"]],
            max_tokens=int(data.get("max_tokens", 256)),
        )


@dataclass(frozen=True)
class EvaluationJob:
    """Everything the GPU eval function needs to run one challenger."""

    image: str
    digest: str
    prompts: list[Prompt]
    model_volume: str = "/models"
    per_prompt_timeout_s: int = 120
    n_warmup: int = 2
    startup_timeout_s: int = 600

    def to_dict(self) -> dict[str, Any]:
        return {
            "image": self.image,
            "digest": self.digest,
            "prompts": [p.to_dict() for p in self.prompts],
            "model_volume": self.model_volume,
            "per_prompt_timeout_s": self.per_prompt_timeout_s,
            "n_warmup": self.n_warmup,
            "startup_timeout_s": self.startup_timeout_s,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvaluationJob:
        return cls(
            image=str(data["image"]),
            digest=str(data["digest"]),
            prompts=[Prompt.from_dict(p) for p in data["prompts"]],
            model_volume=str(data.get("model_volume", "/models")),
            per_prompt_timeout_s=int(data.get("per_prompt_timeout_s", 120)),
            n_warmup=int(data.get("n_warmup", 2)),
            startup_timeout_s=int(data.get("startup_timeout_s", 600)),
        )


@dataclass(frozen=True)
class PerPromptResult:
    """Metrics for a single (non-warmup) prompt."""

    ttft_s: float
    throughput_tps: float
    output_tokens: int
    token_match_rate: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PerPromptResult:
        return cls(
            ttft_s=float(data["ttft_s"]),
            throughput_tps=float(data["throughput_tps"]),
            output_tokens=int(data["output_tokens"]),
            token_match_rate=float(data["token_match_rate"]),
        )


@dataclass(frozen=True)
class EvaluationResult:
    """What the GPU eval function returns after running one challenger.

    Aggregated metrics use median across scored prompts (warmup excluded).
    Improvement values are relative to the pre-computed baseline:
      ttft_improvement    = max(0, (baseline_ttft - miner_ttft) / baseline_ttft)
      throughput_improvement = max(0, (miner_tps - baseline_tps) / baseline_tps)
    """

    success: bool
    ttft_improvement: float = 0.0
    throughput_improvement: float = 0.0
    token_match_rate: float = 0.0
    median_ttft_s: float = 0.0
    median_throughput_tps: float = 0.0
    per_prompt: list[PerPromptResult] = field(default_factory=list)
    aggregation: str = "median"
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "ttft_improvement": self.ttft_improvement,
            "throughput_improvement": self.throughput_improvement,
            "token_match_rate": self.token_match_rate,
            "median_ttft_s": self.median_ttft_s,
            "median_throughput_tps": self.median_throughput_tps,
            "per_prompt": [p.to_dict() for p in self.per_prompt],
            "aggregation": self.aggregation,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvaluationResult:
        per_prompt = [PerPromptResult.from_dict(p) for p in data.get("per_prompt", [])]
        return cls(
            success=bool(data["success"]),
            ttft_improvement=float(data.get("ttft_improvement", 0.0)),
            throughput_improvement=float(data.get("throughput_improvement", 0.0)),
            token_match_rate=float(data.get("token_match_rate", 0.0)),
            median_ttft_s=float(data.get("median_ttft_s", 0.0)),
            median_throughput_tps=float(data.get("median_throughput_tps", 0.0)),
            per_prompt=per_prompt,
            aggregation=str(data.get("aggregation", "median")),
            error=data.get("error"),
        )


# --------------------------------------------------------------------------- #
# EvalJob -- CPU -> GPU handoff via S3
# --------------------------------------------------------------------------- #

EVAL_JOB_FILE = "eval_job.json"


@dataclass(frozen=True)
class ChallengerInfo:
    """Serializable challenger identity for the GPU eval job.

    Mirrors ``chain.CommitmentRecord`` without pulling in chain.py.
    """

    uid: int
    hotkey: str
    commit_block: int
    image: str
    digest: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChallengerInfo:
        return cls(
            uid=int(data["uid"]),
            hotkey=str(data["hotkey"]),
            commit_block=int(data["commit_block"]),
            image=str(data["image"]),
            digest=str(data["digest"]),
        )


@dataclass(frozen=True)
class EvalJob:
    """What the CPU writes to ``eval_job.json`` for the GPU entrypoint.

    Contains block context and a list of challengers. The GPU pod reads
    this file, runs eval for each challenger, and writes results into
    ``state.json``.
    """

    block: int
    block_hash: str
    challengers: list[ChallengerInfo]
    created_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "block": self.block,
            "block_hash": self.block_hash,
            "challengers": [c.to_dict() for c in self.challengers],
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvalJob:
        return cls(
            block=int(data["block"]),
            block_hash=str(data["block_hash"]),
            challengers=[
                ChallengerInfo.from_dict(c) for c in data.get("challengers", [])
            ],
            created_at=float(data.get("created_at", 0.0)),
        )

    def save(self, state_dir: str | Path) -> None:
        path = Path(state_dir) / EVAL_JOB_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(
            "Wrote eval job: %d challenger(s) to %s", len(self.challengers), path
        )

    @classmethod
    def load(cls, state_dir: str | Path) -> EvalJob | None:
        path = Path(state_dir) / EVAL_JOB_FILE
        if not path.exists():
            logger.warning("No eval job file at %s", path)
            return None
        try:
            with open(path) as f:
                data = json.load(f)
            job = cls.from_dict(data)
            logger.info(
                "Loaded eval job: block=%d, %d challenger(s)",
                job.block,
                len(job.challengers),
            )
            return job
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            logger.error("Failed to load eval job from %s: %s", path, exc)
            return None
