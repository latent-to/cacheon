"""Display the selected competition and its retained evaluation history."""

from functools import lru_cache
import json
from pathlib import Path
import sqlite3
from typing import Any

from dashboard.sources import selected

# First GLM-5.3 submission, verified from the retained glm53 mock release URL.
GLM53_FIRST_BLOCK = 9009654


def competition_label(block: int, arena: str = "") -> str:
    """Keep the historical model label across the recorded competition cutover."""
    source = selected.get()
    if source is not None and source.model:
        return source.model
    return arena or ("GLM-5.3" if int(block) >= GLM53_FIRST_BLOCK else "MiniMax-M3")


# Provisioning receipts match these content identities to the pinned HF snapshots.
_CHECKPOINTS = {
    "243b810341c7609234827ba864f4eb107560d9e3a3020be254921fa09eb80428":
        ("Mapika/MiniMax-M3-NVFP4", "668435825700a0047399441720f430bdd8eca0ab"),
    "cab30f10b0039ac9bf0b6caa4b4d8b617093defbcc31778fd282a421d9b814f6":
        ("incoai/GLM-5.3-NVFP4", "54e52520606f96b3d9fc84088ad22882a61648ac"),
}


def checkpoint_for_engine(engine: str) -> dict | None:
    """Use only the selected source's declared checkpoint for its exact engine."""
    source = selected.get()
    if source is not None:
        return (source.checkpoint or {}).get(engine)
    return _legacy_checkpoint_for_engine(engine)


@lru_cache(maxsize=128)
def _legacy_checkpoint_for_engine(engine: str) -> dict | None:
    """Match the submission's engine to retained runtime model content, not its date."""
    if not engine:
        return None
    roots = Path("/root/cacheon-ops")
    paths = list((roots / "remote-worker/state").glob("mainnet-screen-dispatcher-*.json"))
    paths += list((roots / "stage").glob("*/monday-config/mainnet-screen-dispatcher.json"))
    for path in paths:
        try:
            runtime = json.loads(path.read_text())["arena_service_manifest"]["runtime"]
        except (OSError, ValueError, KeyError, TypeError):
            continue
        if runtime.get("base_engine_digest") != engine:
            continue
        content = runtime.get("model_content_digest")
        if content not in _CHECKPOINTS:
            return None
        repo, revision = _CHECKPOINTS[content]
        return {"repo": repo, "revision": revision, "content_digest": content,
                "url": f"https://huggingface.co/{repo}/tree/{revision}"}
    return None


def submission_baseline(con: sqlite3.Connection, reservation_id: str, target_id: str, *, bars=None) -> dict[str, Any]:
    """Describe the baseline a submission is measured on and the best retained result on it."""
    from dashboard.winners import reward_bars

    candidate = con.execute(
        "SELECT candidate_json FROM settlement_candidates WHERE reservation_id=?", (reservation_id,)).fetchone()
    raw: dict[str, Any] = {}
    evaluated = assigned = False
    if candidate is not None:
        doc = json.loads(candidate["candidate_json"] or "{}")
        raw, evaluated = doc.get("primary") or doc, True
    else:
        qualification = con.execute(
            "SELECT qualification_json FROM settlement_qualifications "
            "WHERE reservation_id=? ORDER BY reproduction_index LIMIT 1", (reservation_id,)).fetchone()
        if qualification is not None:
            raw, evaluated = json.loads(qualification["qualification_json"] or "{}"), True
        elif _has_table(con, "reservation_baseline_segments"):
            segment = con.execute(
                "SELECT arena_id,stack_digest,tree_digest,stack_json "
                "FROM reservation_baseline_segments WHERE reservation_id=?", (reservation_id,)).fetchone()
            if segment is not None:
                raw = {"arena_digest": segment["arena_id"], "incumbent_manifest": json.loads(segment["stack_json"]),
                       "incumbent_stack_digest": segment["stack_digest"],
                       "incumbent_tree_digest": segment["tree_digest"]}
                assigned = True
    if not raw:
        return {"evaluated": False, "assigned": False, "artifact_digest": "", "reward_bar": None}

    manifest = raw.get("incumbent_manifest") or {}
    artifact = ((manifest.get("entries") or {}).get(target_id) or {}).get("artifact_digest") or ""
    stack = raw.get("incumbent_stack_digest") or manifest.get("digest") or ""
    arena = raw.get("arena_digest") or manifest.get("arena_digest") or ""
    result: dict[str, Any] = {
        "kind": "incumbent" if manifest.get("entries") else "stock",
        "checkpoint": checkpoint_for_engine(manifest.get("base_engine_digest") or ""),
        "evaluated": evaluated, "assigned": assigned, "artifact_digest": artifact, "reservation_id": None,
        "stack_digest": stack, "tree_digest": raw.get("incumbent_tree_digest") or "", "arena_digest": arena,
        # Pay compares a PASS with the best retained PASS on its own baseline, never with the crown lineage.
        "reward_bar": (reward_bars(con) if bars is None else bars).get((arena, stack)),
    }
    if artifact and _has_table(con, "target_lineage_nodes"):
        scope, predicate = (), ""
        if "competition_arena" in {r["name"] for r in con.execute("PRAGMA table_info(target_lineage_nodes)")}:
            reservation = con.execute(
                "SELECT competition_arena FROM reservations WHERE reservation_id=?", (reservation_id,)).fetchone()
            scope, predicate = (reservation["competition_arena"],), " AND n.competition_arena=?"
        origin = con.execute(
            "SELECT e.reservation_id FROM target_lineage_nodes n "
            "JOIN settlement_events e ON e.event_id=n.transition_event_id "
            "WHERE n.target_id=? AND n.artifact_digest=?" + predicate, (target_id, artifact, *scope)).fetchone()
        if origin is not None:
            result["reservation_id"] = origin["reservation_id"]
    return result


def _has_table(con: sqlite3.Connection, name: str) -> bool:
    return con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone() is not None


# Short human descriptions of the optimization targets ("which op they improved").
TARGET_SUMMARIES = {
    "activation.silu_and_mul": "SiLU-and-multiply activation kernel (SwiGLU MLP gate)",
    "norm.rmsnorm": "RMSNorm normalization kernel",
    "attention.decode": "Attention decode kernel (token generation path)",
    "attention.sdpa": "Scaled dot-product attention (radix/prefill path)",
    "attention.msa_block_score": "MSA block-sparse attention decode scoring kernel",
    "attention.msa_prefill_block_score": "MSA block-sparse attention prefill scoring kernel",
    "collective.all_reduce": "All-reduce collective (multi-GPU tensor sum)",
    "collective.ar_residual_rmsnorm": "Fused all-reduce + residual-add + RMSNorm collective",
    "collective.moe_finalize_ar_rmsnorm": "Fused MoE finalize + all-reduce + RMSNorm collective epilogue",
    "collective.moe_epilogue.v1": "Atomic MoE epilogue (AR-residual-RMSNorm + MoE finalize pair)",
    "moe.fused_experts": "Fused MoE expert GEMM dispatch kernel",
    "moe.fused_experts_reduce": "Fused MoE expert GEMM with built-in reduction collective",
}


def target_summary(target_id: str) -> str:
    if target_id in TARGET_SUMMARIES:
        return TARGET_SUMMARIES[target_id]
    if not target_id:
        return ""
    return target_id.replace("_", " ").replace(".", " › ")
