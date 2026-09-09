"""Display labels for the GLM-5.3 competition and preceding history."""

from decimal import Decimal
from functools import lru_cache
import json
from pathlib import Path
import sqlite3
from typing import Any

# First GLM-5.3 submission, verified from the retained glm53 mock release URL.
GLM53_FIRST_BLOCK = 9009654


def competition_label(block: int) -> str:
    """Keep the historical model label across the recorded competition cutover."""
    return "GLM-5.3" if int(block) >= GLM53_FIRST_BLOCK else "MiniMax-M3"


# Provisioning receipts match these content identities to the pinned HF snapshots.
_CHECKPOINTS = {
    "243b810341c7609234827ba864f4eb107560d9e3a3020be254921fa09eb80428":
        ("Mapika/MiniMax-M3-NVFP4", "668435825700a0047399441720f430bdd8eca0ab"),
    "cab30f10b0039ac9bf0b6caa4b4d8b617093defbcc31778fd282a421d9b814f6":
        ("incoai/GLM-5.3-NVFP4", "54e52520606f96b3d9fc84088ad22882a61648ac"),
}


@lru_cache(maxsize=128)
def checkpoint_for_engine(engine: str) -> dict | None:
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


def submission_baseline(
    con: sqlite3.Connection,
    reservation_id: str,
    target_id: str,
    *,
    lineage_tables_available: bool | None = None,
) -> dict[str, Any]:
    """Describe the baseline used and its relationship to the active tip."""

    candidate = con.execute(
        "SELECT candidate_json FROM settlement_candidates "
        "WHERE reservation_id=?",
        (reservation_id,),
    ).fetchone()
    raw: dict[str, Any] = {}
    evaluated = False
    assigned = False
    if candidate is not None:
        doc = json.loads(candidate["candidate_json"] or "{}")
        raw = doc.get("primary") or doc
        evaluated = True
    else:
        qualification = con.execute(
            "SELECT qualification_json FROM settlement_qualifications "
            "WHERE reservation_id=? ORDER BY reproduction_index LIMIT 1",
            (reservation_id,),
        ).fetchone()
        if qualification is not None:
            raw = json.loads(qualification["qualification_json"] or "{}")
            evaluated = True
        elif con.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' "
            "AND name='reservation_baseline_segments'"
        ).fetchone() is not None:
            segment = con.execute(
                "SELECT arena_id,stack_digest,tree_digest,stack_json "
                "FROM reservation_baseline_segments WHERE reservation_id=?",
                (reservation_id,),
            ).fetchone()
            if segment is not None:
                manifest = json.loads(segment["stack_json"])
                raw = {
                    "arena_digest": segment["arena_id"],
                    "incumbent_manifest": manifest,
                    "incumbent_stack_digest": segment["stack_digest"],
                    "incumbent_tree_digest": segment["tree_digest"],
                }
                assigned = True

    if not raw:
        return {
            "evaluated": False,
            "assigned": False,
            "relationship": "not_evaluated",
            "artifact_digest": "",
            "current_tip_artifact_digest": "",
            "threshold_speedup": None,
        }

    manifest = raw.get("incumbent_manifest") or {}
    entry = (manifest.get("entries") or {}).get(target_id) or {}
    baseline_artifact = entry.get("artifact_digest") or ""
    result: dict[str, Any] = {
        "checkpoint": checkpoint_for_engine(manifest.get("base_engine_digest") or ""),
        "evaluated": evaluated,
        "assigned": assigned,
        "relationship": "no_active_tip",
        "artifact_digest": baseline_artifact,
        "stack_digest": raw.get("incumbent_stack_digest") or manifest.get("digest") or "",
        "tree_digest": raw.get("incumbent_tree_digest") or "",
        "arena_digest": raw.get("arena_digest") or manifest.get("arena_digest") or "",
        "current_tip_artifact_digest": "",
        "threshold_speedup": None,
    }
    if lineage_tables_available is None:
        tables = {
            row["name"]
            for row in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name IN "
                "('target_lineage_tips','target_lineage_nodes')"
            )
        }
        lineage_tables_available = tables == {
            "target_lineage_tips",
            "target_lineage_nodes",
        }
    if not lineage_tables_available:
        return result
    tip = con.execute(
        "SELECT artifact_digest FROM target_lineage_tips WHERE target_id=?",
        (target_id,),
    ).fetchone()
    if tip is None:
        return result
    tip_artifact = str(tip["artifact_digest"])
    result["current_tip_artifact_digest"] = tip_artifact
    if baseline_artifact == tip_artifact:
        result["relationship"] = "current_tip"
        result["threshold_speedup"] = 1.0
        return result

    nodes: list[dict[str, Any]] = []
    artifact = tip_artifact
    seen: set[str] = set()
    while artifact and artifact not in seen:
        seen.add(artifact)
        node = con.execute(
            "SELECT artifact_digest,parent_artifact_digest,winner_speedup "
            "FROM target_lineage_nodes WHERE target_id=? AND artifact_digest=?",
            (target_id, artifact),
        ).fetchone()
        if node is None:
            break
        nodes.append(dict(node))
        artifact = str(node["parent_artifact_digest"])
    nodes.reverse()
    start = next(
        (
            index for index, node in enumerate(nodes)
            if node["parent_artifact_digest"] == baseline_artifact
        ),
        None,
    )
    if start is None:
        result["relationship"] = "outside_active_lineage"
        return result
    threshold = Decimal(1)
    for node in nodes[start:]:
        threshold *= Decimal(str(node["winner_speedup"]))
    result["relationship"] = "ancestor"
    result["threshold_speedup"] = float(threshold)
    return result
