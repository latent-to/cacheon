"""Display labels for the GLM-5.3 competition and preceding history."""

# First GLM-5.3 submission, verified from the retained glm53 mock release URL.
GLM53_FIRST_BLOCK = 9009654


def competition_label(block: int) -> str:
    return "GLM-5.3" if int(block) >= GLM53_FIRST_BLOCK else "MiniMax-M3"


from functools import lru_cache
import json
from pathlib import Path

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
