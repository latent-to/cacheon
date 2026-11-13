"""Read-only validator Python files consumed by the pinned engine image."""

import hashlib
import json
import stat
from pathlib import Path

from cacheon.engine_tree import MaterializedEngineTree
from cacheon.manifest import load_manifest


def _claims_draft_layer(tree: MaterializedEngineTree, model: Path) -> bool:
    """Mirror the node binder's explicit MTP address against the mounted checkpoint."""
    if tree.runtime_manifest is None:
        return False
    slots = [op.slot for op in load_manifest(tree.root).ops]
    if not any(slot.startswith("model.layers.") and slot.split(".")[2].isdigit() for slot in slots):
        return False
    config = json.loads((model / "config.json").read_bytes())
    text = config.get("text_config") or config
    return bool(text.get("num_nextn_predict_layers")) and any(
        slot.startswith(f"model.layers.{text['num_hidden_layers']}.") for slot in slots
    )


def worker_policy_files(
    tree: MaterializedEngineTree, model: Path, worker_destination: str, error: type[Exception],
) -> tuple[list[str], dict[str, str]]:
    """Bind the corrected node adapter only to a tree claiming the checkpoint's MTP layer."""
    worker = Path(__file__).with_name("engine_worker.py")
    files = [(worker, worker_destination, "CACHEON_CONTROLLER_ENGINE_WORKER_SHA256")]
    if _claims_draft_layer(tree, model):
        module = Path("integrations/sglang_nodes.py")
        files.append((worker.parents[1] / module, str(Path(worker_destination).parents[1] / module),
                      "CACHEON_CONTROLLER_NODE_ADAPTER_SHA256"))
    mounts, digests = [], {}
    for source, destination, key in files:
        try:
            if not stat.S_ISREG(source.lstat().st_mode):
                raise error(f"controller policy has an unsafe shape: {source}")
            digests[key] = hashlib.sha256(source.read_bytes()).hexdigest()
        except OSError as exc:
            raise error(f"controller policy is unavailable: {source}: {exc}") from None
        mounts.append(f"--mount=type=bind,src={source},dst={destination},bind-propagation=rprivate,readonly")
    return mounts, digests
