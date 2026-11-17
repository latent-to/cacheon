"""A draft-layer repair must preserve the timed stock and target-only launch arguments."""

import hashlib
import json
import shutil
from pathlib import Path

from cacheon.engine_tree import MaterializedEngineTree
from cacheon.eval.oci_backend import CONTAINER_ENGINE_WORKER_POLICY, OCIBackendError
from cacheon.eval.oci_worker_policy import worker_policy_files


def test_only_explicit_mtp_submodules_mount_the_commissioned_adapter(tmp_path):
    model, root = tmp_path / "model", tmp_path / "tree"
    model.mkdir()
    stock = MaterializedEngineTree(root, "", "", (), None)
    common = worker_policy_files(stock, model, CONTAINER_ENGINE_WORKER_POLICY, OCIBackendError)
    source = Path(__file__).resolve().parents[1] / "cacheon" / "integrations" / "sglang_nodes.py"
    shutil.copytree(source.parents[2] / "examples/miner_node_identity", root)
    manifest = (root / "manifest.toml").read_text()
    tree = MaterializedEngineTree(root, "", "", (), "manifest.toml")
    for depth, slot, draft in [(45, "model.layers.*.mlp", False), (45, "model.layers.44.mlp", False),
                               (45, "model.layers.45.mlp", True), (48, "model.layers.45.mlp", False),
                               (48, "model.layers.48.mlp", True)]:
        (model / "config.json").write_text(json.dumps({"text_config": {"num_hidden_layers": depth, "num_nextn_predict_layers": 1}}))
        (root / "manifest.toml").write_text(manifest.replace("model.layers.*.mlp", slot))
        mounts, digests = worker_policy_files(tree, model, CONTAINER_ENGINE_WORKER_POLICY, OCIBackendError)
        if draft:
            assert mounts[:-1] == common[0] and mounts[-1].endswith("/integrations/sglang_nodes.py,bind-propagation=rprivate,readonly")
            assert digests == {**common[1], "CACHEON_CONTROLLER_NODE_ADAPTER_SHA256": hashlib.sha256(source.read_bytes()).hexdigest()}
        else:
            assert (mounts, digests) == common
