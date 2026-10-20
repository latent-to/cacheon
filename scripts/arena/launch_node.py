"""Start one pair's commission from a rendered operator directory on the validator VM.

Checks the retained image digest, the exported controller source against its metadata and the
render receipt independently, then runs the
rendered ``launch_stage.py`` detached in its own session with its whole output in the task
directory, and records what it started. Arguments after ``--`` go to ``launch_stage`` unchanged:
``--host``, ``--user``, ``--known-hosts``, ``--shared``, ``--tag``, ``--lane``, ``--baseline``,
``--model-input``, ``--model-token`` and, when used, ``--pod-port`` and ``--runtime-seed``. The
stage ends with ``COMMISSIONING_ONLY``: connecting the worker to the intake is a separate step
taken on the owner's order.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--operator", required=True, type=Path, help="directory render_operator.py wrote")
parser.add_argument("--task-dir", required=True, type=Path, help="receipts, state and logs land here")
parser.add_argument("--base", required=True, help="ssh_key, ssh_key.pub and bootstrap-tools/uv")
parser.add_argument("--image-proof", required=True, type=Path, help="durable-images.json of the worker image")
parser.add_argument("--source", required=True, type=Path, help="exported runtime source archive")
parser.add_argument("--source-metadata", required=True, type=Path)
parser.add_argument("--source-revision", required=True, help="full commit the exported controller source must carry")
parser.add_argument("--image-digest", required=True, help="sha256 hex the worker image reference must carry")
args, stage = parser.parse_known_args()
if stage and stage[0] == "--":
    stage.pop(0)
os.umask(0o077)
operator = args.operator.resolve(strict=True)
task = args.task_dir.resolve(strict=True)
receipt = json.loads((operator / "render-receipt.json").read_bytes())
slug = receipt["operator"]["slug"]
proof = json.loads(args.image_proof.read_bytes())
assert proof["worker_image"].endswith("@sha256:" + args.image_digest), proof["worker_image"]
metadata = json.loads(args.source_metadata.read_bytes())
assert metadata["revision"] == args.source_revision and len(metadata["revision"]) == 40
assert sha256(args.source) == metadata["sha256"], "source archive differs from its metadata"
archives = [operator / f"cpu-operator-{slug}.tar", operator / f"retained-bootstrap-inputs-{slug}.tar"]
for path in archives:
    assert receipt["outputs"][path.name] == sha256(path), f"{path.name} differs from the render receipt"
tag = stage[stage.index("--tag") + 1]
argv = ["/usr/bin/python3", "-B", "-u", str(operator / "launch_stage.py"),
        "--base", args.base, "--state-root", str(task / "state"), "--image-proof", str(args.image_proof),
        "--bootstrap-inputs", str(archives[1]), "--operator-archive", str(archives[0]),
        "--source", str(args.source), "--source-metadata", str(args.source_metadata), *stage]
with (task / f"stage-launch-{tag}.log").open("xb", buffering=0) as output:
    child = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
                             start_new_session=True)
started = {"pid": child.pid, "started_at": int(time.time()), "argv": argv, "tag": tag,
           "render_receipt": str(operator / "render-receipt.json"),
           "commissioning_only": True, "gpu_execution_started": False}
(task / f"stage-launch-{tag}-started.json").write_text(json.dumps(started, indent=2) + "\n")
print(json.dumps({"pid": child.pid, "tag": tag, "log": str(task / f"stage-launch-{tag}.log")}))
sys.exit(0)
