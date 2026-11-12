"""Render one arena's commissioning operator from the tracked templates and its operator.json.

The templates are the bytes that commissioned the Muse Spark pair (operator r26, derived from the
DeepSeek-V4.1-Flash operator r25) with every arena-specific value lifted into an ``@NAME@``
placeholder, so a new arena is one ``operator.json`` under ``examples/arena_inputs/<arena>/`` and
one render, and the diff between two arenas' rendered operators is exactly their parameter
difference. The output is the task-directory layout the validator VM launches from:
``operator/`` and ``cpu-operator-<slug>.tar`` (the scripts the pod and the VM run), ``inputs/``
and ``retained-bootstrap-inputs-<slug>.tar`` (the commission inputs extracted into the pod's
shared root), the rendered ``launch_stage.py``, a copy of ``launch_node.py``, and a receipt that
digests every template, retained input and output.

The deployment-owned sealed inputs are not templates and are never fabricated here: the packet
(the fixed qualification capabilities, the numeric judge, the finalize step, the adapter and the
sealed prompt authority), the model provision receipt and the pod transport are read from the
``--retained-inputs`` directory and copied by digest; only the packet's replay load (one session
count, or a list of them for a multi-load arena), its per-load decode floors, scored windows and audit
controls are rewritten to the arena's values, as the operator always did by hand.
"""

import argparse
import ast
import hashlib
import json
import os
import re
import shutil
import subprocess
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEMPLATES = Path(__file__).resolve().with_name("templates")
SCHEMA = "cacheon-arena-operator-v1"
PLACEHOLDER = re.compile(r"@([A-Z][A-Z0-9_]*)@")
KEYS = {"schema", "arena_id", "slug", "model_profile_key", "tensor_parallel_size", "replay_load",
        "tokens_per_prompt", "tokenizer_asset", "model_in_ram", "owner", "audit_control_delta_digests", "node"}
# decode_floors: per-user decode floor per load (tok/s keyed by session count), required with several loads;
# replay_windows: the scored windows per load, replacing the packet's; audit_row_floor_ppm: the node audit's
# sealed per-row floor, absent at the 2% default; load_margin: the per-load regression margin, off when absent.
OPTIONAL_KEYS = {"decode_floors", "replay_windows", "audit_row_floor_ppm", "load_margin"}
NODE_KEYS = {"name", "scheduler_starts", "helper_starts", "client_cpus"}
# engine_cpus: an engine's first GPU -> [DP controller vCPU, frontend vCPU] (oci_cpuset.canonical_cpu_roles).
OPTIONAL_NODE_KEYS = {"engine_cpus", "lifetime_memory_gib"}
PACKET = ("cacheon_b300_qualification_adapter.py", "finalize-authority.py", "numeric_answer_judge.py",
          "prompt-authority.current.json", "qualification_capabilities.py")
# The cpu-operator archive: commission_worker, prepare_worker and runtime_uid_preflight run on the pod;
# await_commission, connect_worker, prepare_cpu_pair and worker_runtime run on the validator VM;
# runtime_service is installed beside the pod transport.
OPERATOR = {
    "await_commission.py": "await_commission.py",
    "commission_worker.py.tmpl": "commission_worker.py",
    "connect_worker.py.tmpl": "connect_worker.py",
    "prepare_cpu_pair.py.tmpl": "prepare_cpu_pair_{slug}.py",
    "prepare_worker.py.tmpl": "prepare_worker.py",
    "runtime_service.py": "runtime_service.py",
    "runtime_uid_preflight.py": "runtime_uid_preflight.py",
    "twin.sh.tmpl": "twin_{slug}.sh",
    "worker_runtime.py.tmpl": "worker_runtime.py",
}
# The retained bootstrap inputs archive, extracted into the pod's shared root by launch_stage.
INPUTS = {
    "commission_pair.sh.tmpl": "commission-inputs/commission_pair_{slug}.sh",
    "fill_deployment.py.tmpl": "commission-inputs/fill_deployment_{slug}.py",
    "req.py": "commission-inputs/req_{slug}.py",
    "deployment.json.tmpl": "inputs/deployment.json",
    "check_runtime_paths.py": "check_runtime_paths.py",
    "preflight_replay_tokenizer.py": "preflight_replay_tokenizer.py",
}
# Member modes of the retained inputs archive, as the commissioned archives carry them.
MODES = {"commission-inputs/commission_pair_{slug}.sh": 0o700, "inputs/deployment.json": 0o400,
         "inputs/pod-remote-worker-service.py": 0o400,
         "commission-inputs/packet/numeric_answer_judge.py": 0o400,
         "commission-inputs/packet/prompt-authority.current.json": 0o400,
         "commission-inputs/packet/qualification_capabilities.py": 0o400}
RECEIPT_MODE = 0o444


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def load_operator(path: Path) -> dict:
    """Read and shape-check one arena's operator.json; policy stays with the owner."""
    operator = json.loads(path.read_bytes())
    if set(operator) - OPTIONAL_KEYS != KEYS or operator["schema"] != SCHEMA:
        raise SystemExit(f"{path}: expected keys {sorted(KEYS)}, optionally {sorted(OPTIONAL_KEYS)}, and schema {SCHEMA}")
    if set(operator["node"]) - OPTIONAL_NODE_KEYS != NODE_KEYS:
        raise SystemExit(f"{path}: node needs {sorted(NODE_KEYS)}, optionally {sorted(OPTIONAL_NODE_KEYS)}")
    if not re.fullmatch(r"[a-z0-9]+", operator["slug"]):
        raise SystemExit(f"{path}: slug must be lowercase letters and digits")
    width, loads = operator["tensor_parallel_size"], operator["replay_load"]
    loads = loads if isinstance(loads, list) else [loads]
    if not all(type(value) is int and value > 0
               for value in (width, operator["tokens_per_prompt"], operator.get("replay_windows", 1),
                             operator["node"].get("lifetime_memory_gib", 1), *loads)
               ) or loads != sorted(set(loads)):
        raise SystemExit(f"{path}: tensor_parallel_size, tokens_per_prompt, replay_windows, node.lifetime_memory_gib "
                         "and the ascending replay_load (one session count or a list) are positive integers")
    floors = operator.get("decode_floors", {})
    if (sorted(floors) != (sorted(map(str, loads)) if floors or len(loads) > 1 else [])
            or not all(re.fullmatch(r"[1-9][0-9]*(\.[0-9]*[1-9])?", value) for value in floors.values())):
        raise SystemExit(f"{path}: decode_floors maps every replay load to a canonical decimal tok/s floor")
    if "load_margin" in operator and (not floors or not re.fullmatch(r"0\.[0-9]*[1-9]", operator["load_margin"])):
        raise SystemExit(f"{path}: load_margin is a canonical decimal in (0, 1), sealed with decode_floors")
    starts, helpers = operator["node"]["scheduler_starts"], operator["node"]["helper_starts"]
    if len(helpers) != len(starts) or not all(type(c) is int for c in (*starts, *helpers)):
        raise SystemExit(f"{path}: helper_starts names one helper vCPU per scheduler start, in GPU order")
    for lane, cpus in operator["node"]["client_cpus"].items():
        devices = [int(value) for value in lane.split(",")]
        if len(devices) != width or devices != sorted(devices) or not cpus:
            raise SystemExit(f"{path}: client_cpus lane {lane!r} must name {width} sorted devices")
    controls = operator["audit_control_delta_digests"]
    if controls != sorted(set(controls)) or not all(re.fullmatch(r"[0-9a-f]{64}", d) for d in controls):
        raise SystemExit(f"{path}: audit_control_delta_digests must be sorted, distinct, lowercase sha256 hex")
    floor = operator.get("audit_row_floor_ppm")
    if floor is not None and (type(floor) is not int or not 0 < floor <= 400_000 or floor == 20_000):
        raise SystemExit(f"{path}: audit_row_floor_ppm is an integer in (0, 400000], absent at the 20000 default")
    return operator


def values(operator: dict, arena_inputs: str) -> dict[str, str]:
    width = operator["tensor_parallel_size"]
    return {
        "ARENA_ID": operator["arena_id"], "ARENA_INPUTS": arena_inputs, "SLUG": operator["slug"],
        "MODEL_PROFILE_KEY": operator["model_profile_key"], "TP": str(width),
        "TOKENS_PER_PROMPT": str(operator["tokens_per_prompt"]),
        "TOKENIZER_ASSET": operator["tokenizer_asset"],
        "OWNER": operator["owner"], "RAM_MODEL_ARGS": " '--ram-model'," if operator["model_in_ram"] else "",
        "SCHEDULER_STARTS": json.dumps(operator["node"]["scheduler_starts"]),
        "HELPER_STARTS": json.dumps(operator["node"]["helper_starts"]),
        "NODE_NAME": operator["node"]["name"], "CLIENT_SETS": repr(operator["node"]["client_cpus"]),
        "ENGINE_CPUS": repr(operator["node"].get("engine_cpus")),
        "LIFETIME_MEMORY": repr(operator["node"]["lifetime_memory_gib"] << 30
                                if "lifetime_memory_gib" in operator["node"] else None),
    }


def render(template: Path, destination: Path, fill: dict[str, str], mode: int) -> None:
    text = template.read_text()
    names = set(PLACEHOLDER.findall(text))
    if names - set(fill):
        raise SystemExit(f"{template.name}: no value for {sorted(names - set(fill))}")
    text = PLACEHOLDER.sub(lambda match: fill[match.group(1)], text)
    check(destination.name, text)
    destination.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
    destination.write_text(text)
    destination.chmod(mode)


def check(name: str, text: str) -> None:
    """Every rendered file parses in the language it is run as."""
    if PLACEHOLDER.search(text):
        raise SystemExit(f"{name}: unrendered placeholder {PLACEHOLDER.search(text).group(0)}")
    if name.endswith(".py"):
        ast.parse(text, filename=name)
    elif name.endswith(".json"):
        json.loads(text)
    elif name.endswith(".sh"):
        subprocess.run(["sh", "-n"], input=text, text=True, check=True)


def copy(source: Path, destination: Path, mode: int) -> None:
    destination.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    destination.chmod(mode)


def archive(output: Path, root: Path, members: list[Path]) -> None:
    with tarfile.open(output, "x") as out:
        for path in members:
            out.add(path, arcname=str(path.relative_to(root)), recursive=False)
    output.chmod(0o600)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arena", required=True, help="directory name under examples/arena_inputs")
    parser.add_argument("--retained-inputs", required=True, type=Path,
                        help="packet/, model-receipts/ and pod-remote-worker-service.py of this arena")
    parser.add_argument("--out", required=True, type=Path, help="new task directory layout to write")
    args = parser.parse_args()
    os.umask(0o077)
    arena = ROOT / "examples" / "arena_inputs" / args.arena
    operator = load_operator(arena / "operator.json")
    fill = values(operator, args.arena)
    slug = operator["slug"]
    retained = args.retained_inputs.resolve(strict=True)
    (receipt,) = sorted((retained / "model-receipts").glob("model-provision-sha256-*.json"))
    json.loads(receipt.read_bytes())
    out = args.out
    out.mkdir(mode=0o700, parents=False, exist_ok=False)
    operator_dir, inputs = out / "operator", out / "inputs"

    for name, target in OPERATOR.items():
        render(TEMPLATES / name, operator_dir / target.format(slug=slug), fill, 0o600)
    copy(ROOT / "scripts" / "bootstrap_worker.py", operator_dir / "bootstrap_worker.py", 0o600)
    render(TEMPLATES / "launch_stage.py.tmpl", out / "launch_stage.py", fill, 0o600)
    copy(Path(__file__).with_name("launch_node.py"), out / "launch_node.py", 0o600)
    for name, target in INPUTS.items():
        render(TEMPLATES / name, inputs / target.format(slug=slug), fill, MODES.get(target, 0o644))
    for name in PACKET:
        target = f"commission-inputs/packet/{name}"
        copy(retained / "packet" / name, inputs / target, MODES.get(target, 0o644))
    capabilities = inputs / "commission-inputs/packet/qualification_capabilities.py"
    text = capabilities.read_text()
    # The registered audit controls are the arena's own bundles (a wrong control for this model's node address):
    # their digests replace the previous arena's in the sealed policy, as the replay load does.
    floor = operator.get("audit_row_floor_ppm")
    controls = lambda match: "".join((*([f'{match["indent"]}"audit_row_floor_ppm": {floor},\n'] if floor else []),
                                      f'{match["indent"]}"audit_control_delta_digests": [\n',
                                      *(f'{match["indent"]}    "{digest}",\n'
                                        for digest in operator["audit_control_delta_digests"]), f'{match["indent"]}]'))
    rewrites = [(r'"load": (?:\d+|\[[\d, ]*\])', lambda _: f'"load": {json.dumps(operator["replay_load"])}'),
                (r'(?P<indent>[ \t]*)"audit_control_delta_digests": \[(?:\s*"[0-9a-f]{64}",)*\s*\]', controls)]
    if "decode_floors" in operator:  # keywords of the packet's GOODPUT_POLICY = GoodputPolicy(..., decode_floor=1.0)
        floors = ", ".join(f"({load}, {operator['decode_floors'][load]})" for load in sorted(operator["decode_floors"], key=int))
        margin = f', load_margin={operator["load_margin"]}' if "load_margin" in operator else ""
        rewrites.append((r"decode_floor=1\.0\)", lambda _: f"decode_floor=1.0, decode_floors=({floors},){margin})"))
    if "replay_windows" in operator:
        rewrites.append((r'"windows": \d+', lambda _: f'"windows": {operator["replay_windows"]}'))
    for pattern, value in rewrites:
        text, count = re.subn(pattern, value, text)
        if count != 1:
            raise SystemExit(f"{capabilities.name}: expected one match of {pattern}, found {count}")
    check(capabilities.name, text)
    capabilities.chmod(0o600)
    capabilities.write_text(text)
    capabilities.chmod(MODES["commission-inputs/packet/qualification_capabilities.py"])
    copy(receipt, inputs / "commission-inputs/model-receipts" / receipt.name, RECEIPT_MODE)
    copy(retained / "pod-remote-worker-service.py", inputs / "inputs/pod-remote-worker-service.py",
         MODES["inputs/pod-remote-worker-service.py"])

    for path in inputs.rglob("*"):
        if path.is_dir():
            path.chmod(0o755)  # the commissioned archives carry world-traversable directories
    archive(out / f"cpu-operator-{slug}.tar", operator_dir, sorted(operator_dir.iterdir()))
    archive(out / f"retained-bootstrap-inputs-{slug}.tar", inputs, sorted(inputs.rglob("*")))
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True)
    report = {
        "schema": "cacheon-arena-operator-render-v1", "arena": args.arena, "operator": operator,
        "source_head": head.stdout.strip() if head.returncode == 0 else None,
        "templates": {path.name: sha256(path) for path in sorted(TEMPLATES.iterdir())},
        "retained_inputs": {str(path.relative_to(retained)): sha256(path)
                            for path in sorted(retained.rglob("*")) if path.is_file()},
        "outputs": {str(path.relative_to(out)): sha256(path)
                    for path in sorted(out.rglob("*")) if path.is_file()},
    }
    (out / "render-receipt.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"arena": args.arena, "slug": slug, "out": str(out),
                      "operator_archive": f"cpu-operator-{slug}.tar",
                      "bootstrap_inputs": f"retained-bootstrap-inputs-{slug}.tar",
                      "files": len(report["outputs"])}))


if __name__ == "__main__":
    main()
