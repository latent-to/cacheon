"""Move an arena's mainnet worker to a replacement pod: from the pod's address and CPU table to a started stage.

``launch`` runs where the repository, the arena's retained render inputs and the operator's SSH access to the pod
are. From a clean checkout of the deployed commit it renders the operator with the pod's measured pin table (only
rendered scripts carry pins, so the source revision stays the deployed one), exports that source, authorizes the
validator's key on the pod, copies everything to a new move directory on the validator and starts ``run`` there,
detached. ``run`` does the rest on the validator with its own key, each step through the existing tool: it pins the
pod's host key, pulls the worker image from its published registry and restores the retained runtime seed on the
pod, commissions the stock stage ``<tag>a`` with ``launch_node.py``, measures its fresh stock pair with
``stock_reference.py``, commissions the mainnet stage ``<tag>b``, imports the pair into it, prepares it with
``connect_worker.py``, maps its reward group onto the previous mainnet stage's (``continuation``), stops the previous
stage's units and starts the new stage with ``connect_worker.py --start``. Every step fails with its original
error; nothing is retried or substituted, and ``run --from STEP`` continues a stopped move at that step.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import sqlite3
import subprocess
import sys
import tempfile
import time

STEPS = ("access", "prepare-pod", "stock-commission", "stock-start", "stock-verify", "stock-stop",
         "mainnet-commission", "import", "connect", "continuation", "start")
# The validator's production interpreter, as connect_worker.py and prepare_cpu_pair run it.
VENV = "/root/cacheon-ops/venv/bin/python"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def option(argv: list[str], name: str) -> str:
    return argv[argv.index(name) + 1]


def checked(argv: list[str], what: str, timeout: int, **kwargs) -> str:
    """Run one command; a failure stops the move with that command's own output."""
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, **kwargs)
    if proc.returncode:
        raise SystemExit(f"{what} exited {proc.returncode}:\n{proc.stdout[-4000:]}{proc.stderr[-4000:]}")
    return proc.stdout


def launch(args: argparse.Namespace) -> None:
    root = Path(__file__).resolve().parents[2]
    git = lambda *argv: subprocess.check_output(["git", "-C", str(root), *argv], text=True).strip()
    if git("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("launch renders from a clean checkout of the deployed commit")
    revision = git("rev-parse", "HEAD")
    fast, helpers = ([int(cpu) for cpu in text.split(",")] for text in (args.fast, args.helpers))
    operator = json.loads((root / "examples/arena_inputs" / args.arena / "operator.json").read_bytes())
    width = operator["tensor_parallel_size"]
    if len(helpers) != width or len(fast) != 2 * width or len(set(fast + helpers)) != 3 * width:
        raise SystemExit(f"the CPU table names {2 * width} distinct fast vCPUs (the schedulers of GPUs 0-{width - 1} "
                         f"in order, then the client) and {width} helpers")
    pins = {"scheduler_starts": fast[:width], "helper_starts": helpers,
            "client_cpus": {",".join(map(str, range(width))): fast[width:]}}
    work = Path(tempfile.mkdtemp(prefix=f"pod-move-{args.tag}-"))
    (work / "pins.json").write_text(json.dumps(pins) + "\n")
    name = f"operator-{operator['slug']}-{revision[:8]}"
    quality = ["--quality-controls", str(args.quality_controls)] if args.quality_controls else []
    checked([sys.executable, "-B", str(root / "scripts/arena/render_operator.py"), "--arena", args.arena,
             "--retained-inputs", str(args.retained_inputs), "--pins", str(work / "pins.json"),
             "--out", str(work / name), *quality], "render", 300, cwd=root)
    source = work / f"source-{revision[:8]}.tar"
    with source.open("xb") as output:
        subprocess.run(["git", "-C", str(root), "archive", "--format=tar", revision], stdout=output, check=True)
    metadata = source.with_suffix(".json")
    metadata.write_text(json.dumps({"revision": revision, "sha256": sha256(source)}) + "\n")
    validator = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", args.validator]
    previous = json.loads(checked(validator + [shlex.join(["cat", args.previous_launch])], "read previous launch", 60))
    public = checked(validator + [shlex.join(["cat", option(previous["argv"], "--base") + "/ssh_key.pub"])],
                     "read the validator key", 60).strip()
    authorize = ('read key; mkdir -p -m 700 ~/.ssh; grep -qxF "$key" ~/.ssh/authorized_keys 2>/dev/null'
                 ' || echo "$key" >> ~/.ssh/authorized_keys; chmod 600 ~/.ssh/authorized_keys')
    checked(["ssh", "-o", "ConnectTimeout=15", "-p", str(args.port), f"{args.user}@{args.host}", authorize],
            "authorize the validator key on the pod", 60, input=public + "\n")
    move = {"tag": args.tag, "host": args.host, "port": args.port, "user": args.user, "operator": name,
            "source": source.name, "source_metadata": metadata.name, "revision": revision, "pins": pins,
            "previous_launch": args.previous_launch, "state_root": args.state_root, "seed_archive": args.seed_archive}
    (work / "move.json").write_text(json.dumps(move, indent=1, sort_keys=True) + "\n")
    checked(validator + [shlex.join(["mkdir", "-m", "700", args.move_dir])], "create the move directory", 30)
    checked(["scp", "-q", "-r", "-o", "BatchMode=yes", str(work / name), str(source), str(metadata),
             str(work / "move.json"), f"{args.validator}:{args.move_dir}/"], "copy the move to the validator", 900)
    script, log = f"{args.move_dir}/{name}/pod_move.py", f"{args.move_dir}/run.log"
    checked(validator[:1] + ["-n"] + validator[1:] + [
        f"setsid nohup /usr/bin/python3 -B -u {shlex.quote(script)} run {shlex.quote(args.move_dir)}"
        f" >> {shlex.quote(log)} 2>&1 < /dev/null &"], "start the move on the validator", 30)
    print(json.dumps({"move_dir": args.move_dir, "log": log, "stages": [args.tag + "a", args.tag + "b"]}))


class Move:
    """One move directory on the validator: move.json, the rendered operator, the exported source and its task."""

    def __init__(self, directory: Path):
        self.dir = directory.resolve(strict=True)
        self.config = c = json.loads((self.dir / "move.json").read_bytes())
        self.previous = json.loads(Path(c["previous_launch"]).read_bytes())["argv"]
        self.operator = self.dir / c["operator"]
        rendered = json.loads((self.operator / "render-receipt.json").read_bytes())["operator"]
        self.slug, self.owner = rendered["slug"], rendered["owner"]
        self.base, self.shared = option(self.previous, "--base"), option(self.previous, "--shared")
        self.proof_path = option(self.previous, "--image-proof")
        self.proof = json.loads(Path(self.proof_path).read_bytes())
        self.seed, self.previous_tag = option(self.previous, "--runtime-seed"), option(self.previous, "--tag")
        self.stock, self.live = c["tag"] + "a", c["tag"] + "b"
        self.known, self.task, self.state = self.dir / "known_hosts", self.dir / "task", Path(c["state_root"])
        self.source, self.metadata = self.dir / c["source"], self.dir / c["source_metadata"]
        self.ssh = ["ssh", "-i", self.base + "/ssh_key", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
                    "-o", f"UserKnownHostsFile={self.known}", "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=10",
                    "-o", "ServerAliveCountMax=6", "-p", str(c["port"]), f"{c['user']}@{c['host']}"]

    def pod(self, script: str, what: str, timeout: int) -> str:
        return checked(self.ssh + [shlex.join(["sudo", "-n", "sh", "-c", script])], what, timeout)

    def worker(self, tag: str) -> str:
        return f"{self.shared}/workers/{tag}"

    def stock_reference(self, action: str, tag: str, *extra: str) -> str:
        python = "$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))[\"controller_python\"])' " \
                 + shlex.quote(self.shared + "/bootstrap-complete.json") + ")"
        return " ".join([python, "-B", "-u", shlex.quote(self.worker(tag) + "/recovery-operator/stock_reference.py"),
                         action, "--shared", shlex.quote(self.shared), "--tag", shlex.quote(tag), *map(shlex.quote, extra)])

    def access(self) -> None:
        c = self.config
        scan = checked(["ssh-keyscan", "-p", str(c["port"]), "-t", "ed25519", c["host"]], "ssh-keyscan", 30)
        lines = [line for line in scan.splitlines() if line and not line.startswith("#")]
        if len(lines) != 1:
            raise SystemExit(f"ssh-keyscan returned {len(lines)} ed25519 keys for the pod")
        self.known.write_text(lines[0] + "\n")
        self.known.chmod(0o600)
        # connect_worker.py and the relay reach the pod without a port argument: the validator's ssh config names it.
        config = Path.home() / ".ssh/config"
        text = config.read_text() if config.exists() else ""
        block = re.search(rf"^Host {re.escape(c['host'])}\n((?:[ \t].*\n?)*)", text, re.M)
        if block is None:
            with config.open("a") as stream:
                stream.write(f"\nHost {c['host']}\n  Port {c['port']}\n  IdentityFile {self.base}/ssh_key\n"
                             "  IdentitiesOnly yes\n")
        elif not re.search(rf"^\s+Port {c['port']}$", block.group(1), re.M):
            raise SystemExit(f"{config} already names {c['host']} with another port")
        self.task.mkdir(mode=0o700, exist_ok=True)
        # The sealed workload and tokenizer are not rendered: the previous stage's operator directory holds them.
        replay = f"replay-inputs-{self.slug}.tar"
        if not (self.operator / replay).exists():
            stage = next(Path(value) for value in self.previous if value.endswith("/launch_stage.py"))
            os.link(stage.parent / replay, self.operator / replay)
        print("POD_BOOT", self.pod("cat /proc/sys/kernel/random/boot_id", "reach the pod", 30).strip(), flush=True)

    def prepare_pod(self) -> None:
        """Pull the worker image from its published registry and restore the retained runtime seed."""
        published, image_id = self.proof["published"], self.proof["worker_image_id"]
        archive = Path(self.config["seed_archive"])
        seed = Path(self.seed)
        q = shlex.quote
        inspect = f"docker image inspect --format '{{{{.Id}}}}' {q(published)}"
        script = "\n".join([
            "set -eu",
            f"{inspect} > /dev/null 2>&1 || docker pull {q(published)}",
            f'test "$({inspect})" = {q(image_id)}',
            'test -e /usr/bin/docker || ln -s "$(command -v docker)" /usr/bin/docker',
            f"if [ ! -d {q(str(seed))} ]; then",
            f"  mkdir -p {q(str(seed.parent))}",
            f"  cd {q(str(archive.parent))} && awk -v name={q(archive.name)} '$2 == name' SHA256SUMS | sha256sum -c -",
            f"  tar -x --numeric-owner -f {q(str(archive))} -C {q(str(seed.parent))}",
            "fi",
            f"test -d {q(str(seed))} && test -f {q(str(seed) + '.json')}",
        ])
        print(self.pod(script, "prepare the pod", 7200)[-2000:], flush=True)

    def commission(self, tag: str) -> None:
        c, prev = self.config, self.previous
        argv = ["/usr/bin/python3", "-B", str(self.operator / "launch_node.py"), "--operator", str(self.operator),
                "--task-dir", str(self.task), "--base", self.base, "--image-proof", self.proof_path,
                "--source", str(self.source), "--source-metadata", str(self.metadata), "--source-revision", c["revision"],
                "--image-digest", self.proof["worker_image"].rsplit("@sha256:", 1)[1], "--",
                "--host", c["host"], "--pod-port", str(c["port"]), "--user", c["user"], "--known-hosts", str(self.known),
                "--shared", self.shared, "--tag", tag, "--lane", option(prev, "--lane"),
                "--model-input", option(prev, "--model-input"), "--model-token", option(prev, "--model-token"),
                "--runtime-seed", self.seed]
        # Without an agent, launch_stage starts the stage's own at /run/cacheon-ssh-<tag>.sock, which its units use.
        env = {key: value for key, value in os.environ.items() if key != "SSH_AUTH_SOCK"}
        pid = json.loads(checked(argv, "launch_node " + tag, 60, env=env))["pid"]
        while Path("/proc", str(pid)).exists():
            time.sleep(5)
        log = (self.task / f"stage-launch-{tag}.log").read_text()
        if "COMMISSIONING_ONLY" not in log:
            raise SystemExit(f"launch_stage {tag} failed:\n{log[-4000:]}")
        root = self.worker(tag)
        deadline = time.monotonic() + 7200
        while not (status := self.pod(f"cat {root}/commission-{tag}.exit 2>/dev/null || true", "read the commission", 30).strip()):
            if time.monotonic() > deadline:
                raise TimeoutError(f"commission {tag} exceeded its wall bound")
            time.sleep(15)
        if status != "0":
            raise SystemExit(f"commission {tag} exited {status}:\n" + self.pod(
                f"tail -n 30 {root}/logs/commission-runner.log; tail -n 20 {root}/logs/*-{tag}.log", "read its logs", 30))
        print("COMMISSIONED", tag, flush=True)

    def stock_commission(self) -> None:
        self.commission(self.stock)

    def stock_start(self) -> None:
        print(self.pod(self.stock_reference("start", self.stock), "start the stock pair", 600)[-1500:], flush=True)

    def stock_verify(self) -> None:
        """Run the verifier detached on the pod, on a vCPU outside the pin table, and wait for its exit status."""
        out = self.worker(self.stock) + "/stock-reference"
        pins = self.config["pins"]
        used = {*pins["scheduler_starts"], *pins["helper_starts"], *(c for cpus in pins["client_cpus"].values() for c in cpus)}
        cpu = min(set(range(len(used) + 1)) - used)
        inner = (f"taskset -c {cpu} nice -n 19 {self.stock_reference('verify', self.stock)} > {out}/verify.log 2>&1;"
                 f" echo $? > {out}/verify.exit")
        self.pod(f"test -e {out}/verify.log || setsid nohup sh -c {shlex.quote(inner)} > /dev/null 2>&1 < /dev/null &",
                 "start the stock verifier", 60)
        deadline = time.monotonic() + 16200
        while not (status := self.pod(f"cat {out}/verify.exit 2>/dev/null || true", "read the stock verifier", 30).strip()):
            if time.monotonic() > deadline:
                raise TimeoutError("the stock pair exceeded its wall bound")
            time.sleep(60)
        tail = self.pod(f"tail -c 3000 {out}/verify.log", "read the stock verifier log", 30)
        if status != "0":
            raise SystemExit(f"stock verification exited {status}:\n{tail}")
        print(tail, flush=True)

    def stock_stop(self) -> None:
        print(self.pod(self.stock_reference("stop", self.stock), "stop the stock pod service", 300), flush=True)

    def mainnet_commission(self) -> None:
        self.commission(self.live)

    def import_(self) -> None:
        print(self.pod(self.stock_reference("import", self.live, "--stock-tag", self.stock), "import the stock pair", 1800),
              flush=True)

    def connect_worker(self, *extra: str) -> None:
        c = self.config
        argv = [VENV, "-uB", str(self.operator / "operator/connect_worker.py"), "--base", self.base,
                "--state-root", str(self.state), "--host", c["host"], "--user", c["user"], "--pod-port", str(c["port"]),
                "--pod-stage", f"{self.worker(self.live)}/node/pair-{self.live}", "--known-hosts", str(self.known),
                "--image-proof", self.proof_path, "--tag", self.live, "--owner", self.owner,
                "--source", str(self.source), "--source-metadata", str(self.metadata), *extra]
        env = {**os.environ, "SSH_AUTH_SOCK": f"/run/cacheon-ssh-{self.live}.sock"}
        print(checked(argv, "connect_worker " + " ".join(extra), 900, env=env)[-3000:], flush=True)

    def connect(self) -> None:
        self.connect_worker()

    def continuation(self) -> None:
        workers = self.state / "workers"
        print(checked([VENV, "-B", __file__, "continuation", "--source",
                       str(self.state / ("source-" + self.config["revision"])), "--stage", str(workers / self.live),
                       "--previous", str(workers / self.previous_tag),
                       "--snapshot", str(self.dir / "intake-before-continuation.sqlite3")],
                      "the reward continuation", 300), flush=True)

    def start(self) -> None:
        def systemctl(verb: str, unit: str) -> str:
            return subprocess.run(["systemctl", verb, unit], capture_output=True, text=True, timeout=30).stdout.strip()

        # The previous stage's supervisor would keep waiting on its (lost) pod; the intake returns its lease to the queue.
        for unit in (f"cacheon-eval-{self.slug}-relay-pair-{self.previous_tag}.service",
                     f"cacheon-eval-{self.slug}-pair-{self.previous_tag}.service"):
            if systemctl("is-active", unit) not in ("inactive", "failed"):
                checked(["systemctl", "stop", unit], "stop " + unit, 300)
        self.connect_worker("--start")
        units = {unit: systemctl("is-active", unit) for unit in (
            f"cacheon-eval-{self.slug}-pair-{self.live}.service",
            f"cacheon-eval-{self.slug}-relay-pair-{self.live}.service", f"cacheon-{self.slug}-intake.service")}
        # A validator reboot left a never-enabled intake down for 20 hours (2026-11-17).
        units["intake enabled"] = systemctl("is-enabled", f"cacheon-{self.slug}-intake.service")
        print("UNITS", json.dumps(units), flush=True)
        if set(units.values()) - {"active", "enabled"} or units["intake enabled"] != "enabled":
            raise SystemExit("the started stage's units are not all active, or the intake unit is not enabled")


def run(directory: Path, first: str) -> None:
    move = Move(directory)
    os.chdir(move.dir)
    for step in STEPS[STEPS.index(first):]:
        print("STEP", step, time.strftime("%H:%M:%SZ", time.gmtime()), flush=True)
        getattr(move, step.replace("-", "_") if step != "import" else "import_")()
    print("MOVE_DONE", move.live, time.strftime("%H:%M:%SZ", time.gmtime()), flush=True)


def continuation(args: argparse.Namespace) -> None:
    """Map the new stage's reward group onto the previous stage's, after checking it on an in-memory copy."""
    sys.path.insert(0, str(args.source))
    from cacheon.chain.evaluation_order import payment_bar, reward_group_continuations
    from cacheon.stack_manifest import EvaluationStackManifest

    def group(stage: Path):
        stack = EvaluationStackManifest.from_dict(json.loads((stage / "qualification-incumbent-stack.json").read_bytes()))
        return stack, f"{stack.arena_digest}:{stack.digest}"

    def declare(db, values: dict) -> None:
        with db:
            db.execute("INSERT INTO metadata(key, value) VALUES('reward_group_continuations', ?) "
                       "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (json.dumps(values, sort_keys=True),))

    (stack, later), (earlier_stack, earlier) = group(args.stage), group(args.previous)
    dispatch = json.loads((args.stage / "mainnet-screen-dispatcher.json").read_bytes())
    arena = dispatch["arena_service_manifest"]["runtime"]["arena_id"]
    live = sqlite3.connect(dispatch["intake_db"], timeout=90)
    trial = sqlite3.connect(":memory:")
    live.backup(trial)
    live.row_factory = trial.row_factory = sqlite3.Row
    values = reward_group_continuations(trial)
    if values.get(later) == earlier:
        print("CONTINUATION_PRESENT", later, "->", earlier)
        return
    if later == earlier or later in values:
        raise SystemExit(f"{later} cannot continue {earlier}: {values}")
    order = "ORDER BY block, event_index, event_subindex, hotkey, content_hash"
    rows = [row[0] for row in trial.execute(
        f"SELECT reservation_id FROM reservations WHERE competition_arena=? AND status='published' {order}", (arena,))]
    rows = rows or [row[0] for row in trial.execute(
        f"SELECT reservation_id FROM reservations WHERE competition_arena=? {order.replace(',', ' DESC,')} DESC LIMIT 1",
        (arena,))]
    before = {rid: payment_bar(trial, earlier_stack, rid) for rid in rows}
    declare(trial, {**values, later: earlier})
    after = {rid: payment_bar(trial, stack, rid) for rid in rows}
    if after != before:
        raise SystemExit(f"the continued stage would read other payment bars: {before} -> {after}")
    snapshot = sqlite3.connect(args.snapshot)
    live.backup(snapshot)
    snapshot.close()
    os.chmod(args.snapshot, 0o600)
    if reward_group_continuations(live) != values:
        raise SystemExit("reward_group_continuations changed during the check")
    declare(live, {**values, later: earlier})
    print("CONTINUATION", later, "->", earlier, "payment bars", json.dumps(after), "snapshot", args.snapshot)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    start = commands.add_parser("launch", help="render, authorize and start the move (operator machine)")
    start.add_argument("--arena", required=True, help="directory name under examples/arena_inputs")
    start.add_argument("--host", required=True)
    start.add_argument("--port", required=True, type=int)
    start.add_argument("--user", default="root")
    start.add_argument("--fast", required=True, help="the pod's fast vCPUs: one scheduler per GPU in order, then the client")
    start.add_argument("--helpers", required=True, help="one helper vCPU per GPU, in GPU order")
    start.add_argument("--tag", required=True, help="stage tag prefix; the stages are <tag>a (stock) and <tag>b (mainnet)")
    start.add_argument("--validator", required=True, help="user@host of the validator VM")
    start.add_argument("--move-dir", required=True, help="new directory on the validator")
    start.add_argument("--previous-launch", required=True,
                       help="validator path of the previous mainnet stage's stage-launch-<tag>-started.json")
    start.add_argument("--state-root", required=True, help="validator state root connect_worker.py uses")
    start.add_argument("--seed-archive", required=True, help="pod path of the retained runtime seed tar, SHA256SUMS beside it")
    start.add_argument("--retained-inputs", required=True, type=Path)
    start.add_argument("--quality-controls", type=Path)
    proceed = commands.add_parser("run", help="carry out a launched move (validator)")
    proceed.add_argument("move_dir", type=Path)
    proceed.add_argument("--from", dest="first", choices=STEPS, default=STEPS[0])
    declare = commands.add_parser("continuation", help="declare the reward continuation (run by `run`)")
    for name in ("source", "stage", "previous", "snapshot"):
        declare.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    os.umask(0o077)
    if args.command == "launch":
        launch(args)
    elif args.command == "run":
        run(args.move_dir, args.first)
    else:
        continuation(args)


if __name__ == "__main__":
    main()
