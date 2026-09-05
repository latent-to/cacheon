"""Freeze the identity-bearing surfaces of the tree and fail when they move.

A refactor that only moves code must not change what the chain, the sealed
packets, the docs checker, or a validator database observe. This script
regenerates ten such surfaces on CPU with no third-party import and compares
them with the committed baseline under ``scripts/surface_baseline/``:

- ``cli``: every argparse help text (root, subcommands, nested lease ops);
- ``seams``: the adapter rows, bindings, gates and target modules;
- ``capability_modules``: the declared private-caller manifest;
- ``sqlite_ddl``: the verbatim DDL of a fresh intake and recoverable store;
- ``catalog``: the target-catalog digest and every target's spec digests;
- ``content_hashes``: ``content_hash`` of every committed example bundle and
  stack fixture;
- ``modules``: ``__all__`` (or the public top-level names) and the exception
  classes of every module;
- ``digest_domains``: every digest domain literal;
- ``continuation_codecs``: every wire type key and field set of the
  continuation codecs, whose keys are ``module.qualname`` and therefore move
  with any file split;
- ``digest_goldens``: empty-store settlement, burn-projection and standalone
  claim digests.

Every section is a flat mapping written one key per line, so a baseline diff
names the exact help text, table, module or wire type that moved.

A non-empty diff is a surface change. When it is intended, run ``--write``
and put a ``surface-change:`` line per changed section in the pull request
body saying why; the reviewer reads the baseline diff. ``content_hashes``
drift is a consensus break or an identity epoch, never a routine refresh:
those vectors were cross-checked byte-identical on macOS arm64 (CPython
3.10/3.11) and Linux x86_64 (CPython 3.12) through 2026-08-31, with the
capture history in Git under ``tests/fixtures/golden_consensus_vectors.json``
before 2026-09-05.

Not captured yet: the settled-lineage golden (crown, lineage tip, settlement
digest). Its builders live in ``tests/test_chain_intake.py`` behind a pytest
import; moving them to ``tests/support/settlement.py`` adds that arm.

Run ``python scripts/surface_snapshot.py --check`` (CI hygiene job) or
``--write``.
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import difflib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASELINE_DIR = ROOT / "scripts" / "surface_baseline"
DIFF_KEYS = 20
DIFF_WIDTH = 400
THIRD_PARTY = ("bittensor", "numpy", "sglang", "torch")
STACK_FIXTURES = ("tests/fixtures/stack_norm_singleton",)

sys.path.insert(0, str(ROOT))
# argparse wraps help to the terminal width; pin it before any format_help.
os.environ["COLUMNS"] = "100"


def cli_surface() -> dict[str, str]:
    import cacheon.cli as cli

    parser = cli.build_parser()
    out = {"": parser.format_help()}

    def walk(node: argparse.ArgumentParser, prefix: str) -> None:
        for action in node._actions:
            if not isinstance(action, argparse._SubParsersAction):
                continue
            for name, child in action.choices.items():
                out[prefix + name] = child.format_help()
                walk(child, prefix + name + " ")

    walk(parser, "")
    return out


def seam_surface() -> dict[str, object]:
    from cacheon import seams

    out: dict[str, object] = {
        "binding_env_gates": dict(sorted(seams.SEAM_BINDING_ENV_GATES.items())),
        "target_modules": sorted(seams.TARGET_MODULES),
    }
    for row in seams.SEAM_ADAPTERS:
        out[f"adapter:{row.name}"] = dataclasses.asdict(row)
    for row in seams.SEAM_BINDINGS:
        out[f"binding:{row.binding_id}"] = dataclasses.asdict(row)
    return out


def capability_surface() -> dict[str, object]:
    from cacheon.capability_manifest import CAPABILITY_MODULES

    return {"modules": sorted(CAPABILITY_MODULES)}


def _fresh_dir() -> Path:
    private = Path(tempfile.mkdtemp()) / "private"
    private.mkdir(mode=0o700)
    return private


def _ddl(db, label: str) -> dict[str, object]:
    # Automatic primary-key indexes carry no DDL and their naming is a library
    # behaviour; the constraints that create them are already in the table SQL.
    return {
        f"{label}:{row[0]}:{row[1]}": {"tbl_name": row[2], "sql": row[3]}
        for row in db.execute(
            "SELECT type,name,tbl_name,sql FROM sqlite_master"
            " WHERE name NOT LIKE 'sqlite_autoindex_%' ORDER BY type,name,tbl_name"
        )
    }


def ddl_surface() -> dict[str, object]:
    from cacheon.chain.intake import FinalizedIntakeStore, IntakePolicy, IntakeScope
    from cacheon.chain.recoverable_intake import RecoverableFinalizedIntakeStore

    scope = IntakeScope("0x" + "0" * 64, 307)
    out: dict[str, object] = {}
    for label, cls in (
        ("intake", FinalizedIntakeStore),
        ("recoverable_intake", RecoverableFinalizedIntakeStore),
    ):
        store = cls(_fresh_dir() / "intake.sqlite3", IntakePolicy(), scope=scope)
        try:
            out[f"{label}:metadata"] = dict(
                store._db.execute("SELECT key,value FROM metadata ORDER BY key")
            )
            out.update(_ddl(store._db, label))
        finally:
            store.close()
    return out


def catalog_surface() -> dict[str, object]:
    from cacheon.target_catalog import default_target_catalog

    catalog = default_target_catalog()
    targets = sorted(row["target_id"] for row in catalog.snapshot()["targets"])
    out: dict[str, object] = {"catalog_digest": catalog.digest}
    for target in targets:
        out[f"target:{target}"] = {
            "spec_digest": catalog.target_spec_digest(target),
            "contract_digest": catalog.contract_digest(target),
        }
    return out


def content_hash_surface() -> dict[str, str]:
    from cacheon.bundle_hash import content_hash

    trees = sorted(
        f"examples/{entry.name}"
        for entry in (ROOT / "examples").iterdir()
        if entry.is_dir() and (entry / "manifest.toml").is_file()
    )
    return {tree: content_hash(ROOT / tree) for tree in (*trees, *STACK_FIXTURES)}


def _tracked_modules() -> list[str]:
    listing = subprocess.run(
        ["git", "ls-files", "cacheon/*.py"], cwd=ROOT, capture_output=True, text=True, check=True
    )
    return sorted(listing.stdout.split())


def _is_exception_class(node: ast.ClassDef) -> bool:
    for base in node.bases:
        name = base.id if isinstance(base, ast.Name) else getattr(base, "attr", "")
        if name.endswith("Error") or name in ("Exception", "BaseException"):
            return True
    return False


def module_surface() -> dict[str, object]:
    out: dict[str, object] = {}
    for relpath in _tracked_modules():
        tree = ast.parse((ROOT / relpath).read_text(encoding="utf-8"))
        exports: list[str] | None = None
        public: list[str] = []
        for node in tree.body:
            targets = (
                node.targets if isinstance(node, ast.Assign)
                else [node.target] if isinstance(node, ast.AnnAssign)
                else []
            )
            names = [target.id for target in targets if isinstance(target, ast.Name)]
            if "__all__" in names and exports is None:
                exports = sorted(ast.literal_eval(node.value))
            public.extend(name for name in names if name.isupper() and not name.startswith("_"))
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if not node.name.startswith("_"):
                    public.append(node.name)
        exceptions = sorted(
            node.name for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef) and _is_exception_class(node)
        )
        module = relpath[:-3].replace("/", ".").removesuffix(".__init__")
        entry: dict[str, object] = {"all": exports, "exceptions": exceptions}
        if exports is None:
            entry["public"] = sorted(set(public))
        out[module] = entry
    return out


def domain_surface() -> dict[str, list[str]]:
    suffixes = ("_DOMAIN", "_AUTHORITY", "_SCHEMA")
    out: dict[str, list[str]] = {}
    for relpath in _tracked_modules():
        found: set[str] = set()
        for node in ast.walk(ast.parse((ROOT / relpath).read_text(encoding="utf-8"))):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "canonical_digest"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                found.add(node.args[0].value)
            if isinstance(node, (ast.Assign, ast.AnnAssign)) and (
                isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)
            ):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                names = [target.id for target in targets if isinstance(target, ast.Name)]
                if any(name == "_domain" or name.endswith(suffixes) for name in names):
                    found.add(node.value.value)
        if found:
            out[relpath] = sorted(found)
    return out


def codec_surface() -> dict[str, object]:
    from cacheon.eval import qualification_continuation, qualification_runner
    from cacheon.eval import resident_count_continuation, resident_execution_evidence
    from cacheon.eval import resident_pair_retirement_checkpoint, resident_pair_speed_witness
    from cacheon.eval.continuation_codec import ContinuationCodec
    from cacheon.eval.resident_pair_crossover import ResidentPairCrossoverEvidence

    codecs = {
        "qualification_continuation": qualification_continuation._codec(),
        "registered_count": qualification_runner._registered_count_codec(),
        "resident_closure": qualification_runner._resident_closure_codec(),
        "resident_execution": resident_execution_evidence.EXECUTION_CODEC,
        "resident_count_raw": resident_count_continuation._RAW_CODEC,
        "speed_witness_slice": resident_pair_speed_witness._SLICE_CODEC,
        "retirement_session": resident_pair_retirement_checkpoint._SESSION_CODEC,
        "retirement_slice": resident_pair_retirement_checkpoint._SLICE_CODEC,
        "crossover": ContinuationCodec((ResidentPairCrossoverEvidence,)),
    }
    out: dict[str, object] = {}
    for name, codec in codecs.items():
        out[f"{name}:roots"] = [f"{r.__module__}.{r.__qualname__}" for r in codec.roots]
        for key in sorted({**codec._dataclasses, **codec._enums}):
            out[f"{name}:{key}"] = sorted(codec._hints.get(key, {}))
    return out


def digest_golden_surface() -> dict[str, object]:
    from cacheon.chain.intake import FinalizedIntakeStore, IntakePolicy, IntakeScope
    from cacheon.economics import (
        EmissionsPolicyManifest,
        GlobalRewardProjectionContext,
        MetagraphMember,
        StandingRewardClaim,
    )
    from cacheon.eval.oci_session_protocol import SlotAuditControl

    scope = IntakeScope("0x" + "0" * 64, 307)
    policy = EmissionsPolicyManifest(100, 20, 100_000)
    context = GlobalRewardProjectionContext(
        scope.digest,
        "validator",
        12,
        "0x" + f"{12:064x}",
        (MetagraphMember(0, "validator"), MetagraphMember(1, "burnsink")),
    )
    store = FinalizedIntakeStore(_fresh_dir() / "intake.sqlite3", IntakePolicy(), scope=scope)
    try:
        burn = store.build_burn_weight_projection(
            policy=policy, context=context, netuid=307, burn_hotkey="burnsink"
        )
        owner = store.build_subnet_owner_burn_weight_projection(
            policy=policy, context=context, netuid=307, burn_hotkey="burnsink",
            owner_coldkey="ownerck", owner_hotkey="burnsink", candidate_uids=(0, 1),
        )
        out: dict[str, object] = {
            "empty_settlement_state_digest": store.settlement_state_digest(),
            "burn_projection": {
                "digest": burn.digest, "authority_digest": burn.evaluation_state_digest,
            },
            "owner_burn_projection": {
                "digest": owner.digest, "authority_digest": owner.evaluation_state_digest,
            },
        }
    finally:
        store.close()
    out["standing_reward_claim"] = StandingRewardClaim(
        arena_digest="1" * 64, target_id="norm.rmsnorm", target_spec_digest="2" * 64,
        contribution_digest="3" * 64, hotkey="miner", speedup_ppm=1_050_000,
        crowned_block=10, retained_evidence_digest="4" * 64,
    ).digest
    out["slot_audit_control"] = SlotAuditControl(
        sample_rate_ppm=1000, minimum_calls=10,
        expected_slots=("norm.rmsnorm",), expected_member_count=1,
    ).digest
    return out


SECTIONS = {
    "cli": cli_surface,
    "seams": seam_surface,
    "capability_modules": capability_surface,
    "sqlite_ddl": ddl_surface,
    "catalog": catalog_surface,
    "content_hashes": content_hash_surface,
    "modules": module_surface,
    "digest_domains": domain_surface,
    "continuation_codecs": codec_surface,
    "digest_goldens": digest_golden_surface,
}


def _compact(value: object) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def _render(section: dict[str, object]) -> str:
    lines = [f" {json.dumps(key)}: {_compact(section[key])}" for key in sorted(section)]
    return "{\n" + ",\n".join(lines) + "\n}\n"


def _report(section: str, expected: str, actual: str) -> None:
    before = json.loads(expected) if expected else {}
    after = json.loads(actual)
    keys = sorted(set(before) | set(after))
    moved = [key for key in keys if before.get(key) != after.get(key)]
    print(f"--- surface change in {section}: {len(moved)} key(s)")
    for key in moved[:DIFF_KEYS]:
        old, new = before.get(key), after.get(key)
        if key not in before:
            print(f"+ {key}: {_compact(new)[:DIFF_WIDTH]}")
        elif key not in after:
            print(f"- {key}: {_compact(old)[:DIFF_WIDTH]}")
        elif isinstance(old, str) and isinstance(new, str):
            print(f"~ {key}:")
            print("\n".join(difflib.unified_diff(
                old.splitlines(), new.splitlines(), lineterm="", n=1,
            )))
        else:
            print(f"~ {key}:\n  - {_compact(old)[:DIFF_WIDTH]}\n  + {_compact(new)[:DIFF_WIDTH]}")
    if len(moved) > DIFF_KEYS:
        print(f"... {len(moved) - DIFF_KEYS} more key(s)")


def build() -> dict[str, str]:
    document = {section: _render(producer()) for section, producer in SECTIONS.items()}
    loaded = sorted(name for name in THIRD_PARTY if name in sys.modules)
    if loaded:
        raise SystemExit(f"surface_snapshot: third-party modules loaded: {loaded}")
    return document


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="freeze or check the identity surfaces")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="compare with the baseline")
    mode.add_argument("--write", action="store_true", help="rewrite the baseline")
    args = parser.parse_args(argv)
    document = build()
    BASELINE_DIR.mkdir(exist_ok=True)
    if args.write:
        for stale in BASELINE_DIR.glob("*.json"):
            if stale.stem not in document:
                stale.unlink()
        for section, text in document.items():
            (BASELINE_DIR / f"{section}.json").write_text(text, encoding="utf-8")
        print(f"surface_snapshot: wrote {len(document)} sections")
        return 0
    changed: list[str] = []
    for section, text in document.items():
        path = BASELINE_DIR / f"{section}.json"
        expected = path.read_text(encoding="utf-8") if path.is_file() else ""
        if expected == text:
            continue
        changed.append(section)
        _report(section, expected, text)
    if not changed:
        print(f"surface_snapshot: {len(document)} sections match the baseline")
        return 0
    if "content_hashes" in changed:
        print(
            "content_hashes moved: a consensus break or an unreviewed identity "
            "epoch, never a routine refresh"
        )
    print(
        f"surface change detected in {', '.join(changed)}; if intended, run "
        "`python scripts/surface_snapshot.py --write` and add a `surface-change:` "
        "line per section to the pull request body"
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
