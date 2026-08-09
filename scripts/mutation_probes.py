#!/usr/bin/env python3
"""Mutation probes: do the tests detect real invariant breaks?

Run from the repository root on a disposable checkout:

    python scripts/mutation_probes.py

A green suite proves the code passes its tests; it cannot prove the
tests would notice the code breaking. Each probe below applies one
targeted invariant break, runs the test files that own that behavior,
and restores the file via git. Run it on hardware sessions after
refactors that move authority code.

Each probe applies one targeted behavior break to a production module,
runs the test files that claim to cover that behavior, and restores the
file. DETECTED means the tests failed under mutation (good). NOT-DETECTED
means the tests passed with the invariant broken (coverage gap).
"""
import subprocess
import sys
from pathlib import Path

REPO = Path.cwd()
PY = sys.executable

PROBES = [
    (
        "P1-post-publication-release-fence",
        "cacheon/chain/evaluation_recovery.py",
        "            phase_permitted = previous.phase in {\n"
        "                RecoveryPhase.CLAIMED,\n"
        "                RecoveryPhase.PREPARED,\n"
        "            }",
        "            phase_permitted = previous.phase in {\n"
        "                RecoveryPhase.CLAIMED,\n"
        "                RecoveryPhase.PREPARED,\n"
        "                RecoveryPhase.REQUEST_READY,\n"
        "            }",
        [
            "tests/test_evaluation_recovery_store.py",
            "tests/test_evaluation_recovery_transitions.py",
        ],
    ),
    (
        "P2-refusal-hmac-verify",
        "cacheon/chain/execution_disposition.py",
        "        or not hmac.compare_digest(\n"
        "            refusal.auth_tag, _refusal_tag(credential, refusal.digest)\n"
        "        )",
        "        or False",
        ["tests/test_execution_disposition.py", "tests/test_remote_worker_request_plan.py"],
    ),
    (
        "P3-judge-tolerance",
        "cacheon/eval/numeric_answer_judge.py",
        "passed = candidate is not None and abs(candidate - gold) <= 1e-4",
        "passed = candidate is not None and abs(candidate - gold) <= 1e6",
        ["tests/test_numeric_answer_judge.py"],
    ),
    (
        "P4-supervisor-exception-to-disposition",
        "cacheon/chain/standing_cpu_supervisor.py",
        "            raise StandingCpuSupervisorError(\n"
        "                f\"stage {stage!r} failed closed without a typed disposition: \"\n"
        "                f\"{type(exc).__name__}: {exc}\"\n"
        "            ) from None",
        "            return SupervisorStageResult(\n"
        "                stage=stage,\n"
        "                progressed=False,\n"
        "                disposition=\"hold\",\n"
        "                request_id=None,\n"
        "                lease_id=None,\n"
        "                hold_reason=\"stage_error\",\n"
        "                phase=SupervisorPhase.HOLD,\n"
        "            )",
        ["tests/test_standing_cpu_supervisor.py"],
    ),
    (
        "P5-count-quality-vacuous",
        "cacheon/eval/resident_count_quality.py",
        "    candidate_correct = rejudge_resident_count_observation(candidate, judge)",
        "    candidate_correct = stock_correct",
        ["tests/test_resident_count_quality.py", "tests/test_registered_resident_count_quality.py"],
    ),
]

ALL_TEST_FILES = sorted({t for probe in PROBES for t in probe[4]})


def run_tests(files: list[str]) -> tuple[int, str]:
    proc = subprocess.run(
        [PY, "-m", "pytest", "-q", *files],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    out = (proc.stdout + proc.stderr).strip()
    tail = out.splitlines()[-1] if out else "no output"
    return proc.returncode, tail


def main() -> None:
    if not (REPO / "cacheon").is_dir() or not (REPO / "tests").is_dir():
        print("ABORT: run from the repository root")
        raise SystemExit(2)
    dirty = subprocess.run(
        ["git", "status", "--short"], cwd=REPO, capture_output=True, text=True
    ).stdout.strip()
    if dirty:
        print(f"ABORT: repo not clean before probes:\n{dirty}")
        return

    code, tail = run_tests(ALL_TEST_FILES)
    print(f"BASELINE ({len(ALL_TEST_FILES)} files): rc={code} :: {tail}")
    if code != 0:
        print("ABORT: baseline is not green; probe verdicts would be meaningless")
        return

    for name, relpath, old, new, testfiles in PROBES:
        path = REPO / relpath
        src = path.read_text()
        if src.count(old) != 1:
            print(f"{name}: PATTERN-ERROR count={src.count(old)}")
            continue
        path.write_text(src.replace(old, new, 1))
        code, tail = run_tests(testfiles)
        verdict = "DETECTED" if code != 0 else "NOT-DETECTED"
        print(f"{name}: {verdict} :: {tail}")
        subprocess.run(["git", "checkout", "--", relpath], cwd=REPO, check=True)

    final = subprocess.run(
        ["git", "status", "--short"], cwd=REPO, capture_output=True, text=True
    ).stdout.strip()
    print(f"RESTORE-CHECK: {'clean' if not final else 'DIRTY: ' + final}")
    print("PROBES-DONE")


if __name__ == "__main__":
    main()
