"""Launch pinned AIPerf with repeatable cache markers and private client storage.

The stock CLI makes a new benchmark UUID even with a fixed random seed. That
UUID changes both prompt tokens and the first-prefix hash used for DP routing.
Use its existing single-run API so paired reads retain identical inputs.
"""

import os
from pathlib import Path
import sys
from importlib.metadata import version


def main() -> None:
    """Run the validator's single load using the existing AIPerf scheduler."""
    identity, output, *arguments = sys.argv[1:]
    # Equal benchmark identities must not share the client's mmap files when
    # incumbent and candidate run concurrently on the same host.
    os.environ["AIPERF_DATASET_MMAP_BASE_PATH"] = str(Path(output) / "client-dataset")
    if version("aiperf") != "0.13.0":
        raise RuntimeError("agent replay requires AIPerf 0.13.0")

    from aiperf.cli_commands.profile import app
    from aiperf.cli_runner import (
        _make_benchmark_run, _run_single_benchmark, _preflight_artifact_dir,
        _preflight_accuracy_deps, _preflight_fd_limit, _preflight_endpoint_ready,
    )
    from aiperf.common.bootstrap import register_sigusr1_faulthandler
    from aiperf.config.flags.resolver import resolve_config
    from aiperf.config.loader import build_benchmark_plan
    from aiperf.orchestrator.orchestrator import resolve_run_seed
    from aiperf.timing.phase.credit_counter import CreditCounter

    def all_returned(self) -> bool:
        # The session-target arm can freeze the sent count one root turn early; closing the phase at that count
        # left the last turn out of the export and voided a stock window (pair dsv41pd-k, 2026-10-27). Services
        # fork from this process, so the timing service inherits this.
        return self._final_requests_sent is not None and (
            self._requests_completed + self._requests_cancelled >= max(self._final_requests_sent, self._requests_sent))

    CreditCounter.check_all_returned_or_cancelled = all_returned
    _, bound, _ = app.parse_args(arguments)
    config = bound.arguments["cli_config"]
    plan = build_benchmark_plan(resolve_config(config, config.config_file))
    if not plan.is_single_run or plan.configs[0].artifacts.auto_plot:
        raise ValueError("agent replay client requires one load without plotting")
    register_sigusr1_faulthandler()
    _preflight_artifact_dir(plan)
    _preflight_accuracy_deps(plan)
    _preflight_fd_limit()
    _preflight_endpoint_ready(plan)
    run = _make_benchmark_run(
        plan.configs[0], benchmark_id=identity, variables=plan.variables,
        random_seed=resolve_run_seed(plan, plan.variations[0]),
    )
    _run_single_benchmark(run)


if __name__ == "__main__":
    main()
