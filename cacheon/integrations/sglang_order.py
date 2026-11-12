"""Serve the order of the scheduler's waiting queue from a bundle and hold the answer to a permutation.

The address ``queue_order`` names a question SGLang's scheduler asks before every
prefill batch: in which order are the waiting requests admitted. A bundle naming it
supplies ``entry(view)``. Stock orders the queue first; the view lists it in that
order, one row of plain numbers per request: tokens still to prefill, tokens the cache
serves, and prefill tokens processed since the request arrived (stock's aging clock,
equal on every rank). No token ids, request ids or output length: the replay fixes
output lengths, so a policy that saw one would read the future. The cache count is
stock's own match, which it runs per waiting request only under a cache-aware policy;
under ``fcfs`` every row reads zero, so an arena opening this address seals one.

The entry returns the positions of the view in the order to admit. Any permutation
yields the same tokens for every request; answers differ only in who waits. Anything
else is refused and receipted as the candidate's: nothing returns to stock's order in
a candidate's name. Stock's admission code then applies the order under its own
budgets, so a bundle chooses who is next, never how much is admitted. Every TP rank
asks with the same view and must get the same answer, so the entry must be a pure
function of the view.
"""

from __future__ import annotations

import functools
import sys

from cacheon import receipts as _receipts
from cacheon.capabilities import CallDescriptor
from cacheon.registry import REGISTRY, KernelRegistry

ADDRESS = "queue_order"
_MODULE = "sglang.srt.managers.schedule_policy"
_STOCK = "_cacheon_stock_order"


def _view(queue, processed_tokens: int) -> tuple[tuple[int, int, int], ...]:
    return tuple(
        (max(0, len(req.origin_input_ids) + len(req.output_ids) - req.num_matched_prefix_tokens),
         int(req.num_matched_prefix_tokens),
         max(0, processed_tokens - getattr(req, "arrival_processed_tokens", processed_tokens)))
        for req in queue
    )


def _ordered(impl, queue, processed_tokens: int):
    """Ask the candidate for the order and apply it, refusing whatever is not a permutation."""

    if impl.prepare is not None:
        failure = TypeError(f"{ADDRESS}: an order entry takes no prepare")
        _receipts.failed(ADDRESS, failure, phase="prepare")
        raise failure
    order = _receipts.invoke(ADDRESS, impl.entry, _view(queue, processed_tokens))
    if (not isinstance(order, (list, tuple)) or len(order) != len(queue)
            or any(type(i) is not int for i in order) or sorted(order) != list(range(len(queue)))):
        failure = ValueError(f"{ADDRESS}: entry must return each of the {len(queue)} queue positions once")
        _receipts.failed(ADDRESS, failure, phase="entry")
        raise failure
    _receipts.completed(ADDRESS)
    return [queue[i] for i in order]


def install(registry: KernelRegistry = REGISTRY) -> None:
    """Wrap the policy's ordering so stock decides first and the candidate reorders its answer."""

    policy = getattr(sys.modules.get(_MODULE), "SchedulePolicy", None)
    stock = getattr(policy, "calc_priority", None)
    if stock is None or hasattr(stock, _STOCK):
        return

    @functools.wraps(stock)
    def calc_priority(self, waiting, *args, **kwargs):
        result = stock(self, waiting, *args, **kwargs)
        impl = registry.select(ADDRESS, CallDescriptor()).impl
        # Engines before the 2026-11 nightlies sort the caller's list and return no queue.
        queue = result if isinstance(result, list) else waiting
        if impl is None or not queue:
            return result
        processed = kwargs.get("processed_tokens", args[1] if len(args) > 1 else 0)
        queue[:] = _ordered(impl, queue, processed)
        return result

    setattr(calc_priority, _STOCK, stock)
    policy.calc_priority = calc_priority
