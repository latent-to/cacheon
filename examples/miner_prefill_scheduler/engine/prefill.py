import sys

from sglang.srt.managers.schedule_policy import CacheAgnosticPolicy, get_disagg


def remaining(req):
    length = len(req.origin_input_ids) + len(req.output_ids)
    # These facts already exist after PD bootstrap. Neither runs another cache lookup nor acquires a page.
    promised = max(req.num_matched_prefix_tokens, req.disagg_decode_prefix_len)
    return max(1, length - promised)


def replace(stock):
    reported = False

    def schedule(policy, waiting, running_batch=None, processed_tokens=0, bootstrap_queue=()):
        nonlocal reported
        rows = stock(policy, waiting, running_batch, processed_tokens, bootstrap_queue)
        if (len(rows) < 2 or get_disagg().disaggregation_mode != "prefill" or
                policy.enable_priority_scheduling or policy.policy != CacheAgnosticPolicy.FCFS):
            return rows
        # The queue itself adopts each result. Restore arrival order before each choice. In every fourth
        # 16k-token interval the earliest arrivals get service even under a stream of cheap returning turns.
        rows.sort(key=lambda req: req.arrival_processed_tokens)
        if (processed_tokens // 16384) % 4 == 0:
            return rows
        first = remaining(rows[0])
        shortest = min(range(len(rows)), key=lambda index: remaining(rows[index]))
        work = remaining(rows[shortest])
        if shortest and first >= max(2 * work, 2 * policy.tree_cache.page_size):
            rows.insert(0, rows.pop(shortest))
            if not reported:
                print("CACHEON-PREFILL-WORK", len(rows), first, work, processed_tokens, file=sys.stderr, flush=True)
                reported = True
        return rows

    return schedule
