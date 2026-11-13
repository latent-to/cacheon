"""Fit the arena's declared quality metrics on GLM control trajectories, using disjoint prompts for validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path


def _read(path, threshold):
    raw = path.read_bytes()
    prompts = {}
    for row in json.loads(raw):
        key = row["key"]
        nll = [-token[0] for token in row["tok"]]
        if key in prompts or not nll or any(not math.isfinite(x) or x < 0 for x in nll):
            raise ValueError(f"{path}: malformed teacher control")
        prompts[key] = {"tokens": len(nll), "mean_nll": statistics.mean(nll),
                        "tail_excess_nll": statistics.mean(max(0, x - threshold) for x in nll),
                        "tail_rate": statistics.mean(x > threshold for x in nll)}
    return prompts, hashlib.sha256(raw).hexdigest()


def _interval(samples, z):
    mean = statistics.mean(samples)
    width = z * statistics.stdev(samples) / math.sqrt(len(samples))
    return mean - width, mean + width


def _number(value):
    return format(value, ".12f").rstrip("0").rstrip(".") or "0"


def _stock_speed_noise(path):
    from cacheon.eval.goodput_runtime import GoodputPolicy, GoodputReadSet
    from cacheon.eval.service_capacity import fixed_work_rate, steady_span_cost
    from cacheon.stack_identity import canonical_json_bytes

    attempt = json.loads(path.read_bytes())
    witness = attempt["speed_witness"] if "speed_witness" in attempt else attempt["reports"][0]["speed_witness"]
    policy = GoodputPolicy.from_dict(witness["resident_policy"]["goodput"])
    reads = GoodputReadSet.from_dict(witness["goodput"])
    count = len(reads.loads) * reads.window_limit
    boots = []
    for i in range(reads.reference_boots):
        costs = {}
        for r in reads.incumbent[i * count:(i + 1) * count]:
            work = reads.work(r.load)
            cost = (steady_span_cost(r, work, policy.steady_span) if policy.steady_span else
                    fixed_work_rate(r, work, wall_time=True).elapsed_s)
            costs.setdefault(r.load, []).append(cost)
        boots.append(statistics.mean(math.log(statistics.mean(v)) for v in costs.values()))
    stock = witness["goodput"]["value"]["incumbent"][:reads.reference_boots * count]
    return hashlib.sha256(canonical_json_bytes(stock)).hexdigest(), statistics.stdev(boots)


def measure(files, policy, speed_witness):
    """Grade paired stock, faithful and damaged controls and return the measured commissioning observations."""
    from cacheon.stack_identity import canonical_json_bytes

    inputs = {arm: _read(path, float(policy["nll_tail_threshold"])) for arm, path in files.items()}
    reference = inputs["baseline"][0]
    if any(set(arm) != set(reference) or any(arm[k]["tokens"] != reference[k]["tokens"] for k in reference)
           for arm, _ in inputs.values()):
        raise ValueError("quality controls differ in prompt membership or token budget")
    # Split on the prompt identity, shared across all model arms. The selection precedes reading any outcome.
    order = sorted(reference, key=lambda k: hashlib.sha256(("glm-pd-quality-calibration-v1/" + k).encode()).digest())
    size = policy["select_count"]
    if len(order) < size * 2:
        raise ValueError("quality controls need separate fitting and validation panels of the commissioned size")
    z = float(policy["familywise_z"])
    stock_sha, speed_noise = _stock_speed_noise(speed_witness)
    observed, validation = [], {}
    for arm in ("negative", "positive", "stock"):
        values, sha = inputs[arm]
        sample = {"speed_noise": _number(speed_noise)}
        checks = []
        for offset in (0, size):
            bounds = {}
            for name in policy["limits"]:
                deltas = [values[key][name] - reference[key][name] for key in order[offset:offset + size]]
                scored = list(map(abs, deltas)) if arm == "stock" else deltas
                low, high = _interval(scored, z)
                limit = float(policy["limits"][name]["stock_envelope" if arm == "stock" else "candidate_delta"])
                bounds[name] = {"lower": low, "upper": high, "limit": limit}
                if offset == 0:
                    # The calibrated observation is the measured one-sided bound over the panel. Zero is the
                    # lower bound on deterioration. Preserve the signed interval in the receipt.
                    sample[name] = _number(max(0, low if arm == "negative" else high))
            passes = all(v["upper"] <= v["limit"] for v in bounds.values())
            fails = any(v["lower"] > v["limit"] for v in bounds.values())
            if (arm == "negative" and not fails) or (arm != "negative" and not passes):
                raise ValueError(f"{arm} control is unresolved or misclassified in panel {offset // size}: {bounds}")
            checks.append({name: {key: _number(value) for key, value in values.items()}
                           for name, values in bounds.items()})
        validation[arm] = checks
        observed.append({"control_kind": arm, "measurements": [
            {"name": m, "values": [v]} for m, v in sorted(sample.items())], "seed_digest": hashlib.sha256(canonical_json_bytes({
                "baseline": inputs["baseline"][1], "comparison": sha, "fit_prompt_keys": order[:size],
                "stock_speed": stock_sha, "nll_tail_threshold": policy["nll_tail_threshold"]})).hexdigest()})
    return {"source_files": {k: sha for k, (_, sha) in inputs.items()}, "policy": policy,
            "stock_speed_sha256": stock_sha, "stock_boot_log_sd": _number(speed_noise),
            "observations": observed, "bounds": validation}


def main():
    """Write the control receipt used by the operator renderer at commissioning."""
    parser = argparse.ArgumentParser(description=__doc__)
    for arm in ("baseline", "negative", "positive", "stock"):
        parser.add_argument("--" + arm, type=Path, required=True)
    parser.add_argument("--operator", type=Path, required=True)
    parser.add_argument("--speed-witness", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    controls = measure({k: getattr(args, k) for k in ("baseline", "negative", "positive", "stock")},
                       json.loads(args.operator.read_bytes())["quality"], args.speed_witness)
    with args.out.open("x") as out:
        json.dump(controls, out, indent=2, allow_nan=False)
        out.write("\n")


if __name__ == "__main__":
    main()
