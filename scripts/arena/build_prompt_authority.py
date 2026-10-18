"""Assemble one arena's sealed prompt authority from its exact-length GSM8K pools.

Runs inside the arena's worker image, where the model tokenizer and the installed cacheon are the
ones the qualification engines use. The previous arena's sealed authority is the reference: the
judge digest function must reproduce its sealed values from its own inputs (self-test), the corpus
commitment must match, and the cell structure and the selection policy digest carry over unchanged.
The engine config is laid out as the pod's prepare_worker rewrites it, owned top-level fields from
the arena's engine-config.json and everything else under ``engine_kwargs``, so that rewrite is a
no-op and the digest reported here is the one the pod seals. The model content digest comes from
the provision receipt the commission binds; nothing is typed by hand.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

CELLS = [{"cell_id": "s8k", "concurrency": 128, "input_tokens": 8192, "output_tokens": 1024, "timed_reads": 2},
         {"cell_id": "l65k", "concurrency": 24, "input_tokens": 65536, "output_tokens": 4096, "timed_reads": 3}]
ROUTING = ["s8k", "l65k", "s8k", "s8k", "l65k", "l65k", "l65k"]
OMITTED_BY_DESIGN = {"deterministic"}  # arena kits leave it out; the authority keeps the reference's value


def sealed(nested):
    """Nested tuples for the judge: batches, prompts, one accepted alternative, token ids."""
    return tuple(tuple(tuple(tuple(int(t) for t in seq) for seq in alt) for alt in batch) for batch in nested)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arena-inputs", required=True, type=Path, help="examples/arena_inputs/<arena>, mounted")
    parser.add_argument("--pools", required=True, type=Path, help="pool_<tokens>.json per cell and gsm8k_train.jsonl")
    parser.add_argument("--judge-dir", required=True, type=Path, help="directory holding numeric_answer_judge.py")
    parser.add_argument("--reference", required=True, type=Path, help="the previous arena's sealed authority")
    parser.add_argument("--model", required=True, type=Path, help="model root (tokenizer) in the worker image")
    parser.add_argument("--model-receipt", required=True, type=Path, help="the model provision receipt")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.judge_dir.resolve(strict=True)))
    import numeric_answer_judge as judge
    from transformers import AutoTokenizer
    from cacheon.eval.b300_arena_definition import engine_template, workload as parse_workload
    from cacheon.eval.b300_deployment import _prompt_batches, _prompt_identity, _stable_json
    from cacheon.eval.tokenizer_identity import tokenizer_identity_digest

    arena = args.arena_inputs.resolve(strict=True)
    operator = json.loads((arena / "operator.json").read_bytes())
    kit_bytes = (arena / "engine-config.json").read_bytes()
    kit = json.loads(kit_bytes)
    content_digest = json.loads(args.model_receipt.read_bytes())["content_digest"]
    corpus = hashlib.sha256((args.pools / "gsm8k_train.jsonl").read_bytes()).hexdigest()
    pools = {}
    for cell in CELLS:
        pool = json.loads((args.pools / f"pool_{cell['input_tokens']}.json").read_bytes())
        assert pool["schema"] == "cacheon-c18-prompt-pool-v2" and pool["source_sha256"] == corpus
        assert pool["input_tokens"] == cell["input_tokens"]
        assert len(pool["prompts"]) == len(pool["golds"]) == pool["pool_size"]
        assert all(isinstance(g, list) and g for g in pool["golds"])
        pools[cell["cell_id"]] = pool
    ref = json.loads(args.reference.read_bytes())
    # Self-test: the judge digest function reproduces the reference authority's sealed values from its own inputs.
    assert judge.numeric_hidden_judge_digest(
        hidden_corpus_commitment=ref["hidden_corpus_commitment"],
        accepted_token_subsequences=sealed(ref["accepted_token_subsequences"])) == ref["hidden_judge_digest"]
    assert judge.hidden_task_policy_digest() == ref["hidden_task_policy_digest"], "task policy constant"
    assert ref["hidden_corpus_commitment"] == corpus, "same GSM8K source as the reference"
    assert CELLS == ref["workload_cells"] and ROUTING == ref["prompt_batch_cells"], "structure mirrors the reference"
    width = {cell["cell_id"]: cell["concurrency"] for cell in CELLS}
    cursor = dict.fromkeys(width, 0)
    batches, accepted = [], []
    for cell in ROUTING:
        pool, w, i = pools[cell], width[cell], cursor[cell]
        prompts, golds = pool["prompts"][i:i + w], pool["golds"][i:i + w]
        assert len(prompts) == w == len(golds), f"{cell}: pool exhausted at {i}"
        batches.append(prompts)
        accepted.append([[list(g)] for g in golds])
        cursor[cell] = i + w
    flat = [p for b in batches for p in b]
    assert len(set(flat)) == len(flat) == sum(width[c] for c in ROUTING)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    lengths = {cell["cell_id"]: cell["input_tokens"] for cell in CELLS}
    assert all(len(tokenizer(p).input_ids) == lengths[c] for b, c in zip(batches, ROUTING) for p in b[:4]), "exact-length recheck"
    tokenizer_digest = tokenizer_identity_digest(tokenizer, content_digest)
    assert len(tokenizer_digest) == 64 and int(tokenizer_digest, 16) >= 0

    owned = set(ref["engine_config"]) - {"engine_kwargs"} | {"max_running_requests"}
    assert owned - set(kit) == OMITTED_BY_DESIGN, owned - set(kit)
    engine_config = {k: v for k, v in ref["engine_config"].items() if k != "engine_kwargs"}
    engine_config.update({k: v for k, v in kit.items() if k in owned})
    engine_config["engine_kwargs"] = {k: v for k, v in kit.items() if k not in owned}
    # The pod's prepare_worker rewrite, replayed: it must leave this layout unchanged.
    replay = dict(engine_config)
    replay.update({k: v for k, v in kit.items() if k in owned})
    replay["engine_kwargs"] = {k: v for k, v in kit.items() if k not in owned}
    assert replay == engine_config, "prepare_worker rewrite is not a no-op"
    cursor_s = sum(width[c] for c in ROUTING if c == "s8k")
    cursor_l = sum(width[c] for c in ROUTING if c == "l65k")
    authority = {
        "accepted_token_subsequences": accepted, "engine_config": engine_config, "hidden_corpus_commitment": corpus,
        "hidden_judge_digest": judge.numeric_hidden_judge_digest(hidden_corpus_commitment=corpus,
                                                                 accepted_token_subsequences=sealed(accepted)),
        "hidden_task_policy_digest": judge.hidden_task_policy_digest(),
        "model_profile_key": operator["model_profile_key"], "prompt_batch_cells": ROUTING, "prompt_batches": batches,
        "prompt_seed_scheme": f"{arena.name}-gsm8k-v1.s{cursor_s}-l{cursor_l}",
        "registered_targets": ["forward_pass", "prefix_cache"], "schema": "cacheon-private-prompt-authority-v1",
        "selection_policy_digest": ref["selection_policy_digest"], "tokenizer_digest": tokenizer_digest,
        "workload_cells": CELLS,
    }
    assert set(authority) == set(ref), set(authority) ^ set(ref)
    payload = json.dumps(authority, sort_keys=True).encode() + b"\n"  # the pod rewrites with exactly this serialization
    with args.out.open("xb") as handle:
        handle.write(payload)
    args.out.chmod(0o400)
    path, value, sha = _stable_json(args.out, "prompt authority")
    parsed = _prompt_batches(value)
    template = engine_template(value)
    workload = parse_workload(value, parsed, sha)
    print(json.dumps({
        "path": str(path), "sha256": sha, "bytes": len(payload), "batches": [len(b) for b in parsed],
        "kit_sha256": hashlib.sha256(kit_bytes).hexdigest(), "inherited_from_reference": sorted(OMITTED_BY_DESIGN),
        "engine_kwargs": sorted(template.engine_kwargs), "engine_env": dict(template.engine_env),
        "max_running_requests": template.max_running_requests, "tp_size": template.tp_size,
        "workload_digest": workload.digest, "seed_scheme": workload.prompt_seed_scheme,
        "cells": [c.cell_id for c in workload.cells], "hidden_judge_digest": authority["hidden_judge_digest"],
        "hidden_corpus_commitment": corpus, "tokenizer_digest": tokenizer_digest,
        "tokenizer_class": type(tokenizer).__qualname__, "identity_keys": sorted(_prompt_identity(value, sha)),
    }))


if __name__ == "__main__":
    main()
