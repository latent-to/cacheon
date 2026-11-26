"""Public request contents determine the established replay's cold order and replica affinity."""

import hashlib
import json

from cacheon.eval.agent_replay import _request_place


def test_ordinary_key_matches_mainnet_while_pause_order_uses_the_trace():
    messages = [{"role": "system", "content": "assistant"},
                {"role": "user", "content": "[rid:abc]\n\nhello"}]
    head = [("system", json.dumps("assistant")), ("user", "hello")]
    mainnet_key = int(hashlib.sha256(json.dumps(head).encode()).hexdigest(), 16)
    first, second = (2, 0, 1.25), (4, 3, 0.0)
    assert _request_place(messages, first, pause_order=False) == (mainnet_key, None, None)
    messages[1]["content"] = "hello"
    messages.append({"role": "user", "content": "continue"})
    assert _request_place(messages, second, pause_order=False) == (mainnet_key, None, None)
    assert _request_place(messages, second, pause_order=True) == second
