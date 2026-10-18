"""Simulated Bittensor SDK for end-to-end validator subprocesses.

Only ``tests/test_validator_end_to_end.py`` puts this directory on a child
process's ``PYTHONPATH``; it is never on the normal import path, so it cannot
shadow an installed SDK. ``--network`` names the JSON state that
``tests.support.chain.write_state`` wrote, and ``--wallet-path`` names the same
file so the public validator wallet is the hotkey that state registered.
"""

from tests.support.chain import reopen_state, state_wallet


def Subtensor(network, **_options):  # noqa: N802 - the SDK's constructor name
    return reopen_state(network)


def Wallet(name=None, hotkey=None, path=None):  # noqa: N802 - the SDK's class name
    return state_wallet(path)
