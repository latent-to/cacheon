"""Simulated Bittensor SDK for end-to-end validator subprocesses.

Only ``tests.support.e2e.chain_env`` puts this directory on a child process's
``PYTHONPATH``; it is never on the normal import path, so it cannot shadow an
installed SDK. ``--network`` names the JSON state that
``tests.support.chain.write_state`` wrote, as a path or as ``wss://`` and that
path for configs that require an endpoint URL; ``--wallet-path`` names the same
file so the public validator wallet is the hotkey that state registered. A
wallet opened without a path is that ``--hotkey`` name's: a miner's, or the
weight gateway's authority.
"""

from tests.support.chain import LiveEndpoint, SimulatedKeypair, state_wallet

Keypair = SimulatedKeypair


def Subtensor(network, **_options):  # noqa: N802 - the SDK's constructor name
    return LiveEndpoint(network.removeprefix("wss://"))


def Wallet(name=None, hotkey=None, path=None):  # noqa: N802 - the SDK's class name
    return state_wallet(path, hotkey)
