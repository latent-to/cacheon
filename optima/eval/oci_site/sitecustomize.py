"""Install Optima's validator-owned import hook in every OCI Python child.

The immutable arena image need not be mutated with a site-packages ``.pth`` file:
the OCI backend prepends this read-only source directory to ``PYTHONPATH``.  Python's
normal site initialization imports this module in the timing worker and every fresh
SGLang multiprocessing interpreter before any engine module is loaded.
"""

import optima.bootstrap  # noqa: F401 - import-time installation is the contract
