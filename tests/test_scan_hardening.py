from pathlib import Path

import pytest

from cacheon.sandbox import scan_source

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


@pytest.mark.parametrize("src", [
    "import os\nx = getattr(os, 'sys' + 'tem')('id')\n",          # dynamic getattr
    "x = __builtins__['eval']('1+1')\n",                          # builtins subscript
    "g = globals()\n",                                            # namespace exposure
    "v = vars()\n",
    "y = ().__class__.__bases__[0]\n",                            # __class__ escape hop
    "import os\nsetattr(os, 'x'+'y', 1)\n",                       # dynamic setattr
    "import os\nf = os.system\nf('id')\n",                        # banned-callable ALIAS (no Call at the access)
    "import os\ncmds = [os.system]\ncmds[0]('id')\n",             # alias via a container
    "import dill\nl = dill.loads\nl(b'')\n",                      # deserializer alias
    "import cacheon.receipts\n",
    "from cacheon import receipts\n",
    "import cacheon as c\n",
])
def test_known_bypasses_are_flagged(src):
    assert not scan_source(src).ok, f"should have flagged: {src!r}"


@pytest.mark.parametrize("src", [
    "import torch\ndef k(x, out):\n    out.copy_(torch.relu(x))\n",
    "class C:\n    pass\nc = C()\nv = getattr(c, 'attr', None)\n",   # LITERAL getattr is fine
    "d = {'a': 1}\nx = d['a']\n",                                    # ordinary subscript fine
    "from cacheon.moe_nvfp4_contract import dequantize_prepare_args\n",
    "from cacheon.slots import Activation\n",
])
def test_legitimate_code_not_flagged(src):
    assert scan_source(src).ok, f"false positive on: {src!r}"


def test_all_example_kernels_still_scan_clean():
    # No false positives on the shipped bundles after hardening.
    for kernel in EXAMPLES.glob("*/kernels/*.py"):
        res = scan_source(kernel.read_text(), filename=kernel.name)
        assert res.ok, f"{kernel}: {res.violations}"
