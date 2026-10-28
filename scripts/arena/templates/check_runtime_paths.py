"""Check the generated OCI paths as the actual runtime identity before model load."""
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

readonly = json.loads(sys.argv[1])
assert (os.getuid(), os.getgid()) == tuple(json.loads(sys.argv[3]))
keys = ('HOME', 'TMPDIR', 'CUDA_CACHE_PATH', 'FLASHINFER_WORKSPACE_BASE', 'HF_HOME',
        'SGLANG_CACHE_DIR', 'TORCH_EXTENSIONS_DIR', 'TORCHINDUCTOR_CACHE_DIR',
        'TRITON_CACHE_DIR', 'TRITON_HOME', 'XDG_CACHE_HOME')
writes = sorted({os.environ[key] for key in keys if key in os.environ} |
                {tempfile.gettempdir(), os.getcwd()} | set(json.loads(sys.argv[2])))
for value in writes:
    directory = Path(value)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory/'operator-path-preflight'
    with path.open('xb') as handle:
        handle.write(b'preflight')
        handle.flush()
        os.fsync(handle.fileno())
    assert path.read_bytes() == b'preflight'
    path.unlink()

reads = []
pending = [Path(value) for value in readonly]
while pending:
    path = pending.pop()
    if path.is_dir():
        # rglob hid an unreadable oci_site mount during the 2026-10-24 cutover.
        pending.extend(path.iterdir())
    else:
        # Hash exactly the mounted bytes, without importing candidate code.
        with path.open('rb') as handle:
            digest = hashlib.file_digest(handle, 'sha256').hexdigest()
        reads.append({'path': str(path), 'bytes': path.stat().st_size, 'sha256': digest})
print(json.dumps({'uid': os.getuid(), 'gid': os.getgid(), 'writes': writes, 'reads': reads}))
