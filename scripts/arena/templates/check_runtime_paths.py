"""Check the generated OCI paths as the actual runtime identity before model load."""
from concurrent.futures import ThreadPoolExecutor
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

files = []
for value in readonly:
    path = Path(value)
    files.extend([path] if path.is_file() else (item for item in path.rglob('*') if item.is_file()))

def check(path):
    """Hash exactly the mounted bytes, without importing candidate code."""
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        while block := handle.read(8*1024*1024):
            digest.update(block)
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': digest.hexdigest()}

with ThreadPoolExecutor(max_workers=8) as pool:
    reads = list(pool.map(check, files))
print(json.dumps({'uid': os.getuid(), 'gid': os.getgid(), 'writes': writes, 'reads': reads}))
