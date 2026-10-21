"""Use the commissioned source's standing worker and its retained commissioning recipe."""
from pathlib import Path
import runpy
import sys

helper = runpy.run_path(str(Path(__file__).with_name('commission_helper.py')))
if sys.argv[1] == 'commission-current-pod':
    raise SystemExit(helper['main']())
helper['_bind_pod_source']()
from cacheon.chain.remote_worker_service import main
paths = helper['_canonical_pod_paths']()
argv = sys.argv[1:]
for flag, value in [('root', paths.root), ('ready-receipt', paths.ready_receipt),
                    ('registration', paths.registration), ('service', paths.service),
                    ('adapter', paths.adapter), ('credential', paths.credential)]:
    argv += ['--pod-' + flag, str(value)]
raise SystemExit(main(argv))
