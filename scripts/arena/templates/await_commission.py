"""Retain commissioning progress on CPU and connect the finished real worker.

Arguments after -- are passed unchanged to connect_worker.py. The existing
commissioner produces the handoff; the existing supervisor consumes it.
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--pod-root', required=True)
parser.add_argument('--timeout', type=int, default=7200)
args, connect = parser.parse_known_args()
if connect and connect[0] == '--':
    connect.pop(0)
values = dict(zip(connect[::2], connect[1::2]))
base = Path(values['--base'])
state = Path(values['--state-root']) / 'commissioning' / values['--tag']
state.mkdir(mode=0o700, parents=True, exist_ok=True)
ssh = ['ssh', '-i', str(base/'ssh_key'), '-o', 'BatchMode=yes',
       '-o', 'StrictHostKeyChecking=yes', '-o', 'UserKnownHostsFile='+values['--known-hosts'],
       '-o', 'ConnectTimeout=12', values['--user']+'@'+values['--host']]
status_program = '''import json,sys
from pathlib import Path
root,tag=Path(sys.argv[1]),sys.argv[2]
exit_path=root/('commission-'+tag+'.exit')
argv=json.loads((root/'commission-started.json').read_bytes())['argv']
shared=Path(argv[argv.index('--shared')+1])
logs=['logs/'+name+'.log' for name in ('image','download','uv','controller-venv','aiperf-venv','model-provision')
      if (shared/'logs'/(name+'.log')).is_file()]
print(json.dumps({'boot':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
 'exit':int(exit_path.read_text()) if exit_path.exists() else None,
 'complete':(root/('commission-'+tag+'.complete')).exists(),
 'bootstrap_root':str(shared),'bootstrap_logs':logs}))
'''
deadline = time.monotonic() + args.timeout
first_boot = None
while time.monotonic() < deadline:
    raw = subprocess.check_output(ssh+[shlex.join(['sudo','-n','python3','-c',
        status_program,args.pod_root,values['--tag']])],timeout=30)
    status = json.loads(raw)
    if first_boot is None:
        first_boot = status['boot']
    if status['boot'] != first_boot:
        raise RuntimeError('worker rebooted during commissioning; retain CPU progress and recommission')
    (state/'status.json').write_bytes(raw)
    # The first interruption retained runner logs but lost the separate image log.
    for name, folder, entries in [
        ('progress',args.pod_root,['logs','commission-started.json','durable-images.json','source-revision.txt']),
        ('bootstrap-logs',status['bootstrap_root'],status['bootstrap_logs'])]:
        if not entries:
            continue
        partial = state/(name+'.partial')
        with partial.open('wb') as output:
            subprocess.run(ssh+[shlex.join(['sudo','-n','tar','-C',folder,'-cf','-',*entries])],
                           stdout=output,check=True,timeout=60)
            output.flush();os.fsync(output.fileno())
        partial.replace(state/(name+'.tar'))
    if status['exit'] is not None:
        if status['exit'] != 0 or not status['complete']:
            raise RuntimeError('commissioning failed; original logs retained in '+str(state/'progress.tar'))
        command = ['/root/cacheon-ops/venv/bin/python','-B','-u',
                   str(Path(__file__).with_name('connect_worker.py')),*connect,'--start']
        subprocess.run(command,check=True,timeout=600)
        (state/'connected.json').write_text(json.dumps({'connected_at':int(time.time()),
            'boot':first_boot,'tag':values['--tag']})+'\n')
        break
    time.sleep(15)
else:
    raise TimeoutError('commissioning exceeded its wall bound')
