"""Run the retained filesystem probe in the commissioned image as its actual runtime UID."""
import ast
import argparse
import json
from pathlib import Path
import subprocess

parser=argparse.ArgumentParser(description=__doc__)
for name in ('shared','root','tag'):parser.add_argument('--'+name,required=True)
args=parser.parse_args()
shared,root=Path(args.shared),Path(args.root)
stage = root / ('node/pair-' + args.tag)
profile = json.loads((stage / 'qualification-authority/deployment.json').read_bytes())
uid, gid = profile['uid'], profile['gid']
assert (uid, gid) == (65534, 65534)
source = root / 'build/src'
names = {'CONTAINER_MODEL', 'CONTAINER_ENGINE_WORKER_POLICY', 'CONTAINER_SITE_BOOTSTRAP', 'CONTAINER_CACHE'}
constants = {node.targets[0].id: ast.literal_eval(node.value)
             for node in ast.parse((source/'cacheon/eval/oci_backend.py').read_text()).body
             if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name)
             and node.targets[0].id in names}
reads = [constants[k] for k in ['CONTAINER_MODEL','CONTAINER_ENGINE_WORKER_POLICY','CONTAINER_SITE_BOOTSTRAP']]
cache = constants['CONTAINER_CACHE']
env = {'HOME': cache+'/home', 'TMPDIR': '/tmp', 'CUDA_CACHE_PATH': cache+'/cuda',
       'FLASHINFER_WORKSPACE_BASE': cache+'/flashinfer', 'HF_HOME': cache+'/huggingface',
       'SGLANG_CACHE_DIR': cache+'/sglang', 'TORCH_EXTENSIONS_DIR': cache+'/torch-extensions',
       'TORCHINDUCTOR_CACHE_DIR': cache+'/torchinductor', 'TRITON_CACHE_DIR': cache+'/triton',
       'TRITON_HOME': cache+'/triton-home', 'XDG_CACHE_HOME': cache+'/xdg'}
command = ['/usr/bin/docker','run','--rm','--name=cacheon-spot-fs-preflight-'+args.tag,
           '--runtime=runc','--network=none','--cap-drop=ALL',
           '--security-opt=no-new-privileges=true',f'--user={uid}:{gid}',
           '--env=NVIDIA_VISIBLE_DEVICES=void','--env=CUDA_VISIBLE_DEVICES=',
           '--workdir=/tmp',f'--tmpfs=/tmp:rw,exec,size=64m,uid={uid},gid={gid},mode=0700',
           f'--tmpfs={cache}:rw,exec,size=8g,uid={uid},gid={gid},mode=0700']
bootstrap=json.loads((shared/'bootstrap-complete.json').read_bytes())
for src, dst in [(Path(bootstrap['model_root']), reads[0]),
                 (source/'cacheon/eval/engine_worker.py', reads[1]),
                 (source/'cacheon/eval/oci_site', reads[2]),
                 (shared/'check_runtime_paths.py', '/probe.py')]:
    command += ['--mount',f'type=bind,src={src},dst={dst},readonly']
command += [f'--env={k}={v}' for k,v in env.items()]
base_command=list(command)
command += ['--entrypoint=/usr/bin/python3.12',profile['image'],'-I','-S','/probe.py',
            json.dumps(reads[1:]),json.dumps([cache]),json.dumps([uid,gid])]
result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=600)
proof = json.loads(result.stdout)
receipt, = (shared/'commission-inputs/model-receipts').glob('model-provision-sha256-*.json')
expected = json.loads(receipt.read_bytes())
container_receipt='/cacheon/'+receipt.name
program="""import json,os,sys
from cacheon.model_provision import reopen_model_provision
r=reopen_model_provision('/cacheon/input/model',sys.argv[1],require_public_read_only=True,workers=8)
print(json.dumps({'uid':os.getuid(),'gid':os.getgid(),'content_digest':r.receipt.content_digest,'receipt_digest':r.receipt.receipt_digest,'files':len(r.receipt.files)}))
"""
model_command=base_command+['--mount',f'type=bind,src={receipt},dst={container_receipt},readonly',
    '--entrypoint=/usr/bin/python3.12',profile['image'],'-B','-c',program,container_receipt]
model_proof=json.loads(subprocess.check_output(model_command,text=True,timeout=1200))
model_proof.update(image=profile['image'],model_root=bootstrap['model_root'])
assert (model_proof['uid'],model_proof['gid']) == (uid,gid)
assert model_proof['content_digest'] == expected['content_digest']
assert model_proof['files'] == len(expected['files'])
proof['model_reopen'] = model_proof
(stage/'runtime-uid-preflight.json').write_text(json.dumps(proof,indent=2)+'\n')
print('RUNTIME_UID_PREFLIGHT_OK',uid,gid,model_proof['files'],len(proof['writes']),flush=True)
