"""Catch replay formatting failures before the commission starts any engines."""
import ast,hashlib,json,sys
from pathlib import Path
from transformers import AutoTokenizer
from cacheon.eval.agent_replay import _chat_input_ids
from cacheon.eval.agent_slice import load_slice_manifest
module=Path(sys.argv[1]);tree=ast.parse(module.read_text())
settings=next(ast.literal_eval(node.value) for node in tree.body if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='REPLAY_SETTINGS' for t in node.targets))
manifest=load_slice_manifest(Path(settings['manifest_path']));assert manifest.digest==settings['slice_digest']
expected=manifest.expected_work(settings['load'])
root=Path(settings['tokenizer_path']);tok=AutoTokenizer.from_pretrained(root,trust_remote_code=True,local_files_only=True)
assert tok.chat_template,'Replay tokenizer has no chat template'
bodies=[{'messages':[{'role':'user','content':'Return the number seven.'}]},{'messages':[{'role':'system','content':'You are a coding assistant.'},{'role':'user','content':'Read the file.'},{'role':'assistant','content':'I read it.'},{'role':'user','content':'Continue.'}]},{'messages':[{'role':'user','content':'Get the weather.'},{'role':'assistant','content':'','tool_calls':[{'id':'t1','type':'function','function':{'name':'weather','arguments':{'city':'Pune'}}}]},{'role':'tool','tool_call_id':'t1','content':'Sunny.'}],'tools':[{'type':'function','function':{'name':'weather','description':'Weather lookup','parameters':{'type':'object','properties':{'city':{'type':'string'}}}}}]}]
counts=[]
for body in bodies:
 ids=_chat_input_ids(tok,body);assert type(ids) is list and ids and all(type(x) is int and 0<=x<len(tok) for x in ids);counts.append(len(ids))
print(json.dumps({'tokenizer_path':str(root),'tokenizer_files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir() if p.is_file()},'slice_digest':manifest.digest,'selected_sessions':len(expected),'sealed_turns':sum(sum(v) for v in expected.values()),'format_cases':['text','multiturn','tools'],'encoded_token_counts':counts,'cpu_preflight_only':True}))
