#!/usr/bin/env python3
import argparse,inspect,json,os,platform,sys,time
from pathlib import Path
from common import ROOT,CLASSES,REV,CONFIG,WEIGHTS,sha,digest,rows,dump,validate

def versions():
 import torch,transformers,peft,numpy,scipy,huggingface_hub
 return {n:v for n,v in [('python',platform.python_version()),('torch',torch.__version__),('transformers',transformers.__version__),('peft',peft.__version__),('numpy',numpy.__version__),('scipy',scipy.__version__),('huggingface_hub',huggingface_hub.__version__)]}
def encoded(proc,prompt):return proc.apply_chat_template([{'role':'user','content':[{'type':'text','text':prompt}]}],add_generation_prompt=True,tokenize=True,return_dict=True,return_tensors='pt')['input_ids'][0].tolist()
def template_hash(proc):return digest([str(getattr(proc,'chat_template',None)),str(proc.tokenizer.chat_template)])
def preflight(a):
 protocol,meta,claims=validate();out=Path(a.out).resolve();adapter=Path(a.adapter).resolve()
 for n,h in [('adapter_config.json',CONFIG),('adapter_model.safetensors',WEIGHTS)]:
  if sha(adapter/n)!=h:raise RuntimeError('Selected verifier hash mismatch: '+n)
 from transformers import AutoProcessor,AutoConfig,Gemma3ForConditionalGeneration
 from huggingface_hub import snapshot_download
 if 'logits_to_keep' not in inspect.signature(Gemma3ForConditionalGeneration.forward).parameters:raise RuntimeError('Missing memory-safe next-token logits API')
 snap=Path(snapshot_download('google/gemma-3-4b-it',revision=REV,local_files_only=True));assert snap.name==REV
 proc=AutoProcessor.from_pretrained(str(snap),local_files_only=True);cfg=AutoConfig.from_pretrained(str(snap),local_files_only=True);context=int(cfg.text_config.max_position_embeddings)
 ids={x:proc.tokenizer(x,add_special_tokens=False)['input_ids'] for x in ['A','B','C']}
 assert all(len(x)==1 for x in ids.values()) and len({x[0] for x in ids.values()})==3
 enc={}
 for i,r in enumerate(claims):
  v=encoded(proc,r['prompt'])
  if len(v)>context:raise RuntimeError('Full evidence exceeds model context at '+r['key']+'; no truncation permitted')
  enc[r['key']]={'ids':v,'prompt_sha256':digest(r['prompt']),'input_ids_sha256':digest(v)}
  if (i+1)%500==0:print('Tokenized',i+1,'/',len(claims),flush=True)
 if out==adapter or adapter in out.parents:raise RuntimeError('Output cannot be inside selected adapter')
 out.mkdir(parents=True,exist_ok=True)
 if (out/'RUN_FREEZE.json').exists() or (out/'ENCODED.json').exists():raise RuntimeError('Run already initialized; preserve freeze')
 dump(out/'ENCODED.json',enc)
 freeze={'status':'FROZEN_BEFORE_ANY_TRIGGER_BRIDGE_MODEL_OUTPUT','created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'protocol_sha256':sha(ROOT/'PROTOCOL.json'),'snapshot':str(snap),'revision':REV,'adapter':str(adapter),'config_sha256':CONFIG,'weights_sha256':WEIGHTS,'versions':versions(),'python':sys.executable,'chat_template_sha256':template_hash(proc),'label_ids':{x:v[0] for x,v in ids.items()},'encoded_sha256':sha(out/'ENCODED.json'),'max_model_context':context,'max_prompt_tokens':max(len(x['ids']) for x in enc.values()),'prompts_above_training_max221':sum(len(x['ids'])>221 for x in enc.values()),'rows':900,'claims':2489,'human_scores_in_model_inputs':False,'new_training':False}
 dump(out/'RUN_FREEZE.json',freeze);print(json.dumps(freeze,indent=2));print('PREFLIGHT PASSED; Slurm smoke next. No login-node inference.')
def classify(model,ids,f):
 import torch
 with torch.inference_mode():
  x=torch.tensor([ids],dtype=torch.long,device='cuda');o=model(input_ids=x,attention_mask=torch.ones_like(x),use_cache=False,logits_to_keep=1)
  z=o.logits[0,-1,[f['label_ids'][c] for c in ['A','B','C']]].float();lp=torch.log_softmax(z,dim=-1)
  if not torch.isfinite(lp).all():raise RuntimeError('Nonfinite NLI logits')
  v=lp.cpu().tolist();del o
 return {'label':CLASSES[max(range(3),key=lambda i:v[i])],'restricted_logprobs':dict(zip(CLASSES,v))}
def inference(a):
 protocol,meta,claims=validate();out=Path(a.out).resolve();f=json.loads((out/'RUN_FREEZE.json').read_text())
 assert sha(ROOT/'PROTOCOL.json')==f['protocol_sha256'] and sha(out/'ENCODED.json')==f['encoded_sha256']
 if versions()!=f['versions'] or sys.executable!=f['python']:raise RuntimeError('Frozen environment mismatch')
 for n,h in [('adapter_config.json',CONFIG),('adapter_model.safetensors',WEIGHTS)]:assert sha(Path(f['adapter'])/n)==h
 import torch
 from transformers import AutoProcessor,Gemma3ForConditionalGeneration
 from peft import PeftModel
 if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise RuntimeError('CUDA bfloat16 GPU required')
 proc=AutoProcessor.from_pretrained(f['snapshot'],local_files_only=True);assert template_hash(proc)==f['chat_template_sha256']
 torch.manual_seed(2026)
 base=Gemma3ForConditionalGeneration.from_pretrained(f['snapshot'],torch_dtype=torch.bfloat16,local_files_only=True).to('cuda');base.config.use_cache=False
 model=PeftModel.from_pretrained(base,f['adapter'],is_trainable=False).eval()
 runtime={'gpu':torch.cuda.get_device_name(0),'cuda':torch.version.cuda,'dtype':'bfloat16','revision':REV,'versions':f['versions'],'task':a.task,'same_hardware_within_each_base_trained_pair':True,'logits_to_keep':1,'use_cache':False}
 print(json.dumps(runtime,indent=2),flush=True);enc=json.loads((out/'ENCODED.json').read_text());bykey={r['key']:r for r in claims}
 def pair(key):
  e=enc[key]
  with model.disable_adapter():b=classify(model,e['ids'],f)
  n=classify(model,e['ids'],f)
  return {'key':key,'prompt_sha256':e['prompt_sha256'],'input_ids_sha256':e['input_ids_sha256'],'base':b,'trained':n}
 if a.task=='smoke':
  keys=list(enc);longest=max(keys,key=lambda k:len(enc[k]['ids']));report={'status':'SMOKE_PASSED','runtime':runtime,'protocol_sha256':f['protocol_sha256'],'pairs':[pair(k) for k in dict.fromkeys([keys[0],longest])],'longest_prompt_tokens':len(enc[longest]['ids'])}
  dump(out/'SMOKE.json',report);print('COMPLETE smoke');return
 smoke=json.loads((out/'SMOKE.json').read_text());assert smoke['status']=='SMOKE_PASSED' and smoke['protocol_sha256']==f['protocol_sha256']
 dest=out/'rows';dest.mkdir(exist_ok=True)
 for i,ex in enumerate(meta):
  path=dest/f'row_{int(ex["row_id"]):06d}.json';keys=ex['claim_keys']
  if path.exists():
   old=json.loads(path.read_text());assert old['row_id']==ex['row_id'] and old['protocol_sha256']==f['protocol_sha256'] and [r['key'] for r in old['claims']]==keys
   for r in old['claims']:assert r['prompt_sha256']==enc[r['key']]['prompt_sha256'] and r['input_ids_sha256']==enc[r['key']]['input_ids_sha256']
   continue
  record={'row_id':ex['row_id'],'protocol_sha256':f['protocol_sha256'],'runtime':runtime,'claims':[pair(k) for k in keys]}
  tmp=path.with_suffix('.tmp');dump(tmp,record);os.replace(tmp,path)
  print('SAVED row',ex['row_id'],'claims',len(keys),'progress',i+1,'/900',flush=True)
 for n,h in [('adapter_config.json',CONFIG),('adapter_model.safetensors',WEIGHTS)]:assert sha(Path(f['adapter'])/n)==h
 dump(out/'COMPLETE.json',{'status':'ALL_900_ROWS_COMPLETE','rows':900,'claims':2489,'protocol_sha256':f['protocol_sha256']});print('COMPLETE infer')
def main():
 p=argparse.ArgumentParser();p.add_argument('task',choices=['preflight','smoke','infer']);p.add_argument('--out',required=True);p.add_argument('--adapter',default=str(Path.home()/'korean_training/lora_experiment/Korean_Verifier_Development_2026-10-19/run_v1/checkpoints/epoch_2'));a=p.parse_args()
 if a.task=='preflight':preflight(a)
 else:inference(a)
if __name__=='__main__':main()
