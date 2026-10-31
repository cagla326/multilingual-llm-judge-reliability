#!/usr/bin/env python3
"""Inference only: fixed adapter, fixed human-labelled data, no training or data generation."""
import argparse,contextlib,hashlib,json,math,os,platform,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
MODEL='google/gemma-3-4b-it'
REVISION='093f9f388b31de276ce2de164bdc2081324b9767'
ADAPTER_CONFIG='5a9bdc5bb8c31f118a3764651dbceeed3957e58ab875aa20eaf34f0c378f27b9'
ADAPTER_WEIGHT='82d05d91d69396034ef9f69ecb36ad69ac313e23c8723c9b98d0ce9543d6ea19'
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 return h.hexdigest()
def read_jsonl(path):return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
def write_new(path,obj):
 with Path(path).open('x') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)
def validate_package():
 spec=json.loads((ROOT/'PROTOCOL.json').read_text())
 for n,h in spec['package_hashes'].items():
  if sha(ROOT/n)!=h:raise RuntimeError('Frozen package changed: '+n)
 return spec
def encode(processor,prompt):
 return processor.apply_chat_template([{'role':'user','content':[{'type':'text','text':prompt}]}],add_generation_prompt=True,tokenize=True,return_dict=True,return_tensors='pt')
def score_record(raw,gold,la,lb):
 pred=raw.strip() if raw.strip() in ['A','B'] else None
 if not math.isfinite(la) or not math.isfinite(lb):raise ValueError('Nonfinite label log probability')
 lp='A' if la>lb else 'B' if lb>la else None
 return {'prediction':pred,'valid':pred is not None,'correct':pred==gold,'logprob_A':la,'logprob_B':lb,'logprob_prediction':lp,'logprob_correct':lp==gold}
def versions():
 import torch,transformers,peft,huggingface_hub,numpy,scipy
 return {'python_version':platform.python_version(),'torch':torch.__version__,'transformers':transformers.__version__,'peft':peft.__version__,'huggingface_hub':huggingface_hub.__version__,'numpy':numpy.__version__,'scipy':scipy.__version__}
def check_adapter(path):
 for name,digest in [('adapter_config.json',ADAPTER_CONFIG),('adapter_model.safetensors',ADAPTER_WEIGHT)]:
  if sha(Path(path)/name)!=digest:raise RuntimeError('Wrong adapter '+name)
def preflight(args):
 spec=validate_package();check_adapter(args.adapter);v=versions()
 from validate_inputs import validate_data
 data_check=validate_data()
 from huggingface_hub import snapshot_download
 from transformers import AutoProcessor
 snap=Path(snapshot_download(MODEL,revision=REVISION,local_files_only=True)).resolve()
 if snap.name!=REVISION:raise RuntimeError('Wrong immutable model snapshot')
 processor=AutoProcessor.from_pretrained(str(snap),local_files_only=True)
 # Tokenize a published, already frozen pair. No model inference on login node.
 row=read_jsonl(ROOT/'data/evaluate.jsonl')[0];enc=encode(processor,row['prompt']);assert enc['input_ids'].shape[-1]>0
 state={'status':'FROZEN_BEFORE_ANY_KLUE_MODEL_OUTPUT','created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'model_id':MODEL,'revision':REVISION,'snapshot':str(snap),'adapter':str(Path(args.adapter).resolve()),'adapter_config_sha256':ADAPTER_CONFIG,'adapter_weights_sha256':ADAPTER_WEIGHT,'python':sys.executable,'versions':v,'chat_template_sha256':hashlib.sha256((str(getattr(processor,'chat_template',None))+'\n'+str(processor.tokenizer.chat_template)).encode()).hexdigest(),'protocol_sha256':sha(ROOT/'PROTOCOL.json'),'package_hashes':spec['package_hashes'],'data_check':data_check,'official_klue_test_claim':False,'model_pretraining_contamination':'not excluded by this study'}
 out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=True);write_new(out/'RUN_FREEZE.json',state);print(json.dumps(state,indent=2));print('PREFLIGHT PASSED. Submit smoke through Slurm; no model inference on login node.')
def runtime(args):
 spec=validate_package();out=Path(args.out).resolve();freeze=json.loads((out/'RUN_FREEZE.json').read_text())
 if sha(ROOT/'PROTOCOL.json')!=freeze['protocol_sha256']:raise RuntimeError('Protocol changed')
 if versions()!=freeze['versions'] or sys.executable!=freeze['python']:raise RuntimeError('Use the exact frozen Python environment')
 check_adapter(freeze['adapter'])
 import torch
 from transformers import AutoProcessor,Gemma3ForConditionalGeneration
 from peft import PeftModel
 if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise RuntimeError('Requires CUDA and bfloat16-capable GPU')
 torch.manual_seed(2026);processor=AutoProcessor.from_pretrained(freeze['snapshot'],local_files_only=True);tok=processor.tokenizer
 th=hashlib.sha256((str(getattr(processor,'chat_template',None))+'\n'+str(tok.chat_template)).encode()).hexdigest()
 if th!=freeze['chat_template_sha256']:raise RuntimeError('Chat template changed')
 base=Gemma3ForConditionalGeneration.from_pretrained(freeze['snapshot'],torch_dtype=torch.bfloat16,local_files_only=True).to('cuda').eval()
 model=PeftModel.from_pretrained(base,freeze['adapter'],is_trainable=False).eval();context=getattr(getattr(base.config,'text_config',base.config),'max_position_embeddings',None)
 meta={'task':args.task,'gpu':torch.cuda.get_device_name(0),'cuda':torch.version.cuda,'dtype':'bfloat16','revision':freeze['revision'],'python':sys.executable,'max_context':context,'system_contrast':'same process; adapter disabled for base, enabled for adapter','max_new_tokens':4,'do_sample':False,'prompt_serialization':'native chat template'}
 mp=out/(args.task+'_runtime.json')
 if mp.exists():
  if json.loads(mp.read_text())!=meta:raise RuntimeError('Runtime/hardware changed on resume; preserve old run')
 else:write_new(mp,meta)
 print(json.dumps(meta,indent=2),flush=True)
 rows=read_jsonl(ROOT/('data/ablation.jsonl' if args.task=='ablation' else 'data/evaluate.jsonl'))
 if args.task=='smoke':rows=rows[:1]
 for system in ['base','adapter']:
  path=out/(args.task+'_'+system+'.jsonl');done={}
  if path.exists():
   for r in read_jsonl(path):
    if r['key'] in done:raise RuntimeError('Duplicate output key')
    done[r['key']]=r
   if not set(done)<=set(r['key'] for r in rows):raise RuntimeError('Unknown output key')
   expected={r['key']:r for r in rows}
   for key,r in done.items():
    source=expected[key]
    if r['system']!=system or r['prompt_sha256']!=hashlib.sha256(source['prompt'].encode()).hexdigest():raise RuntimeError('Wrong checkpoint metadata')
    for k,v in source.items():
     if k!='prompt' and r.get(k)!=v:raise RuntimeError('Changed reference on resume')
  for row in rows:
   if row['key'] in done:continue
   enc={k:v.to('cuda') for k,v in encode(processor,row['prompt']).items()};n=enc['input_ids'].shape[-1]
   if context and n+4>context:raise RuntimeError('Context exceeded; no truncation authorized')
   cm=model.disable_adapter() if system=='base' else contextlib.nullcontext()
   with cm,torch.inference_mode():
    pad=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    generated=model.generate(**enc,do_sample=False,max_new_tokens=4,pad_token_id=pad)
    tokens=generated[0,n:];raw=tok.decode(tokens,skip_special_tokens=True);lp={}
    for label in ['A','B']:
     ids=tok(label,add_special_tokens=False)['input_ids'];assert ids
     ans=torch.tensor([ids],dtype=torch.long,device='cuda');seq=torch.cat([enc['input_ids'],ans],dim=-1)
     result=model(input_ids=seq,attention_mask=torch.ones_like(seq),use_cache=False)
     logits=result.logits[:,n-1:n-1+len(ids),:].float();logp=torch.nn.functional.log_softmax(logits,dim=-1)
     lp[label]=float(logp.gather(-1,ans.unsqueeze(-1)).sum().cpu());del result,logits,logp
   record={k:v for k,v in row.items() if k!='prompt'};record.update(system=system,raw_response=raw,prompt_tokens=int(n),generated_tokens=int(len(tokens)),output_limit=4,last_token_id=int(tokens[-1]) if len(tokens) else None,reached_token_limit=len(tokens)==4,prompt_sha256=hashlib.sha256(row['prompt'].encode()).hexdigest(),serialized_ids_sha256=hashlib.sha256(json.dumps(enc['input_ids'][0].tolist()).encode()).hexdigest())
   record.update(score_record(raw,row['gold'],lp['A'],lp['B']))
   with path.open('a') as f:f.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
   print(args.task,system,row['key'],'saved',flush=True)
 print('COMPLETE',args.task,flush=True)
def main():
 p=argparse.ArgumentParser();p.add_argument('task',choices=['preflight','smoke','evaluate','ablation']);p.add_argument('--out',required=True);p.add_argument('--adapter',default=str(Path.home()/'korean_training/lora_experiment/gemma3_4b_ko_xlsum_FINAL_DPO_train100'));a=p.parse_args()
 if a.task=='preflight':preflight(a)
 else:runtime(a)
if __name__=='__main__':main()
