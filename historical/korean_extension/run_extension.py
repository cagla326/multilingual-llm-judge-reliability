#!/usr/bin/env python3
"""Fixed-adapter inference. Never trains or modifies historical outputs."""
import argparse, contextlib, hashlib, json, math, os, platform, re, sys, time
from pathlib import Path

ROOT=Path(__file__).resolve().parent
MODEL='google/gemma-3-4b-it'
CONFIG_SHA='5a9bdc5bb8c31f118a3764651dbceeed3957e58ab875aa20eaf34f0c378f27b9'
WEIGHT_SHA='82d05d91d69396034ef9f69ecb36ad69ac313e23c8723c9b98d0ce9543d6ea19'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()
def read_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
def write_new(path,obj):
    with Path(path).open('x') as f: json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)
def validate_package():
    spec=json.loads((ROOT/'PROTOCOL.json').read_text())
    for name,expected in spec['package_hashes'].items():
        if sha(ROOT/name)!=expected: raise RuntimeError(f'Package changed: {name}; use a new version, do not overwrite the freeze.')
    return spec

def parse_score(raw):
    """Same whole-object/greedy-object path and missing-score recovery for both systems."""
    obj=None
    try: obj=json.loads(raw.strip())
    except (ValueError,TypeError):
        m=re.search(r'\{.*\}',raw,re.S)
        if m:
            try: obj=json.loads(m[0])
            except ValueError: pass
    val=obj.get('score') if isinstance(obj,dict) else None
    try:
        score=float(val) if val is not None and not isinstance(val,bool) else None
    except (TypeError,ValueError): score=None
    recovered=False
    if score is None:
        m=re.search(r'"score"\s*:\s*([0-9]+(?:\.[0-9]+)?)',raw)
        if m: score=float(m[1]);recovered=True
    if score is None: return dict(score=None,parser_anchor=False,recovered=recovered,score_status='missing')
    if not math.isfinite(score) or not 0<=score<=100:
        return dict(score=None,parser_anchor=False,recovered=recovered,score_status='out_of_range_or_nonfinite')
    return dict(score=score,parser_anchor=isinstance(obj,dict) and not recovered and score in (0,25,50,75,100),recovered=recovered,score_status='available')

def encode(processor,prompt,serialization):
    if serialization=='plain':
        return processor.tokenizer(prompt.strip()+'\n',return_tensors='pt',add_special_tokens=False)
    messages=[{'role':'user','content':[{'type':'text','text':prompt}]}]
    return processor.apply_chat_template(messages,add_generation_prompt=True,tokenize=True,return_dict=True,return_tensors='pt')

def preflight(args):
    spec=validate_package()
    adapter=Path(args.adapter).expanduser().resolve()
    for name,expected in [('adapter_config.json',CONFIG_SHA),('adapter_model.safetensors',WEIGHT_SHA)]:
        if sha(adapter/name)!=expected: raise RuntimeError(f'Wrong adapter file: {name}')
    import torch, transformers, peft, huggingface_hub, numpy, scipy
    from huggingface_hub import snapshot_download
    # Resolve cache once. The resulting immutable snapshot path is frozen for all conditions.
    snap=Path(snapshot_download(MODEL,revision=args.revision,local_files_only=True))
    revision=snap.name
    if not re.fullmatch('[0-9a-f]{40}',revision): raise RuntimeError(f'Cannot establish immutable snapshot revision: {snap}')
    from transformers import AutoProcessor
    processor=AutoProcessor.from_pretrained(str(snap),local_files_only=True)
    smoke='Synthetic plumbing check: the article says the number is seven. Reply A.'
    for serial in ['plain','native']:
        enc=encode(processor,smoke,serial)
        if enc['input_ids'].shape[-1]<1: raise RuntimeError('Empty tokenization')
    out=Path(args.out).expanduser().resolve();out.mkdir(parents=True,exist_ok=True)
    state={'status':'FROZEN_BEFORE_EXTENSION_GENERATION','created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'snapshot':str(snap),'revision':revision,'model_id':MODEL,'adapter':str(adapter),'adapter_config_sha256':CONFIG_SHA,'adapter_weights_sha256':WEIGHT_SHA,'protocol_sha256':sha(ROOT/'PROTOCOL.json'),'package_hashes':spec['package_hashes'],'python':sys.executable,'python_version':platform.python_version(),'torch':torch.__version__,'transformers':transformers.__version__,'peft':peft.__version__,'huggingface_hub':huggingface_hub.__version__,'numpy':numpy.__version__,'scipy':scipy.__version__,'chat_template_sha256':hashlib.sha256((str(getattr(processor,'chat_template',None))+'\n'+str(processor.tokenizer.chat_template)).encode()).hexdigest(),'historical_revision_equivalence':'not established'}
    write_new(out/'RUN_FREEZE.json',state)
    print(json.dumps(state,indent=2));print('Preflight passed. Submit smoke next; do not run model inference on the login node.')

def runtime(args):
    spec=validate_package();out=Path(args.out).expanduser().resolve();state=json.loads((out/'RUN_FREEZE.json').read_text())
    if state['protocol_sha256']!=sha(ROOT/'PROTOCOL.json'): raise RuntimeError('Frozen protocol changed')
    for name,expected in state['package_hashes'].items():
        if sha(ROOT/name)!=expected: raise RuntimeError(f'Frozen package changed: {name}')
    import torch, transformers, peft
    if (torch.__version__,transformers.__version__,peft.__version__)!=(state['torch'],state['transformers'],state['peft']):
        raise RuntimeError('Runtime versions differ from preflight; select the same Python/environment, do not overwrite freeze')
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(): raise RuntimeError('This protocol requires a bf16-capable CUDA GPU')
    from transformers import AutoProcessor,Gemma3ForConditionalGeneration
    from peft import PeftModel
    for name,expected in [('adapter_config.json',CONFIG_SHA),('adapter_model.safetensors',WEIGHT_SHA)]:
        if sha(Path(state['adapter'])/name)!=expected: raise RuntimeError('Adapter changed')
    torch.manual_seed(2026)
    processor=AutoProcessor.from_pretrained(state['snapshot'],local_files_only=True);tok=processor.tokenizer
    template_hash=hashlib.sha256((str(getattr(processor,'chat_template',None))+'\n'+str(tok.chat_template)).encode()).hexdigest()
    if template_hash!=state['chat_template_sha256']: raise RuntimeError('Processor template changed')
    base=Gemma3ForConditionalGeneration.from_pretrained(state['snapshot'],torch_dtype=torch.bfloat16,local_files_only=True).to('cuda').eval()
    model=PeftModel.from_pretrained(base,state['adapter'],is_trainable=False).eval()
    context=getattr(getattr(base.config,'text_config',base.config),'max_position_embeddings',None)
    meta={'task':args.task,'gpu':torch.cuda.get_device_name(0),'cuda':torch.version.cuda,'python':sys.executable,'revision':state['revision'],'device':'cuda:0','dtype':'bfloat16','max_context':context,'condition':'same process, adapter disabled for base; enabled for adapted condition'}
    print(json.dumps(meta,indent=2),flush=True)
    metadata=out/f'{args.task}_runtime.json'
    if metadata.exists():
        previous=json.loads(metadata.read_text())
        if previous!=meta: raise RuntimeError('Task hardware/runtime metadata differs on resume; start a separate run')
    else: write_new(metadata,meta)
    if args.task=='smoke':
        rows=[{'key':'synthetic','prompt':'Article: A box contains seven books.\nSummary A: The box contains seven books.\nSummary B: The box contains nine books.\nChoose the supported summary. Reply only A or B.','gold':'A','semantic_pair_id':'synthetic','orientation':'original'}];serials=['plain','native'];limit=4
    elif args.task=='robustness':
        rows=read_jsonl(ROOT/'data/korean_test100.jsonl');serials=['plain','native'];limit=4
    else:
        rows=read_jsonl(ROOT/'data/nonkorean880.jsonl');serials=['native'];limit=256
    for system in ['base','adapter']:
        for serial in serials:
            path=out/f'{args.task}_{system}_{serial}.jsonl'
            done={}
            if path.exists():
                for r in read_jsonl(path):
                    if r['key'] in done: raise RuntimeError('Duplicate output key')
                    done[r['key']]=r
                if not set(done)<=set(r['key'] for r in rows): raise RuntimeError('Output keys do not match task')
            for row in rows:
                if row['key'] in done: continue
                prompt=row['prompt'];enc=encode(processor,prompt,serial);enc={k:v.to('cuda') for k,v in enc.items()};n=enc['input_ids'].shape[-1]
                if context and n+limit>context: raise RuntimeError(f'Input exceeds context; no truncation permitted: {row["key"]}')
                cm=model.disable_adapter() if system=='base' else contextlib.nullcontext()
                with cm,torch.inference_mode():
                    pad=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
                    generated=model.generate(**enc,do_sample=False,max_new_tokens=limit,pad_token_id=pad)
                    tokens=generated[0,n:];raw=tok.decode(tokens,skip_special_tokens=True);pred=raw.strip() if raw.strip() in ['A','B'] else None
                    rec={k:v for k,v in row.items() if k!='prompt'}
                    rec.update(system=system,serialization=serial,raw_response=raw,prompt_tokens=int(n),generated_tokens=int(len(tokens)),output_limit=limit,last_token_id=int(tokens[-1]) if len(tokens) else None,reached_token_limit=len(tokens)==limit,prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),serialized_ids_sha256=hashlib.sha256(json.dumps(enc['input_ids'][0].tolist()).encode()).hexdigest())
                    if args.task=='transfer': rec.update(parse_score(raw))
                    else:
                        rec.update(prediction=pred,valid=pred is not None,correct=pred==row['gold'])
                        lp={}
                        # Literal A/B sequence score, same answer-token convention as historical evaluator.
                        for label in ['A','B']:
                            ids=tok(label,add_special_tokens=False)['input_ids']
                            if not ids: raise RuntimeError('Empty answer label')
                            answer=torch.tensor([ids],dtype=torch.long,device='cuda');seq=torch.cat([enc['input_ids'],answer],dim=-1)
                            result=model(input_ids=seq,attention_mask=torch.ones_like(seq),use_cache=False)
                            logits=result.logits[:,n-1:n-1+len(ids),:].float();logp=torch.nn.functional.log_softmax(logits,dim=-1)
                            score=float(logp.gather(-1,answer.unsqueeze(-1)).sum().cpu())
                            if not math.isfinite(score): raise RuntimeError('Non-finite label probability')
                            lp[label]=score;del result,logits,logp
                        lp_pred='A' if lp['A']>lp['B'] else 'B' if lp['B']>lp['A'] else None
                        rec.update(logprob_A=lp['A'],logprob_B=lp['B'],logprob_prediction=lp_pred,logprob_correct=lp_pred==row['gold'])
                with path.open('a') as stream:
                    stream.write(json.dumps(rec,ensure_ascii=False,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
                print(args.task,system,serial,row['key'],'saved',flush=True)
    print('COMPLETE',args.task,flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('task',choices=['preflight','smoke','robustness','transfer']);p.add_argument('--out',required=True);p.add_argument('--adapter',default=str(Path.home()/'korean_training/lora_experiment/gemma3_4b_ko_xlsum_FINAL_DPO_train100'));p.add_argument('--revision',default='main');a=p.parse_args()
    if a.task=='preflight':preflight(a)
    else:runtime(a)
if __name__=='__main__':main()
