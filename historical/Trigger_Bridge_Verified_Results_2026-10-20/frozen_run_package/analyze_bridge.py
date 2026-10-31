#!/usr/bin/env python3
import argparse,json,math
from pathlib import Path
from common import ROOT,CLASSES,sha,digest,dump,validate,capped,trigger

def check_prediction(p):
 lp=p['restricted_logprobs'];assert set(lp)==set(CLASSES)
 assert all(math.isfinite(lp[c]) for c in CLASSES)
 assert abs(sum(math.exp(lp[c]) for c in CLASSES)-1)<1e-5
 assert p['label']==max(CLASSES,key=lambda c:lp[c])
def measures(y,s):
 import numpy as np
 from scipy.stats import spearmanr,kendalltau,pearsonr
 y=np.array(y);s=np.array(s);e=s-y
 return {'n':len(y),'spearman':float(spearmanr(y,s).statistic),'kendall':float(kendalltau(y,s).statistic),'pearson':float(pearsonr(y,s).statistic),'mae':float(np.abs(e).mean()),'rmse':float(np.sqrt(np.mean(e**2))),'signed_bias':float(e.mean()),'extreme_over_rate':float((e>=50).mean()),'extreme_under_rate':float((e<=-50).mean())}
def intervals(meta,scorevectors,repeats=2000):
 import numpy as np
 from scipy.stats import rankdata
 langs=sorted({r['lang'] for r in meta});idx={x:np.array([i for i,r in enumerate(meta) if r['lang']==x]) for x in langs};y=np.array([r['human'] for r in meta]);rng=np.random.default_rng(20261019)
 def rho(a,b):return float(np.corrcoef(rankdata(a),rankdata(b))[0,1])
 pairs=[('trained3_selective','original4_selective'),('trained3_selective','base3_selective')];draws={a+' minus '+b:[] for a,b in pairs}
 arrays={k:np.array(v) for k,v in scorevectors.items()}
 for _ in range(repeats):
  ix=np.concatenate([idx[langs[i]] for i in rng.integers(0,len(langs),size=len(langs))]);yy=y[ix]
  for a,b in pairs:
   sa=arrays[a][ix];sb=arrays[b][ix];draws[a+' minus '+b].append([rho(yy,sa)-rho(yy,sb),float(np.mean(abs(sa-yy))-np.mean(abs(sb-yy))),float(np.mean(sa-yy)-np.mean(sb-yy))])
 return {k:{'resamples':repeats,'seed':20261019,'clusters':'45 original source-language groups; 20 rows each','spearman_delta_95_percentile_ci':np.quantile(v,[.025,.975],axis=0)[:,0].tolist(),'mae_delta_95_percentile_ci':np.quantile(v,[.025,.975],axis=0)[:,1].tolist(),'bias_delta_95_percentile_ci':np.quantile(v,[.025,.975],axis=0)[:,2].tolist()} for k,v in draws.items()}
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out);protocol,meta,claims=validate();freeze=json.loads((out/'RUN_FREEZE.json').read_text());assert freeze['protocol_sha256']==sha(ROOT/'PROTOCOL.json') and freeze['encoded_sha256']==sha(out/'ENCODED.json')
 assert json.loads((out/'COMPLETE.json').read_text())['status']=='ALL_900_ROWS_COMPLETE'
 enc=json.loads((out/'ENCODED.json').read_text());files=list((out/'rows').glob('row_*.json'));assert len(files)==900
 vectors={k:[] for k in ['holistic','original4_selective','base3_capped','trained3_capped','base3_selective','trained3_selective']};reportrows=[];inputs={}
 for ex in meta:
  pth=out/'rows'/f'row_{int(ex["row_id"]):06d}.json';r=json.loads(pth.read_text());assert r['row_id']==ex['row_id'] and r['protocol_sha256']==freeze['protocol_sha256'] and [v['key'] for v in r['claims']]==ex['claim_keys'];inputs[str(pth.relative_to(out))]=sha(pth)
  for v in r['claims']:
   assert v['prompt_sha256']==enc[v['key']]['prompt_sha256'] and v['input_ids_sha256']==enc[v['key']]['input_ids_sha256']
   for c in ['base','trained']:check_prediction(v[c])
  bc=capped([v['base']['label'] for v in r['claims']]);nc=capped([v['trained']['label'] for v in r['claims']]);row={'row_id':ex['row_id'],'lang':ex['lang'],'human':ex['human'],'holistic':ex['baseline'],'original4_selective':ex['original_final'],'base3_capped':bc,'trained3_capped':nc,'base3_selective':trigger(ex['baseline'],bc),'trained3_selective':trigger(ex['baseline'],nc),'num_claims':len(r['claims'])}
  for k in vectors:vectors[k].append(row[k])
  reportrows.append(row)
 y=[r['human'] for r in meta];metrics={k:measures(y,v) for k,v in vectors.items()};cis=intervals(meta,vectors)
 changes={}
 for ref in ['original4_selective','base3_selective']:
  same=helped=harmed=tied=0
  for r in reportrows:
   if math.isclose(r['trained3_selective'],r[ref],abs_tol=1e-9):same+=1
   en=abs(r['trained3_selective']-r['human']);eo=abs(r[ref]-r['human'])
   if en<eo-1e-9:helped+=1
   elif en>eo+1e-9:harmed+=1
   else:tied+=1
  changes[ref]={'unchanged_scores':same,'lower_absolute_error':helped,'higher_absolute_error':harmed,'equal_absolute_error':tied}
 primary=metrics['trained3_selective']['spearman']-metrics['original4_selective']['spearman']
 result={'status':'COMPLETE_FROZEN_TRIGGER_BRIDGE_ANALYSIS','primary':'Trained3 selective minus existing original4 selective: global Spearman across all 900 rows','primary_spearman_delta':primary,'metrics':metrics,'paired_language_cluster_intervals':cis,'changes_vs_references':changes,'per_language_descriptive':{},'freeze_sha256':sha(out/'RUN_FREEZE.json'),'input_record_hashes':inputs,'limits':['Three-label replacement changes the old four-label verifier schema; trained3 versus base3 controls this change within the new branch.','Existing English translations and extracted claims are reused, including their historical errors.','Korean short-NLI training is transferred to English evidence and longer context.','All 900 benchmark rows have been inspected in earlier development; this is not fresh external confirmation.','No new per-claim human reference labels; evaluation is downstream against existing summary Q2 scores.','Korean-only n=20 analyses and per-language ranks are descriptive.','No refitting of model, score mapping or trigger thresholds after results.','No inference-cost saving claim: both judgments are computed for every row.']}
 for lang in sorted({r['lang'] for r in reportrows}):
  rr=[r for r in reportrows if r['lang']==lang];yy=[r['human'] for r in rr]
  # Per-language MAE/bias only, avoiding undefined tiny-sample rank estimates.
  result['per_language_descriptive'][lang]={k:{'n':len(rr),'mae':sum(abs(r[k]-r['human']) for r in rr)/len(rr),'signed_bias':sum(r[k]-r['human'] for r in rr)/len(rr)} for k in ['original4_selective','base3_selective','trained3_selective']}
 (out/'FINAL_ROWS.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in reportrows));dump(out/'FINAL_ANALYSIS.json',result)
 print(json.dumps({k:v for k,v in result.items() if k not in ['input_record_hashes','per_language_descriptive']},indent=2))
if __name__=='__main__':main()
