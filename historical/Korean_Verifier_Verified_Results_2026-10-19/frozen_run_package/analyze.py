#!/usr/bin/env python3
"""Independent paired assessment; source-cluster bootstrap, not 600 independent sources."""
import argparse,hashlib,json,math
from pathlib import Path
from data_checks import ROOT,read_rows,validate_data
from verifier import CLASSES,metrics,sha,spec,selected,atomic_json

def checked(rows,reference,system):
 if len(rows)!=len(reference) or len({r['key'] for r in rows})!=len(reference):raise ValueError('Missing/duplicate predictions')
 rr={r['key']:r for r in rows}
 if set(rr)!={r['key'] for r in reference}:raise ValueError('Prediction keys differ from frozen heldout')
 result=[]
 for ref in reference:
  r=rr[ref['key']]
  if any(r[k]!=ref[k] for k in ['premise_hash','genre','human_label']) or r['system']!=system:raise ValueError('Reference metadata mismatch')
  lp=r['restricted_class_logprobs'];prob=r['restricted_class_probabilities']
  if set(lp)!=set(CLASSES) or set(prob)!=set(CLASSES):raise ValueError('Class set mismatch')
  if not all(math.isfinite(lp[c]) and math.isfinite(prob[c]) and abs(math.exp(lp[c])-prob[c])<1e-6 for c in CLASSES):raise ValueError('Invalid class probabilities')
  if abs(sum(prob.values())-1)>1e-5:raise ValueError('Probabilities not normalized')
  if r['prediction']!=max(CLASSES,key=lambda c:lp[c]):raise ValueError('Prediction/argmax mismatch')
  if abs(r['nll']+lp[r['human_label']])>1e-6:raise ValueError('NLL mismatch')
  if r['prompt_sha256']!=hashlib.sha256(ref['prompt'].encode()).hexdigest():raise ValueError('Prompt mismatch')
  result.append(r)
 return result

def clustered_ci(base,new,repeats=2000):
 import numpy as np
 groups=sorted({r['premise_hash'] for r in base});mats=[]
 for rows in [base,new]:
  mat=np.zeros((len(groups),3,3),dtype=np.int64);gix={g:i for i,g in enumerate(groups)}
  for r in rows:mat[gix[r['premise_hash']],CLASSES.index(r['human_label']),CLASSES.index(r['prediction'])]+=1
  mats.append(mat)
 def scores(cm):
  den=cm.sum(axis=0)+cm.sum(axis=1);diag=np.diag(cm);return np.array([diag.sum()/cm.sum(),np.divide(2*diag,den,out=np.zeros(3,dtype=float),where=den!=0).mean()])
 rng=np.random.default_rng(20261019);deltas=[]
 for _ in range(repeats):
  ix=rng.integers(0,len(groups),size=len(groups));deltas.append(scores(mats[1][ix].sum(axis=0))-scores(mats[0][ix].sum(axis=0)))
 ci=np.quantile(deltas,[.025,.975],axis=0)
 return {'source_clusters':len(groups),'resamples':repeats,'seed':20261019,'accuracy_delta_95_percentile_ci':ci[:,0].tolist(),'macro_f1_delta_95_percentile_ci':ci[:,1].tolist()}
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out);spec();validate_data();path,selection=selected(out)
 freeze=json.loads((out/'RUN_FREEZE.json').read_text())
 if freeze['protocol_sha256']!=sha(ROOT/'PROTOCOL.json'):raise RuntimeError('Protocol changed')
 if freeze['encoded_sha256']!=sha(out/'ENCODED.json'):raise RuntimeError('Encoded inputs changed')
 reference=read_rows(ROOT/'data/internal_test.jsonl');base=checked(read_rows(out/'heldout_base.jsonl'),reference,'base');new=checked(read_rows(out/'heldout_new.jsonl'),reference,'new')
 encoded={r['key']:r for r in json.loads((out/'ENCODED.json').read_text())['internal_test']}
 for rows in [base,new]:
  for r in rows:
   if r['input_ids_sha256']!=hashlib.sha256(json.dumps(encoded[r['key']]['prompt_ids']).encode()).hexdigest():raise ValueError('Tokenization mismatch')
 b=metrics(base);n=metrics(new);result={'task':'Internal heldout 3-class KLUE-train subset; not official test score','selected_epoch':selection['selected_epoch'],'epoch0_noop_selected':selection['new_adapter_is_noop_epoch0'],'base':b,'new':n,'accuracy_delta':n['accuracy']-b['accuracy'],'macro_f1_delta':n['macro_f1']-b['macro_f1'],'paired_source_bootstrap':clustered_ci(base,new),'inputs_sha256':{name:sha(out/name) for name in ['RUN_FREEZE.json','TRAIN_COMPLETE.json','heldout_base.jsonl','heldout_new.jsonl']},'interpretation_limits':['One training seed; no claim of seed robustness.','All data are published and may overlap base-model pretraining.','Short premise-hypothesis classification does not establish full-document summary verification.','Internal test is for one final report; do not tune on its results.','Three-label probabilities are not calibrated summary scores.']}
 atomic_json(out/'VERIFIED_ASSESSMENT.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
