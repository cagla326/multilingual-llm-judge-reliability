#!/usr/bin/env python3
"""Verify submitted raw records and recompute the frozen analysis without overwriting it."""
from pathlib import Path
import sys,json,hashlib,collections,math
import numpy as np
from scipy.stats import binomtest
ROOT=Path(__file__).resolve().parent.parent
PKG=ROOT/'frozen_run_package'
sys.path.insert(0,str(PKG))
import run_klue as runner
import analyze_klue as analyzer
from validate_inputs import validate_data
OUT=ROOT/'results_v1';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();read=lambda p:json.loads(p.read_text())
freeze=read(OUT/'RUN_FREEZE.json');assert sha(ROOT/'PROTOCOL.json')==freeze['protocol_sha256'];assert read(ROOT/'PROTOCOL.json')==read(PKG/'PROTOCOL.json')
for n,h in freeze['package_hashes'].items():assert sha(PKG/n)==h,n
assert freeze['status']=='FROZEN_BEFORE_ANY_KLUE_MODEL_OUTPUT'
report={'checks':{'frozen_package_hashes_match':len(freeze['package_hashes']),'official_data':validate_data(),'reference_and_raw_checks':'all records checked by frozen analyzer; results independently summarized below'},'cells':{}}
recomputed=analyzer.analyze(OUT);saved=read(OUT/'KLUE_ANALYSIS.json')
def compare(a,b,p=''):
 if isinstance(a,dict):
  assert set(a)==set(b),p
  for k in a:compare(a[k],b[k],p+'/'+k)
 elif isinstance(a,list):
  assert len(a)==len(b),p
  for i,(x,y) in enumerate(zip(a,b)):compare(x,y,p+'/'+str(i))
 elif isinstance(a,float):assert np.isclose(a,b,atol=1e-12,rtol=1e-12),(p,a,b)
 else:assert a==b,(p,a,b)
compare(recomputed,saved);report['checks']['saved_analysis_reproduced']=True
for task in ['evaluate','ablation']:
 for system in ['base','adapter']:
  rr=runner.read_jsonl(OUT/(task+'_'+system+'.jsonl'));assert len(rr)==len({r['key'] for r in rr})==2000
  for r in rr:assert r['generated_tokens']<=4 and r['reached_token_limit']==(r['generated_tokens']==4)
  report['cells'][task+'_'+system]={'rows':len(rr),'invalid':sum(not r['valid'] for r in rr),'output_limit_flags':sum(r['reached_token_limit'] for r in rr),'max_prompt_tokens':max(r['prompt_tokens'] for r in rr),'original_correct':sum(r['correct'] for r in rr if r['orientation']=='original'),'swapped_correct':sum(r['correct'] for r in rr if r['orientation']=='swapped')}
assert 'COMPLETE all' in (ROOT/'ko_klue_469103.out').read_text()
errors=(ROOT/'ko_klue_469103.err').read_text();assert 'Traceback' not in errors
report['checks']['complete_all_log']=True;report['checks']['no_traceback_in_job_stderr']=True
# Independently reconstruct primary and source-control transitions from stripped raw strings.
by={}
for task in ['evaluate','ablation']:
 by[task]={s:{r['key']:r for r in runner.read_jsonl(OUT/(task+'_'+s+'.jsonl')) if r['orientation']=='original'} for s in ['base','adapter']}
 b=by[task]['base'];a=by[task]['adapter'];assert set(b)==set(a)
 corr=lambda r:r['raw_response'].strip()==r['gold']
 wins=sum(corr(a[k]) and not corr(b[k]) for k in b);losses=sum(corr(b[k]) and not corr(a[k]) for k in b)
 independent={'base_correct':sum(corr(r) for r in b.values()),'adapter_correct':sum(corr(r) for r in a.values()),'wins':wins,'losses':losses,'delta':(wins-losses)/1000,'exact_p':float(binomtest(wins,wins+losses,.5).pvalue) if wins+losses else 1.}
 target=saved['full_context' if task=='evaluate' else 'premise_omitted_secondary_control']['paired_contrasts']['original']
 assert independent['wins']==target['adapter_only_correct'] and independent['losses']==target['base_only_correct'];assert np.isclose(independent['exact_p'],target['exact_two_sided_paired_binomial_p'])
 report[task+'_independent_primary']=independent
# Record changed outputs including invalid labels; do not hide regressions.
case_records=[];pairs={r['semantic_pair_id']:r for r in runner.read_jsonl(PKG/'data/pairs.jsonl')}
for key,b in by['evaluate']['base'].items():
 a=by['evaluate']['adapter'][key]
 if b['correct']!=a['correct']:
  case_records.append({'key':key,'transition':'adapter_corrects' if a['correct'] else 'adapter_regresses','base':b,'adapter':a,'source_record':pairs[b['semantic_pair_id']]})
report['case_transition_count']=len(case_records);assert len(case_records)==20
report['interpretation']={'primary_gain_percentage_points':1.0,'primary_bootstrap_percentage_points':[.2,1.9],'original_context_benefit_base_pp':31.8,'original_context_benefit_adapter_pp':32.0,'adapter_context_gain_interaction_pp':.2,'interaction_interval_pp':[-1.5,1.9],'scope':'human-labelled, pairwise KLUE-NLI development reformulation; not official three-way benchmark score, not multilingual summary-scoring proof','caveat':'Different significance outcomes across context conditions do not establish a significant context interaction.'}
(ROOT/'audit/VERIFICATION.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False))
(ROOT/'audit/CHANGED_CASES.json').write_text(json.dumps(case_records,indent=2,ensure_ascii=False,allow_nan=False))
print(json.dumps(report,indent=2))
