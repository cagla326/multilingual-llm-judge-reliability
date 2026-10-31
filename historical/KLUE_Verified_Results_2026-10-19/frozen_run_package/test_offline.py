#!/usr/bin/env python3
"""Offline fixtures validate code only. They are not model results and are never saved with results."""
import hashlib,json,tempfile
from pathlib import Path
import numpy as np
import run_klue as runner
import analyze_klue as analyzer
from validate_inputs import validate_data

def fixture_outputs(out,task):
 rows=runner.read_jsonl(runner.ROOT/('data/ablation.jsonl' if task=='ablation' else 'data/evaluate.jsonl'))
 for system in ['base','adapter']:
  records=[]
  for row in rows:
   r={k:v for k,v in row.items() if k!='prompt'};gold=row['gold'];la,lb=(0.,-10.) if gold=='A' else (-10.,0.)
   r.update(system=system,raw_response=gold,prompt_sha256=hashlib.sha256(row['prompt'].encode()).hexdigest(),serialized_ids_sha256='OFFLINE_TEST_FIXTURE_'+row['key'],generated_tokens=2,output_limit=4,reached_token_limit=False)
   r.update(runner.score_record(gold,gold,la,lb));records.append(r)
  (out/(task+'_'+system+'.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in records))
 return records

def main():
 runner.validate_package();print(validate_data())
 assert runner.score_record(' A\n','A',0.,-1.)['correct']
 assert not runner.score_record('A\nB','A',0.,-1.)['valid']
 assert runner.score_record('B','A',-1.,0.)['valid'] and not runner.score_record('B','A',-1.,0.)['correct']
 assert runner.score_record('A','A',0.,0.)['logprob_prediction'] is None
 try:runner.score_record('A','A',float('nan'),0.)
 except ValueError:pass
 else:raise AssertionError('Accepted nonfinite probability')
 assert analyzer.bootstrap([0.]*50)['percentile95']==[0.,0.]
 with tempfile.TemporaryDirectory(prefix='klue_offline_fixture_') as td:
  out=Path(td)
  fixture_outputs(out,'evaluate');fixture_outputs(out,'ablation')
  summary,maps,ids=analyzer.task_summary(out,'evaluate');assert len(ids)==1000
  assert summary['paired_contrasts']['original']['strict_difference']['observed']==0.
  assert summary['paired_contrasts']['original']['exact_two_sided_paired_binomial_p']==1.
  assert summary['systems']['base']['semantic_swap_consistency']==1.
  # Full analyzer: temporary fixtures only, never experimental outputs.
  freeze={'protocol_sha256':hashlib.sha256((runner.ROOT/'PROTOCOL.json').read_bytes()).hexdigest()}
  (out/'RUN_FREEZE.json').write_text(json.dumps(freeze))
  meta={k:'OFFLINE_TEST_FIXTURE' for k in ['gpu','cuda','dtype','revision','python','max_new_tokens','do_sample','prompt_serialization']}
  for task in ['evaluate','ablation']:(out/(task+'_runtime.json')).write_text(json.dumps(meta))
  full=analyzer.analyze(out);assert full['secondary_context_gain_interaction']['observed']==0.
  base_path=out/'evaluate_base.jsonl';base_original=base_path.read_text();base_lines=base_original.splitlines();first=json.loads(base_lines[0]);wrong='B' if first['gold']=='A' else 'A';first['raw_response']=wrong;la,lb=(0.,-10.) if wrong=='A' else (-10.,0.);first.update(runner.score_record(wrong,first['gold'],la,lb));base_path.write_text(json.dumps(first)+'\n'+'\n'.join(base_lines[1:])+'\n')
  known=analyzer.analyze(out);assert np.isclose(known['primary_result']['strict_difference']['observed'],.001);assert known['primary_result']['adapter_only_correct']==1;assert np.isclose(known['secondary_context_gain_interaction']['observed'],.001)
  base_path.write_text(base_original)
  path=out/'evaluate_adapter.jsonl';original=path.read_text();lines=original.splitlines()
  def rejected(text):
   path.write_text(text)
   try:analyzer.task_summary(out,'evaluate')
   except (ValueError,AssertionError):return
   raise AssertionError('Bad output accepted')
  rejected('\n'.join(lines[:-1])+'\n');rejected(original+lines[0]+'\n')
  r=json.loads(lines[0]);r['gold']='B' if r['gold']=='A' else 'A';rejected(json.dumps(r)+'\n'+'\n'.join(lines[1:])+'\n')
  r=json.loads(lines[0]);r['correct']=False;rejected(json.dumps(r)+'\n'+'\n'.join(lines[1:])+'\n')
  r=json.loads(lines[0]);r['serialized_ids_sha256']='DIFFERENT';rejected(json.dumps(r)+'\n'+'\n'.join(lines[1:])+'\n');path.write_text(original)
 print('PASS: original source text/labels, orientations, metadata, strict parser, known paired statistics, incomplete/duplicate/reference/raw-score/token-hash rejection. No GPU inference executed.')
if __name__=='__main__':main()
