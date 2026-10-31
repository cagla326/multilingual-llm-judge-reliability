#!/usr/bin/env python3
"""Prepare source-authored human-labelled pairs; never generate or rewrite examples."""
import argparse,collections,csv,hashlib,json,random,re
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
norm=lambda s:re.sub(r'\s+',' ',s).strip()
def main():
 p=argparse.ArgumentParser();p.add_argument('--historical',required=True);p.add_argument('--mface',required=True);args=p.parse_args();hist=Path(args.historical)
 provenance=json.loads((ROOT/'SOURCE_PROVENANCE.json').read_text())
 for f in provenance['files']:assert sha(ROOT/f['path'])==f['sha256'],f['path']
 data=json.loads((ROOT/'data/klue-nli-v1.1_dev.json').read_text());assert len(data)==3000
 assert len({r['guid'] for r in data})==3000
 groups=collections.defaultdict(list)
 for r in data:
  assert r['gold_label'] in ['entailment','contradiction','neutral']
  votes=[r[k] for k in ['author','label2','label3','label4','label5']];assert votes.count(r['gold_label'])>=3,r['guid']
  groups[r['premise']].append(r)
 files=['korean_xlsum_TRAIN50_sources.csv','korean_xlsum_FROZEN_TEST50_sources.csv','korean_xlsum_train_pool_50.csv'];texts=[];inputs=[]
 for name in files:
  path=hist/name
  with path.open() as stream:rr=list(csv.DictReader(stream))
  texts += [norm(r['article']) for r in rr];inputs.append({'filename':name,'sha256':sha(path),'rows':len(rr)})
 path=hist/'korean_train_5k.jsonl';rr=[json.loads(l) for l in path.read_text().splitlines() if l.strip()];texts += [norm(r['document']) for r in rr];inputs.append({'filename':path.name,'sha256':sha(path),'rows':len(rr)})
 path=Path(args.mface)
 with path.open() as stream:rr=list(csv.DictReader(stream))
 texts += [norm(r['input']) for r in rr];inputs.append({'filename':path.name,'sha256':sha(path),'rows':len(rr)})
 texts=sorted(set(texts));eligible=[];exclusions=[]
 for premise,rr in groups.items():
  counts=collections.Counter(r['gold_label'] for r in rr)
  if len(rr)!=3 or counts!={'entailment':1,'contradiction':1,'neutral':1}:
   exclusions.append({'guids':[r['guid'] for r in rr],'reason':'not exactly one validated label of each class'});continue
  if len({r['source'] for r in rr})!=1:raise ValueError('Source disagreement within premise')
  pr=norm(premise)
  if any(pr in t or t in pr for t in texts):
   exclusions.append({'guids':[r['guid'] for r in rr],'reason':'normalized premise exact/substring overlap with recovered prior sources'});continue
  label={r['gold_label']:r for r in rr};assert label['entailment']['hypothesis']!=label['contradiction']['hypothesis']
  eligible.append({'premise':premise,'source':rr[0]['source'],'rows':label,'pair_id':'klue_'+hashlib.sha256(premise.encode()).hexdigest()[:20]})
 eligible.sort(key=lambda r:r['pair_id']);assert len({r['pair_id'] for r in eligible})==len(eligible)
 # Use every eligible source, never a sample chosen from model results.
 rng=random.Random(20261019);positions=['A']*((len(eligible)+1)//2)+['B']*(len(eligible)//2);rng.shuffle(positions)
 template='다음 근거 문장과 두 후보 문장을 읽으세요.\n\n근거 문장:\n{premise}\n\n후보 A:\n{candidate_a}\n\n후보 B:\n{candidate_b}\n\n근거 문장의 내용으로 뒷받침되는 후보 문장을 선택하세요. 문체나 길이가 아니라 근거와의 의미 관계를 판단하세요. A 또는 B 중 하나만 답하세요.'
 # Ablation keeps the exact full prompt structure/instruction; replace only premise with empty string.
 abl_template=template
 originals=[];evaluations=[];ablations=[]
 for r,gold in zip(eligible,positions):
  e=r['rows']['entailment'];c=r['rows']['contradiction'];a,b=(e,c) if gold=='A' else (c,e)
  originals.append({'semantic_pair_id':r['pair_id'],'premise':r['premise'],'source':r['source'],'candidate_a':a['hypothesis'],'candidate_b':b['hypothesis'],'candidate_a_guid':a['guid'],'candidate_b_guid':b['guid'],'gold':gold,'entailment_guid':e['guid'],'contradiction_guid':c['guid'],'neutral_guid_excluded':r['rows']['neutral']['guid'],'original_validated_records':r['rows']})
  for orientation in ['original','swapped']:
   aa,bb,gg=(a,b,gold) if orientation=='original' else (b,a,'B' if gold=='A' else 'A')
   meta={'key':r['pair_id']+':'+orientation,'semantic_pair_id':r['pair_id'],'orientation':orientation,'gold':gg,'source':r['source'],'candidate_a_guid':aa['guid'],'candidate_b_guid':bb['guid']}
   evaluations.append(dict(meta,prompt=template.format(premise=r['premise'],candidate_a=aa['hypothesis'],candidate_b=bb['hypothesis']),context='provided'))
   ablations.append(dict(meta,prompt=abl_template.format(premise='',candidate_a=aa['hypothesis'],candidate_b=bb['hypothesis']),context='omitted'))
 for name,rr in [('pairs.jsonl',originals),('evaluate.jsonl',evaluations),('ablation.jsonl',ablations)]:
  (ROOT/'data'/name).write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in rr))
 report={'status':'DATA_PREPARED_BEFORE_ANY_MODEL_OUTPUT','upstream_commit':provenance['revision'],'upstream_split':'published development split, not hidden official test','input_rows':len(data),'input_premises':len(groups),'human_majority_label_check':'all original gold labels receive at least 3 of 5 recorded votes','eligible_pairs':len(eligible),'presentations_per_model_per_task':len(evaluations),'exclusions':exclusions,'selection':'all eligible premises, sorted by exact-source SHA-256 identifier; no result-based selection','position_seed':20261019,'original_gold_counts':dict(collections.Counter(positions)),'source_groups':dict(collections.Counter(r['source'] for r in eligible)),'prior_source_inputs':inputs,'unique_normalized_prior_sources_checked':len(texts),'overlap_definition':'Either normalized premise is a contiguous substring of a recovered prior source, or converse; not semantic/cross-language deduplication','no_new_example_text':True,'no_new_reference_labels':True,'templates':{'provided':template,'omitted':abl_template},'limitations':['Hypotheses were human-constructed for NLI; do not call them natural summaries.','Author-developed pairwise reformulation of published dev data, not official three-way KLUE-NLI accuracy.','Public data can have appeared in base pretraining.','No semantic overlap guarantee; prior-source file hashes support the recorded lexical checks.','Neutral examples are not evaluated, limiting the result to entailment versus contradiction.']}
 (ROOT/'DATA_AUDIT.json').write_text(json.dumps(report,indent=2,ensure_ascii=False));print(json.dumps({k:v for k,v in report.items() if k in ['input_rows','input_premises','eligible_pairs','presentations_per_model_per_task','original_gold_counts','source_groups','exclusions']},indent=2))
if __name__=='__main__':main()
