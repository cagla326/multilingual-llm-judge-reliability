from pathlib import Path
import pandas as pd,json,hashlib,re,numpy as np
from collections import Counter
P=Path('korean_evidence_2026-10-19/korean_records');O=Path('Korean_Extension_2026-10-19/package');R={}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def jlines(n):return [json.loads(l) for l in (P/n).read_text().splitlines() if l.strip()]
def norm(s):return re.sub(r'\s+',' ',str(s)).strip()
def nh(s):return hashlib.sha256(norm(s).encode()).hexdigest()
R['input_archive_sha256']=sha(Path('upload/korean_records_for_review.zip'));R['manifest_checks']=[]
for mf in P.glob('*SHA256.txt'):
 for line in mf.read_text().splitlines():
  m=re.match(r'^([a-f0-9]{64})\s+\*?(.+)$',line)
  if not m:continue
  h,name=m.groups();p=P/name.strip();R['manifest_checks'].append({'manifest':mf.name,'file':name,'status':'missing' if not p.is_file() else 'match' if sha(p)==h else 'mismatch','actual':sha(p) if p.is_file() else None,'expected':h})
train=pd.read_csv(P/'korean_xlsum_TRAIN50_PAIRS_LOCKED.csv');test=pd.read_csv(P/'korean_xlsum_FROZEN_TEST50_PAIRS_LOCKED.csv');trsrc=pd.read_csv(P/'korean_xlsum_TRAIN50_sources.csv');tesrc=pd.read_csv(P/'korean_xlsum_FROZEN_TEST50_sources.csv');ori=jlines('korean_xlsum_FROZEN_TEST50_ORIGINAL.jsonl');swap=jlines('korean_xlsum_FROZEN_TEST50_SWAPPED.jsonl');aug=jlines('korean_xlsum_TRAIN100_SWAP_AUGMENTED.jsonl');protocol=json.loads((P/'korean_xlsum_FROZEN_TEST50_EVAL_PROTOCOL.json').read_text());template=protocol['prompt_template']
trainhash={nh(x) for x in train.article};testhash={nh(x) for x in test.article};mf=pd.read_csv('cluster_evidence_2026-10-19/fullsource_rerun_input_frozen.csv');mfh={nh(x) for x in mf.input}
R['split']={'train_pairs':len(train),'test_pairs':len(test),'train_unique_sources':len(trainhash),'test_unique_sources':len(testhash),'normalized_train_test_overlap':len(trainhash&testhash),'train_mface900_exact_overlap':len(trainhash&mfh),'test_mface900_exact_overlap':len(testhash&mfh),'train_source_hash_mismatches':sum(nh(row.article)!=row.article_hash for _,row in train.iterrows()),'test_source_hash_mismatches':sum(nh(row.article)!=row.article_hash for _,row in test.iterrows()),'train_sources_vs_pairs_match':{nh(x) for x in trsrc.article}==trainhash,'test_sources_vs_pairs_match_match':{nh(x) for x in tesrc.article}==testhash,'train_preferred':train.preferred.value_counts().to_dict(),'test_gold':dict(Counter(x['gold_label'] for x in ori))}
for group in ['train','test']:
 d=train if group=='train' else test;R['split'][group+'_severity']=d.severity.value_counts().to_dict()
assert len(ori)==len(swap)==50 and len(aug)==100
S={x['semantic_pair_id']:x for x in swap};bad=[]
for x in ori:
 y=S[x['semantic_pair_id']]
 if not (x['article']==y['article'] and x['summary_a']==y['summary_b'] and x['summary_b']==y['summary_a'] and x['gold_label']!=y['gold_label'] and (x['summary_a'] if x['gold_label']=='A' else x['summary_b'])==x['grounded_summary']):bad.append(x['semantic_pair_id'])
R['test_orientation_mismatches']=bad;ag={}
for x in aug:ag.setdefault(x['semantic_pair_id'],[]).append(x)
bad=[]
for k,rs in ag.items():
 if len(rs)!=2:bad.append(k);continue
 a=next(r for r in rs if r['orientation']=='original');b=next(r for r in rs if r['orientation']=='swapped')
 if not (a['article']==b['article'] and a['summary_a']==b['summary_b'] and a['summary_b']==b['summary_a'] and a['chosen']!=b['chosen']):bad.append(k)
 for r in rs:
  if template.format(article=r['article'],summary_a=r['summary_a'],summary_b=r['summary_b'])!=r['prompt']:bad.append(k)
R['train_augmentation_mismatches']=bad
# Frozen predictions: recompute strict parser and conditional choice; no importing historical script (runs at module scope).
pred=pd.read_csv(P/'FINAL_KOREAN_TEST50_EVAL/FINAL_TEST50_all_predictions.csv',keep_default_na=False);R['prediction_checks']={'n':len(pred),'duplicate_presentations':int(pred.duplicated(['system','semantic_pair_id','orientation']).sum())};R['summaries']={}
for sys,dd in pred.groupby('system'):
 out={}
 for orient in ['original','swapped']:
  d=dd[dd.orientation==orient];free=d.free_generated_text.str.strip();valid=free.isin(['A','B']);corr=valid & free.eq(d.gold_label);lp=np.where(d.logprob_A.astype(float)>d.logprob_B.astype(float),'A',np.where(d.logprob_B.astype(float)>d.logprob_A.astype(float),'B',''));lpc=lp==d.gold_label;out[orient]={'n':len(d),'strict_accuracy':float(corr.mean()),'invalid':int((~valid).sum()),'conditional_valid_accuracy':float(corr[valid].mean()) if valid.any() else None,'logprob_accuracy':float(lpc.mean()),'stored_free_correct_mismatches':int((corr.to_numpy()!=d.free_correct.astype(str).str.lower().isin(['true','1']).to_numpy()).sum()),'stored_logprob_correct_mismatches':int((lpc!=d.logprob_correct.astype(str).str.lower().isin(['true','1']).to_numpy()).sum())}
 # semantic swap consistency: valid opposite labels
 a=dd[dd.orientation=='original'].set_index('semantic_pair_id');b=dd[dd.orientation=='swapped'].set_index('semantic_pair_id').loc[a.index];pa=a.free_generated_text.str.strip();pb=b.free_generated_text.str.strip();out['swap_consistency']=float((pa.isin(['A','B'])&pb.isin(['A','B'])&pa.ne(pb)).mean());R['summaries'][sys]=out
R['base_response_examples']=pred[pred.system=='BASE'].free_generated_text.value_counts().head(8).to_dict();R['test_families']=test.corruption_family.value_counts().to_dict()
# Historical prompt-file byte equality
stored=Path('Chapter3_Kathy_Review_2026-10-19/prompts/korean_test50_utf8.txt').read_text();R['korean_template_matches']=stored.strip()==template.strip()
# Artifacts identities relevant to new execution
R['required_hashes']={n:sha(P/n) for n in ['eval_korean_xlsum_FINAL_TEST50.py','train_korean_xlsum_FINAL_DPO.py','korean_xlsum_FROZEN_TEST50_ORIGINAL.jsonl','korean_xlsum_FROZEN_TEST50_SWAPPED.jsonl','korean_xlsum_FROZEN_TEST50_EVAL_PROTOCOL.json','korean_xlsum_TRAIN50_sources.csv','korean_xlsum_TRAIN50_PAIRS_LOCKED.csv','gemma3_4b_ko_xlsum_FINAL_DPO_train100/adapter_config.json']}
(O/'KOREAN_AUDIT.json').write_text(json.dumps(R,indent=2));print(json.dumps({k:v for k,v in R.items() if k not in ['manifest_checks','required_hashes']},indent=2));print('MANIFEST',dict(Counter(x['status'] for x in R['manifest_checks'])));print('MISMATCHES',[x for x in R['manifest_checks'] if x['status']=='mismatch'])
