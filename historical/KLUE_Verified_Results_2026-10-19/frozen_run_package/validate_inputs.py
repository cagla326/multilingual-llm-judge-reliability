#!/usr/bin/env python3
"""Verify all pair text and labels against exact original published records."""
import hashlib,json
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parent
read=lambda p:[json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]
def validate_data():
 provenance=json.loads((ROOT/'SOURCE_PROVENANCE.json').read_text())
 for f in provenance['files']:
  assert hashlib.sha256((ROOT/f['path']).read_bytes()).hexdigest()==f['sha256'],f['path']
 upstream=json.loads((ROOT/'data/klue-nli-v1.1_dev.json').read_text());orig={r['guid']:r for r in upstream};assert len(orig)==len(upstream)==3000
 for r in upstream:assert [r[k] for k in ['author','label2','label3','label4','label5']].count(r['gold_label'])>=3
 pairs=read(ROOT/'data/pairs.jsonl');evaluation=read(ROOT/'data/evaluate.jsonl');ablation=read(ROOT/'data/ablation.jsonl');assert len(pairs)==1000;assert len(evaluation)==len(ablation)==2000
 audit=json.loads((ROOT/'DATA_AUDIT.json').read_text());em={r['key']:r for r in evaluation};am={r['key']:r for r in ablation};assert len(em)==len(evaluation) and set(em)==set(am)
 assert len({r['semantic_pair_id'] for r in pairs})==1000 and len({r['premise'] for r in pairs})==1000
 assert Counter(r['gold'] for r in pairs)=={'A':500,'B':500}
 for p in pairs:
  a=orig[p['candidate_a_guid']];b=orig[p['candidate_b_guid']];e=orig[p['entailment_guid']];c=orig[p['contradiction_guid']];neutral=orig[p['neutral_guid_excluded']]
  assert a['hypothesis']==p['candidate_a'] and b['hypothesis']==p['candidate_b']
  assert a['premise']==b['premise']==p['premise']==neutral['premise']
  assert e['gold_label']=='entailment' and c['gold_label']=='contradiction' and neutral['gold_label']=='neutral'
  assert (a if p['gold']=='A' else b)==e
  assert p['original_validated_records']=={'entailment':e,'contradiction':c,'neutral':neutral}
  for ori in ['original','swapped']:
   key=p['semantic_pair_id']+':'+ori;full=em[key];abl=am[key];aa,bb,gg=(a,b,p['gold']) if ori=='original' else (b,a,'B' if p['gold']=='A' else 'A')
   assert full['gold']==abl['gold']==gg and full['orientation']==abl['orientation']==ori
   assert full['candidate_a_guid']==abl['candidate_a_guid']==aa['guid'];assert full['candidate_b_guid']==abl['candidate_b_guid']==bb['guid']
   assert full['source']==abl['source']==p['source']==a['source']==b['source']
   assert full['prompt']==audit['templates']['provided'].format(premise=p['premise'],candidate_a=aa['hypothesis'],candidate_b=bb['hypothesis'])
   assert abl['prompt']==audit['templates']['omitted'].format(premise='',candidate_a=aa['hypothesis'],candidate_b=bb['hypothesis'])
   assert full['context']=='provided' and abl['context']=='omitted'
 return {'published_rows':len(upstream),'semantic_pairs':len(pairs),'presentations_per_system_per_context':len(evaluation),'human_gold_votes_and_verbatim_text':'all checked','assistant_generated_examples':0,'assistant_generated_reference_labels':0}
if __name__=='__main__':print(json.dumps(validate_data(),indent=2))
