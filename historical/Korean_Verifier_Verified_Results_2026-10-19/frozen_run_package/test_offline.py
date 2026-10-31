#!/usr/bin/env python3
"""Software fixtures are never used as model-training/evaluation examples."""
import copy,hashlib,json,unittest
from data_checks import validate_data,read_rows,ROOT
from verifier import metrics,encode_train,spec
from analyze import checked,clustered_ci
class FakeTokenizer:
 def __call__(self,s,**kw):return {'input_ids':[10]}
 def convert_tokens_to_ids(self,s):return 11
 def convert_ids_to_tokens(self,s):return '<end_of_turn>'
class FakeTensor:
 def __init__(self,ids):self.ids=ids
 def __getitem__(self,k):return self
 def tolist(self):return self.ids
class FakeProcessor:
 tokenizer=FakeTokenizer()
 def apply_chat_template(self,messages,**kw):return {'input_ids':FakeTensor([1,2,3] if len(messages)==1 else [1,2,3,10,11,99])}
class Tests(unittest.TestCase):
 def test_sources_and_protocol(self):
  self.assertEqual(validate_data()['train']['rows'],3000);spec()
 def test_answer_serialization(self):
  e=encode_train(FakeProcessor(),{'key':'software_fixture','prompt':'fixture','answer':'A'})
  self.assertEqual(e['prompt_ids'],[1,2,3]);self.assertEqual(e['answer_ids'],[10,11])
  # A causal model predicts the first answer from final prompt-token position.
  P=len(e['prompt_ids']);L=len(e['answer_ids']);self.assertEqual(list(range(P-1,P-1+L)),[2,3])
 def test_metrics(self):
  rr=[{'human_label':x,'prediction':x,'nll':.1} for x in ['entailment','contradiction','neutral']]
  self.assertEqual(metrics(rr)['accuracy'],1);self.assertEqual(metrics(rr)['macro_f1'],1)
  for i,r in enumerate(rr):r['premise_hash']=str(i)
  ci=clustered_ci(rr,rr,100);self.assertEqual(ci['accuracy_delta_95_percentile_ci'],[0.,0.]);self.assertEqual(ci['macro_f1_delta_95_percentile_ci'],[0.,0.])
 def test_reject_missing_duplicate(self):
  ref=read_rows(ROOT/'data/internal_test.jsonl')
  with self.assertRaises(ValueError):checked([],ref,'base')
  with self.assertRaises(ValueError):checked([{'key':'duplicate'}]*600,ref,'base')
 def test_schedule(self):
  groups=[list(range(3000))[i:i+8] for i in range(0,3000,8)]
  self.assertEqual(len(groups),375);self.assertEqual(sum(map(len,groups)),3000);self.assertEqual(2*len(groups),750)
  for start in [0,50,374,375]:self.assertEqual(list(range(start))+list(range(start,len(groups))),list(range(375)))
if __name__=='__main__':unittest.main()
