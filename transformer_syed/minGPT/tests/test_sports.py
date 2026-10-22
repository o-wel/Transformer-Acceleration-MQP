import json
import unittest
from pathlib import Path
from experiments.sports.data import example
from experiments.sports.scoring import score


class TestSports(unittest.TestCase):
    def test_counterfactual_logic_and_scoring(self):
        m=dict(id='x',teams=['Eagles','Owls'],half=[30,40],final=[85,80])
        row=example(m,'comeback')
        self.assertTrue(score(row,row['answer'])['reference_exact'])
        self.assertEqual(row['primary'],'Yes')
        self.assertEqual(example(dict(m,half=[40,30]),'comeback')['primary'],'No')
        self.assertFalse(score(row,'No. Eagles won 85-80.')['fields_correct'])
        row=example(m,'winner')
        self.assertFalse(score(row,'Eagles did not win 85-80.')['primary_correct'])
        self.assertFalse(score(row,'Eagles won 85-80 in 2026.')['fields_correct'])
        self.assertTrue(score(row,'Eagles won, 85 to 80.')['fields_correct'])
        self.assertEqual(example(dict(m,final=[80,85]),'winner')['primary'],'Owls')

    def test_batch_generation_matches_single(self):
        import torch
        from test_lora import tiny_model
        from sports_experiment import generate_batch
        class Tokenizer:
            eos_token_id=15
            def encode(self,text):
                return list(range(1,2+len(text)%4))
            def decode(self,ids,skip_special_tokens=True):
                return ' '.join(str(i) for i in ids if i!=15)
        torch.set_num_threads(1)
        torch.manual_seed(5)
        model=tiny_model().eval()
        tokenizer=Tokenizer()
        questions=['a','ab','abc','abcd']
        batched=generate_batch(model,tokenizer,questions,'cpu',3)
        single=[generate_batch(model,tokenizer,[q],'cpu',3)[0] for q in questions]
        self.assertEqual(batched,single)

    def test_only_answer_tokens_are_targets(self):
        from lora_experiment import encode
        class Tokenizer:
            eos_token_id=15
            def encode(self,text):
                return [1,2,3] if text.startswith('User:') else [4,5]
        x,y=encode([dict(question='q',answer='a')],Tokenizer(),8)[0]
        self.assertEqual(x.tolist(),[1,2,3,4,5,15,15,15])
        self.assertEqual(y.tolist(),[-1,-1,4,5,15,-1,-1,-1])

    def test_split_integrity(self):
        root=Path(__file__).resolve().parents[1]/'experiments/sports'
        rows={s:[json.loads(x) for x in (root/f'{s}.jsonl').read_text().splitlines()] for s in ['train','validation','test']}
        signatures={s:{tuple(r['match']['half']+r['match']['final']) for r in rs} for s,rs in rows.items()}
        self.assertFalse(signatures['train'] & signatures['validation'])
        self.assertFalse(signatures['train'] & signatures['test'])
        self.assertFalse(signatures['validation'] & signatures['test'])
        names=lambda rs:{t for r in rs for t in r['teams']}
        self.assertFalse(names(rows['train']) & names(rows['test']))
        self.assertFalse({r['answer'] for r in rows['train']} & {r['answer'] for r in rows['test']})


if __name__=='__main__':
    unittest.main()
