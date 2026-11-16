"""Regression checks for matched grid/budgets and SMS target alignment."""
import importlib.util
from pathlib import Path
import sys
import unittest

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
sys.path.insert(0,str(HERE.parents[3]))


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


class ProtocolTests(unittest.TestCase):
    def test_core_grid_matches_fp32(self):
        fp16=module('fp16',HERE/'study.py')
        fp32=module('fp32',HERE.parent/'hyperparameter_search/study.py')
        self.assertEqual(fp16.grid(),fp32.grid())
        self.assertEqual(len(fp16.grid()),90)

    def test_spam_label_alignment_and_groups(self):
        import torch
        from transformers import GPT2Tokenizer
        import spam_task
        import json
        tok=GPT2Tokenizer.from_pretrained('gpt2',local_files_only=True)
        splits={s:[json.loads(l) for l in (HERE.parents[3]/f'experiments/sms_spam/data/{s}.jsonl').read_text().splitlines()] for s in ['train','validation','test']}
        for a,b in [('train','validation'),('train','test'),('validation','test')]:
            self.assertFalse({r['group'] for r in splits[a]} & {r['group'] for r in splits[b]})
        for row in splits['train'][:25]:
            x,y=spam_task.encode([row],tok,128,torch)[0]
            end=len(spam_task.prompt(row,tok))-1
            self.assertEqual(int((y!=-1).sum()),1)
            self.assertEqual(int(y[end]),tok.encode(' '+['ham','spam'][row['label']])[0])
            self.assertEqual(int(x[end+1]),int(y[end]))


if __name__=='__main__':unittest.main()
