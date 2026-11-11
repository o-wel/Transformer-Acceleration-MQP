"""Offline checks for scoring, leakage guards, and classification token positions."""
import hashlib
import json
import sys
import unittest
from pathlib import Path

import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
sys.path.insert(0,str(HERE.parents[1]))
from experiment import metrics, template, rows
from mingpt.model import GPT
from mingpt.measurement import ChangeAudit


class TestProtocol(unittest.TestCase):
    def test_metrics(self):
        r=metrics([0,0,1,1],[0,1,0,1])
        for key in ['accuracy','balanced_accuracy','spam_precision','spam_recall','spam_f1']:
            self.assertEqual(r[key],.5)
        self.assertEqual([r[k] for k in ['tp','tn','fp','fn']],[1,1,1,1])
        self.assertEqual(metrics([0,0,1],[0,0,0])['spam_f1'],0.)

    def test_template_and_splits(self):
        self.assertEqual(template('WIN 123! www.example.com/a'),template('win 456? https://other.org/b'))
        manifest=json.loads((HERE/'data/manifest.json').read_text())
        for name,entry in manifest['splits'].items():
            data=rows(name)
            self.assertEqual(len(data),entry['n'])
            self.assertEqual(sum(r['label'] for r in data),entry['spam'])
            self.assertEqual(hashlib.sha256((HERE/'data'/f'{name}.jsonl').read_bytes()).hexdigest(),entry['sha256'])
        for a,b in [('train','validation'),('train','test'),('validation','test')]:
            for key in ['id','group']:
                self.assertFalse({r[key] for r in rows(a)} & {r[key] for r in rows(b)})

    def test_padding_and_adapter_training(self):
        torch.manual_seed(7);torch.set_num_threads(1)
        cfg=GPT.get_default_config();cfg.model_type=None
        cfg.n_layer=1;cfg.n_head=2;cfg.n_embd=16;cfg.vocab_size=32;cfg.block_size=16
        cfg.embd_pdrop=cfg.resid_pdrop=cfg.attn_pdrop=0.
        model=GPT(cfg).eval()
        x=torch.tensor([[1,2,3,4],[5,6,0,0]])
        ends=torch.tensor([3,1])
        expected=torch.cat([model(x[:1])[0][:,-1],model(x[1:2,:2])[0][:,-1]])
        actual=model(x)[0][torch.arange(2),ends]
        torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)
        model.enable_lora(4,8)
        audit=ChangeAudit(model)
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.01)
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            logits=model(x)[0][torch.arange(2),ends]
            torch.nn.functional.cross_entropy(logits,torch.tensor([7,8])).backward()
            optimizer.step()
        changes=audit.measure(model)
        self.assertEqual(changes['frozen_changed_elements'],0)
        self.assertGreater(changes['changed_elements'],0)


if __name__=='__main__':unittest.main()
