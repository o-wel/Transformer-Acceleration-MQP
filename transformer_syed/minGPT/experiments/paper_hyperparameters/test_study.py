"""Tests for statistical units, selection constraints, cache semantics and frozen data."""
import json
import math
from pathlib import Path
import tempfile
import unittest

from experiments.paper_hyperparameters.analysis import choose_lora,gmean,hierarchical_interval
from experiments.paper_hyperparameters.common import read,verify
from experiments.paper_hyperparameters.protocol import GRID,BASE,WORKLOADS,SEARCH_WORKLOADS,TILES
from experiments.paper_hyperparameters.prepare import prepare


class StudyTests(unittest.TestCase):
    def test_search_grid_and_context_bounds(self):
        self.assertEqual(len(GRID),45);self.assertIn(BASE,GRID);self.assertEqual(len(TILES),24)
        self.assertEqual(len(WORKLOADS),14);self.assertEqual(len(SEARCH_WORKLOADS),6)
        self.assertTrue(all(w['prompt']+w['new_tokens']<=1024 for w in WORKLOADS))
        self.assertEqual(len({(w['batch'],w['prompt'],w['new_tokens']) for w in WORKLOADS}),14)

    def test_quality_constraint_excludes_fast_bad_fit(self):
        configs=[dict(rank=r,alpha=2*r,lr=.001) for r in (2,4,8)]
        rows=[dict(config=c,mean_nll=n,mean_step_ms=t) for c,n,t in zip(configs,[.2,.1,.105],[1,10,8])]
        best,winner=choose_lora(rows,.01)
        self.assertEqual(best['config'],configs[1]);self.assertEqual(winner['config'],configs[2])

    def test_no_forced_bad_control(self):
        r=dict(config=BASE,mean_nll=.1,mean_step_ms=5.)
        best,winner=choose_lora([r],.01)
        self.assertEqual(winner['config'],BASE)

    def test_hierarchical_interval_constant_and_direction(self):
        cube=[[[[math.log(.5)]*20 for _ in range(2)] for _ in range(3)] for _ in range(5)]
        result=hierarchical_interval(cube,200)
        self.assertAlmostEqual(result['speedup'],.5)
        self.assertEqual(result['ci95'],[.5,.5])
        self.assertEqual(result['training_seeds'],5);self.assertEqual(result['allocations_per_seed'],3)

    def test_many_repeats_do_not_erase_seed_variation(self):
        cube=[[[[math.log(x)]*20] for _ in range(3)] for x in [1,1,2,4,4]]
        interval=hierarchical_interval(cube,400)
        self.assertGreater(interval['ci95'][1]-interval['ci95'][0],1.)
        self.assertAlmostEqual(interval['speedup'],gmean([1,1,2,4,4]))

    def test_fresh_data_clusters_and_snapshot_integrity(self):
        with tempfile.TemporaryDirectory(prefix='mqp-paper-test-') as d:
            study=prepare(Path(d)/'study')
            verify(study)
            data=study/'snapshot/repo/experiments/paper_hyperparameters/data'
            test=[json.loads(l) for l in (data/'test.jsonl').read_text().splitlines()]
            cf=[json.loads(l) for l in (data/'counterfactual.jsonl').read_text().splitlines()]
            self.assertEqual(len(test),900);self.assertEqual(len(cf),900)
            groups={r['group_id'] for r in test};self.assertEqual(len(groups),300)
            for group in groups:
                self.assertEqual(sum(r['group_id']==group for r in test+cf),6)
            for a,b in zip(test,cf):
                self.assertEqual(a['group_id'],b['group_id'])
                self.assertNotEqual(a['primary'],b['primary'])
            (study/'protocol.json').write_text('{}')
            with self.assertRaisesRegex(RuntimeError,'protocol'):verify(study)


class CacheTests(unittest.TestCase):
    def test_chunked_generation_matches_dense_on_cpu(self):
        import torch
        from mingpt.model import GPT
        from experiments.paper_hyperparameters.engine import generate
        torch.set_num_threads(1);torch.manual_seed(81)
        c=GPT.get_default_config();c.model_type='gpt-nano';c.vocab_size=32;c.block_size=32
        model=GPT(c).eval();ids=torch.randint(32,(2,13))
        reference=generate(model,ids,19)
        for chunk in [1,4,8,13,1024]:
            with self.subTest(chunk=chunk):self.assertTrue(torch.equal(reference,generate(model,ids,19,True,chunk)))
        with self.assertRaises(ValueError):generate(model,ids,20,True,8)


if __name__=='__main__':unittest.main()
