import importlib.util
from pathlib import Path
import tempfile
import unittest
import json
from protocol import choose, selections, materialize, states, WORKLOADS, ATTENTION, CACHES, SEEDS
from report import build

class ProtocolTests(unittest.TestCase):
    def row(self, backend, cache, ms, status='ok',bm=16):
        return dict(lora='best',workload=WORKLOADS[0],config=dict(backend=backend,bm=bm,bn=32,warps=4,cache_mode=cache),wall_ms=ms,status=status)

    def test_failed_fastest_cannot_win(self):
        pick=choose([self.row('fa1','off',1,'parity_failed'),self.row('fa1','off',9,bm=32),self.row('fa1','off',12)])
        self.assertTrue(pick['eligible']);self.assertEqual(pick['best']['bm'],32)
        self.assertEqual(pick['poor']['bm'],16)

    def test_diagnostic_selection_keeps_failure_explicit(self):
        pick=choose([self.row('fa1','off',1,'parity_failed')])
        self.assertFalse(pick['eligible']);self.assertFalse(pick['distinct'])
        self.assertIsNotNone(pick['best'])
        self.assertIsNone(choose([])['best'])

    def test_independent_and_joint_selection_are_distinct(self):
        rows=[self.row('fa1','off',20,bm=16),self.row('fa1','off',10,bm=32),
              self.row('eager','dynamic',9),self.row('eager','static_context',12),
              self.row('fa1','static_context',3,bm=16),self.row('fa1','dynamic',8,bm=32)]
        pick=selections(rows,'best',WORKLOADS[0])
        combined=materialize(dict(lora='best',attention='best',cache='best'),pick)
        self.assertEqual(combined['bm'],32);self.assertEqual(combined['cache_mode'],'dynamic')
        self.assertEqual(pick['joint']['best']['bm'],16)
        self.assertEqual(pick['joint']['best']['cache_mode'],'static_context')

    def test_coverage_matches_full_factorial(self):
        matrix=states()
        self.assertEqual(len(matrix),27)
        self.assertEqual(len({tuple(r.values()) for r in matrix}),27)
        self.assertEqual(len(WORKLOADS)*3*len(ATTENTION)*len(CACHES),2592)
        self.assertEqual((len(matrix)+4)*len(WORKLOADS)*len(SEEDS),2232)
        self.assertEqual(sum(all(x!='poor' for x in r.values()) for r in matrix),8)

    def test_partial_parity_never_reports_three_seed_speedup(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            out=Path(directory);(out/'plan.json').write_text(json.dumps(dict(task='basketball',precision='fp16',configs={})))
            rows=[]
            for seed in SEEDS:
                for best in (False,True):
                    rows.append(dict(id=str((seed,best)),state=dict(lora='best' if best else 'off',attention='off',cache='off'),
                        workload=WORKLOADS[0],seed=seed,config={},status='parity_failed' if best and seed==44 else 'ok',
                        wall_ms=5 if best else 10,samples=[dict(peak_allocated_bytes=100,wall_ms=5 if best else 10)]*7))
            (out/'confirmation-0.json').write_text(json.dumps(rows));build(out)
            import csv
            with (out/'COMPARISONS.csv').open() as stream:
                result=list(csv.DictReader(stream))
            best=next(r for r in result if r['lora']=='best')
            self.assertEqual(best['passing_seeds'],'2')
            self.assertEqual(best['all_seeds_passing_speed_ratio'],'')

    def test_empty_run_reports_partial_and_builds_pdf(self):
        # Missing measurements must never be represented as a completed study.
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            out=Path(directory);(out/'plan.json').write_text(json.dumps(dict(task='basketball',precision='fp16',configs={})))
            build(out)
            status=json.loads((out/'REPORT_STATUS.json').read_text())
            self.assertEqual(status['status'],'PARTIAL')
            self.assertTrue((out/'PERFORMANCE_REVIEW.pdf').read_bytes().startswith(b'%PDF-1.4'))
            self.assertIn('PARTIAL',(out/'PERFORMANCE_REVIEW.md').read_text())

if __name__=='__main__':unittest.main()
