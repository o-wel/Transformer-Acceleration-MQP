"""Synthetic artifacts exercise selection -> confirmation analysis -> report export."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from experiments.paper_hyperparameters.common import save,read
from experiments.paper_hyperparameters.prepare import prepare
from experiments.paper_hyperparameters.protocol import DESIGN,GRID,BASE,CHUNKS,TILES,SEARCH_SEEDS,CONFIRM_SEEDS,SEARCH_WORKLOADS,WORKLOADS,config_key,workload_key
from experiments.paper_hyperparameters.analysis import select,report


class ArtifactIntegrationTests(unittest.TestCase):
    def test_selection_and_final_export_from_synthetic_artifacts(self):
        with tempfile.TemporaryDirectory(prefix='mqp-paper-integration-') as d:
            study=prepare(Path(d)/'study')
            for c in GRID:
                for seed in SEARCH_SEEDS:
                    r=dict(complete=True,successful_updates=2400,config=c,seed=seed,
                           validation_nll=.1+c['rank']/1000,warm_median_step_ms=10+c['rank'],
                           validation=[dict(step=i,nll=.1) for i in range(300,2401,300)],selected_step=2400,
                           frozen_base_before='same',frozen_base_after='same',schedule_sha256=str(seed),measured_update_seconds=10.,training_elapsed_seconds=20.)
                    save(study/'search-training'/config_key(c)/f'seed-{seed}'/'training.json',r)
            for session in range(3):
                modes={f'cache-{c}':{} for c in CHUNKS}|{'fa-'+'-'.join(map(str,t)):{} for t in TILES}
                gates=dict(cache={f'cache-{c}':dict(valid=True) for c in CHUNKS},fa={'fa-'+'-'.join(map(str,t)):dict(valid=True) for t in TILES})
                blocks={workload_key(w):dict(samples={name:[dict(wall_ms=10.)] for name in modes}) for w in SEARCH_WORKLOADS}
                save(study/'runtime-search'/f'gates-{session}.json',gates)
                save(study/'runtime-search'/f'session-{session}.json',dict(complete=True,workloads=blocks))
            chosen=select(study)
            self.assertEqual(chosen['lora']['rank'],2)
            modes=[f'factorial-{i:03b}' for i in range(8)]+['lora_baseline','lora_tuned','cache_baseline','cache_tuned','fa_baseline','fa_tuned']
            for seed in CONFIRM_SEEDS:
                for c in (BASE,chosen['lora']):
                    save(study/'confirmation-training'/config_key(c)/f'seed-{seed}'/'training.json',dict(complete=True,schedule_sha256=str(seed),warm_median_step_ms=10 if c==BASE else 5,
                         validation_nll=.1,successful_updates=2400,measured_update_seconds=10.,training_elapsed_seconds=20.))
                for session in range(3):
                    blocks={}
                    for w in WORKLOADS:
                        samples={}
                        for name in modes:
                            value=5. if name in ('factorial-111','lora_tuned','cache_tuned','fa_tuned') else 10.
                            samples[name]=[dict(repeat=i,order=0,prompt_seed=i,wall_ms=value,cuda_ms=value,peak_allocated_bytes=1,incremental_peak_bytes=1,output_sha256='synthetic') for i in range(2)]
                        blocks[workload_key(w)]=dict(workload=w,samples=samples)
                    save(study/'confirmation-timing'/f'seed-{seed}-session-{session}.json',dict(complete=True,workloads=blocks))
                models={}
                for label in ('baseline','tuned'):
                    rows=[]
                    for group in range(300):
                        for counter in (False,True):
                            for kind in ('winner','halftime','comeback'):
                                rows.append(dict(group_id=str(group),match_id=str(group)+('-counter-'+kind if counter else ''),kind=kind,
                                                 scores=dict(reference_exact=label=='tuned',fields_correct=label=='tuned')))
                    outputs={variant:dict(answers=rows) for variant in ('eager','combined')}
                    models[label]=dict(outputs=outputs,runtime_nll_gate_passed=True)
                save(study/'confirmation-quality'/f'seed-{seed}.json',dict(complete=True,seed=seed,models=models))
            with patch.dict(DESIGN['stats'],bootstrap_draws=20),patch.dict(DESIGN['timing'],confirmation_repeats=2):
                report(study)
            r=read(study/'results.json')
            self.assertAlmostEqual(r['comparisons']['combined']['heldout']['speedup'],2.)
            self.assertAlmostEqual(r['training_speedup']['speedup'],2.)
            self.assertEqual(r['quality']['reference_exact']['paired_difference'],1.)
            self.assertTrue((study/'TIMINGS.csv').stat().st_size>0)
            self.assertTrue((study/'QUALITY_BY_SEED.csv').stat().st_size>0)
            self.assertTrue((study/'RESULTS.html').stat().st_size>0)


if __name__=='__main__':unittest.main()
