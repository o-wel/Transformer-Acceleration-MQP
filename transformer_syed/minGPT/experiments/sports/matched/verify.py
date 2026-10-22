"""Verify completed matched experiments without relying on rounded report tables."""
import json
from pathlib import Path


def main():
    manifest=json.loads(Path('experiments/sports/matched/selection.json').read_text())
    assert manifest.get('complete')
    reports={m:{s:json.loads((Path(path)/'report.json').read_text()) for s,path in info['runs'].items()} for m,info in manifest['methods'].items()}
    for method,info in manifest['methods'].items():
        chosen=min(info['candidates'],key=lambda c:(c['validation_nll'],c['lr']))
        assert info['selected_lr']==chosen['lr']
        assert len(info['candidates'])==2
        for candidate in info['candidates']:
            report=json.loads((Path(candidate['path'])/'report.json').read_text())
            assert report['config']['steps']==1200
            assert report['selected_validation_nll']==candidate['validation_nll']
            if candidate['path']!=info['runs']['42']:
                assert 'full' not in report and 'lora' not in report and 'base' not in report
        for r in reports[method].values():
            assert r['config']['lr']==info['selected_lr']
            assert r['selected_validation_nll']==min(h['nll'] for h in r['validation_history'])
            assert r['batch_generation_checked'] and r['test_evaluated_after_validation_selection']
            assert len(r[method]['outputs'])==120
            assert len(r[method]['counterfactual_outputs'])==120
            assert r['parameters']['trainable']==(589824 if method=='lora' else 163037184)
            assert r['selected_changes']['changed_elements']>0
            if method=='lora':
                assert all(h['frozen_changed_elements']==0 for h in r['history'])
                assert r['selected_changes']['frozen_changed_elements']==0
    for seed in ['42','43','44']:
        left,right=reports['lora'][seed],reports['full'][seed]
        for key in ['initial_base_sha256','sample_order_sha256','data_sha256','source_sha256','processed_training_tokens','supervised_training_tokens','gpu','torch_version']:
            assert left[key]==right[key],(seed,key)
        for key in ['steps','batch_size','length','seed']:
            assert left['config'][key]==right['config'][key]
        assert left['dtype']==right['dtype']=='float32'
        assert left['tf32']==right['tf32']==False
    print('PASS: equal tuning budgets; validation-only selection; identical base weights, data/order, token budgets, FP32 and GPU; full weights changed; LoRA base frozen; all output/checkpoint checks.')


if __name__=='__main__': main()
