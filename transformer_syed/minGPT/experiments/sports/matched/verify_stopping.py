"""Check final checkpoint selection, stopping criteria and paired training controls."""
import hashlib
import json
import math
import random
from pathlib import Path


def sampled_order_hash(seed,examples,batch_size,steps):
    rng=random.Random(seed)
    fingerprint=hashlib.sha256()
    order=[]
    offset=0
    for _ in range(steps):
        if offset>=len(order):
            order=list(range(examples))
            rng.shuffle(order)
            offset=0
        indices=order[offset:offset+batch_size]
        offset+=len(indices)
        fingerprint.update(json.dumps(indices).encode())
    return fingerprint.hexdigest()


def main():
    root=Path('experiments/sports/stopping')
    summary=json.loads((root/'summary.json').read_text())
    assert summary.get('complete'), 'Comparison is incomplete'
    examples=sum(1 for _ in Path('experiments/sports/train.jsonl').open())
    reports={}
    for run in summary['runs']:
        report=json.loads((Path(run['path'])/'report.json').read_text())
        config=report['config']
        steps=report['completed_steps']
        steps_per_epoch=math.ceil(examples/config['batch_size'])
        assert config['checkpoint_selection']=='final'
        assert report['selected_step']==steps
        assert report['sample_order_sha256']==sampled_order_hash(run['seed'],examples,config['batch_size'],steps)
        assert report['processed_training_tokens']==report['examples_processed']*config['length']
        assert report['completed_epochs']<=summary['epochs']
        assert report['test_evaluated_after_validation_selection']
        if run['stopping']=='epochs':
            assert steps==summary['epochs']*steps_per_epoch
            assert report['stop_reason']=='epoch_limit'
        else:
            history=report['training_nll_history']
            assert all(v['nll']>summary['target_train_loss'] for v in history[:-1])
            if report['stop_reason']=='target_train_loss':
                assert history[-1]['nll']<=summary['target_train_loss']
                assert history[-1]['step']==steps
            else:
                assert report['stop_reason']=='epoch_limit'
                assert steps==summary['epochs']*steps_per_epoch
                assert history[-1]['nll']>summary['target_train_loss']
        if run['method']=='lora':
            assert report['final_changes']['frozen_changed_elements']==0
        reports[run['method'],run['stopping'],run['seed']]=report
    for seed in summary['seeds']:
        paired=[reports[method,rule,seed] for method in ['lora','full'] for rule in ['epochs','train_loss']]
        for key in ['initial_base_sha256','data_sha256','source_sha256','gpu','dtype','tf32']:
            assert all(r[key]==paired[0][key] for r in paired), key
        for method in ['lora','full']:
            fixed=reports[method,'epochs',seed]
            loss=reports[method,'train_loss',seed]
            fixed_history={r['step']:r['train_loss'] for r in fixed['history']}
            assert all(abs(fixed_history[r['step']]-r['train_loss'])<1e-6 for r in loss['history'] if r['step'] in fixed_history)
    (root/'verification.json').write_text(json.dumps(dict(passed=True,runs=len(reports),
        checks=['final checkpoints','first threshold crossing or epoch cap','reconstructed shuffled sample hashes',
                'matching weights, data, source and GPU','paired training losses','frozen LoRA base weights']),indent=2))
    print(f'Passed stopping comparison checks for {len(reports)} runs')


if __name__=='__main__':
    main()
