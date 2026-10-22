import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
import torch
from test_lora import tiny_model
from sports_experiment import adapter, load_parameters
from experiments.sports.matched.run_pipeline import select_candidate


class TestMatched(unittest.TestCase):
    def test_full_and_lora_checkpoint_roundtrip(self):
        for full in [True,False]:
            model=tiny_model().eval()
            restored=tiny_model().eval()
            if not full:
                # Identical pretrained base is required when reloading only the adapters.
                restored.load_state_dict(model.state_dict())
                model.enable_lora(4,8)
                restored.enable_lora(4,8)
            with torch.no_grad():
                for p in model.parameters():
                    if p.requires_grad: p.add_(0.01)
            ckpt={'weights':adapter(model)}
            load_parameters(restored,ckpt)
            x=torch.arange(8).unsqueeze(0)
            torch.testing.assert_close(model(x)[0],restored(x)[0],atol=0,rtol=0)
            ckpt['weights'].pop(next(iter(ckpt['weights'])))
            with self.assertRaises(RuntimeError): load_parameters(restored,ckpt)

    def test_selection_uses_validation_not_test(self):
        candidates=[dict(lr=.001,validation_nll=.4,test_score=1),dict(lr=.0001,validation_nll=.2,test_score=0)]
        self.assertEqual(select_candidate(candidates)['lr'],.0001)

    def test_epoch_and_loss_stopping_evaluate_final_checkpoint(self):
        import sports_experiment
        data=[(torch.arange(8),torch.arange(8)) for _ in range(3)]
        for loss_target,expected_steps,expected_epochs,reason in [(None,4,2,'epoch_limit'),(100,2,1,'target_train_loss')]:
            with tempfile.TemporaryDirectory() as directory:
                out=Path(directory)/'run'
                args=['sports_experiment.py','--out',str(out),'--epochs','2',
                      '--batch-size','2','--eval-every','100','--audit-every','100',
                      '--device','cpu','--checkpoint-selection','final','--skip-test']
                if loss_target is not None:
                    args+=['--target-train-loss',str(loss_target)]
                with patch('sys.argv',args), patch.object(sports_experiment.GPT2Tokenizer,'from_pretrained',return_value=object()), \
                     patch.object(sports_experiment.GPT,'from_pretrained',side_effect=lambda _: tiny_model().eval()), \
                     patch.object(sports_experiment,'encode',return_value=data):
                    sports_experiment.main()
                report=json.loads((out/'report.json').read_text())
                self.assertEqual(report['completed_steps'],expected_steps)
                self.assertEqual(report['selected_step'],expected_steps)
                self.assertEqual(report['completed_epochs'],expected_epochs)
                self.assertEqual(report['examples_processed'],3*expected_epochs)
                self.assertEqual(report['stop_reason'],reason)
                self.assertEqual(report['processed_training_tokens'],3*expected_epochs*128)
                self.assertTrue((out/'adapter.pt').exists())


if __name__=='__main__': unittest.main()
