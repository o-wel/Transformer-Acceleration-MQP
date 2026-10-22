"""Offline checks for LoRA math, training isolation, and checkpoint loading."""
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from mingpt.model import GPT, LoRALinear
from mingpt.trainer import Trainer


def tiny_model():
    config = GPT.get_default_config()
    config.model_type = None
    config.n_layer, config.n_head, config.n_embd = 1, 2, 16
    config.vocab_size, config.block_size = 16, 8
    config.embd_pdrop = config.resid_pdrop = config.attn_pdrop = 0.0
    return GPT(config)


class TestLoRA(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(1)

    def test_linear_matches_effective_weight(self):
        linear = nn.Linear(7, 11).double()
        adapter = LoRALinear(linear, rank=3, alpha=6)
        x = torch.randn(2, 5, 7, dtype=torch.float64)
        torch.testing.assert_close(adapter(x), linear(x), rtol=0, atol=0)
        nn.init.normal_(adapter.B.weight)
        effective = linear.weight + adapter.scale * (adapter.B.weight @ adapter.A.weight)
        torch.testing.assert_close(adapter(x), F.linear(x, effective, linear.bias))

    def test_training_and_checkpoint(self):
        model = tiny_model().eval()
        x = torch.arange(8).repeat(4, 1)
        targets = (x + 1) % 16
        before = model(x)[0].detach().clone()
        model.enable_lora(rank=4, alpha=8)
        torch.testing.assert_close(model(x)[0], before, rtol=0, atol=0)
        self.assertEqual(sum(isinstance(m, LoRALinear) for m in model.modules()), 4)
        frozen = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
        trainable = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
        self.assertTrue(all(n.endswith(('A.weight', 'B.weight')) for n in trainable))
        config = Trainer.get_default_config()
        config.learning_rate = 0.03
        config.weight_decay = 0.0
        optimizer = model.configure_optimizers(config)
        self.assertEqual({id(p) for g in optimizer.param_groups for p in g['params']},
                         {id(p) for p in model.parameters() if p.requires_grad})
        initial = model(x, targets)[1].item()
        for _ in range(40):
            optimizer.zero_grad(set_to_none=True)
            loss = model(x, targets)[1]
            loss.backward()
            optimizer.step()
        final = model(x, targets)[1].item()
        self.assertLess(final, initial)
        for name, param in model.named_parameters():
            if name in frozen:
                torch.testing.assert_close(param, frozen[name], rtol=0, atol=0)
                self.assertIsNone(param.grad)
            else:
                self.assertFalse(torch.equal(param, trainable[name]), name)
        restored = tiny_model().enable_lora(rank=4, alpha=8).eval()
        restored.load_state_dict(model.state_dict())
        torch.testing.assert_close(restored(x)[0], model(x)[0], rtol=0, atol=0)
        print(f'LoRA toy training loss: {initial:.4f} -> {final:.4f}')

    def test_validation_and_base_optimizer(self):
        model = tiny_model()
        optimizer = model.configure_optimizers(Trainer.get_default_config())
        self.assertEqual(sum(len(g['params']) for g in optimizer.param_groups), len(list(model.parameters())))
        for rank in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                model.enable_lora(rank=rank)
        self.assertTrue(all(p.requires_grad for p in model.parameters()))
        model.enable_lora()
        with self.assertRaises(ValueError):
            model.enable_lora()


if __name__ == '__main__':
    unittest.main()
