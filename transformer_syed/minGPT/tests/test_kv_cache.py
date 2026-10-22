"""KV-cache correctness, context-window semantics, and training compatibility."""
import unittest
import torch
from mingpt.model import GPT, LoRALinear


def tiny_model():
    config = GPT.get_default_config()
    config.model_type = None
    config.n_layer, config.n_head, config.n_embd = 2, 2, 16
    config.vocab_size, config.block_size = 31, 12
    return GPT(config)


class TestKVCache(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        torch.set_num_threads(1)

    @torch.no_grad()
    def test_chunked_logits_base_and_nonzero_lora(self):
        for lora in (False, True):
            model = tiny_model().eval()
            if lora:
                model.enable_lora(rank=2, alpha=4)
                for module in model.modules():
                    if isinstance(module, LoRALinear):
                        torch.nn.init.normal_(module.B.weight, std=.05)
            idx = torch.randint(0, 31, (2, 12))
            reference = model(idx)[0]
            cache = None
            outputs = []
            start = 0
            for length in (4, 1, 3, 4):
                old = None if cache is None else [(k.clone(), v.clone()) for k, v in cache]
                logits, loss, new_cache = model(idx[:, start:start+length],
                                               past_key_values=cache, use_cache=True)
                self.assertIsNone(loss)
                if old is not None:
                    for (k, v), (ok, ov) in zip(cache, old):
                        self.assertTrue(torch.equal(k, ok) and torch.equal(v, ov))
                cache = new_cache
                start += length
                for k, v in cache:
                    self.assertEqual(k.shape, (2, 2, start, 8))
                    self.assertEqual(v.shape, k.shape)
                    self.assertFalse(k.requires_grad or v.requires_grad)
                outputs.append(logits)
            torch.testing.assert_close(torch.cat(outputs, dim=1), reference, rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(model(idx, last_token_only=True)[0], reference[:, -1:],
                                       rtol=1e-5, atol=1e-6)

    def test_generation_and_window_rebuild(self):
        model = tiny_model().eval()
        for length in (1, 5, 12, 15):
            idx = torch.randint(0, 31, (2, length))
            for sampling in (False, True):
                torch.manual_seed(567)
                expected = model.generate(idx, 16, do_sample=sampling, top_k=5)
                torch.manual_seed(567)
                actual = model.generate(idx, 16, do_sample=sampling, top_k=5, use_cache=True)
                self.assertTrue(torch.equal(actual, expected), (length, sampling))
            self.assertTrue(torch.equal(model.generate(idx, 0, use_cache=True), idx))
        # A new generation call must not inherit the preceding call's cache.
        self.assertTrue(torch.equal(model.generate(idx, 4), model.generate(idx, 4, use_cache=True)))

    def test_training_and_checkpoint_unchanged(self):
        model = tiny_model()
        keys = set(model.state_dict())
        x = torch.randint(0, 31, (2, 5))
        logits, loss = model(x, x)
        loss.backward()
        self.assertEqual(logits.shape, (2, 5, 31))
        self.assertTrue(all(p.grad is not None for p in model.parameters()))
        model.eval().generate(x, 3, use_cache=True)
        self.assertEqual(keys, set(model.state_dict()))

    def test_invalid_cache_usage(self):
        model = tiny_model()
        x = torch.ones((2, 4), dtype=torch.long)
        with self.assertRaises(ValueError):
            model.generate(x, 1, use_cache=True)
        model.eval()
        with self.assertRaises(ValueError):
            model(x, use_cache=True)  # autograd enabled
        with torch.no_grad():
            _, _, cache = model(x, use_cache=True)
            for kwargs in [dict(past_key_values=cache),
                           dict(past_key_values=cache[:1], use_cache=True),
                           dict(past_key_values=((cache[0][0][:1], cache[0][1][:1]), cache[1]), use_cache=True),
                           dict(targets=x, use_cache=True), dict(targets=x, last_token_only=True)]:
                with self.assertRaises(ValueError):
                    model(x, **kwargs)
            with self.assertRaises(ValueError):
                model(torch.ones((2, 9), dtype=torch.long), past_key_values=cache, use_cache=True)
            with self.assertRaises(ValueError):
                model(x[:, :0], use_cache=True)


if __name__ == '__main__':
    unittest.main()
