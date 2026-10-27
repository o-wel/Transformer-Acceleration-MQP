"""Attention backend parity, offset cache masks, gradients, and checkpoint compatibility."""
import copy
import unittest
import torch
from mingpt.model import GPT, LoRALinear


def tiny():
    c = GPT.get_default_config()
    c.model_type = None
    c.n_layer, c.n_head, c.n_embd = 2, 2, 16
    c.vocab_size, c.block_size = 31, 16
    c.embd_pdrop = c.resid_pdrop = c.attn_pdrop = 0
    return GPT(c)


class TestAttention(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(1)

    def check_backend(self, backend, device='cpu', dtype=torch.float32):
        for lora in (False, True):
            eager = tiny().to(device=device, dtype=dtype)
            if lora:
                eager.enable_lora(2, 4)
                for m in eager.modules():
                    if isinstance(m, LoRALinear):
                        torch.nn.init.normal_(m.B.weight, std=.05)
            other = copy.deepcopy(eager).set_attention_backend(backend)
            self.assertEqual(set(eager.state_dict()), set(other.state_dict()))
            x = torch.randint(31, (2, 12), device=device)
            tol = 2e-2 if dtype != torch.float32 else 2e-5
            for model in (eager, other):
                model(x, x)[1].backward()
            for (n, p), (_, q) in zip(eager.named_parameters(), other.named_parameters()):
                if p.requires_grad:
                    torch.testing.assert_close(p.grad, q.grad, atol=tol, rtol=tol, msg=n)
            eager.eval()
            other.eval()
            with torch.no_grad():
                reference = eager(x)[0]
                cache, pieces, start = None, [], 0
                # Covers prefill, one-token decoding, and multi-token offset masks.
                for length in (4, 1, 3, 4):
                    logits, _, cache = other(x[:, start:start+length], past_key_values=cache, use_cache=True)
                    pieces.append(logits)
                    start += length
                torch.testing.assert_close(torch.cat(pieces, 1), reference, atol=tol, rtol=tol)
                # Both generators must agree when learned positions reset at overflow.
                torch.testing.assert_close(other.generate(x, 8, use_cache=True),
                                           other.generate(x, 8, use_cache=False), atol=0, rtol=0)

    def test_math(self):
        self.check_backend('math')

    def test_automatic(self):
        self.check_backend('sdpa')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_flash(self):
        self.check_backend('flash', 'cuda', torch.float16)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_efficient_fp32(self):
        torch.backends.cuda.matmul.allow_tf32 = False
        self.check_backend('efficient', 'cuda', torch.float32)

    def test_explicit_errors(self):
        model = tiny()
        with self.assertRaises(ValueError):
            model.set_attention_backend('invalid')
        model.set_attention_backend('flash')
        with self.assertRaisesRegex(RuntimeError, 'requires CUDA'):
            model(torch.ones(1, 2, dtype=torch.long))


if __name__ == '__main__':
    unittest.main()
