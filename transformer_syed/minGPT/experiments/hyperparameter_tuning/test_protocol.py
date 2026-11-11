"""Small correctness tests for the new chunked-prefill generation path."""
import unittest
import torch
from mingpt.model import GPT
from experiments.hyperparameter_tuning.run_experiment import generate, comparison, LORA_GRID, BASE_LORA


class ProtocolTests(unittest.TestCase):
    def test_chunked_cache_matches_uncached(self):
        torch.manual_seed(5)
        torch.set_num_threads(1)
        c = GPT.get_default_config()
        c.model_type = 'gpt-nano'
        c.vocab_size = 32
        c.block_size = 32
        model = GPT(c).eval()
        ids = torch.randint(32, (2, 13))
        reference = generate(model, ids, 5, False, 4)
        for chunk in (1, 4, 8, 13, 32):
            with self.subTest(chunk=chunk):
                self.assertTrue(torch.equal(reference, generate(model, ids, 5, True, chunk)))

    def test_generation_refuses_context_overflow(self):
        class Model:
            block_size = 8
        with self.assertRaises(ValueError):
            generate(Model(), torch.zeros((1, 7), dtype=torch.long), 2, True, 4)

    def test_control_in_grid_and_ratio_direction(self):
        self.assertIn(BASE_LORA, LORA_GRID)
        self.assertEqual(len(LORA_GRID), 8)
        base = dict(samples=[dict(wall_ms=x) for x in (10, 10, 10)])
        tuned = dict(samples=[dict(wall_ms=x) for x in (20, 20, 20)])
        result = comparison(base, tuned)
        self.assertEqual(result['speedup'], .5)
        self.assertEqual(result['bootstrap_95'], [.5, .5])


if __name__ == '__main__':
    unittest.main()
