import unittest
import torch
from mingpt.measurement import ChangeAudit, parameter_counts
from test_lora import tiny_model


class TestMeasurement(unittest.TestCase):
    def test_audit_detects_frozen_mutation(self):
        model = tiny_model().enable_lora()
        audit = ChangeAudit(model)
        self.assertEqual(audit.measure(model)['changed_elements'], 0)
        name, parameter = next((n, p) for n, p in model.named_parameters() if not p.requires_grad)
        with torch.no_grad():
            parameter.flatten()[0] += 1
        with self.assertRaisesRegex(RuntimeError, name):
            audit.measure(model)

    def test_count_elements_not_tensors(self):
        model = tiny_model().enable_lora()
        counts = parameter_counts(model)
        self.assertEqual(counts['trainable'], 4 * 16 * 16)
        self.assertEqual(counts['parameter_bytes'], 4*counts['total'])
        audit = ChangeAudit(model)
        param = next(p for p in model.parameters() if p.requires_grad)
        with torch.no_grad():
            param.flatten()[:3] += 1
        changes = audit.measure(model)
        self.assertEqual(changes['changed_elements'], 3)
        self.assertEqual(changes['frozen_changed_elements'], 0)


if __name__ == '__main__':
    unittest.main()
