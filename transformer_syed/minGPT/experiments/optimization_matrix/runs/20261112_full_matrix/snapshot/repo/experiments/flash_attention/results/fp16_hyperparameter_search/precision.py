"""Explicit AMP policy; never describes master weights/reductions as pure FP16."""
from collections import Counter


def install(model, torch, precision):
    enabled = precision == 'fp16'
    expected = torch.float16 if enabled else torch.float32
    audit = dict(compute_dtype=str(expected), master_parameter_dtypes=dict(Counter(str(p.dtype) for p in model.parameters())),
                 policy='FP16 autocast; FP32 master parameters, gradients, optimizer states and stable reductions' if enabled else 'FP32, TF32 disabled',
                 projection_outputs={}, cache_dtype_checks=0, forward_checks=0)
    model._dtype_audit = audit
    # Check every Linear output on the first complete forward, including adapters.
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            def hook(module, inputs, output, name=name):
                if output.dtype != expected:
                    raise AssertionError(f'{name} computed {output.dtype}, expected {expected}')
                audit['projection_outputs'][name] = str(output.dtype)
            handles.append(module.register_forward_hook(hook))
    original = model.forward
    def forward(*args, **kwargs):
        with torch.autocast('cuda', dtype=torch.float16, enabled=enabled):
            result = original(*args, **kwargs)
        assert result[0].dtype == expected, f'LM logits dtype {result[0].dtype}'
        if kwargs.get('use_cache'):
            for k, v in result[2]:
                assert k.dtype == v.dtype == expected, 'KV cache precision mismatch'
            audit['cache_dtype_checks'] += 1
        audit['forward_checks'] += 1
        for handle in handles:
            handle.remove()
        handles.clear()
        return result
    model.forward = forward
