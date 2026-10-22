"""Parameter accounting and exact CPU-side change audits, reusable across experiments."""
import torch


def parameter_counts(model):
    params = list(model.parameters())
    total = sum(p.numel() for p in params)
    trainable = sum(p.numel() for p in params if p.requires_grad)
    return dict(total=total, trainable=trainable, frozen=total-trainable,
                trainable_percent=100*trainable/total,
                parameter_bytes=sum(p.numel()*p.element_size() for p in params),
                trainable_parameter_bytes=sum(p.numel()*p.element_size() for p in params if p.requires_grad),
                buffer_bytes=sum(p.numel()*p.element_size() for p in model.buffers()))


class ChangeAudit:
    def __init__(self, model):
        self.initial = {n: p.detach().cpu().clone() for n, p in model.named_parameters()}
        self.trainable = {n for n, p in model.named_parameters() if p.requires_grad}

    def measure(self, model):
        rows = {}
        for name, param in model.named_parameters():
            current = param.detach().cpu()
            initial = self.initial[name]
            delta = current - initial
            rows[name] = dict(trainable=name in self.trainable, elements=param.numel(),
                              changed_elements=int(torch.count_nonzero(current != initial)),
                              max_abs_delta=float(delta.abs().max()), l2_delta=float(delta.norm()))
            if name not in self.trainable:
                if rows[name]['changed_elements'] or param.grad is not None or param.requires_grad:
                    raise RuntimeError(f'Frozen parameter changed or acquired a gradient: {name}')
        changed = sum(r['changed_elements'] for r in rows.values())
        return dict(changed_elements=changed,
                    changed_percent=100*changed/sum(r['elements'] for r in rows.values()),
                    frozen_changed_elements=sum(r['changed_elements'] for r in rows.values() if not r['trainable']),
                    per_tensor=rows)


def cuda_memory(device):
    if torch.device(device).type != 'cuda':
        return None
    torch.cuda.synchronize(device)
    return dict(allocated_bytes=torch.cuda.memory_allocated(device),
                reserved_bytes=torch.cuda.memory_reserved(device),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
