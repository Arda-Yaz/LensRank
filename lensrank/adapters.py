"""LoRA adapter layers."""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """Drop-in linear layer that keeps the original weight frozen."""

    def __init__(self, base: nn.Linear, rank: int=8, alpha: int=16):
        super().__init__()
        self.base = base
        self.scale = alpha / rank
        for p in self.base.parameters():
            p.requires_grad = False
        self.a = nn.Parameter(torch.empty(rank, base.in_features))
        self.b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.a, a=np.sqrt(5))

    @property
    def weight(self):
        """Expose the base weight for modules that inspect Linear metadata."""
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.a), self.b) * self.scale


def inject_lora(module: nn.Module, rank: int, alpha: int) -> int:
    """Adapt OpenCLIP transformer MLP projections.

    ``nn.MultiheadAttention`` consumes ``out_proj.weight`` directly through its
    functional implementation, so replacing ``out_proj`` with a wrapper breaks
    the module contract. OpenCLIP's ``c_fc``/``c_proj`` layers are invoked
    normally and are therefore safe LoRA targets.
    """
    changed = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and name.lower() in {'c_fc', 'c_proj', 'fc1', 'fc2'}:
            setattr(module, name, LoRALinear(child, rank, alpha))
            changed += 1
        else:
            changed += inject_lora(child, rank, alpha)
    return changed
