""" For the source of this code check-outhttps://github.com/NingMa-AI/AS-MAML/blob/master/models/layersFw.py """

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearModel(nn.Linear):
    """A Simple Linear model implementation for fast weights"""
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__(in_features, out_features, bias=True)
        self.weight.fast = None
        self.bias.fast = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.weight.fast is not None and self.bias.fast is not None:
            return F.linear(x, self.weight.fast, self.bias.fast)
        return super().forward(x)