"""
For the source of this code check-out https://github.com/NingMa-AI/AS-MAML
"""

import torch
import torch.nn as nn


class StopControl(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super(StopControl, self).__init__()
        self.lstm = nn.LSTMCell(input_size=input_size, hidden_size=hidden_size)
        self.output_layer = nn.Linear(hidden_size, 1)
        self.output_layer.bias.data.fill_(0.0)
        self.h_0 = nn.Parameter(torch.randn((hidden_size, ), requires_grad=True))
        self.c_0 = nn.Parameter(torch.randn((hidden_size, ), requires_grad=True))

    def forward(self, inputs, hx) -> torch.Tensor:
        if hx is None:
            hx = (self.h_0.unsqueeze(0), self.c_0.unsqueeze(0))
        
        h, c = self.lstm(inputs, hx)
        return torch.sigmoid(self.output_layer(h).unsqueeze(0)), (h, c)