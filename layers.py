import torch
from torch import nn
from typing import Callable


class DoubleLinear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features: int,
        activation: Callable,
        bias: bool = True,
    ):
        super().__init__()

        in_features_1, in_features_2 = in_features, in_features

        if type(in_features) == tuple:
            in_features_1, in_features_2 = *in_features

        self.linear_1 = nn.Linear(in_features_1, out_features, bias=bias)
        self.linear_2 = nn.Linear(in_features_2, out_features, bias=bias)

    def forward(self, x):
        return activation(self.linear_1(x) + self.linear_2(x))

class LSTMCell(nn.Module):
    def __init__(
            self,
            input_size:int,
            hidden_size:int,
            bias=True,
            activation:Callable = torch.tanh,
            dropout = 0.0):

        self.input_gate_layer = DoubleLinear(())
