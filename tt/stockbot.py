import torch
from torch import nn
from typing import Callable, Tuple
import numpy as np

from functools import reduce

from hyperparameters import *

class DoubleLinear(nn.Module):
    def __init__ (
            self,
            in_features:Tuple[int],
            out_features:int,

            bias:bool=False,
            activation:Callable=nn.Sigmoid()
            ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.activation = activation

        self.ffn = [ nn.Linear(
                in_features=f_i,
                out_features=out_features,
                bias=bias
                ) for f_i in in_features]

    def forward(self, all_x):
        x = [f(x_i) for f,x_i in zip(self.ffn,all_x)]
        x = reduce(lambda y,x: y+x, x)
        x = self.activation(x)
        return x

class LSTM(nn.Module):
    def __init__ (
            self,
            input_size:int,
            hidden_size:int,
            num_layers:int,
            
            bias:bool=False,
            ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias

        self.h = None
        self.c = None

        self.lstm_cell = LSTMCell(
                input_size=input_size,
                hidden_size=hidden_size,
                bias=bias)

    def forward (
            self,
            x, h0=None, c0=None
            ):
        self.h, self.c = h0, c0
        if self.h == None: self.h = torch.zeros(x.shape)
        if self.c == None: self.c = torch.zeros(x.shape)

        for _ in range(self.num_layers):
            x,(self.h, self.c) = self.lstm_cell(x, self.h, self.c)

        return x, (self.h, self.c)


"""
f = sigmoid(W x + U h + b)
in_gate = sigmoid(W x + U h + b)
out_gate = sigmoid(W x + U h + b)
cell_gate = tanh(W x + U h + b)
c_t = f * c_{t-1} + i_t * cell_gate
h_t = o * tanh(c_t)
"""
class LSTMCell(nn.Module):
    def __init__ (
            self,
            input_size:int,
            hidden_size:int,

            bias:bool=False
            ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        self.h = None
        self.c = None

        self.forget_gate_layer = DoubleLinear(
                in_features=(input_size, hidden_size),
                out_features=hidden_size,
                bias=bias)
        self.input_gate_layer = DoubleLinear(
                in_features=(input_size, hidden_size),
                out_features=hidden_size,
                bias=bias)
        self.output_gate_layer = DoubleLinear(
                in_features=(input_size, hidden_size),
                out_features=hidden_size,
                bias=bias)
        self.cell_gate_layer = DoubleLinear(
                in_features=(input_size, hidden_size),
                out_features=hidden_size,
                bias=bias)

        self.tanh = nn.Tanh()


    def forward(self, x, h0=None, c0=None):
        self.h, self.c = h0, c0
        if self.h == None: self.h = torch.zeros(x.shape)
        if self.c == None: self.c = torch.zeros(x.shape)

        forget_gate = self.forget_gate_layer((x, self.h))
        input_gate = self.input_gate_layer((x, self.h))
        output_gate = self.output_gate_layer((x, self.h))
        cell_gate = self.cell_gate_layer((x, self.h))

        self.c = forget_gate * self.c + input_gate * cell_gate
        self.h = output_gate * self.tanh(self.c)

        return self.h,(self.h,self.c)


class StockBot(nn.Module):
    def __init__(
        self,
        past_history: int = PAST_HISTORY,
        forward_look: int = FORWARD_LOOK,
        stack_depth: int = STACK_DEPTH,
        layer_units: int = LAYER_UNITS,

        bias:bool = BIAS,
        hidden_size:int = HIDDEN_SIZE,
        dropout:int = DROPOUT
    ):
        super().__init__()

        self.past_history = past_history
        self.forward_look = forward_look
        self.stack_depth = stack_depth
        self.layer_units = layer_units

        self.layer_norm = nn.LayerNorm((past_history, past_history))
        self.lstm = LSTM(
                input_size=past_history,
                hidden_size=past_history,
                num_layers=stack_depth,
                bias=bias)
        self.ffn1 = nn.Linear(
                in_features=past_history,
                out_features=1,
                bias=bias
                )
        self.ffn2 = nn.Linear(
                in_features=past_history,
                out_features=1,
                bias=bias
                )


    def forward(self, x):
        x = self.layer_norm(x)
        x, (h_n, c_n) = self.lstm(x)
        x = self.ffn1(x)
        x = torch.squeeze(x)
        x = self.ffn2(x)
        x = torch.squeeze(x)

        return x
        
if __name__ == "__main__":
    sb = StockBot()
    print(sb)
