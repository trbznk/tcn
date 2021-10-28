import time
import math

from numpy.core.numeric import identity
import torch.nn as nn
import torch.optim as optim


def minimum_required_blocks(window, dilation_base, kernel_size):
    return math.ceil(math.log((((window-1)*(dilation_base-1))/((kernel_size-1)*2))+1, dilation_base))


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, dropout=0, is_last_block=False):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = self.dilation*(self.kernel_size-1)
        self.dropout = dropout
        self.is_last_block = is_last_block

        _hidden_channels = self.in_channels if self.is_last_block else self.out_channels
        
        if self.in_channels != self.out_channels:
            self.res = nn.Conv1d(self.in_channels, self.out_channels, 1)
        else:
            self.res = None
                 
        layers = [
            nn.ConstantPad1d((self.padding, 0), 0),
            nn.Conv1d(self.in_channels, _hidden_channels, self.kernel_size, dilation=self.dilation),
            nn.BatchNorm1d(_hidden_channels),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.ConstantPad1d((self.padding, 0), 0),
            nn.Conv1d(_hidden_channels, self.out_channels, self.kernel_size, dilation=self.dilation),
            nn.BatchNorm1d(self.out_channels)
        ]

        if not self.is_last_block:
            layers += [
                nn.ReLU(),
                nn.Dropout(p=self.dropout)
            ]

        self.m = nn.Sequential(*layers)

        self.init_weights()
        
    def init_weights(self):
        self.m[1].weight.data.normal_(0, 0.01)
        self.m[6].weight.data.normal_(0, 0.01)
        if self.res is not None:
            self.res.weight.data.normal_(0, 0.01)
        
    def forward(self, x):
        identity = self.res(x) if self.res is not None else x
        out = self.m(x)
        
        return out+identity
    
    
class TCN(nn.Module):
    def __init__(
            self,
            window,
            horizon,
            features,
            kernel_size=3,
            dilation_base=2,
            dropout=0,
            additional_blocks=0,
            hidden_channels_factor=1
        ):
        super().__init__()
        self.window = window
        self.horizon = horizon
        self.features = features
        self.kernel_size = kernel_size
        self.dilation_base = dilation_base
        self.dropout = dropout
        self.hidden_channels_factor = hidden_channels_factor
        self.n_layers = minimum_required_blocks(self.window, self.dilation_base, self.kernel_size)
        self.m = nn.Sequential()
        
        for i in range(self.n_layers):
            is_last_block = False
            if i == 0:
                in_channels, out_channels = self.features, self.hidden_channels
            elif i == self.n_layers-1:
                in_channels, out_channels = self.hidden_channels, self.features
                is_last_block = True
            else:
                in_channels, out_channels = self.hidden_channels, self.hidden_channels

            block = ResidualBlock(
                in_channels,
                out_channels,
                self.kernel_size,
                dilation=self.dilation_base**i,
                dropout=self.dropout,
                is_last_block=is_last_block
            )
            self.m.add_module(str(i), block)
    
    @property
    def hidden_channels(self):
        return self.features*self.horizon*self.hidden_channels_factor
    
    @property
    def amount_parameters(self):
        return sum(p.numel() for p in self.parameters())
    
    def forward(self, x):
        x = self.m(x)
        x = x.squeeze(1)
        return x[:, -self.horizon:]
