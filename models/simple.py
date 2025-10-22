import torch
import torch.nn as nn
from .utils import InitWeights_He


class Block(nn.Module):
    def __init__(self, ic, oc, k):
        super(Block, self).__init__()
        self.conv_block = nn.Sequential(
                nn.Conv2d(ic, oc, kernel_size = k, padding = (k-1)//2, bias=True),
                nn.BatchNorm2d(oc),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Dropout2d(0.1),
            )

    def forward(self, x):
        return self.conv_block(x)


class Simple(nn.Module):
    def __init__(self,  param_list):
        super(Simple, self).__init__()
        ch, layer, k = param_list
        self.head = Block(1, ch, k)
        
        ch, layer, k = int(ch), int(layer), int(k)
        layer_list = []
        for _ in range(layer):
            layer_list.append(Block(ch, ch, k))
        self.body = nn.Sequential(*layer_list)
        self.tail = nn.Conv2d(ch, 1, kernel_size = 1, padding=0, bias=True)
        self.apply(InitWeights_He)
        

    def forward(self, x):
        o = self.head(x)
        o = self.body(o)
        o = self.tail(o)

        return o
