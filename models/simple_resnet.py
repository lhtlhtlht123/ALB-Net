import torch
import torch.nn as nn
import torch.nn.functional as F

from models.utils import InitWeights_Custom


# class BlockRes(nn.Module):
#     def __init__(self, ic, oc, k, use_skip_connection=True):
#         super(BlockRes, self).__init__()
#         self.conv_block = nn.Sequential(
#             nn.Conv2d(ic, oc, kernel_size=k, padding=(k-1)//2, bias=True),
#             nn.BatchNorm2d(oc),
#             nn.LeakyReLU(0.1, inplace=False),
#         )
#         self.skip_connection = nn.Conv2d(ic, oc, kernel_size=1, bias=False) if use_skip_connection and ic != oc else None
#
#     def forward(self, x):
#         identity = x
#         out = self.conv_block(x)
#         if self.skip_connection is not None:
#             identity = self.skip_connection(identity)
#         out += identity
#         return out
#
#
# class SimpleRes(nn.Module):
#     def __init__(self, param_list):
#         super(SimpleRes, self).__init__()
#         ch, layer, k = param_list
#         self.head = BlockRes(1, ch, k)
#
#         ch, layer, k = int(ch), int(layer), int(k)
#         layer_list = []
#         for _ in range(layer):
#             layer_list.append(BlockRes(ch, ch, k))
#         self.body = nn.Sequential(*layer_list)
#         self.tail = nn.Conv2d(ch, 1, kernel_size=1, padding=0, bias=True)
#         self.apply(InitWeights_Simple_Res)
#
#     def forward(self, x):
#         o = self.head(x)
#         o = self.body(o)
#         o = self.tail(o)
#         return o


class BlockRes(nn.Module):
    def __init__(self, ic, oc, k, use_skip_connection=True):
        super(BlockRes, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(ic, oc, kernel_size=k, padding=(k-1)//2, bias=True),
            nn.BatchNorm2d(oc),
            nn.Dropout(0.1)  # 添加 Dropout 层
        )
        self.skip_connection = nn.Conv2d(ic, oc, kernel_size=1, bias=False) if use_skip_connection and ic == oc else None
        self.relu = nn.LeakyReLU(0.1, inplace=False)


    def forward(self, x):
        identity = x
        out = self.conv_block(x)
        if self.skip_connection is not None:
            identity = self.skip_connection(identity)
        out += identity
        out = self.relu(out)
        return out

class SimpleRes(nn.Module):
    def __init__(self, param_list):
        super(SimpleRes, self).__init__()
        ch, layer, k = param_list
        self.head = BlockRes(1, ch, k)

        ch, layer, k = int(ch), int(layer), int(k)
        layer_list = []
        for _ in range(layer):
            layer_list.append(BlockRes(ch, ch, k))
        self.body = nn.Sequential(*layer_list)
        self.tail = nn.Conv2d(ch, 1, kernel_size=1, padding=0, bias=True)
        self.apply(InitWeights_Custom)

    def forward(self, x):
        o = self.head(x)
        o = self.body(o)
        o = self.tail(o)
        return o


# class BlockRes(nn.Module):
#     def __init__(self, ic, oc, k, use_skip_connection=True):
#         super(BlockRes, self).__init__()
#         self.conv_block = nn.Sequential(
#             nn.Conv2d(ic, oc, kernel_size=k, padding=(k-1)//2, bias=True),
#             nn.BatchNorm2d(oc),
#             nn.Dropout(0.1)  # 添加 Dropout 层
#         )
#         self.skip_connection = nn.Conv2d(ic, oc, kernel_size=1, bias=False) if use_skip_connection and ic == oc else None
#         self.relu = nn.LeakyReLU(0.1, inplace=False)
#
#     def forward(self, x):
#         identity = x
#         out = self.conv_block(x)
#         if self.skip_connection is not None:
#             identity = self.skip_connection(identity)
#         out += identity
#         out = self.relu(out)
#         return out
#
# class SimpleRes(nn.Module):
#     def __init__(self, param_list):
#         super(SimpleRes, self).__init__()
#         ch, layer, k = param_list
#         self.head = BlockRes(1, ch, k)
#
#         ch, layer, k = int(ch), int(layer), int(k)
#         self.layers = nn.ModuleList()
#         for _ in range(layer):
#             self.layers.append(BlockRes(ch, ch, k))
#         self.tail = nn.Conv2d(ch, 1, kernel_size=1, padding=0, bias=True)
#         self.apply(InitWeights_Custom)
#
#     def forward(self, x):
#         output = self.head(x)
#         for layer in self.layers:
#             residual = layer(output)
#             output = output + residual  # Add the output of the block to the previous output
#         output = self.tail(output)
#         return output