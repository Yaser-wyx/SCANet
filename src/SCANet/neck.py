#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 12/29/2023 12:09 PM
# @Author  : Yaser
# @Site    : 
# @File    : neck.py
# @Software: PyCharm
import torch
from torch import nn

from src.models.networks import ResidualBlock
from src.SCANet.attentions import ProjAttention, get_attention


class Neck(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        raise NotImplementedError()


class ClassicNeck(Neck):
    def __init__(self, in_channel, out_channel, stride=1, num_conv_layers=2, block_stack=2, atten_name=None):
        super().__init__()

        self.attention = get_attention(atten_name)(out_channel) if atten_name is not None else None

        self.blocks = [ResidualBlock(in_channel, out_channel, stride=stride, num_conv_layers=num_conv_layers)]
        self.blocks += [ResidualBlock(out_channel, out_channel, stride=stride, num_conv_layers=num_conv_layers)
                        for _ in range(block_stack - 1)]
        for idx, block in enumerate(self.blocks):
            self.add_module(f'block_{idx}', block)

        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        if isinstance(x, tuple):
            x = torch.cat(list(x), dim=1)
        assert x.shape[1] == self.in_channel
        out_list = [x]
        for idx, block in enumerate(self.blocks):
            x = block(x)
            out_list.append(x)

        if self.attention is not None:
            out_list.append(self.attention(x))

        return out_list


def build_neck(neck_config) -> Neck:
    neck_name = neck_config['name']
    if neck_name not in globals():
        raise f"Neck: {neck_name} has not found!"

    neck_cls = globals()[neck_name]
    opts = neck_config['opts']
    neck: Neck = neck_cls(**opts)
    return neck
