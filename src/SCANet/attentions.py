#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 12/28/2023 5:34 PM
# @Author  : Yaser
# @Site    :
# @File    : convs.py
# @Software: PyCharm

# GCNet
# 论文名称：《GCNet: Non-local Networks Meet Squeeze-Excitation Networks and Beyond》
# 论文地址： https://arxiv.org/abs/1904.11492
# github地址：https://github.com/xvjiarui/GCNet

import torch
from torch import nn

from src.SCANet.weight_init import kaiming_init, constant_init


def last_zero_init(m):
    if isinstance(m, nn.Sequential):
        constant_init(m[-1], val=0)
        m[-1].inited = True
    else:
        constant_init(m, val=0)
        m.inited = True


class GCNet_Atten(torch.nn.Module):
    '''
    func:
    Parameter
    ---------
    in_dim: int
        输入的特征图的通道数
    reduction: int
        default 4. 通道权重融合过程中的通道减少倍数
    fusion_type: str
        ['add','mul']. 默认'add'
        输入的特征图和self-attention权重的结合方式
        'mul': 输出为权重与X相乘
        'add': 输出为权重 + X
    '''

    def __init__(self, in_dim,
                 reduction=4,
                 fusion_type='add'
                 ):

        self.in_dim = in_dim
        self.fusion_type = fusion_type

        self.out_dim = self.in_dim // reduction if self.in_dim >= reduction else 1

        super(GCNet_Atten, self).__init__()

        assert self.fusion_type in ['add', 'mul']

        self.conv_mask = torch.nn.Conv2d(self.in_dim, 1, kernel_size=1)
        self.softmax = torch.nn.Softmax(dim=2)

        self.channel_conv = torch.nn.Sequential(
            torch.nn.Conv2d(self.in_dim, self.out_dim, kernel_size=1),
            torch.nn.LayerNorm([self.out_dim, 1, 1]),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(self.out_dim, self.in_dim, kernel_size=1))
        self.reset_parameters()

    def reset_parameters(self):
        kaiming_init(self.conv_mask, mode='fan_in')
        self.conv_mask.inited = True
        last_zero_init(self.channel_conv)

    def forward(self, X):
        '''
        X: 4D-Tensor
        '''
        batch, channel, height, width = X.size()

        ##Query
        input_x = X
        input_x = input_x.view(batch, channel, height * width)
        input_x = input_x.unsqueeze(1)  # size = [B,1,C,H*W]

        ##Key
        context_mask = self.conv_mask(X)
        context_mask = context_mask.view(batch, 1, height * width)  # [B, 1, H * W]
        context_mask = self.softmax(context_mask)  # [B, 1, H * W]
        context_mask = context_mask.unsqueeze(-1)  # [B, 1, H*W, 1]

        ##weight
        context = torch.matmul(input_x, context_mask)  # [B,1,C,1]
        context = context.permute(0, 2, 1, 3)  # [B, C, 1, 1]

        context = self.channel_conv(context)  # [B, C, 1, 1]
        if self.fusion_type == 'add':
            y = X + context
        else:
            y = X * torch.sigmoid(context)

        return y


class ProjAttention(nn.Module):
    def __init__(self, in_channel, out_channel, atten_name="GCNet_Atten"):
        super().__init__()
        if atten_name not in globals():
            raise f"Attention: {atten_name} has not found!"
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=1)
        atten_cls = globals()[atten_name]
        self.attention = atten_cls(out_channel)

    def forward(self, x):
        return self.attention(self.conv1(x))

def get_attention(attention_name):
    if attention_name not in globals():
        raise f"Attention: {attention_name} has not found!"
    return globals()[attention_name]
# gcnet = GCNet_Atten(12, fusion_type='add')
# a = torch.randn((2, 12, 20, 20))
# out = gcnet(a)
# print(out.size())
