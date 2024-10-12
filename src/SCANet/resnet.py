#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 12/27/2023 3:42 PM
# @Author  : Yaser
# @Site    : 
# @File    : resnet.py
# @Software: PyCharm
import torch.nn
from torchvision import models
from torch import nn
from torchvision.models import ResNet50_Weights
from torchvision.models._utils import IntermediateLayerGetter
import torch.nn.functional as F


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class ResnetFusion(nn.Module):
    def __init__(self, in_channel, name, pretrained, out_channel=None, train_resnet=False, return_interm_layers=False):
        super().__init__()
        if out_channel is None:
            out_channel = 512 if name in ('resnet18', 'resnet34') else 2048
        backbone = getattr(models, name)(weights=pretrained, norm_layer=FrozenBatchNorm2d)
        if not train_resnet:
            for name, parameter in backbone.named_parameters():
                parameter.requires_grad_(False)
        # 替换第一层
        # conv1 = backbone.conv1
        # fusion_conv = nn.Conv2d(in_channel, out_channels=conv1.out_channels, kernel_size=conv1.kernel_size,
        #                         padding=conv1.padding, stride=conv1.stride, bias=conv1.bias)
        # backbone.conv1 = fusion_conv
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {'layer4': "0"}
        # self.model = torch.nn.Sequential(*list(resnet_raw.children())[:-2])
        self.model = IntermediateLayerGetter(backbone, return_layers=return_layers)  # 去掉最后的两层
        self.out_channel = out_channel

    def forward(self, in_x):
        xs = self.model(in_x)
        out = {}
        for name, x in xs.items():
            out[name] = x
        return out


class Resnet(nn.Module):
    def __init__(self, name, pretrained, out_channel=None, train=True, train_layers=None):
        super().__init__()

        if out_channel is None:
            out_channel = 512 if name in ('resnet18', 'resnet34') else 2048
        backbone = getattr(models, name)(weights=pretrained, norm_layer=FrozenBatchNorm2d)

        train_all_flag = train_layers is None and train
        if not train_all_flag:
            for name, parameter in backbone.named_parameters():
                if train_layers is not None:
                    name_prefix = name.split(".")[0]
                    if name_prefix not in train_layers:
                        parameter.requires_grad_(False)
                else:
                    parameter.requires_grad_(False)

        self.model = IntermediateLayerGetter(backbone, return_layers={'layer4': "out"})  # 去掉最后的两层
        self.out_channel = out_channel

    def forward(self, in_x):
        xs = self.model(in_x)
        return xs["out"]


def build_resnet(name="resnet50", train=True, pretrained=ResNet50_Weights.DEFAULT, train_layers=None):
    return Resnet(name=name, pretrained=pretrained, train=train, train_layers=train_layers)


def build_resnet_fusion(in_channel, name="resnet50", train_resnet=False, return_interm_layers=False,
                        pretrained=ResNet50_Weights.DEFAULT):
    return ResnetFusion(in_channel=in_channel, name=name, pretrained=pretrained,
                        train_resnet=train_resnet,
                        return_interm_layers=return_interm_layers)


if __name__ == '__main__':
    device = torch.device("cuda:2")
    resnet = build_resnet(True, train_layers=['layer3', 'layer4']).to(device)
    x = torch.rand((1, 3, 512, 512)).to(device)
    out = resnet(x)
    print(out.shape)
