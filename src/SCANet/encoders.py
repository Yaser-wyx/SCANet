#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 12/27/2023 4:42 PM
# @Author  : Yaser
# @Site    : 
# @File    : encoders.py
# @Software: PyCharm
import torch
from torch import nn
from torch.nn import Flatten
from torchvision.models import ResNet34_Weights

from src.models.heatmap.lego_hg import AvgPool, simple_voxel_encoder
from src.models.networks import ResidualBlock3D
from src.SCANet.resnet import build_resnet


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        raise NotImplementedError()


torch.autograd.set_detect_anomaly(True)


class VoxelEncoderV5(Encoder):

    def __init__(self, num_features, freeze_mepnet_encoder=True):
        super().__init__()

        self.trans_rot_encoder = nn.Sequential(nn.Linear(6, 128),
                                               nn.ReLU(inplace=True),
                                               nn.Linear(128, 128))
        self.conv = ResidualBlock3D(in_channels=64, out_channels=128, stride=2, num_conv_layers=2)

        self.resnet = build_resnet(name="resnet34", pretrained=ResNet34_Weights.DEFAULT,
                                   train_layers=["layer1", "layer2", "layer3", "layer4"])
        self.mepnet_encoder = simple_voxel_encoder(64, no_out=True)  # 去掉avg与flatten
        self.freeze_mepnet_encoder = freeze_mepnet_encoder
        self.out_layer = nn.Sequential(AvgPool(), Flatten())
        self.project_layer = nn.Sequential(nn.Linear(512, 128),
                                           nn.ReLU(inplace=True),
                                           nn.Linear(128, 128))

        if freeze_mepnet_encoder:
            for param in self.mepnet_encoder.parameters():
                param.requires_grad = False
        # self.conv = ResidualBlock3D(in_channels=64, out_channels=num_features, stride=2, num_conv_layers=2)

    def forward(self, base_shape_tensor, trans_rot_tensor, pred_brick_imgs):
        '''
        :param occ: [B, 1, W, H, D]
        :param theta: [B, 3, 4]
        :return:
        '''

        if self.freeze_mepnet_encoder:
            with torch.no_grad():
                # 使用原论文的编码器权重，且不更新
                brick_out = self.mepnet_encoder(base_shape_tensor)
        else:
            brick_out = self.mepnet_encoder(base_shape_tensor)
        brick_out = self.out_layer(self.conv(brick_out))  # 将维度从64增加到128
        brick_position_embedding = self.trans_rot_encoder(trans_rot_tensor)
        brick_with_position = brick_out + brick_position_embedding
        brick_img_out = self.project_layer(self.out_layer(self.resnet(pred_brick_imgs)))  # 128
        brick_embedding = torch.cat([brick_with_position, brick_img_out], dim=-1)  # 256
        return brick_embedding


class PositionEncoder3D(nn.Module):
    def __init__(self, out_dim, in_dim=6):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(in_dim, out_dim),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(out_dim, out_dim))


def build_encoder(encoder_config) -> Encoder:
    encoder_name = encoder_config['name']
    if encoder_name not in globals():
        raise f"Encoder: {encoder_name} has not found!"
    encoder_cls = globals()[encoder_name]
    opts = encoder_config['opts']
    encoder: Encoder = encoder_cls(**opts)
    return encoder
