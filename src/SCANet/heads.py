#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 12/27/2023 4:39 PM
# @Author  : Yaser
# @Site    :
# @File    : heads.py
# @Software: PyCharm

import torch
from matplotlib import pyplot as plt
from torch import nn, Tensor
import torch.nn.functional as F



class Head(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        raise NotImplementedError()


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN, from DETR)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        x = x["hs"]

        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class CHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, num_layers=1, use_all=False):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [out_dim])
        )
        self.use_all = use_all

    def forward(self, x):
        x = x["hs"] if self.use_all and self.training else x["hs"][-1]

        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)

        return x


class RTHead(Head):
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.FC = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [hidden_dim])
        )
        self.delta_R = nn.Linear(hidden_dim, 4)
        self.delta_T = nn.Linear(hidden_dim, 3)

    def forward(self, x):
        x = x["hs"]

        for i, layer in enumerate(self.FC):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)

        x_R = self.delta_R(x)
        x_T = self.delta_T(x)
        return x_R, x_T


class TransHeadV3(Head):
    def __init__(self, input_dim, hidden_dim, out_dim, ratio=2, use_all=False):
        super().__init__()
        self.hidden_mlp = nn.Linear(input_dim, hidden_dim)
        self.mlp_head_x = nn.Linear(hidden_dim, out_dim * ratio)
        self.mlp_head_y = nn.Linear(hidden_dim, out_dim * ratio)
        self.mlp_head_z = nn.Linear(hidden_dim, out_dim * ratio)
        self.use_all = use_all

    def forward(self, x):
        hs = x["hs"] if self.use_all and self.training else x["hs"][-1]
        x = F.relu(self.hidden_mlp(hs))
        mlp_x_out = self.mlp_head_x(x).unsqueeze(-2)
        mlp_y_out = self.mlp_head_y(x).unsqueeze(-2)
        mlp_z_out = self.mlp_head_z(x).unsqueeze(-2)
        xyz_trans_out = torch.cat([mlp_x_out, mlp_y_out, mlp_z_out], dim=-2)

        return xyz_trans_out


class RHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, num_layers=2, use_all=False):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [out_dim])
        )
        self.use_all = use_all

    def forward(self, x):
        x = x["hs"] if self.use_all and self.training else x["hs"][-1]

        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)

        return x


def show_feature_map(feature_map_list, title="", mean=True):
    for idx, feature_map in enumerate(feature_map_list):
        if mean:
            mean_feature_map = torch.mean(feature_map, dim=1).cpu().detach().numpy()
        else:
            mean_feature_map = feature_map.cpu().detach().numpy()
        plt.imshow(mean_feature_map)
        plt.title(f"{title}_{idx}")
        plt.show()


def _expand(tensor, length: int):
    return tensor.unsqueeze(1).repeat(1, int(length), 1, 1, 1).flatten(0, 1)


def build_head(head_config) -> Head:
    head_name = head_config["name"]
    model_type = head_config["model_type"]

    if model_type not in globals():
        raise f"The head: {head_name} which type is {model_type} has not found!"
    head_cls = globals()[model_type]
    opts = head_config["opts"]
    head: Head = head_cls(**opts)
    return head
