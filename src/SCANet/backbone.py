#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 12/27/2023 4:38 PM
# @Author  : Yaser
# @Site    :
# @File    : fusion_net.py
# @Software: PyCharm

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from src.models.heatmap.lego_hg import VoxelTransformEncoder, simple_voxel_encoder
from src.models.heatmap.models.networks.hourglass import get_hourglass_net


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.voxel_proj_encoder = None
        self.GT_fusion_backbone = None
        self.pred_fusion_backbone = None
        self.error_backbone = None
        self.out_channel = 0
        self.freeze = False

    def forward(self):
        raise NotImplementedError()

    def freeze_parameters(self):
        self.freeze = True
        for p in self.parameters():
            p.requires_grad = False


class Hourglass(nn.Module):
    def __init__(
            self,
            occ_out_channels,
            occ_fmap_size,
            img_size=512,
            brick_emb=64,
            num_stacks=2,
            num_brick=5,
    ):
        super().__init__()
        # Backbone
        self.img_size = img_size
        # 体素编码器，输入为voxel representation，将3D representation转化为2D representation，为原论文中的 fig3(a){i}
        self.occ_encoder = VoxelTransformEncoder(
            occ_out_channels,
            feature_map2d_size=(occ_fmap_size,) * 2,
            use_coordconv=False,
        )

        self.brick_encoder = simple_voxel_encoder(brick_emb) if brick_emb > 0 else None

        # 论文fig3(a){ii}的编解码器
        self.num_bricks_single_forward = num_brick

        self.hg = get_hourglass_net(
            heads=None,
            img_nc=3 + occ_out_channels,
            num_stacks=num_stacks,
            cond_emb_dim=brick_emb * self.num_bricks_single_forward,
        )

    def forward(self, images, shape_tensor, transform_options_tensor, brick_occs=None):
        new_transform_options_list = []

        for transform_id in range(4):
            # 4个transform属性：azims, elevs, obj_scales, obj_centers
            new_transform_options_list.append(
                torch.stack(
                    [
                        transform_options_tensor[batch_id][transform_id]
                        for batch_id in range(len(transform_options_tensor))
                    ]
                )
            )
        # occ_time = time()
        occ_f_map = self.occ_encoder(
            shape_tensor,
            new_transform_options_list[0],
            new_transform_options_list[1],
            new_transform_options_list[2],
            new_transform_options_list[3],
        )  # 对shape进行编码
        # master_only_print("occ time:", time()-occ_time)
        occ_f_map = F.interpolate(
            occ_f_map, size=self.img_size, mode="nearest"
        )  # 将2D feature map进行上采样
        img_concat = torch.cat([images, occ_f_map], dim=1)  # 拼接图像与特征图

        def build_batch_from_list(l):
            """
            将列表中的张量进行批次化处理，假设列表中的每个张量的形状为[brick_num, ...]，并且列表长度为batch_size
            返回一个形状为[batch_size, self.num_bricks_single_forward, ...]的张量
            """
            t = pad_sequence(l, batch_first=True)
            # 将第二维度填充到self.num_bricks_single_forward
            t_padded = F.pad(
                t,
                (0,) * (2 * (len(t.shape) - 2))
                + (0, self.num_bricks_single_forward - t.shape[1]),
            )
            return t_padded

        if brick_occs is not None and self.brick_encoder is not None:
            n_bricks = [len(b) for b in brick_occs]
            brick_occs = [item for sublist in brick_occs for item in sublist]
            brick_occs_concat = torch.cat(brick_occs, dim=0)
            if len(brick_occs_concat.shape) == 4:
                brick_occs_concat.unsqueeze_(1)

            f_brick_concat = self.brick_encoder(
                brick_occs_concat
            )  # 对需要的部件进行编码
            brick_ct = 0
            f_brick_list = []
            for n_brick in n_bricks:
                f_brick_list.append(f_brick_concat[brick_ct: brick_ct + n_brick])
                brick_ct += n_brick
            f_brick_padded = build_batch_from_list(
                f_brick_list
            )  # 对部件的编码进行padding补0
            outputs = self.hg(
                img_concat, f_brick_padded.reshape(f_brick_padded.shape[0], -1)
            )  # 使用Hourglass网络进行预测
        else:
            outputs = self.hg(img_concat)
        return outputs[-1]


class TwinNet(Backbone):
    def __init__(self, hourglass_opts):
        super().__init__()
        self.hourglass = Hourglass(**hourglass_opts)

    def forward(self, x):
        if self.freeze:
            with torch.no_grad():
                return self._forward(x)
        else:
            return self._forward(x)

    def _forward(self, x):
        assert isinstance(x, dict)
        manual_tensor, base_shape_tensor, pred_tensor, pred_shape_tensor = (
            x["manual_tensor"],
            x["base_shape_tensor"],
            x["pred_tensor"],
            x["pred_shape_tensor"],
        )
        transform_options_tensor = x["transform_options_tensor"]
        brick_tensor = x["brick_voxel_raw"]
        manual_num = len(manual_tensor)
        img_cat_tensor = torch.cat([manual_tensor, pred_tensor])
        shape_cat_tensor = torch.cat([base_shape_tensor, pred_shape_tensor])
        transform_options_tensor = transform_options_tensor + transform_options_tensor
        brick_tensor = brick_tensor + brick_tensor
        total_out = self.hourglass(
            img_cat_tensor, shape_cat_tensor, transform_options_tensor, brick_tensor
        )
        manual_out = total_out[:manual_num].contiguous()
        pred_out = total_out[manual_num:].contiguous()

        return manual_out, pred_out


def build_backbone(backbone_config) -> Backbone:
    backbone_name = backbone_config["name"]
    if backbone_name not in globals():
        raise f"Backbone: {backbone_name} has not found!"

    backbone_cls = globals()[backbone_name]
    opts = backbone_config["opts"]
    backbone: Backbone = backbone_cls(**opts)
    if "freeze" in backbone_config and backbone_config["freeze"]:
        backbone.freeze_parameters()
    return backbone
