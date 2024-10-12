#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 12/27/2023 4:23 PM
# @Author  : Yaser
# @Site    :
# @File    : SCANet.py
# @Software: PyCharm
import collections
import os
from itertools import repeat
from time import time
from typing import Any, Optional

import math
import numpy as np
import torch

from src.tu.ddp import (master_only_print, )
from debug.utils import StateFlagNew
from .backbone import build_backbone, Backbone
from .encoders import build_encoder, Encoder, VoxelEncoderV5
from .heads import build_head, Head
from .loss import build_loss
from .neck import Neck, build_neck, ClassicNeck
from .position_encoding import PositionEmbeddingSine
from .transformer import Transformer, build_transformer
import lightning as L
from src.util.common_utils import create_not_exist, get_filename

FLIPPED_ROTATION_ID_MAP = {
    0: [0, 0, 0],
    1: [0, 90, 0],
    2: [180, 0, 180],
    3: [0, -90, 0],
    4: [180, 90, 180]

}


def get_correction_acc(pred_list, gt):
    # 获得纠正率：将错误的纠正对的占所有错误的
    brick_num = 0
    total_correction_cnt = 0  # 进行纠错，并确实纠正对了，有可能出现多纠正了
    position_correct_cnt = 0
    rotation_correct_cnt = 0
    position_incorrect_cnt = 0
    rotation_incorrect_cnt = 0
    total_incorrect_cnt = 0  # 进行纠错，但纠正错了
    already_correct_cnt = 0
    correct2correct = 0
    correct2incorrect = 0
    incorrect2correct = 0
    incorrect2incorrect = 0
    incorrect_cnt = 0
    for batch_idx in range(len(pred_list)):
        pred_item = pred_list[batch_idx]

        gt_position = gt['gt_trans'][batch_idx]
        gt_rotation = gt['gt_rotation_state'][batch_idx]
        gt_brick_state = gt['brick_state'][batch_idx]

        for brick_idx in range(len(gt_brick_state)):
            brick_num += 1
            gt_rotation_idx = FLIPPED_ROTATION_ID_MAP[gt_rotation[brick_idx].item()]
            gt_position_idx = gt_position[brick_idx].cpu().numpy()
            correction_position_idx = pred_item[brick_idx]['position']
            correction_rotation_idx = pred_item[brick_idx]['rotation']
            if correction_position_idx is None:
                correction_position_idx = gt_position_idx
            if correction_rotation_idx is None:
                correction_rotation_idx = gt_rotation_idx
            rotation_correct = np.all(gt_rotation_idx == correction_rotation_idx)
            position_correct = np.allclose(gt_position_idx, correction_position_idx)
            if gt_brick_state[brick_idx].item() == StateFlagNew.CORRECT.value:
                already_correct_cnt += 1
                if rotation_correct and position_correct:
                    correct2correct += 1
                else:
                    correct2incorrect += 1
            else:
                incorrect_cnt += 1
                if rotation_correct and position_correct:
                    incorrect2correct += 1
                else:
                    incorrect2incorrect += 1

            if position_correct and rotation_correct:
                total_correction_cnt += 1
                rotation_correct_cnt += 1
                position_correct_cnt += 1
            elif position_correct and not rotation_correct:
                position_correct_cnt += 1
                rotation_incorrect_cnt += 1
                total_incorrect_cnt += 1
            elif not position_correct and rotation_correct:
                position_incorrect_cnt += 1
                rotation_correct_cnt += 1
                total_incorrect_cnt += 1
            else:
                rotation_incorrect_cnt += 1
                position_incorrect_cnt += 1
                total_incorrect_cnt += 1

    correction_result = {
        "total_correction": total_correction_cnt / brick_num,
        "rotation_correction": rotation_correct_cnt / brick_num,
        "position_correction": position_correct_cnt / brick_num,
        "total_incorrect": total_incorrect_cnt / brick_num,
        "rotation_incorrect": rotation_incorrect_cnt / brick_num,
        "position_incorrect": position_incorrect_cnt / brick_num,
        "already_correct": already_correct_cnt / brick_num,
        "correct2incorrect": correct2incorrect / already_correct_cnt if already_correct_cnt != 0 else 0,
        "correct2correct": correct2correct / already_correct_cnt if already_correct_cnt != 0 else 0,
        "incorrect": incorrect_cnt / brick_num,
        "incorrect2incorrect": incorrect2incorrect / incorrect_cnt if incorrect_cnt != 0 else 0,
        "incorrect2correct": incorrect2correct / incorrect_cnt if incorrect_cnt != 0 else 0
    }
    return correction_result


def decode_pred_out(heads_out, padding_len_list):
    # 根据CHead的结果来判断使用哪个head
    c_out = heads_out['CHead']
    c_confidence = torch.softmax(c_out, dim=-1)
    c_pred = torch.argmax(c_out, dim=-1)
    position_pred = torch.argmax(heads_out['TransHead'], dim=-1)
    rotation_pred = torch.argmax(heads_out['RHead'], dim=-1)
    voxel_translation = torch.as_tensor([65, 65 // 4, 65], device=position_pred.device, dtype=torch.int32)
    brick_num_max = c_out.shape[1]
    pred_list = []
    for batch_idx in range(c_out.shape[0]):
        one_batch_pred = []
        for brick_idx in range(c_out.shape[1]):
            if brick_idx + 1 > brick_num_max - padding_len_list[batch_idx]:
                continue
            position = position_pred[batch_idx, brick_idx] - voxel_translation
            position = position.to(torch.float32)
            position *= 0.5
            position = position.detach().cpu().numpy()
            rotation = FLIPPED_ROTATION_ID_MAP[rotation_pred[batch_idx, brick_idx].item()]
            c_confidence_one = c_confidence[batch_idx, batch_idx, c_pred[batch_idx, brick_idx]]
            one_batch_pred.append({
                'position': position,
                'rotation': rotation,
                'confidence': c_confidence_one,
                'brick_status': c_pred[batch_idx, brick_idx]
            })
        pred_list.append(one_batch_pred)

    return pred_list


class SCANet(L.LightningModule):
    def __init__(
            self,
            backbone,
            neck,
            transformer,
            heads,
            num_component_queries,
            brick_encoder,
            position_embedding,
            hidden_dim,
            loss,
            train_cfg=None,
    ):
        super().__init__()

        self.backbone: Backbone = backbone
        self.neck: Neck = neck
        self.transformer: Transformer = transformer
        self.heads = heads
        self.loss = loss
        self.train_cfg = train_cfg
        self.acc_dict = collections.defaultdict(list)
        if heads is not None:
            for head_name, head in heads.items():
                self.add_module(head_name, head)
        self.brick_encoder: Encoder = brick_encoder

        self.num_component_queries = num_component_queries
        self.hidden_dim = hidden_dim
        self.vis = False
        self.time_static = collections.defaultdict(list)
        self.position_embedding = position_embedding
        if train_cfg is not None:
            self.visualize_dir = self.train_cfg["out_dir"] + "/visualize"
        else:
            self.visualize_dir = "./visualize_all_version2"

        create_not_exist(self.visualize_dir)

    def _forward_brick_encoder_v2_1(self, x, brick_batch_size=25):
        # Extract raw voxel data, predicted brick images, and predicted transformations/rotations from input x

        one_batch_brick_voxel_tensor = x["brick_voxel_raw2"]
        pred_brick_imgs = x["pred_brick_imgs"]
        pred_trans_rotation = x["pred_trans_rotation"]
        # Initialize lists to store brick data, number of bricks per step, and prediction data

        all_brick_list = []
        num_brick_per_step = []
        all_pred_trans_rotation = []
        all_pred_brick_img_list = []
        # Loop through each batch and accumulate brick voxel data and corresponding predictions

        for batch_idx, brick_voxel_list in enumerate(one_batch_brick_voxel_tensor):
            all_brick_list += brick_voxel_list
            all_pred_trans_rotation += pred_trans_rotation[batch_idx]
            all_pred_brick_img_list += pred_brick_imgs[batch_idx]
            num_brick_per_step.append(len(brick_voxel_list))

        # If brick_batch_size is "all", encode all bricks at once
        if brick_batch_size == "all":
            all_brick_tensor = torch.stack(all_brick_list)
            all_pred_trans_rotation = torch.stack(all_pred_trans_rotation)
            all_pred_brick_img_tensor = torch.stack(all_pred_brick_img_list)
            brick_embedding_cat = self.brick_encoder(all_brick_tensor,
                                                     all_pred_trans_rotation,
                                                     all_pred_brick_img_tensor)

        else:
            assert isinstance(brick_batch_size, int)
            # Break bricks into smaller batches to avoid memory issues during inference
            brick_batch_num = math.ceil(len(all_brick_list) / brick_batch_size)
            brick_embedding_list = []
            for brick_batch_id in range(brick_batch_num):
                start_idx = brick_batch_id * brick_batch_size
                end_idx = start_idx + brick_batch_size
                if end_idx > len(all_brick_list):
                    end_idx = len(all_brick_list)
                # Extract brick data for the current batch
                sub_brick_list = all_brick_list[start_idx:end_idx]
                sub_pred_trans_rot_list = all_pred_trans_rotation[start_idx:end_idx]
                sub_pred_brick_img_list = all_pred_brick_img_list[start_idx:end_idx]
                # Stack and pass through the encoder
                sub_brick_tensor = torch.stack(sub_brick_list)
                sub_pred_trans_rot_list = torch.stack(sub_pred_trans_rot_list)
                sub_pred_brick_img_list = torch.stack(sub_pred_brick_img_list)
                brick_embedding = self.brick_encoder(sub_brick_tensor, sub_pred_trans_rot_list,
                                                     sub_pred_brick_img_list)  # 一次性将所有组件进行编码
                brick_embedding_list.append(brick_embedding)
            # Concatenate all encoded brick batches
            brick_embedding_cat = torch.cat(brick_embedding_list, dim=0)
        # Restore the original sequence of bricks and apply padding where necessary
        max_len = max(num_brick_per_step) # Find the maximum number of bricks in a single step
        one_batch_brick_embedding = []
        # padding_time = time()
        padding_len_list = []
        current_sum = 0
        # Loop through each brick step and apply padding to match the max length
        for this_len in num_brick_per_step:
            this_brick = brick_embedding_cat[current_sum: current_sum + this_len]
            current_sum += this_len
            if this_len == max_len:
                this_brick_final = torch.cat([this_brick])
                padding_len_list.append(0)
            else:
                padding_len = max_len - this_len
                padding_len_list.append(padding_len)
                padding_tensor = (
                    torch.zeros((padding_len, self.hidden_dim), device=self.device, dtype=torch.float32)
                )
                this_brick_final = torch.cat([this_brick, padding_tensor])
            one_batch_brick_embedding.append(this_brick_final)
        # Stack the embeddings for all bricks and return the result along with padding lengths
        brick_query_embedding = torch.stack(one_batch_brick_embedding)
        return brick_query_embedding, padding_len_list

    def configure_optimizers(self):
        assert self.train_cfg is not None
        param_dicts = [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if "backbone" not in n and p.requires_grad
                ]
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if "backbone" in n and p.requires_grad
                ],
                "lr": float(self.train_cfg["backbone_lr"]),
            },
        ]
        optimizer = torch.optim.AdamW(
            param_dicts,
            lr=float(self.train_cfg["lr"]),
            weight_decay=float(self.train_cfg["weight_decay"]),
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, self.train_cfg["epoch_nums"]
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch"
            },
        }

    def print_time(self):
        for k, v in self.time_static.items():
            if k == "other_time" or len(v) == 0:
                continue
            master_only_print(f"{k}: {v[-1]}")

    def _forward(self, batch, _, need_loss=True):
        x, y = batch
        backbone_out = self.backbone(x)
        neck_out = self.neck(backbone_out)
        # self.show_feature_map(neck_out[-1], x)
        if isinstance(self.neck, ClassicNeck):
            neck_out_last = neck_out[-1]
        else:
            neck_out_last = neck_out
        position_embedding = self.position_embedding(neck_out_last)
        brick_query_embedding, padding_len_list = self._forward_brick_encoder_v2_1(x)
        hs, memory = self.transformer(
            neck_out_last,
            pos_embed=position_embedding,
            query_embed=brick_query_embedding,
        )

        head_input = {
            "hs": hs,
            "memory": memory,
            "neck_out": neck_out,
            "backbone_out": backbone_out,
        }
        head_out = {}

        for head_name, head in self.heads.items():
            head_out[head_name] = head(head_input)

        if need_loss and y is not None:
            loss_all = self.loss(head_out, y, padding_len_list)

            return loss_all, head_out, padding_len_list
        else:
            return head_out, padding_len_list

    def training_step(self, batch, batch_idx):

        loss_all, head_out, _ = self._forward(batch, batch_idx)
        loss_sum, loss_states = loss_all
        for loss_name, loss_value in loss_states.items():
            loss_name = "train_" + loss_name
            self.log(
                loss_name,
                loss_value,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                batch_size=len(batch),
                sync_dist=True,
            )

        return loss_sum

    def validation_step(self, batch, batch_idx):
        loss_all, head_out, _ = self._forward(batch, batch_idx)
        loss_sum, loss_states = loss_all
        for loss_name, loss_value in loss_states.items():
            loss_name = "val_" + loss_name
            self.log(
                loss_name,
                loss_value,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                batch_size=len(batch),
                sync_dist=True,
            )

        return loss_sum

    def test_step(self, batch, batch_idx):
        loss_all, head_out, _ = self._forward(batch, batch_idx)
        loss_sum, loss_states = loss_all
        for loss_name, loss_value in loss_states.items():
            loss_name = "test_" + loss_name
            self.log(
                loss_name,
                loss_value,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                batch_size=len(batch),
            )

        return loss_sum

    def forward(self, batch, batch_idx=0):
        # only for pred!
        head_out, padding_len_list = self._forward(batch, batch_idx)
        decoded_out = decode_pred_out(head_out, padding_len_list)
        return decoded_out

    def predict_step(self, batch, batch_idx):
        x, y = batch
        loss_all, head_out, padding_len_list = self._forward(batch, batch_idx)
        loss_sum, loss_states = loss_all
        for loss_name, loss_value in loss_states.items():
            if loss_name.endswith("acc"):
                self.acc_dict[loss_name].append(loss_value)
        decoded_out = decode_pred_out(head_out, padding_len_list)
        correction_res = get_correction_acc(decoded_out, y)
        for k, v in correction_res.items():
            self.acc_dict[k].append(round(v * 100, 2))
        return decoded_out


def build_SCANet(
        model_config: dict, train_cfg: dict = None, checkpoint=None
) -> L.LightningModule:
    ############## backbone and neck ##############
    backbone: Backbone = build_backbone(model_config["backbone"])
    hidden_dim = model_config["hidden_dim"]  # 256
    neck: Neck = build_neck(model_config["neck"])
    ############## encoder and decoder ##############
    component_encoder = build_encoder(model_config["component_encoder"])
    position_embedding = PositionEmbeddingSine(
        hidden_dim // 2, normalize=model_config["position_norm"]
    )
    transformer = build_transformer(model_config["transformer"])
    ############## heads ##############
    heads = {}
    loss = build_loss(model_config["loss"])
    for head_config in model_config["heads"]:
        head_name = head_config["name"]
        heads[head_name] = build_head(head_config)
    if checkpoint is not None:
        assert os.path.exists(checkpoint)
        SCA_net = SCANet.load_from_checkpoint(
            checkpoint,
            backbone=backbone,
            neck=neck,
            transformer=transformer,
            heads=heads,
            num_component_queries=model_config["num_component_queries"],
            brick_encoder=component_encoder,
            position_embedding=position_embedding,
            hidden_dim=hidden_dim,
            loss=loss,
            train_cfg=train_cfg
        )
    else:
        SCA_net = SCANet(
            backbone=backbone,
            neck=neck,
            transformer=transformer,
            heads=heads,
            num_component_queries=model_config["num_component_queries"],
            brick_encoder=component_encoder,
            position_embedding=position_embedding,
            hidden_dim=hidden_dim,
            loss=loss,
            train_cfg=train_cfg

        )
        # load pretrain
        model_state_dict = SCA_net.state_dict()
        checkpoint_path = model_config["pretrain"]
        state_dict = torch.load(checkpoint_path)
        loaded_weights_name = []

        for name, param in state_dict.items():
            if name in model_state_dict:
                loaded_weights_name.append(name)
                if model_state_dict[name].shape == param.shape:
                    model_state_dict[name].copy_(param)
                else:
                    print(
                        f"Warning: model layer {name} has shape {param.shape} is not match the state dict "
                        f"{model_state_dict[name].shape}!"
                    )
        print("-" * 20)
        print("loaded weights: ")
        print(*loaded_weights_name, sep="\n")
        print("-" * 20)
    return SCA_net
