#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 1/9/2024 3:43 PM
# @Author  : Yaser
# @Site    :
# @File    : loss.py
# @Software: PyCharm

import einops.einops as enp
from torch import nn
import torch


class LossMix(nn.Module):
    def __init__(self, loss_list):
        super().__init__()
        assert len(loss_list) > 0
        self.loss_list = []
        for loss_dict in loss_list:
            loss_type = loss_dict["loss_type"]
            loss_opts = loss_dict["opts"]
            assert loss_type in globals()
            loss = globals()[loss_type](**loss_opts)
            self.loss_list.append(loss)

    def forward(self, output, target, brick_padding_list):
        loss_sum = 0
        total_loss_dict = {}
        for loss_call in self.loss_list:
            loss_val, loss_dict = loss_call(output, target, brick_padding_list)
            loss_sum += loss_val
            total_loss_dict.update(loss_dict)

        total_loss_dict["loss_sum"] = loss_sum
        return loss_sum, total_loss_dict


class TransLossV5(torch.nn.Module):
    def __init__(self, weight=1.):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.weight = weight

    def forward_one_layer(self, trans_out, target, brick_padding_list):
        trans_cat = enp.rearrange(trans_out, "B N K C -> (B N) K C")
        brick_idx_exclude_pad = []
        max_brick_num = trans_out.size(1)
        for pan_len in brick_padding_list:
            brick_idx_exclude_pad += [True] * (max_brick_num - pan_len) + [
                False
            ] * pan_len
        brick_idx_exclude_pad = torch.as_tensor(brick_idx_exclude_pad)
        trans_cat_exclude_pad = trans_cat[brick_idx_exclude_pad]

        pred_trans_x = trans_cat_exclude_pad[:, 0]
        pred_trans_y = trans_cat_exclude_pad[:, 1]
        pred_trans_z = trans_cat_exclude_pad[:, 2]

        target_trans_x = (
            torch.cat(target["gt_trans_x"], dim=0).squeeze(-1).to(torch.long)
        )
        target_trans_y = (
            torch.cat(target["gt_trans_y"], dim=0).squeeze(-1).to(torch.long)
        )
        target_trans_z = (
            torch.cat(target["gt_trans_z"], dim=0).squeeze(-1).to(torch.long)
        )

        pred_trans_x_arg_max = pred_trans_x.argmax(dim=1)
        pred_trans_y_arg_max = pred_trans_y.argmax(dim=1)
        pred_trans_z_arg_max = pred_trans_z.argmax(dim=1)

        correct = (
                (pred_trans_x_arg_max == target_trans_x)
                & (pred_trans_y_arg_max == target_trans_y)
                & (pred_trans_z_arg_max == target_trans_z)
        )
        correct_sum = torch.sum(correct).item()
        acc = round(correct_sum / len(correct) * 100, 2)

        x_trans_loss = self.criterion(pred_trans_x, target_trans_x)
        y_trans_loss = self.criterion(pred_trans_y, target_trans_y)
        z_trans_loss = self.criterion(pred_trans_z, target_trans_z)

        trans_loss = x_trans_loss + y_trans_loss + z_trans_loss
        return trans_loss * self.weight, acc

    def forward(self, output, target, brick_padding_list):
        trans_out = output["TransHead"]
        dim_len = len(trans_out.shape)
        if dim_len == 5:
            L, B, N, K, C = trans_out.shape
            trans_loss = 0
            acc = 0
            for L_out in trans_out:
                loss_result = self.forward_one_layer(L_out, target, brick_padding_list)
                trans_loss += loss_result[0]
                acc += loss_result[1]
            acc /= L
        else:
            trans_loss, acc = self.forward_one_layer(
                trans_out, target, brick_padding_list
            )

        return trans_loss, {"trans_cls_loss": trans_loss, "trans_acc": acc}


class CLoss(torch.nn.Module):
    def __init__(self, weight=1.):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.weight = weight

    def forward(self, output, target, brick_padding_list):
        c_out = output["CHead"]
        dim_len = len(c_out.shape)
        if dim_len == 4:
            L, B, N, C = c_out.shape
            c_loss = 0
            acc = 0
            for L_out in c_out:
                loss_result = self.forward_one_layer(L_out, target, brick_padding_list)
                c_loss += loss_result[0]
                acc += loss_result[1]
            acc /= L
        else:
            c_loss, acc = self.forward_one_layer(c_out, target, brick_padding_list)

        return c_loss, {"c_loss": c_loss, "c_acc": acc}

    def forward_one_layer(self, c_out, target, brick_padding_list):

        c_target = target["brick_state"]
        c_out_exclude_pad = exclude_pad(c_out, brick_padding_list, c_out.shape[1])

        c_target = torch.cat(c_target, dim=0).squeeze(-1).to(torch.long)
        c_loss = self.criterion(c_out_exclude_pad, c_target) * self.weight
        c_pred = c_out_exclude_pad.argmax(dim=1)
        # 计算预测正确的数量

        correct = (c_pred == c_target).sum().item()
        total = len(c_target)
        acc = round((correct / total) * 100, 2)
        return c_loss, acc


class RLoss(torch.nn.Module):
    def __init__(self, weight=1.):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.weight = weight

    def forward(self, output, target, brick_padding_list):
        r_out = output["RHead"]
        dim_len = len(r_out.shape)
        if dim_len == 4:
            L, B, N, C = r_out.shape
            r_loss = 0
            acc = 0
            for L_out in r_out:
                loss_result = self.forward_one_layer(L_out, target, brick_padding_list)
                r_loss += loss_result[0]
                acc += loss_result[1]
            acc /= L
        else:
            r_loss, acc = self.forward_one_layer(r_out, target, brick_padding_list)

        return r_loss, {"r_loss": r_loss, "r_acc": acc}

    def forward_one_layer(self, r_out, target, brick_padding_list):

        rotation_target = target["gt_rotation_state"]
        r_out_exclude_pad = exclude_pad(r_out, brick_padding_list, r_out.shape[1])
        rotation_target = torch.cat(rotation_target, dim=0).to(torch.long).squeeze(-1)
        r_loss = self.criterion(r_out_exclude_pad, rotation_target) * self.weight
        r_pred = torch.argmax(r_out_exclude_pad, dim=1).squeeze(-1)
        # 计算预测正确的数量
        correct = (r_pred == rotation_target).sum().item()
        total = len(rotation_target)
        acc = round((correct / total) * 100, 2)
        return r_loss, acc


def exclude_pad(x, brick_padding_list, max_brick_num):
    brick_idx_exclude_pad = [
        [True] * (max_brick_num - pan_len) + [False] * pan_len
        for pan_len in brick_padding_list
    ]
    brick_idx_exclude_pad = torch.as_tensor(brick_idx_exclude_pad)
    x = x[brick_idx_exclude_pad]
    return x


def build_loss(loss_config):
    loss_name = loss_config["name"]
    assert loss_name in globals(), f"{loss_name} is not found in globals()"

    loss_cls = globals()[loss_name]
    opts = loss_config["opts"]
    loss = loss_cls(**opts)
    return loss
