#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 3/4/2024 2:50 PM
# @Author  : Yaser
# @Site    : 
# @File    : correct_bricks.py
# @Software: PyCharm
import copy
import os

import torch
import yaml

from src.bricks.brick_info import dict_to_cbrick, get_cbrick_enc_pc, get_brick_class, get_brick_enc_pc_, \
    get_cbrick_keypoint, BricksPC, add_cbrick_to_bricks_pc
from src.datasets.lego_ECA import load_brick_voxel_pc, load_trans_rotation
from debug.utils import render_dict_simple_one_step_only_current_brick, render_dict_simple_one_step
from incorrect_pred_generator import EvalOptions

from src.SCANet.SCANet import build_SCANet
from torchvision import transforms as tfs
from PIL import Image
import numpy as np
from pycocotools import mask as mask_utils
import trimesh.transformations as tr

SCANet = None


class SCANetOptions(EvalOptions):
    def initialize(self, parser):
        parser = super().initialize(parser)
        parser.add_argument('--config_path', type=str, default='./configs/SCANet.yaml')
        parser.add_argument('--seed', type=int, default=0)
        parser.add_argument('--correct_iter_num', type=int, default=1)
        parser.add_argument('--checkpoint_path', type=str, default=None)
        parser.add_argument('--set_for_test', action="store_true")
        parser.add_argument('--set_for_test_path', type=str, default=None)
        parser.add_argument('--without-correct', action="store_true")
        parser.add_argument('--correct_conf', type=float, default=0.1)
        parser.add_argument('--visualize', action="store_true")
        return parser


def get_pred_obj_occ(prev_obj_occ, bricks_pred, opt, x_input, targets, show=False, return_voxel=False):
    current_bricks_pc = BricksPC(grid_size=(65, 65, 65), record_parents=False)
    for i, b in enumerate(bricks_pred):
        rot_quat = tr.quaternion_from_euler(
            *list(map(lambda x: x * np.pi / 180, b["rot_decoded"]))
        )
        if b["bid"] >= 0:
            kp_offset = np.array([0, get_brick_class(b["bid_decoded"]).get_height(), 0])
            current_bricks_pc.add_brick(
                b["bid_decoded"],
                b["trans"] - kp_offset,
                rot_quat,
                op_type=targets[0]["op_type"][i].cpu().numpy(),
                no_check=True,
            )
        else:
            cbrick_canonical = x_input["cbrick"][0][int(-b["bid"]) - 1]
            kp_offset = (
                    get_cbrick_keypoint(
                        cbrick_canonical,
                        policy="brick" if opt.cbrick_brick_kp else "simple",
                    )[0]
                    - cbrick_canonical.position
            )
            cbrick_this = copy.deepcopy(cbrick_canonical)
            cbrick_this.update_position_rotation(b["trans"] - kp_offset, rot_quat)

            assert add_cbrick_to_bricks_pc(
                current_bricks_pc,
                cbrick_this,
                op_type=targets[0]["op_type"],
                no_check=True,
            )

    current_occ = torch.as_tensor(current_bricks_pc.get_occ_with_rotation()[0])
    current_occ |= prev_obj_occ
    if return_voxel:
        return current_occ
    points = torch.nonzero(current_occ, as_tuple=False).float()

    points = points.numpy()

    return points


def replace_poses_one_step(opt, pred_manual_step_i, bricks_pred, subm_cbricks=None):
    subm_ct = 0

    b_step = pred_manual_step_i["bricks"]
    # bricks_pred = pred['bricks_pred']
    has_subm = False
    for i, brick in enumerate(b_step):
        rot_euler = bricks_pred[i]["rot_decoded"]
        rot_quat = tr.quaternion_from_euler(
            *list(map(lambda x: x * np.pi / 180, rot_euler))
        )
        position = np.array(bricks_pred[i]["trans"])
        key_point = bricks_pred[i]["kp"] + [0]  # 预测的结果只有xy轴的坐标
        mask = np.asfortranarray(bricks_pred[i]["mask"]).astype("uint8")
        mask_encode = mask_utils.encode(mask)
        mask_encode["counts"] = mask_encode["counts"].decode("utf-8")
        if "brick_type" in brick:
            brick_type = brick["brick_type"]
            kp_offset = [0, get_brick_class(brick_type).get_height(), 0]
        else:
            if subm_cbricks is not None:
                cbrick = subm_cbricks[subm_ct]
                has_subm = True
                if cbrick is not None:
                    brick["canonical_state"] = cbrick.to_dict()
                else:
                    cbrick = dict_to_cbrick(brick)
            else:
                cbrick = dict_to_cbrick(brick)
            kp_offset = (
                    get_cbrick_keypoint(
                        cbrick, policy="brick" if opt.cbrick_brick_kp else "simple"
                    )[0]
                    - cbrick.position
            )

        if opt.top_center:
            position -= kp_offset
        brick["brick_transform"]["position"] = list(map(float, position))
        brick["brick_transform"]["rotation"] = list(map(float, rot_quat))
        brick["brick_transform"]["keypoint_brick"] = list(map(float, key_point))
        brick["mask"] = mask_encode
        brick["correct"] = bricks_pred[i]["correct"]

        if has_subm:
            subm_ct += 1
    return b_step


def init_SCANet(opt):
    config_path = opt.config_path
    with open(config_path, 'r') as file:
        config_data = yaml.safe_load(file)
    model_config = config_data['model']
    global SCANet
    SCANet = build_SCANet(model_config, checkpoint=opt.checkpoint_path)
    SCANet.eval()


def correct_bricks(opt, device, img_path, manual_pred, step_id, x_input, targets, bricks_pred_result_recovered):
    preprocess = tfs.Compose(
        [
            tfs.ToTensor(),
            tfs.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    bricks_pred = replace_poses_one_step(opt, manual_pred["operations"][str(step_id)],
                                         bricks_pred_result_recovered)

    raw_manual_img = Image.open(img_path).convert("RGB")
    manual_tensor = preprocess(raw_manual_img).unsqueeze(0).to(device)
    raw_pred_img = render_dict_simple_one_step(manual_pred, step_id).convert("RGB")
    pred_tensor = preprocess(raw_pred_img).unsqueeze(0).to(device)

    pred_brick_imgs = render_dict_simple_one_step_only_current_brick(manual_pred, step_id, no_check=True)
    pred_brick_imgs = [preprocess(brick_img).to(device) for brick_img in pred_brick_imgs]
    # pred_brick_imgs = None
    base_shape_tensor = x_input["obj_occ_prev"].float().unsqueeze(0).to(device)
    brick_voxel_raw = [b.to(device) for b in x_input['brick_occs']]
    brick_voxel_raw2 = []
    pred_shape_tensor = get_pred_obj_occ(x_input["obj_occ_prev"][0], bricks_pred_result_recovered, opt,
                                         x_input,
                                         targets,
                                         show=False,
                                         return_voxel=True
                                         ).float().unsqueeze(0).unsqueeze(1).to(device)

    kp_offset_list = []
    for brick in bricks_pred:

        if "brick_type" in brick:
            brick_type = brick["brick_type"]
            kp_offset = np.array([0, get_brick_class(brick_type).get_height(), 0])
        else:
            cbrick = dict_to_cbrick(brick)
            kp_offset = np.array(
                get_cbrick_keypoint(cbrick, policy="brick" if opt.cbrick_brick_kp else "simple")[0]
                - cbrick.position)
        kp_offset_list.append(kp_offset)
    pred_brick_voxel_tensor_transformed = []
    pred_trans_rotation = []
    for brick_info in bricks_pred:
        pred_trans_rotation.append(load_trans_rotation(brick_info).to(device))
        brick_type = -1
        if "brick_type" in brick_info:
            # 直接从brick type加载
            brick_type = brick_info["brick_type"]
            extra_point = [0, get_brick_class(brick_type).get_height(), 0]
            brick_pc_map = get_brick_enc_pc_(brick_type)
            # 只获取点云数据
            brick_pc_map["extra_point"] = extra_point
            brick_pc_map["extra_point_value"] = 2
        else:
            cbrick = dict_to_cbrick(brick_info, reset_pose=True)
            brick_pc_map = get_cbrick_enc_pc(cbrick)
            extra_point = [0, cbrick.get_height(), 0]
            brick_pc_map["extra_point"] = extra_point
            brick_pc_map["extra_point_value"] = 2

        pred_brick_voxel = torch.as_tensor(
            load_brick_voxel_pc(brick_pc_map, brick_type, brick_info, return_pc=False),
            device=device).float().unsqueeze(dim=0)
        pred_brick_voxel_tensor_transformed.append(pred_brick_voxel)

        raw_brick_voxel = torch.as_tensor(
            load_brick_voxel_pc(brick_pc_map, brick_type, return_pc=False), device=device).float()

        grid_size = 65
        min_xyz = [32, 16, 32]
        max_xyz = [d + grid_size for d in min_xyz]
        raw_brick_voxel = raw_brick_voxel[
                          min_xyz[0]: max_xyz[0],
                          min_xyz[1]: max_xyz[1],
                          min_xyz[2]: max_xyz[2],
                          ]

        brick_voxel_raw2.append(raw_brick_voxel.unsqueeze(dim=0))

    transform_options_tensor = [[x_input['azim'][0].to(device), x_input['elev'][0].to(device),
                                 x_input['obj_scale'][0].to(device), x_input['obj_center'][0].to(device)]]
    batch_data_x = {
        'manual_tensor': manual_tensor,
        'pred_tensor': pred_tensor,
        'base_shape_tensor': base_shape_tensor,
        'pred_shape_tensor': pred_shape_tensor,
        'pred_brick_voxel_tensor_transformed': [pred_brick_voxel_tensor_transformed],
        'transform_options_tensor': transform_options_tensor,
        'brick_voxel_raw': brick_voxel_raw,
        'brick_voxel_raw2': [brick_voxel_raw2],
        'pred_trans_rotation': [pred_trans_rotation],
        'pred_brick_imgs': [pred_brick_imgs],
        "raw_pred_img":raw_pred_img,
        "raw_manual_img":raw_manual_img

    }
    batch_data = (batch_data_x, None)
    global SCANet
    decoded_out = SCANet(batch_data)[0]
    for decoded_out_idx, kp_offset in zip(decoded_out, kp_offset_list):
        decoded_out_idx['position'] = np.array(decoded_out_idx['position']) + kp_offset

    return decoded_out
