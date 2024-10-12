#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 1/7/2024 2:50 PM
# @Author  : Yaser
# @Site    :
# @File    : lego_incorrect_dataset.py
# @Software: PyCharm
import argparse
import json
import os
import pickle
import random
from itertools import repeat
from multiprocessing.pool import ThreadPool, Pool
from os.path import exists

import numpy as np
from PIL import Image

from torchvision import transforms

import torch

from src.tu.ddp import get_rank

# torch.multiprocessing.set_sharing_strategy("file_system")

from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm

from src.bricks.brick_info import (
    get_brick_class,
    get_brick_enc_voxel_,
    dict_to_cbrick,
    get_cbrick_enc_voxel,
    get_brick_enc_pc_,
    get_cbrick_enc_pc,
    brick_pc2voxel,
)
from src.datasets.definition import gdef
from src.models.heatmap.utils.image import draw_umich_gaussian, draw_umich_gaussian_1d
from src.util.common_utils import (load_json, create_not_exist, )
import torch.utils.data as data

THREAD_NUM_WORKERS = 8
ROTATION_ID_MAP = {
    "0,0,0": 0,
    "0,90,0": 1,
    "180,0,180": 2,
    "0,-90,0": 3,
    "180,90,180": 4,
}


def _prepare_data(sub_dir_path, train_valid_split):
    # Read the pred_info.json and info.json files
    pred_info_json_path = os.path.join(sub_dir_path, "pred_info.json")
    info_path = os.path.join(sub_dir_path, "info.json")
    # Ensure that both files exist
    assert exists(pred_info_json_path) and exists(info_path), f"{pred_info_json_path} or {info_path} does not exist!"
    pred_info_data = load_json(pred_info_json_path)
    gt_info_data = load_json(info_path)
    # Load the point cloud data
    base_pc_data, pred_pc_data = load_pc(sub_dir_path)
    # Extract object scale and center from the ground truth data
    obj_scale = gt_info_data["obj_scale"]
    obj_center = gt_info_data["obj_center"]
    # Extract operation information from both prediction and ground truth data
    pred_info_operations = pred_info_data["operations"]
    gt_info_operations = gt_info_data["operations"]
    # Separate the dataset into training and validation sets based on the split provided

    train_id_set = set(train_valid_split["train"])
    valid_id_set = set(train_valid_split["valid"])
    view_directions = []
    obj_transforms = []
    GT_img_path_list = []
    pred_img_path_list = []
    base_shape_pc_list = []
    pred_shape_pc_list = []
    pred_bricks_info_list = []
    gt_bricks_info_list = []
    train_valid_flag = []
    # Iterate through each predicted operation
    for pred_operation_id, pred_operation in pred_info_operations.items():
        if (
                pred_operation_id not in train_id_set
                and pred_operation_id not in valid_id_set
        ):
            continue
        # Extract the corresponding ground truth operation ID
        gt_operation_id = int(pred_operation_id.split("_")[0])
        view_directions.append(pred_operation["view_direction"])
        obj_transforms.append([obj_scale, obj_center])

        gt_img_path, pred_img_path = get_gt_pred_img_path(
            sub_dir_path, pred_operation_id
        )
        GT_img_path_list.append(gt_img_path)
        pred_img_path_list.append(pred_img_path)
        # Add point cloud data for both base (ground truth) and predicted shapes
        base_shape_pc_list.append(
            np.array(base_pc_data[gt_operation_id - 1], dtype=int)
            if gt_operation_id > 0
            else None
        )
        pred_shape_pc_list.append(np.array(pred_pc_data[pred_operation_id], dtype=int))
        # Add brick information for each step from both predicted and ground truth operations
        pred_bricks_info_list.append(pred_info_operations[pred_operation_id]["bricks"])
        gt_bricks_info_list.append(gt_info_operations[str(gt_operation_id)]["bricks"])
        train_valid_flag.append(pred_operation_id in train_id_set)

    return {
        "view_directions": view_directions,
        "obj_transforms": obj_transforms,
        "GT_img_path_list": GT_img_path_list,
        "pred_img_path_list": pred_img_path_list,
        "base_shape_pc_list": base_shape_pc_list,
        "pred_shape_pc_list": pred_shape_pc_list,
        "pred_bricks_info_list": pred_bricks_info_list,
        "gt_bricks_info_list": gt_bricks_info_list,
        "train_valid_flag": train_valid_flag,
    }


def load_img(img_path):
    img = Image.open(img_path).convert("RGB")
    img.load()
    return img


def load_voxel(brick_type_step, brick_pc_cache_list, brick_info_step=None):
    brick_num = len(brick_type_step)
    bytes_size = 0
    if brick_info_step is None:
        visited_brick = set()
        pred_brick_voxel_raw_list = []
        for brick_idx, btype_tuple in zip(range(brick_num), brick_type_step):
            brick_type = btype_tuple[1]
            brick_pc_map = brick_pc_cache_list[brick_type]
            # 加载预测的体素数据，但不应用RT矩阵
            if brick_type not in visited_brick:
                # 此处模仿mepnet，每一个型号的组件只添加一次
                brick_voxel = load_brick_voxel_pc(
                    brick_pc_map, brick_type, return_pc=False
                )
                # 按照mepnet，此处对组件的voxel进行crop
                grid_size = 65
                min_xyz = [32, 16, 32]
                max_xyz = [d + grid_size for d in min_xyz]
                pred_brick_voxel_raw = brick_voxel[
                                       min_xyz[0]: max_xyz[0],
                                       min_xyz[1]: max_xyz[1],
                                       min_xyz[2]: max_xyz[2],
                                       ]

                pred_brick_voxel_raw_tensor = (
                    torch.as_tensor(pred_brick_voxel_raw).float().unsqueeze(dim=0)
                )
                bytes_size += (
                        pred_brick_voxel_raw_tensor.element_size()
                        * pred_brick_voxel_raw_tensor.numel()
                )  # 计算占用的内存大小

                pred_brick_voxel_raw_list.append(
                    pred_brick_voxel_raw_tensor
                )  # 体素数据都需要加一个维度表示channel
                visited_brick.add(brick_type)
        return (
            pred_brick_voxel_raw_list,
            None,
            bytes_size,
        )  # 使用None来保证返回的数据长度一样，方便处理
    else:
        brick_voxel_transformed_list = []
        brick_pc_list = []
        for brick_idx, btype_tuple, brick_info in zip(
                range(brick_num), brick_type_step, brick_info_step
        ):
            brick_type = btype_tuple[1]
            brick_pc_map = brick_pc_cache_list[brick_type]
            brick_voxel, brick_pc = load_brick_voxel_pc(
                brick_pc_map, brick_type, brick_info
            )

            brick_voxel_tensor = torch.as_tensor(brick_voxel).float().unsqueeze(dim=0)
            brick_pc_tensor = torch.as_tensor(brick_pc).float()
            # 计算占用的内存大小
            bytes_size += brick_voxel_tensor.element_size() * brick_voxel_tensor.numel()
            bytes_size += brick_pc_tensor.element_size() * brick_pc_tensor.numel()

            brick_voxel_transformed_list.append(
                brick_voxel_tensor
            )  # 体素数据都需要加一个维度表示channel
            brick_pc_list.append(brick_pc_tensor)
        return brick_voxel_transformed_list, brick_pc_list, bytes_size


def get_gt_pred_img_path(sub_dir_path, pred_operation_id):
    gt_operation_id = int(pred_operation_id.split("_")[0])
    pred_operation_inner_id = pred_operation_id.split("_")[-1]
    gt_img_name = f"{gt_operation_id:03d}.png"
    pred_img_name = f"{gt_operation_id:03d}_{pred_operation_inner_id}_pred.png"
    gt_img_path = os.path.join(sub_dir_path, gt_img_name)
    pred_img_path = os.path.join(sub_dir_path, pred_img_name)

    assert exists(gt_img_path) and exists(pred_img_path)
    return gt_img_path, pred_img_path


def load_pc(sub_dir_path):
    """
    初始化并加载该文件夹的点云数据
    Args:
        sub_dir_path: 子文件夹路径
    """
    assert exists(sub_dir_path)
    base_shape_path = os.path.join(sub_dir_path, "occs.pkl")
    with open(base_shape_path, "rb") as file:
        base_pc_data = pickle.load(file)
    pred_shape_path = os.path.join(sub_dir_path, "pred_point_clouds.json")
    with open(pred_shape_path, "r") as file:
        pred_pc_data = json.load(file)
    return base_pc_data, pred_pc_data


def load_brick_voxel_pc(brick_pc_map, brick_type, brick_info=None, return_pc=True):
    if brick_info is not None:
        position = brick_info["brick_transform"]["position"]
        rotation = brick_info["brick_transform"]["rotation"]
    else:
        position = None
        rotation = None
    voxel_pc_result = brick_pc2voxel(
        brick_type, brick_pc_map, position, rotation, return_pc=return_pc
    )  # 可能是cbrick或brick，这里统一视为brick
    return voxel_pc_result


def shape_pc2voxel(pc_list):
    shape_voxel = np.zeros([130, 130, 130], dtype=bool)
    if pc_list is not None:
        # base shape在第一个step时为None
        shape_voxel[pc_list[:, 0], pc_list[:, 1], pc_list[:, 2]] = True
    return torch.as_tensor(shape_voxel).float().unsqueeze(dim=0)


def load_kp_classify_xy(brick_info, output_w, output_h):
    keypoint = brick_info["brick_transform"]["keypoint_brick"][:2]
    ct = (np.array(keypoint, dtype=np.float32) / gdef.down_ratio) * gdef.scale_ration
    ct_int = ct.astype(np.int32)
    radius = gdef.hm_gauss * 3
    hm = np.zeros(
        (output_h * gdef.scale_ration, output_w * gdef.scale_ration), dtype=np.float32
    )
    hm = torch.as_tensor(draw_umich_gaussian(hm, ct_int, radius), dtype=torch.float32)
    hm_x, _ = torch.max(hm, dim=0)
    hm_y, _ = torch.max(hm, dim=1)
    ind = torch.as_tensor(ct_int[1] * output_w + ct_int[0], dtype=torch.int64)

    # reg = torch.as_tensor(ct - ct_int, dtype=torch.float32)
    return hm, hm_x, hm_y, ind


def load_trans_rotation(brick_info):
    trans = np.asarray(brick_info["brick_transform"]["position"], dtype=np.float32) * 2
    voxel_translation = np.asarray([65, 65 // 4, 65])
    trans += voxel_translation
    trans = torch.as_tensor(trans, dtype=torch.float32) / 130.0
    rotation = torch.as_tensor(
        brick_info["brick_transform"]["rotation_euler"], dtype=torch.float32
    )
    rotation = (rotation - torch.tensor([0, -90, 0])) / torch.tensor([180, 180, 180])
    return torch.cat([trans, rotation], dim=0)


def load_trans_xyz_kl(gt_brick_info):
    trans = (
            np.asarray(gt_brick_info["brick_transform"]["position"], dtype=np.float32) * 2
    )
    voxel_translation = np.asarray([65, 65 // 4, 65])
    trans += voxel_translation
    trans_x = np.zeros((gdef.occ_size[0], 1), dtype=np.float32)
    trans_y = np.zeros((gdef.occ_size[1], 1), dtype=np.float32)
    trans_z = np.zeros((gdef.occ_size[2], 1), dtype=np.float32)
    radius = gdef.hm_gauss * 3
    assert np.min(trans) >= 0 and np.max(trans) < 130, f"trans_dist: {trans}"
    trans_x = torch.as_tensor(
        draw_umich_gaussian_1d(trans_x, trans[0], radius), dtype=torch.float32
    ).squeeze(-1)
    trans_y = torch.as_tensor(
        draw_umich_gaussian_1d(trans_y, trans[1], radius), dtype=torch.float32
    ).squeeze(-1)
    trans_z = torch.as_tensor(
        draw_umich_gaussian_1d(trans_z, trans[2], radius), dtype=torch.float32
    ).squeeze(-1)
    return trans_x, trans_y, trans_z


def load_kp_heat_map(brick_info, output_w, output_h):
    keypoint = brick_info["brick_transform"]["keypoint_brick"][:2]
    ct = np.array(keypoint, dtype=np.float32)
    ct_int = ct.astype(np.int32)
    radius = gdef.hm_gauss * 3
    hm = np.zeros(
        (output_h * gdef.scale_ration, output_w * gdef.scale_factor), dtype=np.float32
    )
    hm = torch.as_tensor(draw_umich_gaussian(hm, ct_int, radius), dtype=torch.float32)
    hm_x, _ = torch.max(hm, dim=0)
    hm_y, _ = torch.max(hm, dim=1)
    ind = torch.as_tensor(ct_int[1] * output_w + ct_int[0], dtype=torch.int64)
    reg = torch.as_tensor(ct - ct_int, dtype=torch.float32)
    return hm, hm_x, hm_y, ind, reg


def load_brick_rotation_state(gt_brick_info):
    gt_rotation = gt_brick_info["brick_transform"]["rotation_euler"]
    gt_rotation_str = map(str, gt_rotation)
    gt_rotation_str = ",".join(gt_rotation_str)
    return ROTATION_ID_MAP[gt_rotation_str]


def load_trans_xyz(gt_brick_info, return_one=False):
    trans = (
            np.asarray(gt_brick_info["brick_transform"]["position"], dtype=np.float32) * 2
    )
    voxel_translation = np.asarray([65, 65 // 4, 65])
    trans += voxel_translation
    trans_x = torch.as_tensor(trans[0], dtype=torch.long)
    trans_y = torch.as_tensor(trans[1], dtype=torch.long)
    trans_z = torch.as_tensor(trans[2], dtype=torch.long)
    if not return_one:
        return trans_x, trans_y, trans_z
    else:
        return torch.cat([trans_x, trans_y, trans_z], dim=0)


def load_dist_xyz(pred_brick_info, gt_brick_info):
    gt_pos = np.asarray(gt_brick_info["brick_transform"]["position"], dtype=np.float32)
    pred_pos = np.asarray(
        pred_brick_info["brick_transform"]["position"], dtype=np.float32
    )
    trans_dist = (gt_pos - pred_pos) * 2
    # voxel_translation = np.asarray([65, 65 // 4, 65])
    # trans_dist += voxel_translation
    symbol = [0 if dist >= 0 else 1 for dist in trans_dist]
    dist_x = torch.as_tensor([symbol[0], abs(trans_dist[0])], dtype=torch.long)
    dist_y = torch.as_tensor([symbol[1], abs(trans_dist[1])], dtype=torch.long)
    dist_z = torch.as_tensor([symbol[2], abs(trans_dist[2])], dtype=torch.long)
    assert 0 <= dist_x[1] < 130 and 0 <= dist_y[1] < 130 and 0 <= dist_z[1] < 130
    return dist_x, dist_y, dist_z


def _cache_img(img_list, preprocess, prefix=""):
    pool = ThreadPool(THREAD_NUM_WORKERS)
    results = pool.imap(load_img, img_list)
    pbar = tqdm(
        enumerate(results),
        total=len(img_list),
        disable=get_rank() != 0,
    )
    cached_img_list = []
    gb = 0
    GB = 1024 ** 3
    for idx, result in pbar:
        cached_img_list.append(preprocess(result))
        gb += np.asarray(cached_img_list[idx]).nbytes
        pbar.desc = f"Caching {prefix}_images ({gb / GB:.1f}GB)..."
    # pbar.close()
    pool.close()
    return cached_img_list


def _cache_voxel(
        btype_cache_list,
        brick_pc_cache_list,
        brick_info_list=None,
        return_pc=True,
        prefix="",
):
    pool = Pool(THREAD_NUM_WORKERS)

    results = pool.imap(
        lambda x: load_voxel(*x),
        zip(
            btype_cache_list,
            repeat(brick_pc_cache_list),
            brick_info_list if brick_info_list is not None else repeat(None),
        ),
    )
    pbar = tqdm(
        enumerate(results), total=len(btype_cache_list), disable=get_rank() != 0
    )
    cached_voxel_list = []
    cached_pc_list = []
    gb = 0
    GB = 1024 ** 3
    for idx, result in pbar:
        brick_voxel_list, brick_pc_list, bytes_size = result
        cached_voxel_list.append(brick_voxel_list)
        cached_pc_list.append(brick_pc_list)
        gb += bytes_size
        pbar.desc = f"Caching {prefix}_voxel ({gb / GB:.1f}GB)..."
    # pbar.close()
    pool.close()

    if return_pc:
        return cached_voxel_list, cached_pc_list
    else:
        return cached_voxel_list


class LegoECADataset(Dataset):
    def __init__(self, opt, set_for_train, train_valid_json):
        self.opt = opt
        self.dataset_root = f"{os.path.abspath(opt.dataset_root)}"

        assert os.path.isdir(self.dataset_root)
        self.set_for_train = set_for_train
        sub_dir_path_list = [
            os.path.join(self.dataset_root, set_id) for set_id in set_for_train
        ]

        self.GT_img_path_list = []
        self.pred_img_path_list = []
        self.base_shape_pc_list = []
        self.pred_shape_pc_list = []
        self.pred_bricks_info_list = []
        self.gt_bricks_info_list = []
        self.pred_brick_img_path_list = []
        self.obj_transforms = []
        self.preprocess = transforms.Compose(
            [
                # transforms.Resize((526, 526)),
                # transforms.CenterCrop((512, 512)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        self.view_directions = []
        self.brick_pc_cache_list = (
            {}
        )  # 缓存所有需要用到组件的点云数据，这些点云数据是预测与gt共用的，因此保存的是未应用RT前的数据
        self.btype_cache = []
        self.train_idx = []
        self.valid_idx = []

        self.train_valid_json = train_valid_json
        self.dataset_cache_path = os.path.join(
            self.dataset_root, opt.dataset_cache_name
        )
        predpare_data_args = [
            [sub_dir_path, self.train_valid_json[os.path.split(sub_dir_path)[1]]]
            for sub_dir_path in sub_dir_path_list
        ]
        thread_pool = ThreadPool(THREAD_NUM_WORKERS)
        prepare_result_list = thread_pool.imap(
            lambda x: _prepare_data(*x), predpare_data_args
        )
        train_valid_flag_list = []
        for data_result in tqdm(
                prepare_result_list,
                desc=f"Preparing for the training data...",
                total=len(predpare_data_args),
        ):
            for k, v in data_result.items():
                if k != "train_valid_flag":
                    k_attr = self.__getattribute__(k)
                    assert isinstance(k_attr, list)
                    k_attr.extend(v)
                else:
                    train_valid_flag_list.extend(v)
        for idx, train_valid_flag in enumerate(train_valid_flag_list):
            if train_valid_flag:
                self.train_idx.append(idx)
            else:
                self.valid_idx.append(idx)
        self.cache_brick_pc()

        self.gt_img_cache_list = None
        self.pred_img_cache_list = None
        self.raw_brick_voxel_list = None
        self.pred_brick_voxel_pc_tuple = None
        self.gt_brick_voxel_pc_tuple = None
        self.cache_img = opt.cache_img
        self.cache_voxel = opt.cache_voxel
        self.cache = opt.cache_img or opt.cache_voxel

    def __len__(self):
        return len(self.pred_img_path_list)

    def __getitem__(self, index):
        # 加载图片
        pred_img_path = self.pred_img_path_list[index]
        gt_img_path = self.GT_img_path_list[index]
        # start_time = time()

        pred_tensor = self.preprocess(Image.open(pred_img_path).convert("RGB"))
        manual_tensor = self.preprocess(Image.open(gt_img_path).convert("RGB"))
        pred_brick_imgs = []

        btype_cache_list = self.btype_cache[index]
        brick_num = len(btype_cache_list)

        pred_brick_info_list_step = self.pred_bricks_info_list[index]
        gt_brick_info_list_step = self.gt_bricks_info_list[index]
        not_visited_list = []

        if self.cache_voxel:
            pred_brick_voxel_transformed_list = self.pred_brick_voxel_pc_tuple[0][index]
            pred_brick_pc_list = self.pred_brick_voxel_pc_tuple[1][index]
            brick_voxel_raw_list = self.raw_brick_voxel_list[index]
            gt_brick_pc_list = self.gt_brick_voxel_pc_tuple[1][index]
        else:
            pred_brick_voxel_transformed_list = []
            brick_voxel_raw_list = []
            pred_brick_pc_list = []
            gt_brick_pc_list = []

        visited_brick = set()

        output_h = gdef.input_h // gdef.down_ratio
        output_w = gdef.input_w // gdef.down_ratio
        hm = torch.zeros(
            (brick_num, output_h * gdef.scale_ration, output_w * gdef.scale_ration),
            dtype=torch.float32,
        )  # keypoint的heatmap
        reg = torch.zeros(
            (brick_num, 2), dtype=torch.float32
        )  # 针对关键点后的修正值（关键点位置为int）
        ind = torch.zeros((brick_num, 1), dtype=torch.int64)
        # keypoint的一维索引，值为从左上角开始从左到右依次计数
        hm_x = torch.zeros(
            (brick_num, output_w * gdef.scale_ration), dtype=torch.float32
        )
        hm_y = torch.zeros(
            (brick_num, output_h * gdef.scale_ration), dtype=torch.float32
        )
        # pred2gt_dist = torch.zeros((brick_num, 3), dtype=torch.float32)
        pred_trans_rotation = []

        gt_trans = torch.zeros((brick_num, 3), dtype=torch.float32)

        gt_trans_x = torch.zeros((brick_num, 1), dtype=torch.float32)
        gt_trans_y = torch.zeros((brick_num, 1), dtype=torch.float32)
        gt_trans_z = torch.zeros((brick_num, 1), dtype=torch.float32)

        gt_trans_x_kl = torch.zeros((brick_num, gdef.occ_size[0]), dtype=torch.float32)
        gt_trans_y_kl = torch.zeros((brick_num, gdef.occ_size[1]), dtype=torch.float32)
        gt_trans_z_kl = torch.zeros((brick_num, gdef.occ_size[2]), dtype=torch.float32)

        dist_trans_x = torch.zeros((brick_num, 2), dtype=torch.float32)
        dist_trans_y = torch.zeros((brick_num, 2), dtype=torch.float32)
        dist_trans_z = torch.zeros((brick_num, 2), dtype=torch.float32)

        brick_state = torch.zeros((brick_num, 1), dtype=torch.long)
        rotation_state = torch.zeros((brick_num, 1), dtype=torch.long)
        # 加载组件体素以及点云数据，此处会将RT矩阵应用在点云数据上
        for brick_idx, btype_tuple, pred_brick_info, gt_brick_info in zip(
                range(brick_num),
                btype_cache_list,
                pred_brick_info_list_step,
                gt_brick_info_list_step,
        ):
            if not self.cache_voxel:
                brick_type = btype_tuple[1]
                brick_pc_map = self.brick_pc_cache_list[brick_type]
                # 加载预测的体素数据，但不应用预测的RT矩阵
                if brick_type not in visited_brick:
                    not_visited_list.append(brick_idx)
                    visited_brick.add(brick_type)
                # 此处模仿mepnet，每一个型号的组件只添加一次
                brick_occ = load_brick_voxel_pc(
                    brick_pc_map, brick_type, return_pc=False
                )
                # 按照mepnet，此处对组件的voxel进行crop
                grid_size = 65
                min_xyz = [32, 16, 32]
                max_xyz = [d + grid_size for d in min_xyz]
                pred_brick_voxel_raw = brick_occ[
                                       min_xyz[0]: max_xyz[0],
                                       min_xyz[1]: max_xyz[1],
                                       min_xyz[2]: max_xyz[2],
                                       ]
                brick_voxel_raw_list.append(
                    torch.as_tensor(pred_brick_voxel_raw).float().unsqueeze(dim=0)
                )  # 体素数据都需要加一个维度表示channel
                # 加载预测的体素数据，并应用预测的RT矩阵
                pred_brick_voxel, pred_brick_pc = load_brick_voxel_pc(
                    brick_pc_map, brick_type, pred_brick_info
                )
                pred_brick_voxel_transformed_list.append(
                    torch.as_tensor(pred_brick_voxel).float().unsqueeze(dim=0)
                )  # 体素数据都需要加一个维度表示channel
                pred_brick_pc_list.append(torch.as_tensor(pred_brick_pc).float())
                # 加载GT的体素数据，并应用GT的RT矩阵（GT只需要应用GT矩阵后的点云）
                _, gt_brick_pc = load_brick_voxel_pc(
                    brick_pc_map, brick_type, gt_brick_info
                )
                gt_brick_pc_list.append(torch.as_tensor(gt_brick_pc).float())

            # 加载从pred到gt的trans
            gt_trans[brick_idx] = torch.tensor(
                gt_brick_info["brick_transform"]["position"], dtype=torch.float32
            )
            (
                dist_trans_x[brick_idx],
                dist_trans_y[brick_idx],
                dist_trans_z[brick_idx],
            ) = load_dist_xyz(pred_brick_info, gt_brick_info)
            gt_trans_x[brick_idx], gt_trans_y[brick_idx], gt_trans_z[brick_idx] = (
                load_trans_xyz(gt_brick_info)
            )
            pred_trans_rotation.append(load_trans_rotation(pred_brick_info))
            brick_state[brick_idx] = pred_brick_info["correct"][-1]
            rotation_state[brick_idx] = load_brick_rotation_state(gt_brick_info)
            # 处理单独的brick图像数据
            brick_img_path = pred_img_path.replace(
                "pred.png", f"pred_brick_{brick_idx}.png"
            )
            pred_brick_imgs.append(
                self.preprocess(Image.open(brick_img_path).convert("RGB"))
            )

        # print(f"load time: {time() - start_time}")
        # 加载base shape的体素数据与pred的体素数据
        base_shape_voxel = shape_pc2voxel(self.base_shape_pc_list[index])
        pred_shape_voxel = shape_pc2voxel(self.pred_shape_pc_list[index])

        #################################可视化训练数据#################################

        # base_shape_voxel = base_shape_voxel.unsqueeze(0).to("cuda:0")
        # show_img(Image.open(gt_img_path).convert("RGB"), "Ground Truth")
        # camera_transform(base_shape_voxel, title="base_shape_voxel")
        # pred_shape_voxel = pred_shape_voxel.unsqueeze(0).to("cuda:0")
        # show_img(Image.open(pred_img_path).convert("RGB"), "prediction")
        # camera_transform(pred_shape_voxel, title="pred_shape_voxel")
        #################################可视化训练数据#################################

        transform_options_tensor = [
            *self.view_directions[index],
            *self.obj_transforms[index],
        ]
        transform_options_tensor = [
            torch.as_tensor(item) for item in transform_options_tensor
        ]
        # print(f"data load time: {time() - start_time}s")
        return {
            "manual_tensor": manual_tensor,
            "pred_tensor": pred_tensor,
            "base_shape_tensor": base_shape_voxel,
            "pred_shape_tensor": pred_shape_voxel,
            "pred_brick_voxel_tensor_transformed": pred_brick_voxel_transformed_list,
            "brick_voxel_raw": [
                brick_voxel_raw_list[sub_idx] for sub_idx in not_visited_list
            ],
            "brick_voxel_raw2": brick_voxel_raw_list,
            "transform_options_tensor": transform_options_tensor,
            "pred_brick_pc_tensor": pred_brick_pc_list,  # 保留预测的点云，是为了方便将预测的RT矩阵应用在上面
            "gt_brick_pc_tensor": gt_brick_pc_list,
            "pred_info": pred_brick_info_list_step,
            "gt_info": gt_brick_info_list_step,
            "pred_img_path": pred_img_path,
            "gt_img_path": gt_img_path,
            "kp_hm": hm,
            "kp_hm_x": hm_x,
            "kp_hm_y": hm_y,
            "kp_ind": ind,
            "kp_reg": reg,
            "dist_trans_x": dist_trans_x,
            "dist_trans_y": dist_trans_y,
            "dist_trans_z": dist_trans_z,
            "brick_state": brick_state,
            "pred_trans_rotation": pred_trans_rotation,
            "pred_brick_imgs": pred_brick_imgs,
            "gt_trans": gt_trans,
            "gt_trans_x": gt_trans_x,
            "gt_trans_y": gt_trans_y,
            "gt_trans_z": gt_trans_z,
            "gt_trans_x_kl": gt_trans_x_kl,
            "gt_trans_y_kl": gt_trans_y_kl,
            "gt_trans_z_kl": gt_trans_z_kl,
            "gt_rotation_state": rotation_state,
        }

    def cache_brick_pc(self):
        for brick_info_list in tqdm(
                self.pred_bricks_info_list,
                desc=f"Caching the bricks...",
        ):
            btype_cache_list = []
            for brick_info in brick_info_list:
                if "brick_type" in brick_info:
                    # 直接从brick type加载
                    b_type = brick_info["brick_type"]
                    btype_cache_list.append(("brick", b_type))
                    if b_type in self.brick_pc_cache_list:
                        continue
                    extra_point = [0, get_brick_class(b_type).get_height(), 0]
                    brick_pc_map = get_brick_enc_pc_(b_type)

                    # 只获取点云数据
                    brick_pc_map["extra_point"] = extra_point
                    brick_pc_map["extra_point_value"] = 2
                    self.brick_pc_cache_list[b_type] = brick_pc_map
                else:
                    bricks_pc = dict(brick_info["canonical_state"]["bricks_pc"])
                    # 进行hash化
                    bricks_hash = hash(json.dumps(bricks_pc, sort_keys=True))
                    btype_cache_list.append(("cbrick", bricks_hash))
                    if bricks_hash in self.brick_pc_cache_list:
                        continue
                    cbrick = dict_to_cbrick(brick_info, reset_pose=True)
                    cbrick_pc_map = get_cbrick_enc_pc(cbrick)
                    extra_point = [0, cbrick.get_height(), 0]
                    cbrick_pc_map["extra_point"] = extra_point
                    cbrick_pc_map["extra_point_value"] = 2
                    self.brick_pc_cache_list[bricks_hash] = cbrick_pc_map
            self.btype_cache.append(btype_cache_list)

    @staticmethod
    def modify_commandline_options(parser: argparse.ArgumentParser):
        parser.add_argument("--dataset_root", type=str)
        parser.add_argument("--dataset_name", type=str, default="train_valid_v2")
        parser.add_argument("--dataset_size", type=str, default="tiny")
        parser.add_argument("--limit", type=int, default=None)
        parser.add_argument("--start", type=int, default=0)
        parser.add_argument("--max_brick", type=int, default=5)
        parser.add_argument("--cache_img", action="store_true")
        parser.add_argument("--cache_voxel", action="store_true")
        parser.add_argument("--random_select", action="store_true")
        parser.add_argument("--dataset_cache_name", type=str, default="dataset_cache")
        parser.add_argument("--use-cache", action="store_true")
        parser.add_argument("--load_set_path", type=str, default=None)

        return parser


def list_data_to_device(list_data, device):
    assert len(list_data) > 0
    if isinstance(list_data[0], list):
        list_data_device = []
        for data in list_data:
            list_data_device.append(list_data_to_device(data, device))
    elif isinstance(list_data[0], torch.Tensor):
        list_data_device = []
        for data in list_data:
            list_data_device.append(data.to(device))
    else:
        raise "Not support data type!"
    return list_data_device


def data_to_device(batch_data, device):
    feed_data, target = batch_data
    feed_data_device = {}
    target_data_device = {}

    for name, data in feed_data.items():
        if isinstance(data, list):
            feed_data_device[name] = list_data_to_device(data, device)
        else:
            assert isinstance(data, torch.Tensor)
            feed_data_device[name] = data.to(device)

    for name, data in target.items():
        if name.endswith("path") or name.endswith("info"):
            continue
        target_data_device[name] = list_data_to_device(data, device)
    return feed_data_device, target_data_device


def collect_fn(batch):
    d = {}
    targets = {}
    for k in batch[0].keys():
        data = [b[k] for b in batch]
        if k in [
            "gt_brick_pc_tensor",
            "pred_brick_pc_tensor",
            "pred_info",
            "gt_info",
            "pred_img_path",
            "gt_img_path",
            "kp_hm",
            "kp_reg",
            "kp_ind",
            "kp_hm_x",
            "kp_hm_y",
            "pred2gt_dist",
            "gt_trans",
            "gt_trans_x_kl",
            "gt_trans_y_kl",
            "gt_trans_z_kl",
            "dist_trans_x",
            "dist_trans_y",
            "dist_trans_z",
            "brick_state",
            "gt_rotation_state",
            "gt_trans_x",
            "gt_trans_y",
            "gt_trans_z",
        ]:
            targets[k] = data
        elif isinstance(batch[0][k], torch.Tensor):
            stack_tensor = torch.stack(data)
            d[k] = stack_tensor
        else:
            d[k] = data
    return d, targets


def build_dataloader(opts, config, log_dir):
    # 选择进行训练的set
    create_not_exist(log_dir)
    dataset_root = f"{os.path.abspath(opts.dataset_root)}"
    dataset_cache_path = f"{dataset_root}/{opts.dataset_cache_name}.pkl"
    use_dataset_cache = opts.use_cache

    if use_dataset_cache and os.path.exists(dataset_cache_path):
        print(
            "Dataset cache exist! Loading cached dataset from",
            dataset_cache_path,
            "...",
        )
        with open(dataset_cache_path, "rb") as f:
            dataset: LegoECADataset = pickle.load(f)
        set_for_train = dataset.set_for_train
        if len(dataset.btype_cache) == 0:
            dataset.cache_brick_pc()
            with open(dataset_cache_path, "wb") as f:
                pickle.dump(dataset, f)
    else:
        train_valid_json = f"{dataset_root}/{opts.dataset_name}.json"
        with open(train_valid_json, "r") as f:
            train_valid_json_data = json.load(f)
        if opts.load_set_path is not None and os.path.exists(opts.load_set_path):
            with open(opts.load_set_path, "r") as f:
                set_for_train = sorted(list(json.load(f)))
        else:
            set_for_train = sorted(list(train_valid_json_data.keys()))
            if opts.random_select:
                random.seed(opts.seed)
                random.shuffle(set_for_train)
            if opts.start > 0:
                set_for_train = set_for_train[opts.start:]
            if opts.limit is not None:
                limit_sz = min(opts.limit, len(set_for_train))
                set_for_train = set_for_train[:limit_sz]

        dataset = LegoECADataset(opts, set_for_train, train_valid_json_data)
        if use_dataset_cache:
            with open(dataset_cache_path, "wb") as f:
                pickle.dump(dataset, f)
                print(f"Dataset cache saved to {dataset_cache_path}.")

    set_for_train_path = os.path.join(log_dir, "set_for_train.json")
    with open(set_for_train_path, "w") as f:
        json.dump(set_for_train, f)

    train_set, valid_set = Subset(dataset, dataset.train_idx), Subset(
        dataset, dataset.valid_idx
    )

    num_workers = 0 if opts.debug else config["num_works"]
    train_dataloader = DataLoader(
        train_set,
        batch_size=config["batch_size"],
        collate_fn=collect_fn,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        valid_set,
        batch_size=config["batch_size"],
        collate_fn=collect_fn,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_dataloader, test_dataloader

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser = LegoIncorrectDataset.modify_commandline_options(parser)
#     opts = parser.parse_args()
#
#     for item in dataloader:
#         print(item)
