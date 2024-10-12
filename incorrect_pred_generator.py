import copy
import json
import os
import time
import random

import shutil
import sys
import warnings
from enum import Enum

import cv2
from PIL import Image
from matplotlib import pyplot as plt
from pytorch3d.renderer import look_at_view_transform

from src.datasets.legokps_shape_cond_dataset import (
    img_transform,
    load_rot_symmetry,
    rot_symmetry_idx,
)
from src.tu.train_setup import set_seed
from src.util.common_utils import create_not_exist, list_files, copy2dir

import numpy as np
import pytorch3d.transforms as pt
import torch
import trimesh.transformations as tr
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
import torchvision.transforms.functional as TF
from pycocotools import mask as mask_utils

from src.bricks.brick_info import (
    get_brick_class,
    get_cbrick_keypoint,
    dict_to_cbrick,
    add_cbrick_to_bricks_pc,
    CBrick,
)
from src.datasets import create_dataset
from debug.utils import render_dict_simple, render_dict_simple_one_step, StateFlagNew, StateFlagNew, \
    render_dict_simple_one_step_only_current_brick
from src.models import create_model
from src.models.utils import Meters
from src.options.test_options import TestOptions

warnings.filterwarnings("ignore", category=UserWarning)

SYMMETRY_DICT = None


def add_state_flag(
        bricks_pred, bricks_pred_matched, position_error_set, rotation_error_set
):
    # Add a flag indicating whether the prediction is correct or wrong
    # Replace the original color:
    # - Use green if the prediction is correct,
    # - Use red for position errors,
    # - Use yellow for rotation errors,
    # - Use purple if both position and rotation are wrong.
    for b_pred, b_pred_match in zip(bricks_pred, bricks_pred_matched):
        bid = b_pred_match["bid"]
        if isinstance(bid, np.ndarray):  # Check if bid is a numpy array
            print(bid)
            assert bid.size == 1  # Ensure bid contains only one element
            bid = bid[-1]  # Get the last element of the array
        bid = int(bid)  # Convert bid to an integer
        # Check the error sets and assign the corresponding flag
        if bid not in position_error_set and bid not in rotation_error_set:
            b_pred["state_flag"] = StateFlagNew.CORRECT  # Prediction is correct
        elif bid in position_error_set and bid not in rotation_error_set:
            b_pred["state_flag"] = StateFlagNew.POSITION_ERROR  # Only position is wrong
        elif bid not in position_error_set and bid in rotation_error_set:
            b_pred["state_flag"] = StateFlagNew.ROTATION_ERROR  # Only rotation is wrong
        else:
            b_pred["state_flag"] = StateFlagNew.POSITION_ROTATION_ERROR  # Both position and rotation are wrong
    return bricks_pred


def replace_poses_one_step(opt, pred_manual_step_i, bricks_pred, subm_cbricks=None):
    start_time = time.time()
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
        key_point = bricks_pred[i]["kp"] + [0]
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
    # print(f"replace_poses_one_step: {time.time() - start_time}")
    return b_step


subm_dependency_all = {
    "classics": {
        1: [0],
        4: [3],
        7: [6],
        12: [9, 10, 11],
        17: [16],
    },
    "architecture": {
        4: [0, 1, 2, 3],
        7: [6],
        15: [8, 9, 10, 11, 12, 13, 14],
    },
}

from src.bricks.brick_info import get_cbrick_enc_voxel


def get_brick_occ(opt, cbrick):
    extra_point = None
    if opt.brick_voxel_embedding_dim > 0:
        extra_value = 1
    else:
        extra_value = 2
        extra_point = [0, cbrick.get_height(), 0]
    brick_occ = torch.as_tensor(
        get_cbrick_enc_voxel(
            cbrick, extra_point=extra_point, extra_point_value=extra_value
        )
    )

    if opt.brick_voxel_embedding_dim == 0:
        brick_occ = brick_occ.float()
    else:
        brick_occ = brick_occ.long()
    if opt.crop_brick_occs:
        grid_size = 65
        min_xyz = [32, 16, 32]
        max_xyz = [d + grid_size for d in min_xyz]
        brick_occ = brick_occ[
                    min_xyz[0]: max_xyz[0], min_xyz[1]: max_xyz[1], min_xyz[2]: max_xyz[2]
                    ]
    return brick_occ


from src.bricks.brick_info import BricksPC


def zip_keys(
        exs, features=["trans", "rot_decoded", "bid", "bid_decoded", "kp", "mask"], n=-1
):
    res = []
    if n == -1:
        n = len(exs[features[0]])
    for idx in range(n):
        ex = {}
        for k in features:
            if k in exs:
                ex[k] = exs[k][idx]
                if k != "mask":
                    if isinstance(ex[k], torch.Tensor):
                        ex[k] = ex[k].cpu().tolist()
                    if k == "bid" and isinstance(ex[k], float):
                        ex[k] = int(ex[k])
                else:
                    ex[k] = ex[k].cpu().numpy()
        res.append(ex)
    return res


class EvalOptions(TestOptions):
    def initialize(self, parser):
        parser = TestOptions.initialize(self, parser)

        parser.add_argument(
            "--start_set", type=int, default=0, help="The set start to be evaluated."
        )
        parser.add_argument("--output_pred_json", action="store_true")
        parser.add_argument('--set_load_path', type=str, default=None)
        parser.add_argument("--random", action="store_true")
        parser.add_argument("--output_lpub3d_ldr", action="store_true")
        parser.add_argument("--render_pred_json", action="store_true")
        parser.add_argument("--single_json", action="store_true")
        parser.add_argument("--autoregressive_inference", action="store_true")
        parser.add_argument(
            "--oracle_percentage",
            type=float,
            default=0,
            help="Percentage of steps that are assumed to be true.",
        )
        parser.add_argument(
            "--k",
            type=int,
            default=1,
            help="Maximum number of error generation per step.",
        )
        parser.add_argument(
            "--symmetry_dict_path",
            type=str,
            default="./data/RotationSymmetry.csv",
            help="Path to the rotation symmetry statistics table.",
        )
        return parser


def manual_disturbance(input_images, img_type, k=1):
    # Iterating through input images and adding noise
    new_images = []
    for image in input_images:
        opencv_image = np.array(image)
        if k > 0:
            # Generate Gaussian noise with a mean and stddev based on k
            mean = random.random() * 0.5
            stddev = (k + random.random()) * 0.8  # Increases noise as k increases
            gaussian_noise = np.random.normal(mean, stddev, opencv_image.shape).astype('uint8')
            noisy_image = cv2.add(opencv_image, gaussian_noise)
        else:
            noisy_image = opencv_image
        # plt.imshow(noisy_image)
        # plt.axis('off')
        # plt.savefig("./noise_img.png")
        # plt.show()
        img = Image.fromarray(noisy_image)
        if img_type in ["gray_scale", "laplacian"]:
            img = img.convert("L").convert("RGB")
        img = img_transform(img, img_type)
        new_images.append(img)
    return torch.stack(new_images)


def check_rot_in_symmetry(bid, gt_rotation, pred_rotation):
    assert SYMMETRY_DICT is not None and isinstance(SYMMETRY_DICT, dict)
    if bid not in SYMMETRY_DICT:
        warnings.warn(f"bid: {bid} is not in symmetry dict!")
        return np.allclose(gt_rotation, pred_rotation)
    brick_symmetry_matrix = SYMMETRY_DICT[bid]
    gt_rotation_encode = "_".join(map(str, gt_rotation))
    pred_rotation_encode = "_".join(map(str, pred_rotation))
    assert (
            gt_rotation_encode in rot_symmetry_idx
            and pred_rotation_encode in rot_symmetry_idx
    )
    return brick_symmetry_matrix[rot_symmetry_idx[gt_rotation_encode]][
        rot_symmetry_idx[pred_rotation_encode]
    ]



def step_result_analyse(bricks_gt, bricks_pred_matched):
    # Analyze the predicted results by comparing them with ground truth

    # Iterate over ground truth bricks and matched predicted bricks
    for b_gt, b_pred in zip(bricks_gt, bricks_pred_matched):
        # Assign the ground truth brick ID (bid_decoded) to the predicted brick
        b_pred['bid_decoded'] = b_gt['bid_decoded']
        bid_decoded = b_pred['bid_decoded']

        # Check if rotation is correctly predicted, considering symmetry in the rotation
        # (the original code does not handle rotational symmetry)
        rot_result = check_rot_in_symmetry(bid_decoded, b_gt['rot_decoded'], b_pred['rot_decoded'])

        # Check if the translation is correct using np.allclose to account for floating-point precision issues
        # (for example, predicted trans[0] might be 3.9999999999999996 and ground truth is 4.0)
        trans_result = np.allclose(b_gt['trans'], b_pred['trans'])

        # Align the predicted translation with the ground truth if it's correct
        # (this ensures that slight differences due to precision or rotational symmetry don't cause mismatches)
        if trans_result:
            b_pred['trans'] = b_gt['trans']

        # Determine if both rotation and translation are correct
        if rot_result and trans_result:
            b_pred['correct'] = StateFlagNew.CORRECT  # Mark as correct if both are correct
        else:
            b_pred['correct'] = StateFlagNew.POSITION_ERROR  # Default to position error
            if not rot_result:
                b_pred['correct'] = StateFlagNew.ROTATION_ERROR  # Mark as rotation error if rotation is incorrect
            # If both position and rotation are wrong, mark as position and rotation error
            if not trans_result and b_pred['correct'] == StateFlagNew.ROTATION_ERROR:
                b_pred['correct'] = StateFlagNew.POSITION_ROTATION_ERROR

    # Return the modified predicted bricks with correctness flags
    return bricks_pred_matched

def show_points(points, is_coord=False):
    # Get the coordinates of elements that are True in the tensor
    if not is_coord:
        # If not coordinate data, convert to coordinates of True elements
        point_cloud = torch.nonzero(points, as_tuple=False)
    else:
        point_cloud = points  # Use the coordinates directly

    point_cloud = point_cloud.numpy()  # Convert to NumPy array

    # Create a 3D figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Visualize the point cloud data
    ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], s=1)

    # Set axis labels
    ax.set_xlabel("X Label")
    ax.set_ylabel("Y Label")
    ax.set_zlabel("Z Label")

    # Show the plot
    plt.show()


def get_pred_obj_occ(prev_obj_occ, bricks_pred, opt, x_input, targets, show=False):
    start_time = time.time()
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

    points = torch.nonzero(current_occ, as_tuple=False).float()
    if show:
        # Define rotation matrices
        theta_x = torch.tensor([[1, 0, 0],
                                [0, 0, -1],
                                [0, 1, 0]]).float()
        theta_z = torch.tensor([[-1, 0, 0],
                                [0, -1, 0],
                                [0, 0, 1]]).float()

        # Transpose points to shape (3, N)
        point_cloud = points.t()

        # Apply rotations
        rotated_point_cloud = torch.mm(theta_z, torch.mm(theta_x, point_cloud))
        rotated_point_cloud = rotated_point_cloud.t()

        # Visualize the rotated point cloud
        show_points(rotated_point_cloud, True)

    indices = points.numpy()

    return indices


def render_pred_img(step_id, manual_pred, title=None):
    rendered_raw_img, rendered_state_img = render_dict_simple_one_step(
        manual_pred, step_id, no_check=True, state_img=True
    )

    rendered_brick_img_list = render_dict_simple_one_step_only_current_brick(manual_pred, step_id, no_check=True)

    return {
        "pred_img_name": f"{step_id:03d}_predict.png",
        "pred_img": rendered_raw_img,
        "pred_brick_img_list": rendered_brick_img_list,
        "state_img": rendered_state_img,
        "gt_img_name": f"{step_id:03d}.png",
    }


def out_pred_info(result_list_this, manual_GT, output_dir, set_path):
    base, set_name = os.path.split(set_path)
    # Setting up output directory and copying files from set_path

    out_dir_with_set_path = f"{output_dir}/{set_name}"
    create_not_exist(out_dir_with_set_path)

    file_list = list_files(set_path)
    copy2dir(file_list, out_dir_with_set_path) # Copy ground truth files

    GT_operations = manual_GT["operations"]
    new_operations = {}
    pred_obj_occ = {}
    position_rotation_set = set()
    new_inner_id = 0
    pre_step_id = 0
    # Iterating over prediction results and updating operations

    for result in result_list_this:
        step_id = result["step_id"]
        if pre_step_id != step_id:
            pre_step_id = step_id
            new_inner_id = 0
        # For each result, modify manual_GT's operations for the step
        new_operations_id = f"{step_id}_{new_inner_id}"
        GT_operation_with_step_id = copy.deepcopy(GT_operations[str(step_id)])
        position_list = []
        rotation_list = []
        correct_cnt = 0
        for brick_pred in result["bricks_pred"]:
            state_flag: StateFlagNew = brick_pred["correct"]
            if state_flag == StateFlagNew.CORRECT:
                correct_cnt += 1
            brick_pred["correct"] = [state_flag.name, state_flag.value]
            position_list.append(brick_pred["brick_transform"]["position"])
            rotation_list.append(brick_pred["brick_transform"]["rotation"])
        if len(position_list) > 1:
            position_data = np.array(position_list)
            rotation_data = np.array(rotation_list)
            position_sorted = map(str,
                                  position_data[np.lexsort((position_data[:, 2], position_data[:, 1],
                                                            position_data[:, 0]))].flatten().tolist())
            rotation_sorted = map(str,
                                  rotation_data[np.lexsort((rotation_data[:, 3],
                                                            rotation_data[:, 2],
                                                            rotation_data[:, 1],
                                                            rotation_data[:, 0],))].flatten().tolist())
        else:
            position_sorted = map(str, position_list[0])
            rotation_sorted = map(str, rotation_list[0])
        position_rotation_encode = (
            f"ID_{step_id}@P_{','.join(position_sorted)}@R_{','.join(rotation_sorted)}"
        )
        if position_rotation_encode in position_rotation_set:
            # Skip if duplicate
            continue
        else:
            position_rotation_set.add(position_rotation_encode)

        GT_operation_with_step_id["bricks"] = result["bricks_pred"]
        new_operations[new_operations_id] = GT_operation_with_step_id

        # Render and save predicted images
        render_img_dict = result["render_img"]
        render_img = render_img_dict["pred_img"]
        render_img_name = render_img_dict["pred_img_name"]
        render_img_name = render_img_name.replace("_predict.", f"_{new_inner_id}_pred.")
        render_img.save(os.path.join(out_dir_with_set_path, render_img_name))
        pred_brick_img_list = render_img_dict['pred_brick_img_list']
        # Save individual brick images
        for brick_i, brick_img in enumerate(pred_brick_img_list):
            brick_img_path = render_img_name.replace("pred.png", f"pred_brick_{brick_i}.png")
            brick_img.save(os.path.join(out_dir_with_set_path, brick_img_path))

        # Save predicted point cloud data
        pred_obj_occ[new_operations_id] = result["pred_obj_occ"].tolist()
        new_inner_id += 1

    manual_GT["operations"] = new_operations
    json_path = os.path.join(out_dir_with_set_path, "pred_info.json")
    with open(json_path, "w") as file:
        json.dump(manual_GT, file, indent=4)
    point_clouds_path = os.path.join(out_dir_with_set_path, "pred_point_clouds.json")
    with open(point_clouds_path, "w") as file:
        json.dump(pred_obj_occ, file, indent=4)


def infer_set(
        opt,
        model,
        set_path,
        output_dir,
        is_subm=False,
        subm_cbricks=None,
        subm_meters=None,
        manual_GT=None,
):
    result_list = []
    opt.load_set = set_path
    opt.serial_batches = True

    opt.batch_size = 1

    # meters_this = Meters()
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options

    print(f"Loading from {set_path}, number of steps {len(dataset)}")

    k_times = opt.k
    for i, input_data in tqdm(enumerate(dataset), total=len(dataset) // opt.batch_size):
        x_input, targets = input_data
        raw_imgs = x_input["img"]
        for inner_id in range(k_times):

            manual_pred = copy.deepcopy(manual_GT)
            x_input["img"] = manual_disturbance(raw_imgs, opt.img_type, k=inner_id)
            model.set_input(input_data)
            model.test()
            assert x_input["img"].shape[0] == 1
            for j in range(x_input["img"].shape[0]):
                target = targets[j]["ordered"]
                num_bricks = len(target["bid"])
                bricks_gt = zip_keys(target, n=num_bricks)
                detection = model.detections[j]
                idxs = detection["bid"].argsort().cpu().numpy()
                for k, v in detection.items():
                    if isinstance(v, (torch.Tensor, np.ndarray)):
                        detection[k] = detection[k][idxs]
                    else:
                        detection[k] = [detection[k][idx] for idx in idxs]
                bid_ct = target["bid_ct"]
                bricks_pred = zip_keys(detection, n=num_bricks)
                bricks_pred_matched = []
                brick_ct = 0
                # Perform Hungarian matching
                for _, ct in bid_ct:
                    trans_gt = np.array(target["trans"])[brick_ct: brick_ct + ct, None]
                    trans_pred = np.array(detection["trans"])[
                                 None, brick_ct: brick_ct + ct
                                 ]
                    trans_mat = ((trans_gt - trans_pred) ** 2).sum(axis=-1)
                    trans_mat = (trans_mat < 0.1).astype(np.int8)
                    _, pred_idxs = linear_sum_assignment(-trans_mat)
                    for idx in pred_idxs:
                        bricks_pred_matched.append(bricks_pred[brick_ct + idx])
                    brick_ct += ct

                img_path = x_input["img_path"][j]
                img_fname = os.path.basename(img_path)
                if "_" in img_fname:
                    step_id = int(img_fname[:-9])
                else:
                    step_id = int(img_fname[:-4])

                set_id = img_path.split("/")[-2:]
                brick_pred_result_list = step_result_analyse(
                    bricks_gt, bricks_pred_matched
                )
                bricks_pred_result_recovered = [
                    brick_pred_result_list[idx] for idx in target["reverse_idxs"]
                ]
                current_result = {
                    "set_id": set_id,
                    "step_id": step_id,
                    "inner_id": inner_id,
                    "bricks_pred": replace_poses_one_step(
                        opt,
                        manual_pred["operations"][str(step_id)],
                        bricks_pred_result_recovered,
                        subm_cbricks=subm_cbricks,
                    ),
                    "render_img": render_pred_img(step_id, manual_pred),
                    "pred_obj_occ": get_pred_obj_occ(
                        x_input["obj_occ_prev"][0],
                        bricks_pred_result_recovered,
                        opt,
                        x_input,
                        targets,
                        show=False,
                    ),
                }

                result_list.append(current_result)

    if is_subm:
        return result_list, None
    return result_list


def main():
    opt = EvalOptions().parse()  # get training options
    opt.serial_batches = True
    opt.eval_mode = True
    global SYMMETRY_DICT
    SYMMETRY_DICT = load_rot_symmetry(opt.symmetry_dict_path)
    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers
    model.eval()
    output_dir = os.path.join(opt.results_dir, opt.dataset_alias)
    create_not_exist(output_dir)

    data_dir = os.path.abspath(opt.dataroot)
    if opt.set_load_path is not None and os.path.exists(opt.set_load_path):
        with open(opt.set_load_path, "r") as f:
            set_paths = list(sorted(json.load(f)))
    else:
        set_paths = list(sorted(os.listdir(data_dir)))

        for n in ["occs", "metadata"]:
            if n in set_paths:
                set_paths.remove(n)
    if opt.start_set > 0:
        set_paths = set_paths[opt.start_set:]
    if opt.random:
        random.shuffle(set_paths)
    if opt.n_set >= 0:
        set_paths = set_paths[: opt.n_set]

    from collections import defaultdict

    subm_cbricks_d = defaultdict(list)
    subm_meters_d = defaultdict(Meters)
    if "architecture" in opt.dataset_alias:
        subm_dependency = subm_dependency_all["architecture"]
    elif "classics" in opt.dataset_alias:
        subm_dependency = subm_dependency_all["classics"]
    else:
        subm_dependency = None
        subm_cbricks_d = None
        subm_meters_d = None

    subm_rev_map = {}
    if subm_dependency is not None:
        for k, subms in subm_dependency.items():
            for s in subms:
                subm_rev_map[s] = k

    for s in tqdm(set_paths):
        try:
            base, dir = os.path.split(s)
            set_id = int(dir)
            gt_json_path = os.path.join(data_dir, s, "info.json")
            set_out = f"{output_dir}/{dir}"
            if os.path.exists(set_out):
                print(f"skipping {set_out}...")
                continue
            else:
                create_not_exist(set_out)
            with open(gt_json_path) as f:
                manual_GT = json.load(f)
            if subm_dependency is not None:
                if set_id in subm_rev_map.keys():
                    main_id = subm_rev_map[set_id]
                    result_list_this, subm_cbrick = infer_set(opt, model, s, output_dir, is_subm=True,
                                                              manual_GT=manual_GT)
                    # This set corresponds to a submodule and will be leveraged in other sets.
                    subm_cbricks_d[main_id].append(subm_cbrick)
                    # only in this set a submodule is used in multiple steps
                    if "architecture" in opt.dataset_alias and main_id == 7:
                        subm_cbricks_d[main_id].append(subm_cbrick)
                else:
                    if set_id in subm_dependency.keys():
                        result_list_this = infer_set(
                            opt,
                            model,
                            s,
                            output_dir,
                            subm_cbricks=subm_cbricks_d[set_id],
                            subm_meters=subm_meters_d[set_id],
                            manual_GT=manual_GT,
                        )
                    else:
                        result_list_this = infer_set(opt, model, s, output_dir, manual_GT=manual_GT)
            else:
                result_list_this = infer_set(opt, model, s, output_dir, manual_GT=manual_GT)

            out_pred_info(result_list_this, manual_GT, output_dir, os.path.join(data_dir, s))
        except Exception as e:
            print(e)
            continue


if __name__ == "__main__":
    print("command line:", " ".join(sys.argv))
    set_seed(0)
    main()
