import copy
import csv
import json
import os

from PIL import Image

from correct_bricks import SCANetOptions, init_SCANet, correct_bricks, replace_poses_one_step
import sys
import warnings

from src.datasets.legokps_shape_cond_dataset import (
    img_transform,
    load_rot_symmetry,
    rot_symmetry_idx,
)
from src.tu.train_setup import set_seed
from src.util.common_utils import create_not_exist, list_files, copy2dir, get_filename

import numpy as np
import pytorch3d.transforms as pt
import torch
import trimesh.transformations as tr
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from src.bricks.brick_info import (
    get_brick_class,
    get_cbrick_keypoint,
    dict_to_cbrick,
    add_cbrick_to_bricks_pc,
    CBrick, get_brick_enc_pc_, get_cbrick_enc_pc,
)
from src.datasets import create_dataset
from debug.utils import render_dict_simple, render_dict_simple_one_step, StateFlagNew, \
    render_dict_simple_one_step_only_current_brick
from src.lego.utils.camera_utils import get_cameras, get_scale
from src.lego.utils.inference_utils import recompute_conns
from src.models import create_model
from src.models.utils import Meters
from src.options.test_options import TestOptions
from src.util.util import mkdirs

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

warnings.filterwarnings("ignore", category=UserWarning)

SYMMETRY_DICT = None

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


def replace_poses(opt, d, pose_list, replace_per_step=True, subm_cbricks=None):
    n_steps = len(d['operations'])
    b_steps_orig = [copy.deepcopy(d['operations'][str(i)]['bricks']) for i in range(n_steps)]
    subm_ct = 0
    for ex in pose_list:
        b_step = d['operations'][str(ex['step_id'])]['bricks']
        bricks_pred = ex['bricks_pred']
        has_subm = False
        for i, b in enumerate(b_step):
            rot_euler = bricks_pred[i]['rot_decoded']
            rot_quat = tr.quaternion_from_euler(*list(map(lambda x: x * np.pi / 180, rot_euler)))
            position = np.array(bricks_pred[i]['trans'])
            if 'brick_type' in b:
                brick_type = b['brick_type']
                kp_offset = [0, get_brick_class(brick_type).get_height(), 0]
            else:
                if subm_cbricks is not None:
                    cbrick = subm_cbricks[subm_ct]
                    has_subm = True
                    if cbrick is not None:
                        b['canonical_state'] = cbrick.to_dict()
                    else:
                        cbrick = dict_to_cbrick(b)
                else:
                    cbrick = dict_to_cbrick(b)
                kp_offset = get_cbrick_keypoint(cbrick, policy='brick' if opt.cbrick_brick_kp else 'simple')[
                                0] - cbrick.position

            if opt.top_center:
                position -= kp_offset
            b['brick_transform']['position'] = list(map(float, position))
            b['brick_transform']['rotation'] = list(map(float, rot_quat))
        if has_subm:
            subm_ct += 1

    ks = list(d.keys())
    ks.remove('operations')
    d_template = {k: d[k] for k in ks}
    ds = []
    if replace_per_step:
        for i in range(n_steps):
            d_this = copy.deepcopy(d_template)
            if i > 0:
                d['operations'][str(i - 1)]['bricks'] = b_steps_orig[i - 1]
            d_this['operations'] = copy.deepcopy({str(idx): d['operations'][str(idx)] for idx in range(i + 1)})
            ds.append(d_this)
        return ds
    else:
        return d


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


def infer_set(opt, model, set_path, manual_GT=None, is_subm=False, subm_cbricks=None,
              subm_meters=None):
    # Initialize lists for storing images and results

    manual_img_list = []
    before_img_list = []
    after_img_list = []
    result_list = []
    opt.load_set = set_path
    opt.serial_batches = True
    autoregressive = opt.autoregressive_inference

    if autoregressive:
        # If autoregressive, set batch size to 1 and initialize bricks point cloud

        opt.batch_size = 1
        current_bricks_pc = BricksPC(grid_size=(65, 65, 65), record_parents=False)
        if subm_cbricks is not None:
            subm_ct = 0

    meters_this = Meters()
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options

    print(f"Loading from {set_path}, number of steps {len(dataset)}")
    if is_subm:
        oracle_ct = 0  # disable oracle in subm
    else:
        oracle_ct = len(dataset) * opt.oracle_percentage

    for i, input_data in tqdm(enumerate(dataset), total=len(dataset) // opt.batch_size):
        manual_img_list.append(Image.open(input_data[0]['img_path'][-1]))

        if i > oracle_ct and autoregressive:
            input_data[0]['obj_occ_prev'] = torch.as_tensor(current_bricks_pc.get_occ_with_rotation()[0])[None]

            input_data[0]['bricks'] = [copy.deepcopy(current_bricks_pc)]
            transforms = pt.Transform3d().translate(-input_data[0]['obj_center']).scale(
                input_data[0]['obj_scale'][0]).cuda()
            cameras = get_cameras(azim=input_data[0]['azim'][0], elev=input_data[0]['elev'][0])
            input_data[1][0]['conns'] = recompute_conns(
                current_bricks_pc, op_type=0, transforms=transforms, cameras=cameras, scale=get_scale())

        if subm_cbricks is not None:
            # Use the predicted submodules as input
            for j, bid_counter in enumerate((input_data[0]['bid_counter'])):
                has_subm = False
                for k, (bid, ct) in enumerate(bid_counter):
                    if bid < 0:
                        has_subm = True
                        input_data[0]['brick_occs'][j][k] = get_brick_occ(opt, subm_cbricks[subm_ct])
                        input_data[0]['cbrick'][j][-bid - 1] = subm_cbricks[subm_ct]

                if has_subm:
                    subm_ct += 1
        x_input, targets = input_data
        manual_pred = copy.deepcopy(manual_GT)
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

            result_list.append({
                'set_id': set_id,
                'step_id': step_id,
                'bricks_pred': [
                    bricks_pred_matched[idx] for idx in target['reverse_idxs']
                ] if i > oracle_ct or (not autoregressive) else [
                    bricks_gt[idx] for idx in target['reverse_idxs']
                ]
            })
            # Analyze the step result to check if each brick is correct
            brick_pred_result_list = step_result_analyse(
                bricks_gt, bricks_pred_matched
            )
            bricks_pred_result_recovered = [
                brick_pred_result_list[idx] for idx in target["reverse_idxs"]
            ]  # Recover brick order
            bricks_gt_result_recovered = [
                bricks_gt[idx] for idx in target["reverse_idxs"]
            ]
            # Replace positions for one step and render the result
            replace_poses_one_step(opt, manual_pred["operations"][str(step_id)],
                                   bricks_pred_result_recovered, subm_cbricks=subm_cbricks)
            before_img = render_dict_simple_one_step(step_id=step_id, manual_pred=manual_pred)
            before_img_list.append(before_img)
            if opt.without_correct:
                brickwise_correct_ct = 0
                all_correct = True
                # FOR CORRECT
                for b_gt, b_pred, in zip(bricks_gt_result_recovered, bricks_pred_result_recovered):
                    b_pred['bid_decoded'] = b_gt['bid_decoded']
                    bid_decoded = b_pred['bid_decoded']
                    rot_result = check_rot_in_symmetry(bid_decoded, b_gt['rot_decoded'], b_pred['rot_decoded'])
                    if rot_result and np.allclose(b_gt['trans'], b_pred['trans']):
                        brickwise_correct_ct += 1
                    else:
                        all_correct = False

                meters_this.update('brickwise_acc', brickwise_correct_ct, num_bricks)
                meters_this.update('stepwise_acc', int(all_correct), 1)
            else:
                # Process model output with corrections and update metrics

                device = model.device
                pre_decoded_out_list = bricks_pred_result_recovered
                manual_pred["operations"][str(step_id)] = copy.deepcopy(manual_GT["operations"][str(step_id)])
                correct_brick_out_list = correct_bricks(opt, device, img_path, manual_pred, step_id, x_input,
                                                        targets, pre_decoded_out_list)
                # Adjust the predicted bricks based on corrections
                for correct_brick_out, pre_brick_out in zip(correct_brick_out_list, pre_decoded_out_list):
                    conf = correct_brick_out['confidence']
                    if correct_brick_out['brick_status'] == StateFlagNew.CORRECT.value or \
                            (correct_brick_out[
                                 'brick_status'] != StateFlagNew.CORRECT.value and conf <= opt.correct_conf):
                        correct_brick_out['rotation'] = pre_brick_out['rot_decoded']
                        correct_brick_out['position'] = pre_brick_out['trans']
                    else:
                        # Adjust the predicted bricks based on corrections
                        if correct_brick_out['brick_status'] == StateFlagNew.POSITION_ERROR.value:
                            correct_brick_out['rotation'] = pre_brick_out['rot_decoded']
                        elif correct_brick_out['brick_status'] == StateFlagNew.ROTATION_ERROR.value:
                            correct_brick_out['position'] = pre_brick_out['trans']

                    pre_brick_out['trans'] = correct_brick_out['position']
                    pre_brick_out['rot_decoded'] = correct_brick_out['rotation']

                brickwise_correct_ct = 0
                all_correct = True
                error_brick_ct = 0
                correct_brick_ct = 0
                e2c_ct = 0  # Error to correct count
                c2e_ct = 0 # Correct to error count
                # FOR CORRECT
                for b_gt, b_pred, b_correct in zip(bricks_gt_result_recovered,
                                                   bricks_pred_result_recovered, correct_brick_out_list):
                    b_pred['bid_decoded'] = b_gt['bid_decoded']
                    bid_decoded = b_gt['bid_decoded']

                    if b_pred['correct'] != StateFlagNew.CORRECT:
                        error_brick_ct += 1
                        raw_brick_state = 0  # 0 represent wrong, 1 represent correct
                    else:
                        correct_brick_ct += 1
                        raw_brick_state = 1
                    rot_result = check_rot_in_symmetry(bid_decoded, b_gt['rot_decoded'], b_correct['rotation'])

                    if rot_result and np.allclose(b_gt['trans'], b_correct['position']):
                        brickwise_correct_ct += 1
                        if raw_brick_state == 0:
                            e2c_ct += 1  # success correcting the error brick
                    else:
                        if raw_brick_state == 1:
                            c2e_ct += 1  # The correct thing was mistakenly corrected.
                        all_correct = False
                    if not np.allclose(b_pred['rot_decoded'], b_correct['rotation']) or not np.allclose(
                            b_pred['trans'], b_correct['position']):
                        print("ok")
                    b_pred['rot_decoded'] = b_correct['rotation']
                    b_pred['trans'] = b_correct['position']

                meters_this.update('brickwise_acc', brickwise_correct_ct, num_bricks)
                meters_this.update('stepwise_acc', int(all_correct), 1)
                meters_this.update('CR', e2c_ct, error_brick_ct)
                meters_this.update('MPR', c2e_ct, correct_brick_ct)
            replace_poses_one_step(opt, manual_pred["operations"][str(step_id)],
                                   bricks_pred_result_recovered, subm_cbricks=subm_cbricks)
            after_img = render_dict_simple_one_step(step_id=step_id, manual_pred=manual_pred)
            after_img_list.append(after_img)

            if autoregressive:
                bs = result_list[-1]['bricks_pred']
                for i, b in enumerate(bs):
                    rot_quat = tr.quaternion_from_euler(*list(map(lambda x: x * np.pi / 180, b['rot_decoded'])))
                    if b['bid'] >= 0:
                        kp_offset = np.array([0, get_brick_class(b['bid_decoded']).get_height(), 0])
                        current_bricks_pc.add_brick(b['bid_decoded'], b['trans'] - kp_offset, rot_quat,
                                                    op_type=targets[0]['op_type'][i].cpu().numpy(), no_check=True)
                    else:
                        cbrick_canonical = x_input['cbrick'][0][int(-b['bid']) - 1]
                        kp_offset = \
                            get_cbrick_keypoint(cbrick_canonical, policy='brick' if opt.cbrick_brick_kp else 'simple')[
                                0] - cbrick_canonical.position
                        cbrick_this = copy.deepcopy(cbrick_canonical)
                        cbrick_this.position = b['trans'] - kp_offset
                        cbrick_this.rotaiton = rot_quat
                        assert add_cbrick_to_bricks_pc(current_bricks_pc, cbrick_this, op_type=targets[0]['op_type'],
                                                       no_check=True)

    n_steps = meters_this.n_d['stepwise_acc']
    if subm_cbricks is not None:
        n_steps += subm_meters.n_d['stepwise_acc']
    mtc = n_steps - meters_this.sum_d['stepwise_acc']
    if subm_cbricks is not None:
        mtc -= subm_meters.sum_d['stepwise_acc']

    meters_this.update('mtc_raw', mtc)
    meters_this.update('mtc_norm', mtc / n_steps)
    meters_this.update('CR_norm', meters_this.avg("CR"))
    meters_this.update('MPR_norm', meters_this.avg("MPR"))
    meters_this.update('setwise_acc', int(mtc == 0))

    if is_subm:
        if autoregressive:
            cbrick = CBrick(current_bricks_pc, [0, 0, 0], [1, 0, 0, 0])
        else:
            cbrick = None
        return cbrick, meters_this, result_list, before_img_list, after_img_list, manual_img_list

    return meters_this, result_list, before_img_list, after_img_list, manual_img_list


def main(opt):
    seed = opt.seed
    set_seed(seed)

    if not opt.without_correct:
        base, checkpoint_file = os.path.split(opt.checkpoint_path)
        train_root = os.path.dirname(base)
        init_SCANet(opt)
    else:
        checkpoint_file = "MEPNet_raw"
        train_root = opt.checkpoint_path
        create_not_exist(train_root)

    opt.serial_batches = True
    opt.eval_mode = True
    global SYMMETRY_DICT
    SYMMETRY_DICT = load_rot_symmetry(opt.symmetry_dict_path)
    assert len(SYMMETRY_DICT) > 0
    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers
    model.eval()

    output_dir = os.path.join(opt.results_dir, opt.dataset_alias, 'correction')
    result_dir = os.path.join(opt.results_dir, opt.name, opt.dataset_alias)
    mkdirs(result_dir)
    create_not_exist(output_dir)

    data_dir = os.path.abspath(opt.dataroot)
    if opt.set_for_test:
        set_for_train_path = f"{train_root}/set_for_train.json"
        with open(set_for_train_path, 'r') as file:
            set_for_test = json.load(file)
    elif opt.set_for_test_path is not None and os.path.exists(opt.set_for_test_path):
        with open(opt.set_for_test_path, 'r') as f:
            set_for_test = json.load(f)
    else:
        set_for_test = list(sorted(os.listdir(data_dir)))
    set_paths = list(sorted(set_for_test))
    if opt.start_set > 0:
        set_paths = set_paths[opt.start_set:]
    if opt.n_set > 0:
        set_paths = set_paths[:opt.n_set]
    for n in ["occs", "metadata"]:
        if n in set_paths:
            set_paths.remove(n)

    meters = Meters()
    print_keys = ['brickwise_acc_sum', 'brickwise_acc_n', 'brickwise_acc', 'stepwise_acc_sum', 'stepwise_acc_n',
                  'stepwise_acc', 'mtc_norm', "CR_norm", "MPR_norm", "CR", "MPR"]

    from collections import defaultdict

    subm_cbricks_d = defaultdict(list)
    subm_meters_d = defaultdict(Meters)
    if 'architecture' in opt.dataset_alias:
        subm_dependency = subm_dependency_all['architecture']
    elif 'classics' in opt.dataset_alias:
        subm_dependency = subm_dependency_all['classics']
    else:
        subm_dependency = None
        subm_cbricks_d = None
        subm_meters_d = None

    subm_rev_map = {}
    if subm_dependency is not None:
        for k, subms in subm_dependency.items():
            for s in subms:
                subm_rev_map[s] = k

    csv_name = f"eval_result@{set_paths[0]}-{set_paths[-1]}"
    info_dataset_dir_path = f"{result_dir}/dataset"
    if opt.set_for_test_path is not None and os.path.exists(opt.set_for_test_path):
        csv_name += f"@{os.path.basename(opt.set_for_test_path).split('.')[0]}@iter_{opt.correct_iter_num}"
        info_dataset_dir_path += f"@{os.path.basename(opt.set_for_test_path).split('.')[0]}@iter_{opt.correct_iter_num}"

    if opt.without_correct:
        csv_name += "@without_correct"
    csv_path = f"{result_dir}/{csv_name}.csv"
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["set_id"] + print_keys + ["checkpoint"])
    metric_list = []
    for s in tqdm(set_paths):
        gt_json_path = os.path.join(data_dir, s, "info.json")
        with open(gt_json_path) as f:
            manual_GT = json.load(f)
        _, dir_name = os.path.split(s)
        set_id = int(dir_name)
        if subm_dependency is not None:
            if set_id in subm_rev_map.keys():
                main_id = subm_rev_map[set_id]
                (subm_cbrick, metric_this_step, result_list, before_img_list,
                 after_img_list, manual_img_list) = infer_set(opt, model, s, is_subm=True)
                # This set corresponds to a submodule and will be leveraged in other sets.
                subm_cbricks_d[main_id].append(subm_cbrick)
                # only in this set a submodule is used in multiple steps
                if 'architecture' in opt.dataset_alias and main_id == 7:
                    subm_cbricks_d[main_id].append(subm_cbrick)
                subm_meters_d[main_id].merge_from(metric_this_step)
            else:
                if set_id in subm_dependency.keys():
                    (metric_this_step, result_list, before_img_list,
                     after_img_list, manual_img_list) = infer_set(opt, model, s,
                                                                  subm_cbricks=subm_cbricks_d[set_id],
                                                                  subm_meters=subm_meters_d[set_id],
                                                                  manual_GT=manual_GT)
                else:
                    (metric_this_step, result_list, before_img_list,
                     after_img_list, manual_img_list) = infer_set(opt, model, s, manual_GT=manual_GT)
        else:
            (metric_this_step, result_list, before_img_list,
             after_img_list, manual_img_list) = infer_set(opt, model, s, manual_GT=manual_GT)

        subm_cbricks = None if subm_dependency is None else subm_cbricks_d[set_id]
        d_new = replace_poses(opt, manual_GT, result_list, replace_per_step=False, subm_cbricks=subm_cbricks)
        info_json_dir_path = os.path.join(info_dataset_dir_path, s)
        create_not_exist(info_json_dir_path)
        json_path = os.path.join(info_json_dir_path, 'info.json')
        d_new['gt_json_path'] = gt_json_path
        with open(json_path, 'w') as f:
            json.dump(d_new, f, indent=4)

        metric_dict = metric_this_step.get_dict()
        csv_data_result = [str(s)]
        for k in print_keys:
            if k in metric_dict:
                csv_data_result.append(metric_dict[k])
            else:
                csv_data_result.append("None")

        csv_data_result.append(checkpoint_file)
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(csv_data_result)
        metric_list.append(metric_this_step)

    for metric_this_step in metric_list:
        meters.merge_from(metric_this_step)
    print(f"csv result saved to {csv_path}")
    metric_dict = meters.get_dict()
    csv_data_result = ["total:"]
    for k in print_keys:
        if k in metric_dict:
            csv_data_result.append(metric_dict[k])
        else:
            csv_data_result.append("None")
    if opt.visualize:
        # visualize results
        img_dir_path = os.path.join(info_dataset_dir_path, s)
        if opt.without_correct:
            img_dir_path += "@without_correct"
        create_not_exist(img_dir_path)
        for i, (before_img, after_img, manual_img) in enumerate(zip(before_img_list, after_img_list, manual_img_list)):
            img_fname = f'{i:03d}.png'
            manual_img.save(os.path.join(img_dir_path, img_fname), overwrite=True)
            img_fname = img_fname.replace('.png', '_before.png')
            before_img.save(os.path.join(img_dir_path, img_fname))
            img_fname = f'{i:03d}.png'
            img_fname = img_fname.replace('.png', '_after.png')
            after_img.save(os.path.join(img_dir_path, img_fname))
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['-------------', '-------------', '-------------', '-------------', '-------------'])
        writer.writerow(csv_data_result + [checkpoint_file])

    for name, v in meters.avg_dict().items():
        if name in print_keys:
            print('test/' + name, ':', v)


if __name__ == "__main__":
    print("command line:", " ".join(sys.argv))
    opt = SCANetOptions().parse()  # get training options
    main(opt)
