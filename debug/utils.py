import json
import os
import pickle
import random
import time
from copy import deepcopy
from enum import Enum
from functools import partial
from typing import Union, List

import numpy as np
import pytorch3d.transforms as pt
import torch
import trimesh
import trimesh.transformations as tr
from PIL import Image
from matplotlib import pyplot as plt
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVOrthographicCameras, TexturesVertex,
)
from pytorch3d.structures import join_meshes_as_scene, Meshes

from trimesh.collision import CollisionManager
from pytorch3d.ops.iou_box3d import box3d_overlap
from trimesh.proximity import signed_distance

from src.bricks.brick_info import BricksPC, Brick, HBrick, VBrick, CBrick, add_cbrick_to_bricks_pc, dict_to_cbrick
from common import DEBUG_DIR
from src.datasets.utils import bricks2meshes, brick2p3dmesh, render_lego_scene, get_brick_masks, highlight_edge, \
    transform_mesh, pymesh2trimesh

from src.lego.utils.camera_utils import get_cameras
from src.lego.utils.data_generation_utils import flatten_nested_list, unflatten_nested_list


# import pytorch3d.renderer.cameras as prc


class StateFlagNew(Enum):
    CORRECT = 0
    POSITION_ERROR = 1
    ROTATION_ERROR = 2
    POSITION_ROTATION_ERROR = 3


class StateFlag(Enum):
    CORRECT = 0
    ERROR = 1
    POSITION_ERROR = 2
    ROTATION_ERROR = 3
    POSITION_ROTATION_ERROR = 4


def sample_colors(n, allow_repeats):
    colors = []

    for j in range(n):
        while True:
            rgb = [np.random.randint(0, 256) for _ in range(3)]
            if allow_repeats or rgb not in colors:
                break
        colors.append(rgb)

    return colors


def get_cam_params(mesh):
    # elev, azim = 40, -35
    elev, azim = 30, 225
    # R, T = look_at_view_transform(dist=2000, elev=elev, azim=azim, at=((0, 0, 0),))
    # cameras = FoVOrthographicCameras(device=mesh.device, R=R, T=T,
    #                                  scale_xyz=[(0.0024, 0.0024, 0.0024)])
    cameras = get_cameras(elev=elev, azim=azim)
    bbox = mesh.get_bounding_boxes()[0]
    center = (bbox[:, 1] + bbox[:, 0]) / 2
    bbox_oct = torch.cartesian_prod(bbox[0], bbox[1], bbox[2])
    screen_points = cameras.get_full_projection_transform().transform_points(bbox_oct)[:, :2]
    min_screen_points = screen_points.min(dim=0).values
    max_screen_points = screen_points.max(dim=0).values
    size_screen_points = max_screen_points - min_screen_points
    margin = 0.05
    scale_screen_points = (2 - 2 * margin) / size_screen_points
    return scale_screen_points.min().item(), center


def visualize_bricks(bricks: List[Brick], highlight=False, adjust_camera=True):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def expand_cbrick(brick):
        if isinstance(brick, Brick):
            return brick
        else:
            assert isinstance(brick, CBrick)
            return brick.bricks_raw

    bricks_template = list(map(expand_cbrick, bricks))
    flat_bricks = flatten_nested_list(bricks_template, bricks_template)

    colors = sample_colors(len(flat_bricks), allow_repeats=True)
    mask_colors = sample_colors(len(bricks), allow_repeats=False)

    colors = unflatten_nested_list(colors, bricks_template)

    brick_meshes = bricks2meshes(bricks, colors)

    if adjust_camera:
        obj_scale, obj_center = get_cam_params(join_meshes_as_scene(brick_meshes))
        transform = pt.Transform3d().translate(*(-obj_center)).scale(obj_scale).cuda()
        brick_meshes = list(map(partial(transform_mesh, transform=transform), brick_meshes))

    R, T = look_at_view_transform(dist=2000, elev=30, azim=225, at=((0, 0, 0),))
    cameras = FoVOrthographicCameras(device=device, R=R, T=T, scale_xyz=[[0.0024, 0.0024, 0.0024]])
    # cameras = prc.OpenGLOrthographicCameras(device=device, R=R, T=T, scale_xyz=[(0.0024,) * 3, ])

    mesh = join_meshes_as_scene(brick_meshes)

    image, depth_map = render_lego_scene(mesh, cameras)
    image[:, :, :, 3][image[:, :, :, 3] == 0] = 0
    image[:, :, :, 3][image[:, :, :, 3] > 0] = 1
    image_pil = Image.fromarray((image[0, :, :].detach().cpu().numpy() * 255).astype(np.uint8))
    image_pil = image_pil.resize((512, 512))

    if highlight:
        mask_brick_meshes = [brick2p3dmesh(bricks[i], mask_colors[i]).to(device) for i in range(len(bricks))]
        mask_brick_meshes = list(map(partial(transform_mesh, transform=transform), mask_brick_meshes))
        masks_step, image_shadeless = get_brick_masks(mask_brick_meshes, mask_colors, range(len(bricks)), cameras)
        image_pil = highlight_edge(image_pil, depth_map, image_shadeless)

    return image_pil


def visualize_bricks_pc(bs: BricksPC, highlight=False, adjust_camera=True):
    return visualize_bricks(get_elements(bs.bricks), highlight=highlight, adjust_camera=adjust_camera)


def visualize_brick(b_or_b_type: Union[str, Brick], highlight=False, adjust_camera=True):
    if isinstance(b_or_b_type, Brick):
        return visualize_bricks([b_or_b_type], highlight=highlight, adjust_camera=adjust_camera)

    return visualize_bricks([Brick(b_or_b_type, (0, 0, 0), (1, 0, 0, 0))],
                            highlight=highlight, adjust_camera=adjust_camera)


# helpers

def get_save_path(name, ext, add_timestamp=False, verbose=True):
    if add_timestamp:
        path = os.path.join(DEBUG_DIR, f"{name}_{time.time()}{ext}")
    else:
        # overwrites!
        path = os.path.join(DEBUG_DIR, f"{name}{ext}")

    return path


# BricksPC


def load_bricks_pc_from_dict(d: dict, return_steps=False) -> Union[BricksPC, List[BricksPC]]:
    # ignore object_rotation_quat

    if return_steps:
        bs_steps = []
    bs = BricksPC(np.array(d['grid_size']))

    for i_str, op in d['operations'].items():
        b_step = op['bricks']
        for j in range(len(b_step)):
            if 'canonical_state' in b_step[j]:
                b_state = b_step[j]['canonical_state']
                cls = b_state.pop('cls')
                if cls == 'HBrick':
                    b = HBrick(**b_state)
                    assert bs.add_hbrick(b, b_step[j]['op_type'])
                elif cls == 'VBrick':
                    b = VBrick(**b_state)
                    assert bs.add_vbrick(b, op_type=b_step[j]['op_type'])
                elif cls == 'CBrick':
                    rec_bricks_pc = b_state.pop('bricks_pc')
                    rec_bricks_pc = load_bricks_pc_from_dict(rec_bricks_pc)
                    b = CBrick(rec_bricks_pc, **b_state)
                    assert add_cbrick_to_bricks_pc(bs, b, op_type=b_step[j]['op_type'])
                else:
                    raise NotImplementedError(cls)
            else:
                if not bs.add_brick(b_step[j]['brick_type'], b_step[j]['canonical_position'],
                                    b_step[j]['canonical_rotation'],
                                    b_step[j]['op_type'], canonical=True, verbose=False):
                    print('Cannot add brick at #', i_str)
                    import ipdb;
                    ipdb.set_trace()
                    print()
        if return_steps:
            bs_steps.append(deepcopy(bs))

    if return_steps:
        return bs_steps

    return bs


def save_bricks_pc(bs: BricksPC, add_timestamp=False, as_dict=True, name='bs', info=None):
    ext = '.json' if as_dict else '.pkl'
    path = get_save_path(name, ext, add_timestamp=add_timestamp)

    if not as_dict:
        with open(path, 'wb') as f:
            pickle.dump((bs, info), f)
        return

    with open(path, 'w') as f:
        json.dump((bs.to_dict(), info), f, indent=4)


def load_bricks_pc(path=None, return_info=False):
    if path is None:
        path = os.path.join(DEBUG_DIR, 'bs.json')
    if os.path.splitext(path)[1] == '.pkl':
        with open(path, 'rb') as f:
            ret = pickle.load(f)
            bs, info = ret

    else:
        with open(path, 'r') as f:
            ret = json.load(f)
            if isinstance(ret, tuple):
                bs, info = ret
                bs = load_bricks_pc_from_dict(bs)
            else:
                bs, info = ret, None
                bs = load_bricks_pc_from_dict(bs)

    print('info from saved bricks pc', info)

    if return_info:
        return bs, info

    return bs


def save_bricks_pc_image(bs: BricksPC, add_timestamp=False, highlight=False, name='bs'):
    path = get_save_path(name, '.png', add_timestamp=add_timestamp)
    im = visualize_bricks_pc(bs, highlight)
    im.save(path)


def get_elements(bricks):
    return sum([[b] if isinstance(b, Brick) else b.bricks for b in bricks], [])


# Brick


def save_brick(brick, add_timestamp=False, as_dict=True, name='brick'):
    ext = '.json' if as_dict else '.pkl'
    path = get_save_path(name, ext, add_timestamp=add_timestamp)

    if not as_dict:
        with open(path, 'wb') as f:
            pickle.dump(brick, f)
        return

    with open(path, 'w') as f:
        json.dump(dict(brick_type=brick.brick_type,
                       position=list(map(float, brick.position)),
                       rotation=list(map(float, brick.rotation))), f, indent=4)


def load_brick(path=None):
    if path is None:
        path = os.path.join(DEBUG_DIR, 'brick.json')
    if os.path.splitext(path)[1] == '.pkl':
        with open(path, 'rb') as f:
            return pickle.load(f)

    with open(path, 'r') as f:
        d = json.load(f)
    return Brick(d['brick_type'], d['position'], d['rotation'])


# List[Brick]

def save_bricks_image(bricks: List[Brick], add_timestamp=False, highlight=False, name='bricks'):
    path = get_save_path(name, '.png', add_timestamp=add_timestamp)
    im = visualize_bricks(bricks, highlight)
    im.save(path)


def mesh_IoU(mesh1, mesh2):
    intersection = mesh1.intersection(mesh2)
    union = mesh1.union(mesh2)
    return intersection, union
    # IoU = .volume() / mesh1.union(mesh2).volume()
    # return IoU


def show_meshes(mesh_list, title="all", cmap_list=None):
    if not isinstance(mesh_list, list):
        mesh_list = [mesh_list]
    mesh_list_len = len(mesh_list)
    if cmap_list is None or mesh_list_len != len(cmap_list):
        if cmap_list is None:
            cmap_list = []
        cmap_list += get_cmap_list(mesh_list_len - len(cmap_list))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for mesh, cmap in zip(mesh_list, cmap_list):
        # Plot the mesh object
        ax.plot_trisurf(mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.vertices[:, 2], triangles=mesh.faces,
                        cmap=cmap)

    # Set the axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title(title)
    # Show the plot
    plt.show()


def get_cmap_list(cmap_num):
    default_cmap = ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'Greys', 'Purples', 'Blues', 'Greens',
                    'Oranges', 'Reds', 'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu', 'GnBu', 'PuBu', 'YlGnBu',
                    'PuBuGn', 'BuGn', 'YlGn']
    cmap_list = random.sample(default_cmap, cmap_num)
    return cmap_list


def py_bounding_box_to_box3d(bounding_box_tensor):
    # 将pytorch3d get_bounding_boxes的tensor转化为box3d_overlap的boxes输入
    # bounding_box_tensor shape (N,3,2)
    # boxes shape (B,8,3)
    converted_box = []
    for bounding_box in bounding_box_tensor:
        converted_box.append(torch.tensor(
            [[bounding_box[0, 0], bounding_box[1, 0], bounding_box[2, 0]],
             [bounding_box[0, 1], bounding_box[1, 0], bounding_box[2, 0]],
             [bounding_box[0, 1], bounding_box[1, 1], bounding_box[2, 0]],
             [bounding_box[0, 0], bounding_box[1, 1], bounding_box[2, 0]],

             [bounding_box[0, 0], bounding_box[1, 0], bounding_box[2, 1]],
             [bounding_box[0, 1], bounding_box[1, 0], bounding_box[2, 1]],
             [bounding_box[0, 1], bounding_box[1, 1], bounding_box[2, 1]],
             [bounding_box[0, 0], bounding_box[1, 1], bounding_box[2, 1]],
             ]))

    converted_boxes = torch.stack(converted_box)
    return converted_boxes


def cal_mesh_iou(pymesh_list: [Meshes], cur_brick_states):
    if len(pymesh_list) > 1:
        is_error_state = [brick_state != StateFlagNew.CORRECT for brick_state in cur_brick_states]
        is_error_state_tensor = torch.tensor(is_error_state)
        if sum(is_error_state_tensor) > 0:
            box_list = [mesh.get_bounding_boxes() for mesh in pymesh_list]
            converted_boxes = [py_bounding_box_to_box3d(box) for box in box_list]
            converted_boxes_tensor = torch.stack(converted_boxes)
            converted_boxes_tensor = converted_boxes_tensor.squeeze(dim=1).cuda()

            # error_boxes = []
            # for brick_idx, brick_state in enumerate(cur_brick_states):
            #     if brick_state != StateFlag.CORRECT:
            #         error_boxes.append(converted_boxes[brick_idx])
            # error_boxes_tensor = torch.stack(error_boxes)
            # error_boxes_tensor = error_boxes_tensor.squeeze(dim=1)
            _, IoU = box3d_overlap(converted_boxes_tensor, converted_boxes_tensor.clone())
            is_error_state_tensor = is_error_state_tensor.cuda()
            error_IoU = IoU[is_error_state_tensor]
            idx = torch.where(error_IoU > 0.1)
            print(idx)


def collision_detect(pymesh_list: [Meshes], cur_brick_states):
    if len(pymesh_list) > 1:
        is_error_state = [brick_state != StateFlagNew.CORRECT for brick_state in cur_brick_states]
        is_error_state_tensor = torch.tensor(is_error_state)
        if sum(is_error_state_tensor) > 0:
            trimesh_list = list(map(pymesh2trimesh, pymesh_list))  # 将Pytorch3d的mesh转为trimesh
            box_list = [mesh.bounding_box for mesh in trimesh_list]
            box_collision_manager = CollisionManager()
            # 添加所有的box
            for box_idx, box in enumerate(box_list):
                box_collision_manager.add_object(f"box_{box_idx}", box)
            for brick_idx, brick_state in enumerate(cur_brick_states):
                if brick_state != StateFlagNew.CORRECT:
                    box_collision_manager.remove_object(f"box_{brick_idx}")
                    box = box_list[brick_idx]
                    is_collision = box_collision_manager.in_collision_single(box)
                    distance = box_collision_manager.min_distance_single(box)
                    print("=" * 10)
                    print(f"brick_idx: {brick_idx}")
                    print(f"is_collision: {is_collision}")
                    print(f"distance: {distance}")
                    box_collision_manager.add_object(f"box_{brick_idx}", box)


dist_cal_map = {}  # TODO


def occupy_detect(pymesh_list: [Meshes], cur_brick_states, occupy_color=None):
    if occupy_color is None:
        occupy_color = [73, 165, 254]
    global dist_cal_map

    def modify_color(_mesh_idx):
        verts_rgb = torch.zeros(pymesh_list[_mesh_idx].verts_packed().shape).unsqueeze(dim=0)
        verts_rgb[..., :] = torch.Tensor(np.array(occupy_color) / 255)
        textures = TexturesVertex(verts_features=verts_rgb).to("cuda")
        pymesh_list[_mesh_idx].textures = textures

    if len(pymesh_list) > 1:
        is_error_state = [brick_state != StateFlagNew.CORRECT for brick_state in cur_brick_states]
        is_error_state_tensor = torch.tensor(is_error_state)
        if sum(is_error_state_tensor) > 0:
            trimesh_list = list(map(pymesh2trimesh, pymesh_list))
            for mesh_idx, (mesh, state_flag) in enumerate(zip(trimesh_list, cur_brick_states)):
                if state_flag != StateFlagNew.CORRECT:
                    box_collision_manager = CollisionManager()
                    box_collision_manager.add_object(f"outer_box_{mesh_idx}", mesh.bounding_box)
                    for mesh_idx2, mesh2 in enumerate(trimesh_list):
                        if mesh_idx == mesh_idx2:
                            continue
                        cal_id = f"{mesh_idx}@{mesh_idx2}"
                        if cal_id in dist_cal_map:
                            if dist_cal_map[cal_id]:
                                # 修改颜色为占据色
                                modify_color(mesh_idx)
                            else:
                                continue

                        is_collision = box_collision_manager.in_collision_single(mesh2.bounding_box)
                        dist_cal_map[cal_id] = False
                        if is_collision:
                            distance = signed_distance(mesh, mesh2.vertices)
                            result = distance > 1.0
                            if sum(result) >= 2:
                                dist_cal_map[cal_id] = True
                                # 错误的mesh至少有两个point在另一个mesh里
                                # 修改该mesh的纹理
                                # print(pymesh_list[mesh_idx].verts_packed().shape)
                                # verts_rgb = torch.zeros(pymesh_list[mesh_idx].verts_packed().shape).unsqueeze(dim=0)
                                # verts_rgb[..., :] = torch.Tensor(np.array(occupy_color) / 255)
                                # textures = TexturesVertex(verts_features=verts_rgb).to("cuda")
                                # pymesh_list[mesh_idx].textures = textures
                                modify_color(mesh_idx)
                                # mesh.visual.texture.set_texture(occupy_color)
                                print(distance[result])

    return pymesh_list


def pred_bricks_is_valid(mesh_list, colors):
    if len(mesh_list) > 1:
        # 检查预测的bricks的6D位姿是否合法
        trimesh_list = list(map(pymesh2trimesh, mesh_list))  # 将Pytorch3d的mesh转为trimesh
        # box_list = [trimesh.convex.convex_hull(mesh.vertices) for mesh in trimesh_list]
        box_list = [mesh.bounding_box for mesh in trimesh_list]

        cmap_list = get_cmap_list(len(trimesh_list))
        show_meshes(trimesh_list, cmap_list=cmap_list)
        # show_meshes(box_list[:2])

        for box1_idx, mesh1 in enumerate(trimesh_list):
            # box1 = mesh1.get_bounding_boxes()
            points = mesh1.vertices
            for box2_idx, box2 in enumerate(box_list[box1_idx + 1:]):
                # print(iou)
                box2_idx = box1_idx + box2_idx + 1
                if box2.is_watertight:
                    result = box2.contains(points)
                    point_in = sum(result)
                    if box2.contains(points).any():
                        print(f"box2 contains {point_in / len(result)} points!")
                        show_meshes([mesh1, trimesh_list[box2_idx]],
                                    title=f"box_{box1_idx},box_{box2_idx}, {round(point_in / len(result), 4)}%",
                                    cmap_list=[cmap_list[box1_idx], cmap_list[box2_idx]])

        # box_collision_manager = CollisionManager()
        # for box_idx, box in enumerate(box_list):
        #     box_collision_manager.add_object(f"box_{box_idx}", box)
        # is_collision, collision_names, contact_data = box_collision_manager.in_collision_internal(return_names=True,
        #                                                                                           return_data=True)
        # distance, closest_names, distance_data = box_collision_manager.min_distance_internal(return_names=True,
        #                                                                                      return_data=True)
        # print("=" * 10)
        # print(f"is_collision: {is_collision}")
        # print(f"collision_names: {collision_names}")
        # print(f"distance: {distance}")
        # print(f"closest_names: {closest_names}")
        # print("=" * 10)

        # for mesh1_idx, mesh1 in enumerate(trimesh_list):
        #     # points = mesh1.vertices
        #     collision_manager = CollisionManager()
        #
        #     box1 = trimesh.primitives.Box(mesh1.bounding_box.extents)
        #     collision_manager.add_object(f"box{mesh1_idx}", box1)
        #     for mesh2_idx, mesh2 in enumerate(trimesh_list[mesh1_idx + 1:]):
        #         box2 = trimesh.primitives.Box(mesh2.bounding_box.extents)
        #         collision_manager.add_object("box2", box2)

        # broken = trimesh.repair.broken_faces(mesh2, color=[255, 0, 0, 255])
        # mesh2.remove_degenerate_faces()
        # mesh2.remove_duplicate_faces()
        # mesh2.remove_infinite_values()
        # is_water = mesh2.fill_holes()
        # broken = trimesh.repair.broken_faces(mesh2, color=[255, 0, 0, 255])
        # if not is_water:
        #     broken = trimesh.repair.broken_faces(mesh2, color=[255, 0, 0, 255])
        #     print(len(broken))
        #     # mesh2.show(smooth=False)
        # else:
        # trimesh.intersections.box_box
        # if mesh2.contains(points).any():
        #     print(f"mesh_{mesh1_idx + mesh2_idx + 1} contains mesh_{mesh1_idx} ")
        # 判断任意两个mesh是否发生碰撞

        # collision_manager = CollisionManager()
        # for mesh_idx, mesh in enumerate(trimesh_list):
        #     collision_manager.add_object(f"mesh_{mesh_idx}", mesh)
        # is_collision, collision_names, contact_data = collision_manager.in_collision_internal(return_names=True,
        #                                                                                       return_data=True)
        # distance, closest_names, distance_data = collision_manager.min_distance_internal(return_names=True,
        #                                                                                  return_data=True)
        # print("=" * 10)
        # print(f"is_collision: {is_collision}")
        # print(f"collision_names: {collision_names}")
        # print(f"distance: {distance}")
        # print(f"closest_names: {closest_names}")
        # print("=" * 10)


STATE_COLOR_MAP = {
    StateFlagNew.CORRECT: [51, 193, 146],  # 正确的，绿色
    StateFlagNew.POSITION_ERROR: [255, 116, 105],  # 只是位置错误，红色
    StateFlagNew.ROTATION_ERROR: [249, 248, 113],  # 只是旋转错误，黄色
    StateFlagNew.POSITION_ROTATION_ERROR: [147, 103, 172]  # 全错，紫色
}


@torch.no_grad()
def render_bricks(brick_dict_list, transform_options_tensor, incorrect_R=None, incorrect_T=None):
    # 渲染输入的组件列表，如果incorrectRT不是None，则使用该RT矩阵进行纠正
    # transform_options_tensor: azims, elevs, obj_scales, obj_centers
    azim, elev, obj_scale, obj_center = transform_options_tensor
    # 将brick dict中的所有组件加载进来
    bricks = []
    colors = []
    for brick_info in brick_dict_list:
        colors.append(tuple(brick_info['color']))
        if "brick_type" in brick_info:
            brick_type = brick_info["brick_type"]
            rotation = brick_info['brick_transform']['rotation']
            position = brick_info['brick_transform']['position']
            bricks.append(Brick(brick_type, position, rotation))
        else:
            cbrick = dict_to_cbrick(brick_info)
            bricks.append(cbrick)
    brick_meshes = bricks2meshes(bricks, colors)

    if incorrect_R is not None and incorrect_T is not None:
        for i in range(len(brick_meshes)):
            brick_mesh = brick_meshes[i]
            trans = incorrect_T[i]
            R = torch.as_tensor(tr.quaternion_matrix(incorrect_R[i].cpu().detach().numpy())[:3, :3].T).to(trans.device)
            transform = pt.Transform3d().compose(pt.Rotate(R)).compose(pt.Translate(*trans)).to(trans.device)
            brick_meshes[i] = transform_mesh(brick_mesh, transform)
    elif incorrect_T is not None:
        for i in range(len(brick_meshes)):
            brick_mesh = brick_meshes[i]
            position = incorrect_T[i]
            scale_d = {'x': 20, 'y': 8, 'z': 20}
            trans = [position[0] * scale_d['x'], position[1] * scale_d['y'], position[2] * scale_d['z']]

            transform = pt.Transform3d().compose(pt.Translate(*trans)).to(position.device)
            brick_meshes[i] = transform_mesh(brick_mesh, transform)

    transform = pt.Transform3d().translate(*(-obj_center)).scale(obj_scale).cuda()
    # transform = pt.Transform3d().scale(obj_scale).cuda()
    for i in range(len(brick_meshes)):
        brick_mesh, = brick_meshes[i]
        brick_meshes[i] = transform_mesh(brick_mesh, transform)

    scale_xyz = np.array([0.0024] * 3)
    mesh = join_meshes_as_scene(brick_meshes)

    R, T = look_at_view_transform(dist=2000, elev=elev, azim=azim, at=((0, 0, 0),))
    T = T.cuda()
    cameras = FoVOrthographicCameras(device=brick_meshes[0].device, R=R, T=T,
                                     scale_xyz=[scale_xyz])

    image, depth_map = render_lego_scene(mesh, cameras)  # 渲染场景
    image[:, :, :, 3][image[:, :, :, 3] == 0] = 0  # 处理透明度信息
    image[:, :, :, 3][image[:, :, :, 3] > 0] = 1  # 处理透明度信息
    image_pil = Image.fromarray((image[0, :, :].detach().cpu().numpy() * 255).astype(np.uint8))  # 转换为PIL图像
    image_pil = image_pil.resize((512, 512))  # 调整大小
    return image_pil


@torch.no_grad()
def render_dict_simple_one_step(manual_pred, step_id, no_check=False, state_img=False):
    bricks = []
    state_colors = []
    colors = []
    rendered_raw_img = None
    rendered_state_img = None
    brick_state_flag_list = []

    for i_str, op in manual_pred['operations'].items():
        if int(i_str) > step_id:
            break
        b_step = manual_pred['operations'][i_str]['bricks']
        bricks_num = len(b_step)
        # 加载当前步骤的bricks
        for j in range(bricks_num):
            if "correct" not in b_step[j]:
                b_step[j]['correct'] = StateFlagNew.CORRECT

            state_flag = b_step[j]['correct']
            rotation = b_step[j]['brick_transform']['rotation']
            position = b_step[j]['brick_transform']['position']
            rotation_euler = np.round(np.array(tr.euler_from_quaternion(rotation)) / np.pi * 180).astype(int)
            rotation_euler = list(map(int, rotation_euler))
            b_step[j]['brick_transform']['rotation_euler'] = rotation_euler
            state_colors.append(tuple(STATE_COLOR_MAP[state_flag]))  # 转换颜色
            colors.append(tuple(b_step[j]['color']))
            brick_state_flag_list.append(state_flag)
            if 'brick_type' in b_step[j]:
                brick_type = b_step[j]['brick_type']
                bricks.append(Brick(brick_type, position, rotation))  # 添加砖块信息
            else:
                cbrick = dict_to_cbrick(b_step[j], no_check=no_check)
                bricks.append(cbrick)
    mask_colors = []
    for j in range(len(bricks)):
        while True:
            r, g, b = [np.random.randint(0, 256) for _ in range(3)]
            if not (r, g, b) in mask_colors:
                break
        mask_colors.append((r, g, b))

    brick_meshes = bricks2meshes(bricks, colors)
    state_brick_meshes = bricks2meshes(bricks, state_colors)
    mask_brick_meshes = bricks2meshes(bricks, mask_colors)

    obj_scale, obj_center = manual_pred['obj_scale'], np.array(manual_pred['obj_center'])

    transform = pt.Transform3d().translate(*(-obj_center)).scale(obj_scale).cuda()
    # transform = pt.Transform3d().scale(obj_scale).cuda()
    for i in range(len(brick_meshes)):
        brick_mesh, mask_brick_mesh, state_brick_mesh = brick_meshes[i], mask_brick_meshes[i], state_brick_meshes[i]
        brick_meshes[i] = transform_mesh(brick_mesh, transform)
        mask_brick_meshes[i] = transform_mesh(mask_brick_mesh, transform)
        state_brick_meshes[i] = transform_mesh(state_brick_mesh, transform)

    scale_xyz = np.array([0.0024] * 3)

    def index_list(l, idxs):
        return [l[i] for i in sorted(list(idxs))]

    brick_ct = 0
    for i_str, op in manual_pred['operations'].items():
        if int(i_str) > step_id:
            break
        b_step = manual_pred['operations'][i_str]['bricks']
        step_brick_ct = len(b_step)
        cur_brick_idxs = list(range(brick_ct + step_brick_ct))
        step_idxs = [brick_ct + j for j in range(step_brick_ct)]
        brick_ct += step_brick_ct
        if int(i_str) != step_id:
            continue
        azim = op['view_direction'][0]  # for example: 227
        elev = op['view_direction'][1]  # for example: 39

        cur_mask_brick_meshes = index_list(mask_brick_meshes, cur_brick_idxs)
        cur_mask_colors = index_list(mask_colors, cur_brick_idxs)
        # cur_brick_states = index_list(brick_state_flag_list, cur_brick_idxs)
        cur_brick_meshes = index_list(brick_meshes, cur_brick_idxs)
        cur_state_brick_meshes = index_list(state_brick_meshes, cur_brick_idxs) if state_img else None

        # pred_bricks_is_valid(cur_brick_meshes, colors)
        # cal_mesh_iou(cur_brick_meshes, cur_brick_states)
        # collision_detect(cur_brick_meshes, cur_brick_states)
        # cur_brick_meshes = occupy_detect(cur_brick_meshes, cur_brick_states)
        mesh = join_meshes_as_scene(cur_brick_meshes)
        state_mesh = join_meshes_as_scene(cur_state_brick_meshes) if state_img else None

        R, T = look_at_view_transform(dist=2000, elev=elev, azim=azim, at=((0, 0, 0),))
        T = T.cuda()
        cameras = FoVOrthographicCameras(device=brick_meshes[0].device, R=R, T=T,
                                         scale_xyz=[scale_xyz])
        masks_step, image_shadeless = get_brick_masks(cur_mask_brick_meshes, cur_mask_colors, step_idxs,
                                                      cameras)  # 获取遮罩和无阴影图像

        image, depth_map = render_lego_scene(mesh, cameras)  # 渲染场景
        image[:, :, :, 3][image[:, :, :, 3] == 0] = 0  # 处理透明度信息
        image[:, :, :, 3][image[:, :, :, 3] > 0] = 1  # 处理透明度信息
        image_pil = Image.fromarray((image[0, :, :].detach().cpu().numpy() * 255).astype(np.uint8))  # 转换为PIL图像
        image_pil = image_pil.resize((512, 512))  # 调整大小
        rendered_raw_img = highlight_edge(image_pil, depth_map, image_shadeless)  # 突出边缘
        if state_img:
            image, depth_map = render_lego_scene(state_mesh, cameras)  # 渲染场景
            image[:, :, :, 3][image[:, :, :, 3] == 0] = 0  # 处理透明度信息
            image[:, :, :, 3][image[:, :, :, 3] > 0] = 1  # 处理透明度信息
            image_pil = Image.fromarray((image[0, :, :].detach().cpu().numpy() * 255).astype(np.uint8))  # 转换为PIL图像
            image_pil = image_pil.resize((512, 512))  # 调整大小
            rendered_state_img = highlight_edge(image_pil, depth_map, image_shadeless)  # 突出边缘
    if state_img:
        return rendered_raw_img, rendered_state_img  # 返回图像
    else:
        return rendered_raw_img


def index_list(l, idxs):
    return [l[i] for i in sorted(list(idxs))]


@torch.no_grad()
def render_dict_simple_one_step_only_current_brick(manual_pred, step_id, no_check=False):
    b_step = manual_pred['operations'][str(step_id)]['bricks']
    bricks_num = len(b_step)
    op = manual_pred['operations'][str(step_id)]
    one_brick_img_list = []
    # 加载当前步骤的bricks
    for j in range(bricks_num):
        bricks = []
        colors = []
        if "correct" not in b_step[j]:
            b_step[j]['correct'] = StateFlagNew.CORRECT

        rotation = b_step[j]['brick_transform']['rotation']
        position = b_step[j]['brick_transform']['position']
        rotation_euler = np.round(np.array(tr.euler_from_quaternion(rotation)) / np.pi * 180).astype(int)
        rotation_euler = list(map(int, rotation_euler))
        b_step[j]['brick_transform']['rotation_euler'] = rotation_euler
        colors.append(tuple(b_step[j]['color']))
        if 'brick_type' in b_step[j]:
            brick_type = b_step[j]['brick_type']
            bricks.append(Brick(brick_type, position, rotation))  # 添加砖块信息
        else:
            cbrick = dict_to_cbrick(b_step[j], no_check=no_check)
            bricks.append(cbrick)
        mask_colors = []
        for j in range(len(bricks)):
            while True:
                r, g, b = [np.random.randint(0, 256) for _ in range(3)]
                if not (r, g, b) in mask_colors:
                    break
            mask_colors.append((r, g, b))

        brick_meshes = bricks2meshes(bricks, colors)
        mask_brick_meshes = bricks2meshes(bricks, mask_colors)

        obj_scale, obj_center = manual_pred['obj_scale'], np.array(manual_pred['obj_center'])

        transform = pt.Transform3d().translate(*(-obj_center)).scale(obj_scale).cuda()
        for i in range(len(brick_meshes)):
            brick_mesh, mask_brick_mesh = brick_meshes[i], mask_brick_meshes[i]
            brick_meshes[i] = transform_mesh(brick_mesh, transform)
            mask_brick_meshes[i] = transform_mesh(mask_brick_mesh, transform)

        scale_xyz = np.array([0.0024] * 3)

        azim = op['view_direction'][0]  # for example: 227
        elev = op['view_direction'][1]  # for example: 39

        cur_mask_brick_meshes = mask_brick_meshes
        cur_mask_colors = mask_colors
        cur_brick_meshes = brick_meshes

        mesh = join_meshes_as_scene(cur_brick_meshes)

        R, T = look_at_view_transform(dist=2000, elev=elev, azim=azim, at=((0, 0, 0),))
        T = T.cuda()
        cameras = FoVOrthographicCameras(device=brick_meshes[0].device, R=R, T=T,
                                         scale_xyz=[scale_xyz])

        masks_step, image_shadeless = get_brick_masks(cur_mask_brick_meshes, cur_mask_colors, [0],
                                                      cameras)  # 获取遮罩和无阴影图像，耗时长

        image, depth_map = render_lego_scene(mesh, cameras)  # 渲染场景
        image[:, :, :, 3][image[:, :, :, 3] == 0] = 0  # 处理透明度信息
        image[:, :, :, 3][image[:, :, :, 3] > 0] = 1  # 处理透明度信息
        image_pil = Image.fromarray((image[0, :, :].detach().cpu().numpy() * 255).astype(np.uint8))  # 转换为PIL图像
        image_pil = image_pil.resize((512, 512))  # 调整大小
        rendered_brick_img = highlight_edge(image_pil, depth_map, image_shadeless)  # 突出边缘
        rendered_brick_img = rendered_brick_img.convert('RGB')
        one_brick_img_list.append(rendered_brick_img)

    return one_brick_img_list


@torch.no_grad()
def render_dict_simple(d, only_final=False, no_check=False):
    '''
    :param d: bricks dict
    :param azims: If given, overwrite dict camera parameters
    :param elevs: If given, overwrite dict camera parameters
    :return:
    '''
    bricks = []

    images = []
    # occs = []
    colors = []
    for i_str, op in d['operations'].items():
        # 加载所有需要的brick
        b_step = d['operations'][i_str]['bricks']
        bricks_num = len(b_step)
        for j in range(bricks_num):
            rotation = b_step[j]['brick_transform']['rotation']
            position = b_step[j]['brick_transform']['position']
            rotation_euler = np.round(np.array(tr.euler_from_quaternion(rotation)) / np.pi * 180).astype(int)
            rotation_euler = list(map(int, rotation_euler))
            b_step[j]['brick_transform']['rotation_euler'] = rotation_euler
            colors.append(tuple(b_step[j]['color']))
            if 'brick_type' in b_step[j]:
                brick_type = b_step[j]['brick_type']
                bricks.append(Brick(brick_type, position, rotation))  # 添加砖块信息
            else:
                cbrick = dict_to_cbrick(b_step[j], no_check=no_check)
                bricks.append(cbrick)

    mask_colors = []
    for j in range(len(bricks)):
        while True:
            r, g, b = [np.random.randint(0, 256) for _ in range(3)]
            if not (r, g, b) in mask_colors:
                break
        mask_colors.append((r, g, b))

    brick_meshes = bricks2meshes(bricks, colors)
    mask_brick_meshes = bricks2meshes(bricks, mask_colors)

    obj_scale, obj_center = d['obj_scale'], np.array(d['obj_center'])

    transform = pt.Transform3d().translate(*(-obj_center)).scale(obj_scale).cuda()
    # transform = pt.Transform3d().scale(obj_scale).cuda()
    for i in range(len(brick_meshes)):
        brick_mesh, mask_brick_mesh = brick_meshes[i], mask_brick_meshes[i]
        brick_meshes[i] = transform_mesh(brick_mesh, transform)
        mask_brick_meshes[i] = transform_mesh(mask_brick_mesh, transform)

    scale_xyz = np.array([0.0024] * 3)

    def index_list(l, idxs):
        return [l[i] for i in sorted(list(idxs))]

    brick_ct = 0

    for i_str, op in d['operations'].items():
        # 渲染每一步
        b_step = d['operations'][i_str]['bricks']
        step_brick_ct = len(b_step)
        cur_brick_idxs = list(range(brick_ct + step_brick_ct))
        step_idxs = [brick_ct + j for j in range(step_brick_ct)]
        brick_ct += step_brick_ct
        if only_final and int(i_str) != len(d['operations']) - 1:
            continue

        azim = op['view_direction'][0]  # for example: 227
        elev = op['view_direction'][1]  # for example: 39

        cur_mask_brick_meshes = index_list(mask_brick_meshes, cur_brick_idxs)
        cur_mask_colors = index_list(mask_colors, cur_brick_idxs)

        cur_brick_meshes = index_list(brick_meshes, cur_brick_idxs)
        # pred_bricks_is_valid(cur_brick_meshes, colors)
        # cal_mesh_iou(cur_brick_meshes)
        mesh = join_meshes_as_scene(cur_brick_meshes)
        R, T = look_at_view_transform(dist=2000, elev=elev, azim=azim, at=((0, 0, 0),))
        T = T.cuda()
        cameras = FoVOrthographicCameras(device=brick_meshes[0].device, R=R, T=T,
                                         scale_xyz=[scale_xyz])
        op['camera'] = {
            'T': list(map(float, T[0].detach().cpu().numpy())),
            'R': list(map(float, tr.quaternion_from_matrix(R[0].detach().cpu().numpy())))
        }

        masks_step, image_shadeless = get_brick_masks(cur_mask_brick_meshes, cur_mask_colors, step_idxs,
                                                      cameras)  # 获取遮罩和无阴影图像

        # show_meshes([pymesh2trimesh(mesh)])
        image, depth_map = render_lego_scene(mesh, cameras)  # 渲染场景
        image[:, :, :, 3][image[:, :, :, 3] == 0] = 0  # 处理透明度信息
        image[:, :, :, 3][image[:, :, :, 3] > 0] = 1  # 处理透明度信息
        image_pil = Image.fromarray((image[0, :, :].detach().cpu().numpy() * 255).astype(np.uint8))  # 转换为PIL图像
        image_pil = image_pil.resize((512, 512))  # 调整大小
        image_pil = highlight_edge(image_pil, depth_map, image_shadeless)  # 突出边缘
        images.append(image_pil)  # 添加图像

    if only_final:
        return images[0]  # 返回最终图像
    else:
        return images  # 返回图像列表


def show_img(img_pil, title=""):
    img_np = np.array(img_pil)
    plt.imshow(img_np)
    plt.title(title)
    plt.show()


from src.models.networks import ndc_meshgrid
import torch.nn.functional as F


def plt_points(points, title=""):
    from mpl_toolkits.mplot3d import Axes3D



    # 调整相机位置
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c=z, cmap='viridis', s=50, alpha=0.6)
    ax.view_init(elev=140, azim=0)

    # plt.xlim(0, 250)
    # plt.ylim(0, 250)

    # 设置坐标轴标签
    # ax.set_xlabel('X Label')
    # ax.set_ylabel('Y Label')
    # ax.set_zlabel('Z Label')
    ax.axis('off')
    plt.savefig("./base_shape.png", dpi=300)
    # # 显示图形
    plt.show()

    # # 生成示例点云数据
    # num_points = 1000
    # x = np.random.normal(size=num_points)
    # y = np.random.normal(size=num_points)
    # z = np.random.normal(size=num_points)
    #
    # # 创建 3D 图形
    # fig = plt.figure(figsize=(10, 8))
    # ax = fig.add_subplot(111, projection='3d')
    #
    # # 绘制点云
    # ax.scatter(x, y, z, c=z, cmap='viridis', s=50, alpha=0.6)
    #
    # # 设置坐标轴标签
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    #
    # # 显示图形
    # plt.title('3D Point Cloud Visualization')
    # plt.show()


def camera_transform(x, obj_scale, obj_center, azim=233, elev=21, title="",visualize_data_dir=None):
    R, T = look_at_view_transform(dist=2000, elev=elev, azim=azim, at=((0, 0, 0),))
    cameras = FoVOrthographicCameras(device=R.device, R=R, T=T, scale_xyz=[(0.0024, 0.0024, 0.0024), ])
    # obj_scale, obj_center = torch.tensor([2.158754587173462]), torch.tensor([[1.0, 146.0, -0.000263214111328125]])

    transforms = pt.Transform3d().translate(-obj_center).scale(obj_scale).cuda().inverse()

    grid_size = (256, 256, 128)
    grid = ndc_meshgrid((1, 1, *grid_size)).cuda()
    grid = grid.reshape(1, -1, 3).cuda()

    unprojection_transform = cameras.get_full_projection_transform().inverse().cuda()
    canonical_transform = pt.Translate(-65, -65 // 4, -65).cuda()
    scale_d = [20, 8, 20]
    s_mat = pt.Scale(scale_d[0] * 0.5, scale_d[1] * 0.5, scale_d[2] * 0.5).cuda()
    canonical_transform = canonical_transform.compose(s_mat).cuda()
    canonical_transform = canonical_transform.inverse().cuda()
    grid_s = unprojection_transform.compose(transforms).compose(canonical_transform).transform_points(grid)

    grid_s = grid_s.reshape(1, *grid_size, 3)

    grid_s = grid_s / (torch.as_tensor(x.shape[2:], device=grid_s.device).expand_as(grid_s) // 2) - 1  # normalize
    grid_s = grid_s[:, :, :, :, [2, 1, 0]]  # change to [D, H, W]

    # from  [B, C, W', H', D'] to   [B, C, H', W', D']
    f_map = F.grid_sample(x, grid_s, mode='bilinear', align_corners=True).transpose(-3, -2)
    occ_map = f_map
    # return occ_map
    # points = occ_map[0][0]
    # points = torch.nonzero(points, as_tuple=False).float()
    # center_point = torch.mean(points, dim=0).int()
    #
    occ_inds = occ_map.argmax(dim=-1).unsqueeze(-1)
    # print(torch.max(occ_map), torch.min(occ_map))
    occ_img = occ_inds[0, 0, ..., 0].cpu().numpy()
    # occ_img[occ_img != 0] = 1
    # occ_img[center_point[0], center_point[1]] = 5
    occ_img = occ_img.astype(np.uint8)
    img_pil = Image.fromarray(occ_img)
    # img_pil = img_pil.resize((512, 512))
    img_np = np.array(img_pil)
    #
    plt.imshow(img_np)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"{visualize_data_dir}/{title}.png" if visualize_data_dir is not None else f"./shape.png")
    plt.show()
    #
    # occ_inds = occ_map.argmax(dim=-3).unsqueeze(-3)
    # # # print(torch.max(occ_map), torch.min(occ_map))
    # occ_img = occ_inds[0, 0, ..., 0, :, :].cpu().numpy()
    # occ_img[occ_img != 0] = 1
    # # occ_img[center_point[1], center_point[2]] = 5
    # occ_img = occ_img.astype(np.uint8)
    # img_pil = Image.fromarray(occ_img)
    # # img_pil = img_pil.resize((512, 256))
    # img_np = np.array(img_pil)
    # #
    # plt.imshow(img_np)
    # plt.title("yz")
    # plt.show()
    #
    # occ_inds = occ_map.argmax(dim=-2).unsqueeze(-2)
    # # print(torch.max(occ_map), torch.min(occ_map))
    # occ_img = occ_inds[0, 0, ..., 0, :].cpu().numpy()
    # occ_img[occ_img != 0] = 1
    # occ_img[center_point[0], center_point[2]] = 5
    # occ_img = occ_img.astype(np.uint8)
    # img_pil = Image.fromarray(occ_img)
    # # img_pil = img_pil.resize((512, 256))
    # img_np = np.array(img_pil)
    #
    # plt.imshow(img_np)
    # plt.title("xz")
    # plt.show()
    #
    # points = occ_map[0][0].cpu()
    # points = torch.nonzero(points, as_tuple=False).float()
    #
    # plt_points(points, "after")
    # plot_voxels(f_map.squeeze().cpu())


def plot_voxels(voxels):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.view_init(azim=225, elev=35)

    # Find non-zero indices
    indices = torch.nonzero(voxels)
    # ax.scatter(x, y, z, c=z, cmap='viridis', s=50, alpha=0.6)
    ax.scatter(indices[:, 2], indices[:, 1], indices[:, 0], c= indices[:, 0],marker='.',  cmap='viridis', s=50, alpha=0.6)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Voxel Data')

    plt.show()