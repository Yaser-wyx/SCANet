# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from .loss import KPLoss, MaskLoss
from .models.networks.hourglass import get_hourglass_net


class LegoHourGlass(nn.Module):
    """
    Main class for Generalized R-CNN.

    Arguments:
        backbone (nn.Module):
        rpn (nn.Module):
        roi_heads (nn.Module): takes the features + the proposals from the RPN and computes
            detections / masks from it.
        transform (nn.Module): performs the data transformation from the inputs to feed into
            the model
    """

    def __init__(self, opt, num_classes, occ_out_channels,
                 occ_fmap_size, shape_condition=False, brick_emb=64, one_hot_brick_emb=False,
                 num_bricks_single_forward=10,
                 brick_voxel_embedding_num=0,
                 brick_voxel_embedding_dim=0,
                 projection_brick_encoder=False,
                 num_rots=4,
                 predict_masks=False,
                 predict_trans=False,
                 predict_trans_seperate=False,
                 ):
        super().__init__()

        # Backbone
        occ_out_channels = occ_out_channels  # 输出的Feature dims
        occ_fmap_size = occ_fmap_size  # feature map size
        self.img_size = 512
        no_coordconv = False if not hasattr(opt, 'no_coordconv') else opt.no_coordconv
        # 体素编码器，输入为voxel representation，将3D representation转化为2D representation，为原论文中的 fig3(a){i}
        self.occ_encoder = VoxelTransformEncoder(occ_out_channels, feature_map2d_size=(occ_fmap_size,) * 2,
                                                 use_coordconv=not no_coordconv)
        self.shape_condition = shape_condition  # 默认为true
        self.one_hot_brick_emb = one_hot_brick_emb  # 默认为false
        self.projection_brick_encoder = projection_brick_encoder  # 默认为false
        if self.shape_condition:  # 如果使用component-conditioned Hourglass，默认为true
            if one_hot_brick_emb:
                # 默认不使用one hot编码来表示brick
                self.brick_encoder = nn.Embedding(num_classes, brick_emb)
            else:
                if brick_voxel_embedding_dim == 0:
                    if projection_brick_encoder:  # 默认不使用该编码器，内部实现为：VoxelTransformEncoder+[conv, bn, relu]*3
                        self.brick_encoder = BrickVoxelProjectionEncoder(num_features=brick_emb)
                    else:  # 默认使用该编码器，为fig3(a){ii}的components输入
                        self.brick_encoder = simple_voxel_encoder(brick_emb)
                else:
                    self.brick_encoder = simple_voxel_encoder(brick_emb,
                                                              num_embedding=brick_voxel_embedding_num,
                                                              embedding_dim=brick_voxel_embedding_dim)

        if shape_condition:
            cond_emb_dim = brick_emb * num_bricks_single_forward  # 等于原论文中的 C_2 * K_max (64*5)
            self.num_bricks_single_forward = num_bricks_single_forward
        else:
            cond_emb_dim = 0
        # 设置预测头
        heads = {'hm': num_bricks_single_forward if shape_condition else num_classes, 'reg': 2, 'rot': num_rots}
        self.predict_masks = predict_masks
        if predict_masks:
            heads['mask'] = num_bricks_single_forward + 1  # 五个不同型号的组件的mask以及一个背景
            heads['assoc_emb'] = num_bricks_single_forward
        self.predict_trans = predict_trans  # 无用
        self.predict_trans_separate = predict_trans_seperate  # 无用
        if predict_trans:
            if predict_trans_seperate:
                raise NotImplementedError()
            else:
                heads['trans'] = 3
        # 论文fig3(a){ii}的编解码器以及{iii}的预测头
        self.hg = get_hourglass_net(heads=heads,
                                    img_nc=3 + occ_out_channels,
                                    num_stacks=opt.num_stacks,
                                    cond_emb_dim=cond_emb_dim)
        self.loss = KPLoss(opt)
        if predict_masks:
            self.mask_loss = MaskLoss(opt)

    def forward(self, images, occ_prev,
                azims, elevs, obj_q, obj_scales, obj_centers,
                target,
                brick_occs=None,
                bids=None
                ):
        '''
        :param images: [B, C, H, W] 图像数据，B为batch size，C为通道数，H为高度，W为宽度
        :param occ_prev:  [B, 1, W, H, D] 上一帧的物体遮挡情况，B为batch size，W为宽度，H为高度，D为深度
        :param azims: [B] 摄像机的方位角，B为batch size
        :param elevs: [B] 摄像机的仰角，B为batch size
        :param obj_q: [B, 4] 物体的四元数表示，B为batch size
        :param obj_scales: [B] 物体的尺度，B为batch size
        :param obj_centers: [B, 3] 物体的中心点坐标，B为batch size
        :param target: Dict 目标数据，包含多个键值对
        :param brick_occs: List[num_brick_types, 1, W, H, D] 不同类型的积木的遮挡情况，num_brick_types为积木类型数目，W为宽度，H为高度，D为深度
        :return: outputs: 预测结果，loss_sum: 总损失函数，loss_dict: 损失函数字典
        '''
        occ_f_map = self.occ_encoder(
            occ_prev,
            azims, elevs,
            obj_scales,
            obj_centers,
        )  # 对上一步的物体遮挡情况进行编码，这个是直接用GT得到的
        occ_f_map = F.interpolate(occ_f_map, size=self.img_size, mode='nearest')  # 将2D feature map进行上采样
        img_concat = torch.cat([images, occ_f_map], dim=1)  # 拼接图像与特征图

        def merge_dict(d1, d2, f):
            return {k: f(d1[k], d2[k]) for k in d1.keys()}

        if self.shape_condition:
            if self.num_bricks_single_forward == 1:  # 该段代码不执行
                targets = [{k: v[i] for k, v in target.items()} for i in range(len(brick_occs))]

                loss_dict = None
                outputs = None
                loss_sum = 0
                batch_size = len(brick_occs)
                for i in range(batch_size):
                    if self.one_hot_brick_emb:
                        f_brick = self.brick_encoder(bids[i].long())
                    else:
                        f_brick = self.brick_encoder(brick_occs[i])
                    img_concat_this = img_concat[[i]].expand(brick_occs[i].shape[0], -1, -1, -1)
                    outputs_this = self.hg(img_concat_this, f_brick)
                    for output_this in outputs_this:
                        output_this['hm'] = output_this['hm'].squeeze(dim=1)
                    loss_sum_this, loss_dict_this = self.loss(outputs_this, targets[i])
                    loss_sum += loss_sum_this
                    if i == 0:
                        loss_dict = loss_dict_this
                        outputs = [{k: [v] for k, v in outputs_this[i].items()} for i in range(len(outputs_this))]
                    else:
                        loss_dict = merge_dict(loss_dict, loss_dict_this, lambda x, y: x + y)
                        outputs = [merge_dict(outputs[i], outputs_this[i], lambda x, y: x + [y])
                                   for i in range(len(outputs_this))]
                loss_sum /= batch_size
                loss_dict = {k: v / batch_size for k, v in loss_dict.items()}

            else:
                def build_batch_from_list(l):
                    '''
                    将列表中的张量进行批次化处理，假设列表中的每个张量的形状为[brick_num, ...]，并且列表长度为batch_size
                    返回一个形状为[batch_size, self.num_bricks_single_forward, ...]的张量
                    '''
                    t = pad_sequence(l, batch_first=True)
                    # 将第二维度填充到self.num_bricks_single_forward
                    t_padded = F.pad(t,
                                     (0,) * (2 * (len(t.shape) - 2)) + (0, self.num_bricks_single_forward - t.shape[1]))
                    return t_padded

                # 将目标数据中的hm字段进行批次化处理
                target['hm'] = build_batch_from_list(target['hm'])

                n_bricks = [b.shape[0] for b in brick_occs]
                brick_occs_concat = torch.cat(brick_occs, dim=0)
                if self.projection_brick_encoder:  # pass
                    azims_expand = torch.cat([azims[i].expand(n_bricks[i] * 4) for i in range(azims.shape[0])], dim=0)
                    elevs_expand = torch.cat([elevs[i].expand(n_bricks[i] * 4) for i in range(azims.shape[0])], dim=0)
                    obj_scales_expand = torch.cat(
                        [obj_scales[i].expand(n_bricks[i] * 4) for i in range(azims.shape[0])], dim=0)
                    obj_centers_expand = torch.cat(
                        [obj_centers[i].expand(n_bricks[i] * 4, -1) for i in range(azims.shape[0])], dim=0)
                    f_brick_concat = self.brick_encoder(brick_occs_concat, azims_expand, elevs_expand,
                                                        obj_scales_expand, obj_centers_expand)
                else:
                    f_brick_concat = self.brick_encoder(brick_occs_concat)  # 对需要的部件进行编码
                brick_ct = 0
                f_brick_list = []
                for n_brick in n_bricks:
                    f_brick_list.append(f_brick_concat[brick_ct:brick_ct + n_brick])
                    brick_ct += n_brick
                f_brick_padded = build_batch_from_list(f_brick_list)  # 对部件的编码进行padding补0
                outputs = self.hg(img_concat, f_brick_padded.reshape(f_brick_padded.shape[0], -1))  # 使用Hourglass网络进行预测
                loss_sum, loss_dict = self.loss(outputs, target)

        else:  # 这段不执行
            outputs = self.hg(img_concat)
            loss_sum, loss_dict = self.loss(outputs, target)

        if self.predict_masks:
            loss_sum_mask, loss_dict_mask = self.mask_loss(outputs, target)
            loss_sum += loss_sum_mask
            for k, v in loss_dict_mask.items():
                if k == 'loss_sum':
                    continue
                loss_dict[k] = v

        return outputs, loss_sum, loss_dict


#
# class LegoHourGlassMask(nn.Module):
#     def __init__(self, opt, num_classes, occ_out_channels,
#                  occ_fmap_size, brick_emb=64,
#                  num_bricks_single_forward=10,
#                  brick_voxel_embedding_num=0,
#                  brick_voxel_embedding_dim=0,
#                  projection_brick_encoder=False,
#                  assoc_emb=False,
#                  ):
#         super().__init__()
#
#         # Backbone
#         occ_out_channels = occ_out_channels
#         occ_fmap_size = occ_fmap_size
#         self.img_size = 512
#         self.occ_encoder = VoxelTransformEncoder(occ_out_channels, feature_map2d_size=(occ_fmap_size,) * 2)
#         self.projection_brick_encoder = projection_brick_encoder
#         if brick_voxel_embedding_dim == 0:
#             if projection_brick_encoder:
#                 self.brick_encoder = BrickVoxelProjectionEncoder(num_features=brick_emb)
#             else:
#                 self.brick_encoder = simple_voxel_encoder(brick_emb)
#         else:
#             self.brick_encoder = simple_voxel_encoder(brick_emb,
#                                                       num_embedding=brick_voxel_embedding_num,
#                                                       embedding_dim=brick_voxel_embedding_dim)
#
#         cond_emb_dim = brick_emb * num_bricks_single_forward
#         self.num_bricks_single_forward = num_bricks_single_forward
#         heads = {'mask': num_bricks_single_forward + 1}
#         if assoc_emb:
#             heads['assoc_emb'] = num_bricks_single_forward
#         self.hg = get_hourglass_net(heads=heads,
#                                     img_nc=3 + occ_out_channels,
#                                     num_stacks=opt.num_stacks,
#                                     cond_emb_dim=cond_emb_dim)
#         self.loss = MaskLoss(opt)
#
#     def forward(self, images, occ_prev,
#                 azims, elevs, obj_q, obj_scales, obj_centers,
#                 target,
#                 brick_occs=None,
#                 bids=None
#                 ):
#         '''
#         :param images: [B, C, H, W]
#         :param occ_prev:  [B, 1, W, H, D]
#         :param azims: [B]
#         :param elevs: [B]
#         :param obj_q: [B, 4]
#         :param obj_scales: [B]
#         :param obj_centers: [B, 3]
#         :param target: Dict
#         :param brick_occs: List[num_brick_types, 1, W, H, D]
#         :return:
#         '''
#         occ_f_map = self.occ_encoder(
#             occ_prev,
#             azims, elevs,
#             obj_q,
#             obj_scales,
#             obj_centers,
#         )
#         occ_f_map = F.interpolate(occ_f_map, size=self.img_size, mode='nearest')
#         img_concat = torch.cat([images, occ_f_map], dim=1)
#
#         def build_batch_from_list(l):
#             '''
#             assuming each tesnor in the list has shape [brick_num, ...], and len(l) = batch_size
#             we build a tensor of [batch_size, self.num_bricks_single_forward, ...]
#             '''
#             t = pad_sequence(l, batch_first=True)
#             # pad the second dimensino to self.num_bricks_single_forward
#             t_padded = F.pad(t, (0,) * (2 * (len(t.shape) - 2)) + (0, self.num_bricks_single_forward - t.shape[1]))
#             return t_padded
#
#         n_bricks = [b.shape[0] for b in brick_occs]
#         brick_occs_concat = torch.cat(brick_occs, dim=0)
#         if self.projection_brick_encoder:
#             azims_expand = torch.cat([azims[i].expand(n_bricks[i] * 4) for i in range(azims.shape[0])], dim=0)
#             elevs_expand = torch.cat([elevs[i].expand(n_bricks[i] * 4) for i in range(azims.shape[0])], dim=0)
#             obj_scales_expand = torch.cat([obj_scales[i].expand(n_bricks[i] * 4) for i in range(azims.shape[0])], dim=0)
#             obj_centers_expand = torch.cat([obj_centers[i].expand(n_bricks[i] * 4, -1) for i in range(azims.shape[0])],
#                                            dim=0)
#             f_brick_concat = self.brick_encoder(brick_occs_concat, azims_expand, elevs_expand, obj_scales_expand,
#                                                 obj_centers_expand)
#         else:
#             f_brick_concat = self.brick_encoder(brick_occs_concat)
#         brick_ct = 0
#         f_brick_list = []
#         for n_brick in n_bricks:
#             f_brick_list.append(f_brick_concat[brick_ct:brick_ct + n_brick])
#             brick_ct += n_brick
#         f_brick_padded = build_batch_from_list(f_brick_list)
#         outputs = self.hg(img_concat, f_brick_padded.reshape(f_brick_padded.shape[0], -1))
#         loss_sum, loss_dict = self.loss(outputs, target)
#
#         return outputs, loss_sum, loss_dict


from ..coordconv import AddCoords3D
from ..networks import conv2d, conv3d, occ_inds2fmap, upconv3d
import pytorch3d.renderer.cameras as prc
import pytorch3d.transforms as pt
import os

DEBUG_FMAP = bool(os.getenv("DEBUG_FMAP", 0) == '1')


# output meshgrid of the camera space
# sizes [..., H, W, D]
def ndc_meshgrid(size,device="cpu"):
    x = torch.linspace(1, -1, size[-3],device=device)
    y = torch.linspace(1, -1, size[-2],device=device)
    # set according training data's depth range
    z = torch.linspace(0.03, 0.06, size[-1],device=device)
    coords = torch.meshgrid(x, y, z)
    coords = torch.stack(coords, dim=-1)
    coords = coords.unsqueeze(0).expand(*size, 3)
    return coords


# 该编码器包含两个conv3d以及将3D特征图转为2D特征图的transform，每一个conv3d为[conv, bn, relu]，注意conv与bn为3D版的
class VoxelTransformEncoder(nn.Module):
    def __init__(self, num_features, feature_map2d_size, cuda=True, use_color=False, depth=128, mode='bilinear',
                 canonical_transform=pt.Translate(-65, -65 // 4, -65), stack_4rots=False, use_coordconv=True
                 ):
        '''
        :param stack_4rots: stacking occ for four different rotations
        '''
        super().__init__()

        self.addcoords = AddCoords3D()  # 添加XYZ坐标，默认不使用

        self.use_color = use_color
        num_input_features = (3 if use_color else 1)  # 默认1
        if use_coordconv:
            num_input_features += 3
        self.use_coordconv = use_coordconv

        self.stack_4rots = stack_4rots
        cnn_layers = conv3d(num_input_features, num_features,
                            stride=1)  # as we add xyz coordinate channels，输入为 1 channel，输出为 8 channel
        cnn_layers.extend(conv3d(num_features, num_features, stride=1))

        self.conv = nn.Sequential(*cnn_layers)

        self.feature_map2d_size = feature_map2d_size  # 2d feature map size
        self.depth = depth
        self.mode = mode

        # Scale each voxel to match the size of a real lego voxel
        scale_d = [20, 8, 20]
        device = 'cuda' if cuda else 'cpu'
        canonical_transform.to(device)  # 用于将特征图中的体素缩放到与真实乐高体素相匹配的大小
        # * 0.5 because the occupancy is 2x the original size
        s_mat = pt.Scale(scale_d[0] * 0.5, scale_d[1] * 0.5, scale_d[2] * 0.5, device=device)
        # Transform the voxel to world space
        canonical_transform = canonical_transform.compose(s_mat)
        self.canonical_transform = canonical_transform.inverse()
        # init model parameters
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, occ, azims, elevs, obj_scales, obj_centers, occ_color=None):
        '''
        :param occ: [B, 1, W, H, D]
        :param theta: [B, 3, 4]
        :return:
        '''
        self.canonical_transform = self.canonical_transform.to(occ.device)

        b_orig, _, w, h, d = occ.shape
        occ_trans = occ

        if self.use_coordconv:
            x = self.addcoords(occ_trans)
        else:
            x = occ_trans
        # 使用两个conv3d- batchNorm - relu模块提取3D表示
        x = self.conv(x)

        # Compute sampling grid
        b, c, w, h, d = x.shape
        grid_size = self.feature_map2d_size + (self.depth,)
        grid = ndc_meshgrid((b, 1, *grid_size), occ.device)
        R, T = prc.look_at_view_transform(dist=2000, elev=elevs, azim=azims, at=((0, 0, 0),), device=elevs.device)
        cameras = prc.OpenGLOrthographicCameras(device=R.device, R=R, T=T, scale_xyz=[(0.0024, 0.0024, 0.0024), ])
        transforms = pt.Transform3d(device=occ.device).translate(-obj_centers).scale(obj_scales).inverse()
        grid = grid.reshape(b, -1, 3)

        # Note that quaternion_to_matrix returns matrix that applies to column vectors
        # So we do not need to invert the quaternions because it will be transposed
        # when passed to Rotate class
        unprojection_transform = cameras.get_full_projection_transform().inverse()
        grid_s = unprojection_transform.compose(transforms).compose(self.canonical_transform).transform_points(grid)

        grid_s = grid_s.reshape(b, *grid_size, 3)
        grid_s = grid_s / torch.div(torch.as_tensor(occ.shape[2:], device=grid_s.device).expand_as(grid_s), 2,
                                    rounding_mode='floor') - 1  # normalize
        grid_s = grid_s[:, :, :, :, [2, 1, 0]]  # change to [D, H, W]

        # transpose: [B, C, W, H, D] -> [B, C, H', W', D']
        f_map = F.grid_sample(x, grid_s, mode=self.mode, align_corners=True).transpose(-3, -2)
        occ_map = (F.grid_sample(occ_trans, grid_s, mode='bilinear', align_corners=True) > 0.1).long().transpose(-3, -2)
        if self.stack_4rots:
            f_map = f_map.reshape(b_orig, 4, -1, *f_map.shape[-3:])
            occ_map = occ_map.reshape(b_orig, 4, 1, *f_map.shape[-3:])

        # masks for disabling voxel without occupancy
        occ_inds = occ_map.argmax(dim=-1).unsqueeze(-1)
        if DEBUG_FMAP:
            occ_inds2fmap(occ_inds.squeeze(dim=2), overlap=False)

        occ_inds = occ_inds.expand(*f_map.shape[:-1], 1)
        f_map = f_map.gather(dim=-1, index=occ_inds).squeeze(dim=-1)

        return f_map


class AvgPool(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], -1).mean(dim=-1)
        return x


class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.reshape(x.shape[0], -1)


class BrickVoxelProjectionEncoder(nn.Module):
    def __init__(self, num_features=64, feature_map_size=(32, 32)):
        super().__init__()
        num_features = num_features // 4  # as we compute features for all 4 rotations and then concatenate them.
        self.voxel_projector = VoxelTransformEncoder(num_features, feature_map_size,
                                                     canonical_transform=pt.Translate(-33, 0, -33),
                                                     stack_4rots=True)
        convs = []
        for _ in range(3):
            convs.extend(conv2d(num_features, num_features, stride=2))
        convs.append(AvgPool())
        self.conv = nn.Sequential(*convs)

    def forward(self, occ, azims, elevs, obj_scales, obj_centers):
        f_map = self.voxel_projector(occ, azims, elevs, None, obj_scales, obj_centers)
        b, n, c, w, h = f_map.shape
        f_map = self.conv(f_map.reshape(-1, c, w, h)).reshape(b, -1)
        return f_map


class VoxelEmbedding(nn.Module):
    def __init__(self, num_embedding, embedding_dim):
        super().__init__()
        # 0 corresponds to empty voxel
        self.emb = nn.Embedding(num_embedding, embedding_dim, padding_idx=0)

    def forward(self, v):
        # [batch_size, d, h, w, embedding_dim]
        v_emb = self.emb(v).squeeze(1)
        return v_emb.permute((0, 4, 1, 2, 3))


# 简易版的体素编码器，[Coords3D, conv3d * 5, (), AvgPool, flatten]
def simple_voxel_encoder(num_features, num_input_features=1, addconv=True, n_layers=4, avg_pool=True, num_embedding=0,
                         embedding_dim=0, no_out=False):
    # feature_map has size [130, 130, 130]
    layers = []
    if num_embedding > 0:
        layers.append(VoxelEmbedding(num_embedding, embedding_dim))
        num_input_features = embedding_dim
    if addconv:
        layers.append(AddCoords3D())
        layers.extend(conv3d(num_input_features + 3, num_features, stride=2))
    else:
        layers.extend(conv3d(num_input_features, num_features))
    for i in range(n_layers):
        layers.extend(conv3d(num_features, num_features, stride=2))
    if not no_out:
        if avg_pool:
            layers.extend([AvgPool(), Flatten()])
        else:
            layers.extend([Flatten(), nn.Linear(5 ** 3 * num_features, num_features)])

    return nn.Sequential(*layers)


def simple_voxel_decoder(num_features, num_output_features=1, n_layers=4):
    layers = []
    for i in range(n_layers):
        layers.extend(upconv3d(num_features, num_features, stride=2, output_padding=0))
    layers.extend(upconv3d(num_features, num_output_features, final_layer=True, stride=2, output_padding=0))
    return nn.Sequential(*layers)
