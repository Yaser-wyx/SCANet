from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn


class convolution(nn.Module):
    def __init__(self, k, inp_dim, out_dim, stride=1, with_bn=True):
        super(convolution, self).__init__()

        pad = (k - 1) // 2
        self.conv = nn.Conv2d(inp_dim, out_dim, (k, k), padding=(pad, pad), stride=(stride, stride), bias=not with_bn)
        self.bn = nn.BatchNorm2d(out_dim) if with_bn else nn.Sequential()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        conv = self.conv(x)
        bn = self.bn(conv)
        relu = self.relu(bn)
        return relu


class fully_connected(nn.Module):
    def __init__(self, inp_dim, out_dim, with_bn=True):
        super(fully_connected, self).__init__()
        self.with_bn = with_bn

        self.linear = nn.Linear(inp_dim, out_dim)
        if self.with_bn:
            self.bn = nn.BatchNorm1d(out_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        linear = self.linear(x)
        bn = self.bn(linear) if self.with_bn else linear
        relu = self.relu(bn)
        return relu


class residual(nn.Module):
    def __init__(self, k, inp_dim, out_dim, stride=1, with_bn=True):
        super(residual, self).__init__()

        self.conv1 = nn.Conv2d(inp_dim, out_dim, (3, 3), padding=(1, 1), stride=(stride, stride), bias=False)
        self.bn1 = nn.BatchNorm2d(out_dim)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_dim, out_dim, (3, 3), padding=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(out_dim)

        self.skip = nn.Sequential(
            nn.Conv2d(inp_dim, out_dim, (1, 1), stride=(stride, stride), bias=False),
            nn.BatchNorm2d(out_dim)
        ) if stride != 1 or inp_dim != out_dim else nn.Sequential()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        conv1 = self.conv1(x)
        bn1 = self.bn1(conv1)
        relu1 = self.relu1(bn1)

        conv2 = self.conv2(relu1)
        bn2 = self.bn2(conv2)

        skip = self.skip(x)
        return self.relu(bn2 + skip)


def make_layer(k, inp_dim, out_dim, modules, layer=convolution, **kwargs):
    layers = [layer(k, inp_dim, out_dim, **kwargs)]
    for _ in range(1, modules):
        layers.append(layer(k, out_dim, out_dim, **kwargs))
    return nn.Sequential(*layers)


def make_layer_revr(k, inp_dim, out_dim, modules, layer=convolution, **kwargs):
    layers = []
    for _ in range(modules - 1):
        layers.append(layer(k, inp_dim, inp_dim, **kwargs))
    layers.append(layer(k, inp_dim, out_dim, **kwargs))
    return nn.Sequential(*layers)


class MergeUp(nn.Module):
    def forward(self, up1, up2):
        return up1 + up2


def make_merge_layer(dim):
    return MergeUp()


# def make_pool_layer(dim):
#     return nn.MaxPool2d(kernel_size=2, stride=2)

def make_pool_layer(dim):
    return nn.Sequential()


def make_unpool_layer(dim):
    return nn.Upsample(scale_factor=2)


def make_kp_layer(cnv_dim, curr_dim, out_dim):
    return nn.Sequential(
        convolution(3, cnv_dim, curr_dim, with_bn=False),
        nn.Conv2d(curr_dim, out_dim, (1, 1))
    )


def make_inter_layer(dim):
    return residual(3, dim, dim)


def make_cnv_layer(inp_dim, out_dim):
    return convolution(3, inp_dim, out_dim)


class kp_module(nn.Module):
    def __init__(
            self, n, dims, modules, layer=residual,
            make_up_layer=make_layer, make_low_layer=make_layer,
            make_hg_layer=make_layer, make_hg_layer_revr=make_layer_revr,
            make_pool_layer=make_pool_layer, make_unpool_layer=make_unpool_layer,
            make_merge_layer=make_merge_layer,
            cond_emb_dim=0,
            **kwargs
    ):
        super(kp_module, self).__init__()

        self.n = n
        self.cond_emb_dim = cond_emb_dim

        curr_mod = modules[0]
        next_mod = modules[1]

        curr_dim = dims[0]
        next_dim = dims[1]

        self.up1 = make_up_layer(
            3, curr_dim, curr_dim, curr_mod,
            layer=layer, **kwargs
        )
        self.max1 = make_pool_layer(curr_dim)
        self.low1 = make_hg_layer(
            3, curr_dim, next_dim, curr_mod,
            layer=layer, **kwargs
        )

        self.low2 = kp_module(
            n - 1, dims[1:], modules[1:], layer=layer,
            make_up_layer=make_up_layer,
            make_low_layer=make_low_layer,
            make_hg_layer=make_hg_layer,
            make_hg_layer_revr=make_hg_layer_revr,
            make_pool_layer=make_pool_layer,
            make_unpool_layer=make_unpool_layer,
            make_merge_layer=make_merge_layer,
            cond_emb_dim=cond_emb_dim,
            **kwargs
        ) if self.n > 1 else \
            make_low_layer(
                3, next_dim + cond_emb_dim, next_dim, next_mod,
                layer=layer, **kwargs
            )
        self.low3 = make_hg_layer_revr(
            3, next_dim, curr_dim, curr_mod,
            layer=layer, **kwargs
        )
        self.up2 = make_unpool_layer(curr_dim)

        self.merge = make_merge_layer(curr_dim)

    def forward(self, x, cond_emb=None):
        up1 = self.up1(x)
        max1 = self.max1(x)
        low1 = self.low1(max1)  # 进行一次下采样
        if self.cond_emb_dim == 0:
            low2 = self.low2(low1)
        else:
            if self.n == 1:
                # broadcast cond_emb to match the shape of feature map
                cond_emb_exp = cond_emb.reshape(*cond_emb.shape, 1, 1)  # 进行broadcast
                cond_emb_exp = cond_emb_exp.expand(-1, -1, low1.shape[-2], low1.shape[-1])
                low1 = torch.cat([low1, cond_emb_exp], dim=1)  # 将图像的特征图与component的feature map进行拼接，只会拼接一次，也就是递归的最深层时融合一次
                low2 = self.low2(low1)
            else:
                low2 = self.low2(low1, cond_emb=cond_emb)  # 递归当前模块，递归三次，也就是会进行3次降采样（2^3），以及3次的上采样（2^3）
        low3 = self.low3(low2)
        up2 = self.up2(low3)  # 上采样
        return self.merge(up1, up2)  # 相加操作


class exkp(nn.Module):
    def __init__(
            self, n, nstack, dims, modules, heads, pre=None, cnv_dim=256,
            make_tl_layer=None, make_br_layer=None,
            make_cnv_layer=make_cnv_layer, make_heat_layer=make_kp_layer,
            make_tag_layer=make_kp_layer, make_regr_layer=make_kp_layer,
            make_up_layer=make_layer, make_low_layer=make_layer,
            make_hg_layer=make_layer, make_hg_layer_revr=make_layer_revr,
            make_pool_layer=make_pool_layer, make_unpool_layer=make_unpool_layer,
            make_merge_layer=make_merge_layer, make_inter_layer=make_inter_layer,
            kp_layer=residual,
            img_nc=3,
            # If this is not 0, the hourglass net receives an additional 1 dimension embedding that is concatenated
            # to the feature map with lowest resolution.
            cond_emb_dim=0,
            downsample_2x=False
    ):
        super(exkp, self).__init__()

        self.cond_emb_dim = cond_emb_dim
        self.nstack = nstack
        self.heads = heads

        curr_dim = dims[0]
        # fig3(a){ii}的encoder
        self.pre = nn.Sequential(
            convolution(7, img_nc, 128, stride=1 if downsample_2x else 2),
            residual(3, 128, 256, stride=2)
        ) if pre is None else pre

        self.kps = nn.ModuleList([
            kp_module(
                n, dims, modules, layer=kp_layer,
                make_up_layer=make_up_layer,
                make_low_layer=make_low_layer,
                make_hg_layer=make_hg_layer,
                make_hg_layer_revr=make_hg_layer_revr,
                make_pool_layer=make_pool_layer,
                make_unpool_layer=make_unpool_layer,
                make_merge_layer=make_merge_layer,
                cond_emb_dim=cond_emb_dim
            ) for _ in range(nstack)
        ])
        self.cnvs = nn.ModuleList([
            make_cnv_layer(curr_dim, cnv_dim) for _ in range(nstack)
        ])

        self.inters = nn.ModuleList([
            make_inter_layer(curr_dim) for _ in range(nstack - 1)
        ])

        self.inters_ = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(curr_dim, curr_dim, (1, 1), bias=False),
                nn.BatchNorm2d(curr_dim)
            ) for _ in range(nstack - 1)
        ])
        self.cnvs_ = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(cnv_dim, curr_dim, (1, 1), bias=False),
                nn.BatchNorm2d(curr_dim)
            ) for _ in range(nstack - 1)
        ])

        # keypoint heatmaps
        if heads is not None:
            for head in heads.keys():
                if 'hm' in head:
                    module = nn.ModuleList([
                        make_heat_layer(
                            cnv_dim, curr_dim, heads[head]) for _ in range(nstack)
                    ])
                    self.__setattr__(head, module)
                    for heat in self.__getattr__(head):
                        heat[-1].bias.data.fill_(-2.19)
                else:
                    module = nn.ModuleList([
                        make_regr_layer(
                            cnv_dim, curr_dim, heads[head]) for _ in range(nstack)
                    ])
                    self.__setattr__(head, module)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, image, cond_emb=None):
        if self.cond_emb_dim > 0:
            assert cond_emb is not None
        else:
            assert cond_emb is None

        inter = self.pre(image)  # top-down encoder，下采样4倍
        outs = []

        for ind in range(self.nstack):  # 两个堆叠结构Stacked Hourglass Networks
            kp_, cnv_ = self.kps[ind], self.cnvs[ind]
            kp = kp_(inter, cond_emb)  # 解码器，输入encoder的中间结果以及组件的编码结果
            cnv = cnv_(kp)  # 将解码器的结果再进行一次卷积
            # 预测头的输出
            out = {}
            if self.heads is not None:
                for head in self.heads:
                    layer = self.__getattr__(head)[ind]
                    y = layer(cnv)
                    out[head] = y
            else:
                out = cnv
            outs.append(out)
            if ind < self.nstack - 1:
                inter = self.inters_[ind](inter) + self.cnvs_[ind](cnv)
                inter = self.relu(inter)
                inter = self.inters[ind](inter)
        return outs


def make_hg_layer(kernel, dim0, dim1, mod, layer=convolution, **kwargs):
    layers = [layer(kernel, dim0, dim1, stride=2)]
    layers += [layer(kernel, dim1, dim1) for _ in range(mod - 1)]
    return nn.Sequential(*layers)


class HourglassNet(exkp):
    def __init__(self, heads, num_stacks=2, img_nc=3, cond_emb_dim=0, downsample_2x=False):
        # n = 5
        # dims = [256, 256, 384, 384, 384, 512]
        # modules = [2, 2, 2, 2, 2, 4]
        n = 3
        dims = [256, 384, 384, 512]
        modules = [1, 1, 1, 1]

        super(HourglassNet, self).__init__(
            n, num_stacks, dims, modules, heads,
            make_tl_layer=None,
            make_br_layer=None,
            make_pool_layer=make_pool_layer,
            make_hg_layer=make_hg_layer,
            kp_layer=residual, cnv_dim=256, img_nc=img_nc,
            cond_emb_dim=cond_emb_dim,
            downsample_2x=downsample_2x
        )


def get_hourglass_net(heads, img_nc, num_stacks=2, cond_emb_dim=0, downsample_2x=False):
    model = HourglassNet(heads, num_stacks, img_nc=img_nc, cond_emb_dim=cond_emb_dim, downsample_2x=downsample_2x)
    return model
