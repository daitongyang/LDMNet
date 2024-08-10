# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.cnn import ConvModule, build_norm_layer
from mmengine.model import BaseModule

from mmseg.models.utils import DAPPM, BasicBlock, Bottleneck, resize, DDRFusionModule
from mmseg.registry import MODELS
from mmseg.utils import OptConfigType
import math
import cv2
import math

import cv2
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, build_norm_layer
from mmengine.model import BaseModule
from mmengine.visualization import Visualizer

from mmseg.models.utils import DAPPM, BasicBlock, Bottleneck, resize, PAPPM
from mmseg.registry import MODELS
from mmseg.utils import OptConfigType

@MODELS.register_module()
class LTbANetL(BaseModule):
    """LTbANet backbone.

    Args:
        in_channels (int): Number of input image channels. Default: 3.
        channels: (int): The base channels of DDRNet. Default: 32.
        ppm_channels (int): The channels of PPM module. Default: 128.
        align_corners (bool): align_corners argument of F.interpolate.
            Default: False.
        norm_cfg (dict): Config dict to build norm layer.
            Default: dict(type='BN', requires_grad=True).
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU', inplace=True).
        init_cfg (dict, optional): Initialization config dict.
            Default: None.
    """

    def __init__(self,
                 in_channels: int = 3,
                 channels: int = 32,
                 ppm_channels: int = 128,
                 align_corners: bool = False,
                 norm_cfg: OptConfigType = dict(type='BN', requires_grad=True),
                 act_cfg: OptConfigType = dict(type='ReLU', inplace=True),
                 init_cfg: OptConfigType = None):
        super().__init__(init_cfg)

        self.in_channels = in_channels
        self.ppm_channels = ppm_channels
        self.channels=channels
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.align_corners = align_corners

        # stage 0-2
        self.stem = self._make_stem_layer(in_channels, channels, num_blocks=2)
        self.relu = nn.ReLU()

        # low resolution(context) branch
        self.context_branch_layers = nn.ModuleList()
        for i in range(4):
            self.context_branch_layers.append(
                self._make_layer(
                    block=BasicBlock if i < 3 else Bottleneck,
                    inplanes=channels * (2 ** (i + 1)),
                    planes=channels * 8 if i > 2 else channels * (2 ** (i + 2)),
                    num_blocks=2 if i < 3 else 1,
                    stride=2))

        # bilateral fusion
        # self.compression_1 = ConvModule(
        #     channels * 4,
        #     channels * 2,
        #     kernel_size=1,
        #     norm_cfg=self.norm_cfg,
        #     act_cfg=None)
        self.down_1 = ConvModule(
            channels * 2,
            channels * 4,
            kernel_size=3,
            stride=2,
            padding=1,
            norm_cfg=self.norm_cfg,
            act_cfg=None)

        #*16
        # self.down_2 = nn.Sequential(
        #     ConvModule(
        #         channels * 2,
        #         channels * 8,
        #         kernel_size=3,
        #         stride=2,
        #         padding=1,
        #         norm_cfg=self.norm_cfg,
        #         act_cfg=self.act_cfg),
        #     ConvModule(
        #         channels * 8,
        #         channels * 16,
        #         kernel_size=3,
        #         stride=2,
        #         padding=2,
        #         dilation=2,
        #         norm_cfg=self.norm_cfg,
        #         act_cfg=None))
        # self.down_3 = nn.Sequential(
        #     ConvModule(
        #         channels * 4,
        #         channels * 8,
        #         kernel_size=3,
        #         stride=2,
        #         padding=1,
        #         norm_cfg=self.norm_cfg,
        #         act_cfg=self.act_cfg),
        #     ConvModule(
        #         channels * 8,
        #         channels * 16,
        #         kernel_size=3,
        #         stride=2,
        #         padding=2,
        #         dilation=2,
        #         norm_cfg=self.norm_cfg,
        #         act_cfg=None))
        # self.down_4 = nn.Sequential(
        #     ConvModule(
        #         channels * 2,
        #         channels * 8,
        #         kernel_size=3,
        #         stride=2,
        #         padding=1,
        #         norm_cfg=self.norm_cfg,
        #         act_cfg=self.act_cfg),
        #     ConvModule(
        #         channels * 8,
        #         channels * 12,
        #         kernel_size=3,
        #         stride=2,
        #         padding=1,
        #         norm_cfg=self.norm_cfg,
        #         act_cfg=self.act_cfg),
        #     ConvModule(
        #         channels * 12,
        #         channels * 16,
        #         kernel_size=3,
        #         stride=2,
        #         padding=2,
        #         dilation=2,
        #         norm_cfg=self.norm_cfg,
        #         act_cfg=None)
        # )
        # self.conv_s = ConvModule(
        #     channels * 2,
        #     channels * 8,
        #     kernel_size=3,
        #     stride=1,
        #     padding=1,
        #     norm_cfg=self.norm_cfg,
        #     act_cfg=None)
        # self.compression_1 = ConvModule(
        #     channels * 8,
        #     channels * 2,
        #     kernel_size=1,
        #     norm_cfg=self.norm_cfg,
        #     act_cfg=None)
        # self.compression_2 = ConvModule(
        #     channels * 16,
        #     channels * 8,
        #     kernel_size=1,
        #     norm_cfg=self.norm_cfg,
        #     act_cfg=None)
        # self.compression_3 = ConvModule(
        #     channels * 16,
        #     channels * 16,
        #     kernel_size=1,
        #     norm_cfg=self.norm_cfg,
        #     act_cfg=None)
        # self.compression_4 = ConvModule(
        #     channels * 16,
        #     channels * 16,
        #     kernel_size=1,
        #     norm_cfg=self.norm_cfg,
        #     act_cfg=None)
        # self.compression_5 = ConvModule(
        #     channels * 8,
        #     channels * 4,
        #     kernel_size=1,
        #     norm_cfg=self.norm_cfg,
        #     act_cfg=None)

        #*32
        self.down_2 = nn.Sequential(
            ConvModule(
                channels * 2,
                channels * 8,
                kernel_size=3,
                stride=2,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg),
            ConvModule(
                channels * 8,
                channels * 16,
                kernel_size=3,
                stride=2,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=None))
        self.down_3 = nn.Sequential(
            ConvModule(
                channels * 4,
                channels * 16,
                kernel_size=3,
                stride=2,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg),
            ConvModule(
                channels * 16,
                channels * 32,
                kernel_size=3,
                stride=2,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=None))
        self.down_4 = nn.Sequential(
            ConvModule(
                channels * 2,
                channels * 8,
                kernel_size=3,
                stride=2,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg),
            ConvModule(
                channels * 8,
                channels * 16,
                kernel_size=3,
                stride=2,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg),
            ConvModule(
                channels * 16,
                channels * 32,
                kernel_size=3,
                stride=2,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=None)
        )
        self.conv_s = ConvModule(
            channels * 2,
            channels * 8,
            kernel_size=3,
            stride=1,
            padding=1,
            norm_cfg=self.norm_cfg,
            act_cfg=None)
        self.compression_1 = ConvModule(
            channels * 8,
            channels * 2,
            kernel_size=1,
            norm_cfg=self.norm_cfg,
            act_cfg=None)
        self.compression_2 = ConvModule(
            channels * 16,
            channels * 8,
            kernel_size=1,
            norm_cfg=self.norm_cfg,
            act_cfg=None)
        self.compression_3 = ConvModule(
            channels * 32,
            channels * 16,
            kernel_size=1,
            norm_cfg=self.norm_cfg,
            act_cfg=None)
        self.compression_4 = ConvModule(
            channels * 32,
            channels * 16,
            kernel_size=1,
            norm_cfg=self.norm_cfg,
            act_cfg=None)
        self.compression_5 = ConvModule(
            channels * 8,
            channels * 4,
            kernel_size=1,
            norm_cfg=self.norm_cfg,
            act_cfg=None)
        self.compression_6 = ConvModule(
            channels * 16,
            channels * 4,
            kernel_size=1,
            norm_cfg=self.norm_cfg,
            act_cfg=None)
        # high resolution(spatial) branch
        self.spatial_branch_layers = nn.ModuleList()
        for i in range(3):
            self.spatial_branch_layers.append(
                self._make_layer(
                    block=BasicBlock if i < 2 else Bottleneck,
                    inplanes=channels * 2,
                    planes=channels * 2,
                    num_blocks=2 if i < 2 else 1,
                ))

        # self.spp = DAPPM(
        #     channels * 16, ppm_channels, channels * 4, num_scales=5)
        self.spp = PAPPM(
            channels * 16, ppm_channels, channels * 4, num_scales=5)
        self.DDRFusionModule = DDRFusionModule(channels * 4, channels * 4, channels * 4)

    def _make_stem_layer(self, in_channels, channels, num_blocks):
        layers = [
            ConvModule(
                in_channels,
                channels,
                kernel_size=3,
                stride=2,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg),
            ConvModule(
                channels,
                channels,
                kernel_size=3,
                stride=2,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
        ]

        layers.extend([
            self._make_layer(BasicBlock, channels, channels, num_blocks),
            nn.ReLU(),
            self._make_layer(
                BasicBlock, channels, channels * 2, num_blocks, stride=2),
            nn.ReLU(),
        ])

        return nn.Sequential(*layers)

    def _make_layer(self, block, inplanes, planes, num_blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False),
                build_norm_layer(self.norm_cfg, planes * block.expansion)[1])

        layers = [
            block(
                in_channels=inplanes,
                channels=planes,
                stride=stride,
                downsample=downsample)
        ]
        inplanes = planes * block.expansion
        for i in range(1, num_blocks):
            layers.append(
                block(
                    in_channels=inplanes,
                    channels=planes,
                    stride=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg_out=None if i == num_blocks - 1 else self.act_cfg))

        return nn.Sequential(*layers)

    def forward(self, x):
        """Forward function."""
        out_size = (math.ceil(x.shape[-2] / 8), math.ceil(x.shape[-1] / 8))
        # out_size = (x.shape[-2] // 8, x.shape[-1] // 8)
        # stage 0-2
        x = self.stem(x)

        # stage3
        x_c = self.context_branch_layers[0](x)
        x_s = self.spatial_branch_layers[0](x)
        # x_c = x_c + self.down_1(self.relu(x_s))
        # comp_c = self.compression_1(self.relu(x_c))
        # x_s = x_s + resize(
        #     comp_c,
        #     size=out_size,
        #     mode='bilinear',
        #     align_corners=self.align_corners)

        x_c_res3 = x_c.clone()  # [4, 256, 64, 64]
        x_s_res3 = x_s.clone()  # [4,128,128,128]

        if self.training:
            temp_context = x_s.clone()

        # stage4
        x_c = self.context_branch_layers[1](self.relu(x_c))
        x_s = self.spatial_branch_layers[1](self.relu(x_s))
        x_c = x_c + self.compression_2(self.down_2(self.relu(x_s_res3)))

        comp_c = self.compression_1(self.relu(x_c))
        x_s = x_s + resize(
            comp_c,
            size=out_size,
            mode='bilinear',
            align_corners=self.align_corners)
        x_s_res4 = x_s.clone()

        # stage5
        x_s = self.spatial_branch_layers[2](self.relu(x_s))
        x_c = self.context_branch_layers[2](self.relu(x_c))
        x_c = x_c + self.compression_4(self.down_4(self.relu(x_s_res4)))
        # 添加上下文分支残差
        x_c_res_comp = self.compression_3(self.down_3(self.relu(x_c_res3)))  # [4, 1024, 16, 16]
        x_c = x_c + x_c_res_comp

        # 添加高分辨率分支残差
        x_s_res_comp = self.compression_5(self.conv_s(self.relu(x_s_res3)))
        # 空间高分辨率分支的相加
        x_s = x_s + x_s_res_comp
        x_c_m = x_c.clone()
        # stage6
        x_c1 = self.context_branch_layers[3](self.relu(x_c))
        x_c1=self.compression_6(self.relu(x_c1))
        # x = self.context_branch_layers[3](self.relu(x_c))
        # x = self.DDRFusionModule((x_s, x_c_m, x_c))
        x_c = self.spp(x_c_m)
        # print(x_s.shape)
        # print(x_c.shape)
        # print(x_c1.shape)
        out_size_s = (x_s.shape[-2], x_s.shape[-1])
        x = x_s + resize(
            x_c,
            size=out_size_s,
            mode='bilinear',
            align_corners=None)+resize(
            x_c1,
            size=out_size_s,
            mode='bilinear',
            align_corners=None)
        # x_c = resize(
        #     x_c,
        #     size=out_size,
        #     mode='bilinear',
        #     align_corners=self.align_corners)

        # image = cv2.imread("E:\\daity\\mmSegmentationOri\\mmsegmentation\\demo\\aachen_000001_000019_leftImg8bit.png")
        # visualizer = Visualizer()
        # drawn_img = visualizer.draw_featmap(x_fusion[0], image,
        #                                     channel_reduction=None, topk=6, arrangement=(3, 3))
        # visualizer.show(drawn_img)
        # visualizer.add_image('feat', drawn_img)

        # feats = [x_s, x_c_m, x_c]
        # visualizer = Visualizer()
        # for feat in feats:
        #     drawn_img = visualizer.draw_featmap(feat[0], channel_reduction='select_max')
        #     visualizer.show(drawn_img)
        #     visualizer.add_image('feat' + str(feat.shape) , drawn_img)

        return (temp_context, x) if self.training else x
