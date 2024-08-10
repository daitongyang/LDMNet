import torch
from torch import nn
import torch.nn.functional as F


class AlignModule(nn.Module):
    def __init__(self, inplane: int = 256,
                 outplane: int = 512):
        super(AlignModule, self).__init__()
        self.inplane = inplane
        self.outplane = outplane
        self.down_h = nn.Conv2d(inplane * 2, outplane, 1, bias=False)    # 低分辨率
        self.down_l = nn.Conv2d(inplane, outplane, 1, bias=False)     # 高分辨率
        self.flow_make = nn.Conv2d(outplane * 2, 2, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        # print("AlignModule.self.inplane",self.inplane)
        # print("AlignModule.self.outplane",self.outplane)

        low_feature, h_feature = x  # low_feature 对应分辨率较高的特征图，h_feature即为低分辨率的high-level feature
        # print("low_feature.shape",low_feature.shape)
        # print("h_feature.shape",h_feature.shape)
        h_feature_orign = h_feature
        h, w = low_feature.size()[2:]
        size = (h, w)
        # 将high-level 和 low-level feature分别通过两个1x1卷积进行压缩
        low_feature = self.down_l(low_feature)
        h_feature = self.down_h(h_feature)
        # 将high-level feature进行双线性上采样
        h_feature = F.interpolate(h_feature, size=size, mode="bilinear", align_corners=False)
        # 预测语义流场 === 其实就是输入一个3x3的卷积
        flow = self.flow_make(torch.cat([h_feature, low_feature], 1))
        # 将Flow Field warp 到当前的 high-level feature中
        h_feature = self.flow_warp(h_feature_orign, flow, size=size)

        return h_feature

    @staticmethod
    def flow_warp(inputs, flow, size):
        out_h, out_w = size  # 对应高分辨率的low-level feature的特征图尺寸
        n, c, h, w = inputs.size()  # 对应低分辨率的high-level feature的4个输入维度

        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(inputs).to(inputs.device)
        # 从-1到1等距离生成out_h个点，每一行重复out_w个点，最终生成(out_h, out_w)的像素点
        w = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        # 生成w的转置矩阵
        h = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        # 展开后进行合并
        grid = torch.cat((h.unsqueeze(2), w.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(inputs).to(inputs.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm
        # grid指定由input空间维度归一化的采样像素位置，其大部分值应该在[ -1, 1]的范围内
        # 如x=-1,y=-1是input的左上角像素，x=1,y=1是input的右下角像素。
        # 具体可以参考《Spatial Transformer Networks》，下方参考文献[2]
        output = F.grid_sample(inputs, grid)
        return output
