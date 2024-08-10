from mmcv.ops import DeformConv2d as dcn_v2
import torch
# import torchvision as tv
# import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
import fvcore.nn.weight_init as weight_init

# 使用方法与官方DCNv2一样，只不过deformable_groups参数名改为deform_groups即可，例如：
# dconv2 = DCN(in_channel, out_channel, kernel_size=(3, 3), stride=(2, 2), padding=1, deform_groups=2)
class FeatureSelectionModule(nn.Module):
    def __init__(self, in_chan, out_chan, norm="GN"):
        super(FeatureSelectionModule, self).__init__()
        self.conv_atten = ConvModule(in_chan, in_chan, kernel_size=1, bias=False, norm_cfg=norm)
        self.sigmoid = nn.Sigmoid()
        self.conv = ConvModule(in_chan, out_chan, kernel_size=1, bias=False, norm_cfg=None)
        weight_init.c2_xavier_fill(self.conv_atten)
        weight_init.c2_xavier_fill(self.conv)

    def forward(self, x):
        atten = self.sigmoid(self.conv_atten(F.avg_pool2d(x, x.size()[2:])))
        feat = torch.mul(x, atten)
        x = x + feat
        feat = self.conv(x)
        return feat


class FeatureAlign_V2(nn.Module):  # FaPN full version
    def __init__(self, in_nc, out_nc, norm=None):
        super(FeatureAlign_V2, self).__init__()
        self.in_nc=in_nc
        self.out_nc=out_nc
        # self.lateral_conv = FeatureSelectionModule(in_nc, out_nc, norm="")
        self.offset = ConvModule(in_nc * 3, 18, kernel_size=1, stride=1, padding=0, bias=False, norm_cfg=norm)
        self.dcpack_L2 = dcn_v2(out_nc, out_nc, 3, stride=1, padding=1, dilation=1)
        self.relu = nn.ReLU(inplace=True)
        # weight_init.c2_xavier_fill(self.offset)

    def forward(self, x):
        feat_l, feat_s=x
        # 这里是对低分辨率特征进行插值到与高分辨率特征的shape相同
        HW = feat_l.size()[2:]
        if feat_l.size()[2:] != feat_s.size()[2:]:
            feat_up = F.interpolate(feat_s, HW, mode='bilinear', align_corners=False)
        else:
            feat_up = feat_s
        feat_arm = feat_l  # 0~1 * feats
        # print('self.in_nc',self.in_nc)  # 128
        # print('self.out_nc', self.out_nc)   # 256
        # print('feat_arm.shape',feat_arm.shape)
        # print('feat_up.shape',feat_up.shape)
        feat_all = torch.cat([feat_arm, feat_up * 2], dim=1)
        # print('feat_all',feat_all.shape)
        # 两个特征沿通道方向堆叠，然后进行offset
        offset = self.offset(feat_all)  # concat for offset by compute the dif
        # print('offset',offset.shape)
        feat_align=self.dcpack_L2(feat_up, offset)
        feat_align = self.relu(feat_align)  # [feat, offset]
        return feat_align
