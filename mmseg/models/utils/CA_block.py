import cv2
import torch
from mmengine.visualization import Visualizer
from torch import nn
from mmseg.models.utils import resize
from mmcv.cnn import ConvModule

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, groups=32):
        super(CoordAtt, self).__init__()
        self.inp=inp    # 512
        self.oup=oup    # 512
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // groups)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.conv2 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.relu = h_swish()
        # self.compress_1 = nn.Conv2d(inp, inp // 2, kernel_size=1, stride=1, padding=0)
        self.compression_1 = ConvModule(
            inp, inp // 2,
            kernel_size=1,
            norm_cfg=None,
            act_cfg=None)

    def forward(self, x, residual):
        # print("ca.self.inp",self.inp)
        # print("ca.self.oup",self.oup)
        out_size = (x.shape[-2], x.shape[-1])
        # residual = self.compression_1(residual)
        # print(x.shape)  # [2, 72, 32, 64]
        # print(residual.shape)  # [2, 144, 16, 32]
        # c = resize(
        #     residual,
        #     size=out_size,
        #     mode='bilinear',
        #     align_corners=False)
        # print(c.shape)  # [2, 144, 16, 32]
        x = x + resize(
            residual,
            size=out_size,
            mode='bilinear',
            align_corners=False)
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        # print("ca中的x_h，x_w的shape")
        # print(x_h.shape, x_w.shape)
        y = torch.cat([x_h, x_w], dim=2)
        # print("连接后的y的shape：")
        # print(y.shape)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.relu(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        x_h = self.conv2(x_h).sigmoid()
        x_w = self.conv3(x_w).sigmoid()
        x_h = x_h.expand(-1, -1, h, w)
        x_w = x_w.expand(-1, -1, h, w)

        wei = x_w * x_h
        x1 = 2 * x * wei
        x2 = 2 * residual * (1 - wei)

        xo = x1 + x2

        return xo

class CoordAtt2(nn.Module):
    def __init__(self, inp, oup, groups=32):
        super(CoordAtt2, self).__init__()
        self.inp=inp    # 512
        self.oup=oup    # 512
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // groups)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.conv2 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.relu = h_swish()
        # self.compress_1 = nn.Conv2d(inp, inp // 2, kernel_size=1, stride=1, padding=0)
        self.compression_1 = ConvModule(
            inp, inp // 2,
            kernel_size=1,
            norm_cfg=None,
            act_cfg=None)

    def forward(self, x, residual):
        # print("ca.self.inp",self.inp)
        # print("ca.self.oup",self.oup)
        out_size = (x.shape[-2], x.shape[-1])
        # residual = self.compression_1(residual)
        # print(x.shape)  # [2, 72, 32, 64]
        # print(residual.shape)  # [2, 144, 16, 32]
        # c = resize(
        #     residual,
        #     size=out_size,
        #     mode='bilinear',
        #     align_corners=False)
        # print(c.shape)  # [2, 144, 16, 32]
        x = x + resize(
            residual,
            size=out_size,
            mode='bilinear',
            align_corners=False)
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        # print("ca中的x_h，x_w的shape")
        # print(x_h.shape, x_w.shape)
        y = torch.cat([x_h, x_w], dim=2)
        # print("连接后的y的shape：")
        # print(y.shape)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.relu(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        # print('x_h, x_w1',x_h.shape, x_w.shape)
        x_h = self.conv2(x_h)
        x_w = self.conv3(x_w)
        #         print('x_h, x_w2', x_h.shape, x_w.shape)
        #         # x_h = x_h.expand(-1, -1, h, w)
        #         # x_w = x_w.expand(-1, -1, h, w)
        # print('x_h, x_w3', x_h.shape, x_w.shape)
        wei = x_w * x_h
        # print('wei', wei.shape)
        wei = wei.sigmoid()
        # print('wei', wei.shape)
        # print('x', x.shape)
        x1 = 2 * x * wei
        x2 = 2 * residual * (1 - wei)

        xo = x1 + x2

        return xo

class CoordAtt3(nn.Module):
    def __init__(self, inp, oup, groups=32):
        super(CoordAtt3, self).__init__()
        self.inp=inp    # 512
        self.oup=oup    # 512
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveMaxPool2d((1, None))

        mip = max(8, inp // groups)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.conv2 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.relu = h_swish()
        self.compression_1 = ConvModule(
            inp, inp // 2,
            kernel_size=1,
            norm_cfg=None,
            act_cfg=None)

    def forward(self, x_, residual):
        out_size = (x_.shape[-2], x_.shape[-1])
        x = x_ + resize(
            residual,
            size=out_size,
            mode='bilinear',
            align_corners=False)
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.relu(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        x_h = self.conv2(x_h)
        x_w = self.conv3(x_w)
        # x_h = x_h.expand(-1, -1, h, w)
        # x_w = x_w.expand(-1, -1, h, w)
        wei = x_w * x_h
        wei = wei.sigmoid()
        x1 = 2 * x_ * wei
        x2 = 2 * residual * (1 - wei)

        xo = x1 + x2
        # image = cv2.imread("E:\\daity\\dataset\\MaSTr1325\\images\\train\\0463.jpg")
        # feats = [x, x_, residual, x1, x2, xo]
        # visualizer = Visualizer()
        # for feat in feats:
        #     drawn_img = visualizer.draw_featmap(feat[0], image,
        #                                         channel_reduction=None, topk=9, arrangement=(3, 3))
        #     visualizer.show(drawn_img)
        #     visualizer.add_image('feat', drawn_img)

        # feats = [x, x_, residual, x1, x2, xo]
        # visualizer = Visualizer()
        # for feat in feats:
        #     drawn_img = visualizer.draw_featmap(feat[0], image, channel_reduction='select_max')
        #     visualizer.show(drawn_img)
        #     visualizer.add_image('feat' + str(feat.shape) , drawn_img)
        return xo

class AFF(nn.Module):
    '''
    多特征融合 AFF
    '''

    def __init__(self, channels, r=4):
        super(AFF, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()
        # self.FeatureAlign = utils.FeatureAlign_V2(channels,channels)

    def forward(self, x, residual):
        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        x1 = 2 * x * wei
        x2 = 2 * residual * (1 - wei)

        xo = x1 + x2

        # image = cv2.imread("E:\\daity\\dataset\\MaSTr1325\\images\\train\\0463.jpg")
        # feats = [xa, x, residual, x1, x2, xo]
        # visualizer = Visualizer()
        # for feat in feats:
        #     drawn_img = visualizer.draw_featmap(feat[0], image,
        #                                         channel_reduction=None, topk=9, arrangement=(3, 3))
        #     visualizer.show(drawn_img)
        #     visualizer.add_image('feat', drawn_img)

        # feats = [x, x_, residual, x1, x2, xo]
        # visualizer = Visualizer()
        # for feat in feats:
        #     drawn_img = visualizer.draw_featmap(feat[0], image, channel_reduction='select_max')
        #     visualizer.show(drawn_img)
        #     visualizer.add_image('feat' + str(feat.shape) , drawn_img)
        return xo


class Att_Module3(nn.Module):
    def __init__(self, inp, oup, groups=32):
        super(Att_Module3, self).__init__()
        mip = int(inp // groups)
        self.globalAvg = nn.AdaptiveAvgPool2d((None, 1))
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.conv2 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.relu = h_swish()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        x = x + residual
        n, c, h, w = x.size()
        # x_h = torch.split(x, 1, dim=2)
        #
        # x_h = torch.cat(x_h, dim=2)
        x_h = self.globalAvg(x)
        # print('x_h.shape',x_h.shape)
        x_w = torch.split(x, 1, dim=2)
        # print('x_w.shape', x_w[1].shape)
        # x_w = torch.cat(x_w, dim=3)
        # print('x_wx_h', x_w.shape, x_h.shape)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.relu(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        x_h = self.conv2(x_h).sigmoid()
        x_w = self.conv3(x_w).sigmoid()
        x_h = x_h.expand(-1, -1, h, w)
        x_w = x_w.expand(-1, -1, h, w)

        wei = x_w * x_h
        x1 = 2 * x * wei
        x2 = 2 * residual * (1 - wei)

        xo = x1 + x2

        return xo


class Att_Module4(nn.Module):
    def __init__(self, inp, oup, groups=32):
        super(Att_Module4, self).__init__()
        mip = int(inp // groups)
        self.globalAvg = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.conv2 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.relu = h_swish()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        x = x + residual
        n, c, h, w = x.size()
        x_h = torch.split(x, 1, dim=2)
        x_h = torch.cat(x_h, dim=2)
        x_w = torch.split(x, 1, dim=3)
        x_w = torch.cat(x_w, dim=3)
        x_h = self.globalAvg(x_h)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.relu(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        x_h = self.conv2(x_h)
        x_w = self.conv3(x_w)

        x_h = x_h.expand(-1, -1, h, w)
        x_w = x_w.expand(-1, -1, h, w)
        wei = torch.cat((x_h, x_w), dim=2)
        wei = self.sigmoid(wei)
        # wei = x_w * x_h

        x1 = 2 * x * wei
        x2 = 2 * residual * (1 - wei)

        xo = x1 + x2

        return xo
