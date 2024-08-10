import torch
from torch import nn
from mmseg.models.utils import CoordAtt, AlignModule, AFF, Att_Module3, Att_Module4, CoordAtt2, CoordAtt3, \
    FeatureAlign_V2


class AttentionFusion1(nn.Module):
    def __init__(self, inplane, outplane):
        super(AttentionFusion1, self).__init__()
        self.inplane = inplane
        self.outplane = outplane
        self.attention_block = CoordAtt(inplane, outplane)
        self.align_block = AlignModule(inplane, outplane)

    def forward(self, x_l, x_h):
        w = self.attention_block(x_l, x_h)
        x_l = x_l * w
        x_h = x_h * w
        # print("AttentionFusion1，我运行了！")
        x = self.align_block((x_h, x_l))
        x = x_h + x

        return x


class AttentionFusion2(nn.Module):
    def __init__(self, inplane, outplane):
        super(AttentionFusion2, self).__init__()
        self.inplane = inplane      # 256
        self.outplane = outplane      # 512
        # self.attention_block = CoordAtt(inplane*2, outplane)
        # self.attention_block = CoordAtt2(inplane*2, outplane)
        self.attention_block = CoordAtt3(inplane*2, outplane)
        # self.attention_block = AFF(inplane*2)
        # self.attention_block = Att_Module3(inplane * 2, outplane)
        # self.attention_block = Att_Module4(inplane * 2, outplane)
        # self.align_block = FeatureAlign_V2(inplane, outplane)
        self.align_block = AlignModule(inplane, outplane)

        self.conv1 = nn.Conv2d(inplane, outplane, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(outplane*2, outplane, kernel_size=1, stride=1, padding=0)
    def forward(self,  x_h,x_l):
        # print("AttentionFusion2，我运行了！")
        # print("AttentionFusion2，self.inplane",self.inplane)
        # print("AttentionFusion2，self.outplane",self.outplane)

        x = self.align_block((x_h, x_l))

        # print("x.shape",x.shape)
        # print("x_h.shape", x_h.shape)
        x_h = self.conv1(x_h)
        # x = torch.cat([x, x_h], dim=1)
        #
        # x = self.conv2(x)
        # x = x + x_h
        w = self.attention_block(x, x_h)
        # print("w.shape",w.shape)
        # print("x_l.shape",x_l.shape)

        # x_l = x * w
        # x_h = x_h * (1-w)
        # x = x_l + x_h
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
        return w

class AttentionFusion3(nn.Module):
    def __init__(self, inplane, outplane):
        super(AttentionFusion3, self).__init__()
        self.inplane = inplane  # 256
        self.outplane = outplane  # 512
        # self.attention_block = CoordAtt(inplane*2, outplane)
        # self.attention_block = CoordAtt2(inplane*2, outplane)
        self.attention_block = CoordAtt3(inplane * 2, outplane)
        # self.attention_block = AFF(inplane*2)
        # self.attention_block = Att_Module3(inplane * 2, outplane)
        # self.attention_block = Att_Module4(inplane * 2, outplane)
        # self.align_block = FeatureAlign_V2(inplane, outplane)
        self.align_block = AlignModule(inplane, outplane)

        self.conv1 = nn.Conv2d(inplane, outplane, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(outplane * 2, outplane, kernel_size=1, stride=1, padding=0)

    def forward(self, x_h, x_l):

        x = self.align_block((x_h, x_l))

        # print("x.shape",x.shape)
        # print("x_h.shape", x_h.shape)
        x_h = self.conv1(x_h)
        x = torch.cat([x, x_h], dim=1)

        x = self.conv2(x)
        # x = x + x_h
        w = self.attention_block(x, x_l)
        # print("w.shape",w.shape)
        # print("x_l.shape",x_l.shape)

        # x_l = x * w
        # x_h = x_h * (1-w)
        # x = x_l + x_h
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
        return w
