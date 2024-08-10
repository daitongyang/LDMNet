import torch
from torch import nn
from mmseg.models.utils import AttentionFusion1, AttentionFusion2, BAPPM, resize, PAPPM, AttentionFusion3, DAPPM


class DDRFusionModule(nn.Module):
    def __init__(self,
                 inplane: int = 128,
                 ppmplane: int = 256,
                 outplane: int = 128):
        super(DDRFusionModule, self).__init__()
        self.inplane = inplane
        self.ppmplane = ppmplane
        # self.attention_fusion_block1_1 = AttentionFusion1((inplane // 15) * 4, (inplane // 15) * 4)  # (72,)
        # self.attention_fusion_block1_2 = AttentionFusion1((inplane // 15) * 2, (inplane // 15) * 2)
        # self.attention_fusion_block1_3 = AttentionFusion1(inplane // 15, inplane // 15)
        self.attention_fusion_block2_1 = AttentionFusion2(inplane, inplane *2)  # (144,144)
        self.attention_fusion_block2_2 = AttentionFusion2(inplane, inplane * 2)  # (72,36)
        # self.attention_fusion_block2_1 = AttentionFusion3(inplane, inplane * 2)  # (144,144)
        # self.attention_fusion_block2_2 = AttentionFusion3(inplane, inplane * 2)  # (72,36)
        # self.attention_fusion_block2_3 = AttentionFusion2((inplane // 15) * 2, (inplane // 15) * 2)  # (36,36)
        self.ppm_block = PAPPM(inplane * 4, ppmplane, inplane * 4, num_scales=5)  # (72,144,72)
        self.conv1 = nn.Conv2d(inplane * 4, inplane * 2, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(inplane * 2, inplane, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(inplane * 4, inplane*2, kernel_size=1, stride=1, padding=0)
        self.conv4 = nn.Conv2d(inplane * 2, outplane, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x_h,x_m,x_l=x
        # print("x_h,x_m,x_l",x_h.shape,x_m.shape,x_l.shape)
        out_size=(x_m.shape[-2],x_m.shape[-1])
        x_m=x_m+resize(
            x_l,
            size=out_size,
            mode='bilinear',
            align_corners=None)
        x_m=self.ppm_block(x_m)
        x_m=self.conv1(x_m)
        x_m=self.attention_fusion_block2_1(x_h,x_m)
        x_m=self.conv2(x_m)
        x_l=self.conv3(x_l)
        x_l=self.attention_fusion_block2_2(x_m,x_l)
        x_l=self.conv4(x_l)
        x=x_h+x_l

        return x

