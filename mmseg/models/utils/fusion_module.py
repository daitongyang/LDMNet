from torch import nn
from mmseg.models.utils import AttentionFusion1, AttentionFusion2, BAPPM


class FusionModule(nn.Module):
    def __init__(self,
                 inplane: int = 270,
                 ppmplane: int = 144):
        super(FusionModule, self).__init__()
        self.inplane = inplane
        self.ppmplane = ppmplane
        self.attention_fusion_block1_1 = AttentionFusion1((inplane // 15) * 4, (inplane // 15) * 4)  # (72,)
        self.attention_fusion_block1_2 = AttentionFusion1((inplane // 15) * 2, (inplane // 15) * 2)
        self.attention_fusion_block1_3 = AttentionFusion1(inplane // 15, inplane // 15)
        self.attention_fusion_block2_1 = AttentionFusion2((inplane // 15) * 8, (inplane // 15) * 8)  # (144,144)
        self.attention_fusion_block2_2 = AttentionFusion2((inplane // 15) * 4, (inplane // 15) * 4)  # (72,36)
        self.attention_fusion_block2_3 = AttentionFusion2((inplane // 15) * 2, (inplane // 15) * 2)  # (36,36)
        self.ppm_block = BAPPM((inplane // 15) * 4, ppmplane, (inplane // 15) * 4, num_scales=5)  # (72,144,72)
        self.conv1 = nn.Conv2d((inplane // 15) * 8, (inplane // 15) * 4, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d((inplane // 15) * 4, (inplane // 15) * 2, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d((inplane // 15) * 2, (inplane // 15), kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        print("FusionModule.self.inplane", self.inplane)
        print("FusionModule.self.ppmplane", self.ppmplane)

        x_h, x_m1, x_m2, x_l = x
        print("x_h.shape", x_h.shape)
        print("x_m1.shape", x_m1.shape)

        x_l = self.attention_fusion_block2_1(x_m2, x_l)
        x_l = self.conv1(x_l)
        x_l = self.ppm_block(x_l)
        x = x_l + x_m2

        print("FusionModule，我运行了！")
        x_m2 = self.attention_fusion_block2_2(x_m1, x)
        x_m2=self.conv2(x_m2)
        x = x_m2 + x_m1

        x_m1 = self.attention_fusion_block2_3(x_h, x)
        x_m1=self.conv3(x_m1)
        x = x_m1 + x_h

        return x
