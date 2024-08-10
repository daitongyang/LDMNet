import torch
import torch.nn as nn

# 定义一个输入特征
input = torch.randn(1, 3, 32, 32)

# 使用膨胀率为2的膨胀卷积
dilated_conv = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=2, dilation=2)
output = dilated_conv(input)

print(output.shape)  # 输出特征图的形状
