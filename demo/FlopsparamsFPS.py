import torch
import torchvision
from thop import profile
from ptflops import get_model_complexity_info
from torchstat import stat
from mmseg.models.backbones import LTbANet, DDRNet,PIDNet,    \
    ResNet, ResNetV1c, ResNetV1d, ResNeXt, HRNet, FastSCNN,\
    ResNeSt, MobileNetV2, UNet, CGNet, MobileNetV3,\
    VisionTransformer, SwinTransformer, MixVisionTransformer,\
    BiSeNetV1, BiSeNetV2, ICNet, TIMMBackbone, ERFNet, PCPVT,\
    SVT, STDCNet, STDCContextPathNet, BEiT, MAE, PIDNet, MSCAN,\
    DDRNet, VPD
import time
import numpy as np

print('==> Building model..')
model = LTbANet()
# model = MobileNetV3()
net = model
input = torch.randn(1, 3, 1024, 1024)
flops, params = profile(model, (input,))
print('flops: %.2f M, params: %.2f M' % (flops / 1e6, params / 1e6))



# model = torchvision.models.alexnet(pretrained=False)
# flops, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True, print_per_layer_stat=True)
# print('flops: ', flops, 'params: ', params)



# print('==> Building model..')
# # model = torchvision.models.alexnet(pretrained=False)
#
# stat(model, (3, 1024, 1024))



# model = torchvision.models.alexnet(pretrained=False)
# nelement()：统计Tensor的元素个数
#.parameters()：生成器，迭代的返回模型所有可学习的参数，生成Tensor类型的数据
total = sum([param.nelement() for param in model.parameters()])
print("Number of parameter: %.2fM" % (total/1e6))

model.cuda()
model.eval()
imgs = input.cuda()
start = time.time()
outputs, *outputs_aux = model(imgs)
end = time.time()
print("latency: ", end - start)
print("fps: ", 1 / (end - start))
print(torch.cuda.is_available())

net.eval()

# x是输入图片的大小
x = torch.zeros((1,3,1024,1024)).cuda()
t_all = []

for i in range(100):
    t1 = time.time()
    y = net(x)
    t2 = time.time()
    t_all.append(t2 - t1)


print('average time:', np.mean(t_all) / 1)
print('average fps:',1 / np.mean(t_all))

t_all = np.array([x for x in t_all if x != 0])

print('fastest time:', min(t_all) / 1)
print('fastest fps:',1 / min(t_all))

print('slowest time:', max(t_all) / 1)
print('slowest fps:',1 / max(t_all))
# 将模型设置为评估模式

model.eval()

# 运行模型，并计算FPS
with torch.no_grad():
    num_iterations = 100  # 迭代次数
    start_time = time.time()
    for _ in range(num_iterations):
        output = model(x)
    end_time = time.time()

# 计算FPS
fps = num_iterations / (end_time - start_time)
print("FPS: {:.2f}".format(fps))