import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# 读取图像-转化为灰度图像-转化为numpy数组
myim = Image.open("data/chap2/Lenna.jpg")
myimgray = np.array(myim.convert("L"), dtype=np.float32)  # "L"代表灰度图像
# 可视化图片
plt.figure(figsize=(6, 6))
plt.imshow(myimgray, cmap=plt.cm.gray)
# cmap: 颜色图谱（colormap), 默认绘制为RGB(A)颜色空间。
plt.axis("off")
plt.show()

# 将数组转化为张量
# 在使用pytorch进行卷积操作之前，需要将其转化为1×1×512×512的张量
imh, imw = myimgray.shape
# 使用torch.from_numpy()函数将Numpy数组转化为Pytorch张量
myimgray_t = torch.from_numpy(myimgray.reshape((1, 1, imh, imw)))
print(myimgray_t.shape)

# 卷积时需要将图像转化为四维表示[batch, channel, h, w]
# 在对图像进行卷积操作后，获得两个特征映射，第一个特征映射使用图像轮廓提取卷积核获取
# 第二个特征映射使用的卷积核为随机数，大小为5×5，对图像的边缘不使用0填充

# 对灰度图像进行卷积提取图像轮廓
kersize = 5  # 定义边缘检测卷积核，并将维度处理为1*1*5*5
ker = torch.ones(kersize, kersize, dtype=torch.float32) * -1
ker[2, 2] = 24
ker = ker.reshape((1, 1, kersize, kersize))
# 进行卷积操作
conv2d = nn.Conv2d(1, 2, (kersize, kersize), bias=False)
# 设置卷积时使用的核，第一个核使用边缘检测核
conv2d.weight.data[0] = ker
# 对灰度图像进行卷积操作
imconv2dout = conv2d(myimgray_t)
# 对卷积后的输出进行维度压缩
imconv2dout_im = imconv2dout.data.squeeze()  # 移除所有维度为1的维度
print("卷积后的尺寸：", imconv2dout_im.shape)  # torch.Size([2,508,508])
# 可视化卷积后的图像
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(imconv2dout_im[0], cmap=plt.cm.gray)
plt.axis("off")
plt.subplot(1, 2, 2)
plt.imshow(imconv2dout_im[1], cmap=plt.cm.gray)
plt.axis("off")
plt.show()

# 对卷积后的结果进行最大池化
maxpool2 = nn.MaxPool2d(2, stride=2)
pool2_out = maxpool2(imconv2dout)
pool2_out_im = pool2_out.squeeze()
print(pool2_out.shape)  # torch.Size([1, 2, 254, 254])
# 可视化最大池化后的结果
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(pool2_out_im[0].data, cmap=plt.cm.gray)
plt.axis("off")
plt.subplot(1, 2, 2)
plt.imshow(pool2_out_im[1].data, cmap=plt.cm.gray)
plt.axis("off")
plt.show()

# 对卷积后的结果进行平均池化
avgpool2 = nn.AvgPool2d(2, stride=2)
pool2_out = avgpool2(imconv2dout)
pool2_out_im = pool2_out.squeeze()
print(pool2_out.shape)  # torch.Size([1, 2, 254, 254])
# 可视化平均池化的结果
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(pool2_out_im[0].data, cmap=plt.cm.gray)
plt.axis("off")
plt.subplot(1, 2, 2)
plt.imshow(pool2_out_im[1].data, cmap=plt.cm.gray)
plt.axis("off")
plt.show()

# 对卷积后的结果进行自适应平均池化
AdaAvgpool2 = nn.AdaptiveAvgPool2d(output_size=(100, 100))
pool2_out = AdaAvgpool2(imconv2dout)
pool2_out_im = pool2_out.squeeze()
print(pool2_out.shape) # torch.Size([1, 2, 100, 100])
# 可视化自适应平均池化后的结果
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(pool2_out_im[0].data, cmap=plt.cm.gray)
plt.axis("off")
plt.subplot(1, 2, 2)
plt.imshow(pool2_out_im[1].data, cmap=plt.cm.gray)
plt.axis("off")
plt.show()


