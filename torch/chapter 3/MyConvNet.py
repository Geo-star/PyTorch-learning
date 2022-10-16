import torch
import torch.nn as nn
import torchvision
import torchvision.utils as utils
import torch.utils.data as Data
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# 使用手写字体数据，准备训练数据集
train_data = torchvision.datasets.MNIST(
	root="../data/MNIST",  # 数据的路径
	train=True,  # 只使用训练集
	# 将数据转化为torch使用的张量，取值范围为[0,1]
	transform=torchvision.transforms.ToTensor(),
	download=False  # 下载数据
)
# 定义一个数据加载器
train_loader = Data.DataLoader(
	dataset=train_data,  # 使用的数据集
	batch_size=128,  # 批处理样本大小
	shuffle=True,  # 每次迭代前打乱数据
	num_workers=2,  # 使用两个进程
)
# 准备需要使用的测试数据集
test_data = torchvision.datasets.MNIST(
	root="../data/MNIST",
	train=False,
	download=False
)
# 为数据添加一个通道维度，并且取值范围缩放到0~1之间
test_data_x = test_data.data.type(torch.FloatTensor) / 255.0
test_data_x = torch.unsqueeze(test_data_x, dim=1)
test_data_y = test_data.targets  # 测试集的标签
print("test_data_x.shape", test_data_x.shape)
print("test_data_y.shape", test_data_y.shape)


# 搭建一个卷积神经网络，该网络用于展示如何使用相关包来可视化其网络结构
class ConvNet(nn.Module):
	def __init__(self):
		super(ConvNet, self).__init__()
		# 定义第一个卷积层: Conv2d + RELU + AvgPool2d
		self.conv1 = nn.Sequential(
			nn.Conv2d(
				in_channels=1,  # 输入的feature map
				out_channels=16,  # 输出的feature map
				kernel_size=3,  # 卷积核尺寸3*3
				stride=1,  # 卷积核步长
				padding=1,  # 填充边缘,避免数据丢失;值为1表示填充1层边缘像素,默认用0值填充;padding的值一般是卷积核尺寸的一半(向下取整)
			),
			nn.ReLU(),  # 激活函数
			nn.AvgPool2d(
				kernel_size=2,  # 平均值池化层，使用2*2
				stride=2,  # 池化步长为2
			),
		)
		# 定义第二个卷积层
		self.conv2 = nn.Sequential(
			nn.Conv2d(16, 32, 3, 1, 1),
			nn.ReLU(),
			nn.MaxPool2d(2, 2)  # 最大值池化层
		)
		# 定义全连接层
		self.fc = nn.Sequential(
			nn.Linear(
				in_features=32 * 7 * 7,
				out_features=128,
			),
			nn.ReLU(),
			nn.Linear(128, 64),
			nn.ReLU()
		)
		self.out = nn.Linear(64, 10)  # 最后的分类层

	# 定义网络的向前传播路径
	def forward(self, x):
		x = self.conv1(x)
		x = self.conv2(x)  # tensor尺寸为[batchsize,channels,h,w]
		x = x.view(x.size(0), -1)  # 展平多维的卷积图层；展平为[batchsize,channels*h*w]
		'''
		用在卷积和池化之后、全连接层之前；将多维转成一维，因为fc要求输入数据为一维
		卷积或者池化之后的tensor的维度为(batchsize，channels，x，y)，
		其中x.size(0)指batchsize的值，最后通过x.view(x.size(0), -1)
		将tensor的结构转换为了(batchsize, channels*x*y)，即将（channels，x，y）拉直，
		然后就可以和fc层连接了
		'''
		x = self.fc(x)
		output = self.out(x)
		return output


# 输出网络结构
MyConvnet = ConvNet()
print(MyConvnet)
