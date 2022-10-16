import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os


# 以一个卷积层为例，先定义一个从3个特征映射到16个特征映射的卷积层，（3*3卷积核）然后使用标准正态分布的随机数进行初始化
# 针对一个层进行权重初始化
conv1 = torch.nn.Conv2d(3, 16, 3)
# 使用标准正态分布初始化权重
torch.manual_seed(12)  # 随机数初始化种子
torch.nn.init.normal(conv1.weight, mean=0, std=1)

# 使用直方图可视化conv1.weight的分布情况
plt.figure(figsize=(8, 6))
plt.hist(conv1.weight.data.numpy().reshape((-1, 1)), bins=30)
plt.show()

torch.nn.init.constant(conv1.bias, val=0.1)


# torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
class TestNet(nn.Module):
	def __init__(self):
		super(TestNet, self).__init__()
		# 定义隐藏层
		self.hidden = nn.Sequential(
			nn.Linear(100, 100),
			nn.ReLU(),
			nn.Linear(100, 50),
			nn.ReLU(),
		)
		# 定义预测回归层
		self.cla = nn.Linear(50, 10)

	# 定义网络的向前传播路径
	def forward(self, x):
		x = self.hidden(x)
		x = x.view(x.shape[0], -1)
		x = self.hidden(x)
		output = self.cla(x)
		# 输出为output
		return output


# 输出我们的网络
testnet = TestNet()
print(testnet)


# 针对不同类型的层使用不同的初始化方法
def init_weights(m):
	# 如果是卷积层
	if type(m) == nn.Conv2d:
		torch.nn.init.normal_(m.weight, mean=0, std=0.5)
	# 如果是全连接层
	if type(m) == nn.Linear:
		torch.nn.init.uniform_(m.weight, a=-0.1, b=0.1)
		m.bias.data.fill_(0.01)


# 使用网络的apply方法进行权重初始化
torch.manual_seed(13)
testnet.apply(init_weights)
