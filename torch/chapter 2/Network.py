# 全连接神经网络回归
import torch
import torch.nn as nn  # 对网络中的层的使用
from torch.optim import SGD
import torch.utils.data as Data  # 数据预处理
from sklearn.datasets import load_boston  # 导入数据
from sklearn.preprocessing import StandardScaler  # 数据标准化
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 读取数据
boston_X, boston_y = load_boston(return_X_y=True)
print("boston_X.shape:", boston_X.shape)  # boston_X.shape: (506, 13)
plt.figure()
plt.hist(boston_y, bins=20)  # bins将数据分为20等份
plt.show()

# 数据标准化处理
ss = StandardScaler(with_mean=True, with_std=True)
boston_Xs = ss.fit_transform(boston_X)  # 对数据先进行拟合，然后标准化
# 将数据预处理为可以使用PyTorch进行批量训练的形式
# 训练集X转化为张量
train_xt = torch.from_numpy(boston_Xs.astype(np.float32))
# 训练集y转化为张量
train_yt = torch.from_numpy(boston_y.astype(np.float32))
# 将训练集转化为张量后，使用TensorDataset将X和Y整理到一起
train_data = Data.TensorDataset(train_xt, train_yt)

# 定义一个数据加载器，将训练数据集进行批量处理
train_loader = Data.DataLoader(
	dataset=train_data,  # 使用的数据集
	batch_size=128,  # 批处理样本大小
	shuffle=True,  # 每次迭代前打乱数据
	num_workers=0,  # 使用两个进程
)


# 使用继承Module的方式定义全神经网络
class MLPmodel(nn.Module):
	def __init__(self):
		super(MLPmodel, self).__init__()
		# 定义第一个隐藏层
		self.hidden1 = nn.Linear(
			in_features=13,  # 第一个隐藏层的输入，数据的特帧数
			out_features=10,  # 第一个隐藏层的输出，神经元的数量
			bias=True,  # 默认会有偏置
		)
		self.active1 = nn.ReLU()
		# 定义第二个隐藏层
		self.hidden2 = nn.Linear(10, 10)
		self.active2 = nn.ReLU()
		# 定义预测回归层
		self.regression = nn.Linear(10, 1)

	# 定义网络的前向传播路径
	def forward(self, x):
		x = self.hidden1(x)
		x = self.active1(x)
		x = self.hidden2(x)
		x = self.active2(x)
		output = self.regression(x)
		return output


# 输出我们的网络结构
mlp1 = MLPmodel()
print(mlp1)

# 对回归模型mlp1进行训练并输出损失函数的变化情况，定义优化器和损失函数
optimizer = SGD(mlp1.parameters(), lr=0.001)
loss_func = nn.MSELoss()  # 最小均方根误差
train_loss_all = []  # 输出每个批次训练的损失函数
# 进行训练，并输出每次迭代的损失函数
for epoch in range(30):
	# 对训练数据的加载器进行迭代运算
	for step, (b_x, b_y) in enumerate(train_loader):
		output = mlp1(b_x).flatten()  # MLP在训练batch上的输出
		train_loss = loss_func(output, b_y)  # 均方根误差
		optimizer.zero_grad()  # 每次迭代的梯度初始化为0
		train_loss.backward()  # 损失的后向传播，计算梯度
		optimizer.step()  # 使用梯度进行优化
		train_loss_all.append(train_loss.item())

plt.figure()
plt.plot(train_loss_all, "r-")
plt.title("Train loss per iteration")
plt.show()


# 使用定义网络时使用nn.Sequential的形式
class MLPmodel2(nn.Module):
	def __init__(self):
		super(MLPmodel2, self).__init__()
		# 定义隐藏层
		self.hidden = nn.Sequential(
			nn.Linear(13, 10),
			nn.ReLU(),
			nn.Linear(10, 10),
			nn.ReLU(),
		)
		# 预测回归层
		self.regression = nn.Linear(10, 1)

	# 定义网络的前向传播路径
	def forward(self, x):
		x = self.hidden(x)
		output = self.regression(x)
		return output


# 输出网络结构
mlp2 = MLPmodel2()
print(mlp2)

# 对回归模型mlp2进行训练并输出损失函数的变化情况，定义优化器和损失函数
optimizer = SGD(mlp2.parameters(), lr=0.001)
loss_func = nn.MSELoss()  # 最小均方根误差
train_loss_all = []  # 输出每个批次训练的损失函数
# 进行训练，并输出每次迭代的损失函数
for epoch in range(30):
	# 对训练数据的加载器进行迭代运算
	for step, (b_x, b_y) in enumerate(train_loader):
		output = mlp2(b_x).flatten()  # MLP在训练batch上的输出
		train_loss = loss_func(output, b_y)  # 均方根误差
		optimizer.zero_grad()  # 每次迭代的梯度初始化为0
		train_loss.backward()  # 损失的后向传播，计算梯度
		optimizer.step()  # 使用梯度进行优化
		train_loss_all.append(train_loss.item())

plt.figure()
plt.plot(train_loss_all, "r-")
plt.title("Train loss per iteration")
plt.show()

# 保存整个模型
# …/代表当前所在目录的父目录下的某个文件夹或文件
torch.save(mlp2, '../data/chap3/mlp2.pkl')  # 导入保存的模型
mlp2load = torch.load('../data/chap3/mlp2.pkl')
