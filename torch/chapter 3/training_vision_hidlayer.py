import time
import torch
import torchvision
import torch.nn as nn
import hiddenlayer as hl
import torchvision.utils as utils
import matplotlib.pyplot as plt
import torch.utils.data as Data
from sklearn.metrics import accuracy_score
from MyConvNet import ConvNet

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
	num_workers=0,  # 使用两个进程
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

# 初始化MyConvnet
MyConvnet = ConvNet()
print(MyConvnet)
# 定义优化器
optimizer = torch.optim.Adam(MyConvnet.parameters(), lr=0.0003)
loss_func = nn.CrossEntropyLoss()  # 损失函数
# 记录训练过程的指标
history1 = hl.History()
# 使用Canvas进行可视化
canvas1 = hl.Canvas()
print_step = 100  # 每经过100次迭代后，输出损失
# 对模型进行迭代训练，对所有的数据训练EPOCH轮
for epoch in range(5):
	# 对训练数据的加载器进行迭代计算
	for step, (b_x, b_y) in enumerate(train_loader):
		# 计算每个batch的损失
		output = MyConvnet(b_x)  # CNN在batch上的输出
		loss = loss_func(output, b_y)  # 交叉熵损失函数
		optimizer.zero_grad()  # 每个迭代步的梯度初始化为0
		loss.backward()  # 损失的后向传播，计算梯度
		optimizer.step()  # 使用梯度进行优化
		# 计算迭代次数
		# 计算每经过print_step次迭代后的输出
		if step % print_step == 0:
			# 计算在测试集上的精度
			output = MyConvnet(test_data_x)
			_, pre_lab = torch.max(output, 1)
			acc = accuracy_score(test_data_y, pre_lab)
			# 计算每个epoch和step的模型输出特征
			history1.log((epoch, step),
						 train_loss=loss,  # 训练集损失
						 test_acc=acc,  # 测试集精度
						 # 第二个全连接层权重
						 hidden_weight=MyConvnet.fc[2].weight)
			# 可视化网络训练过程
			with canvas1:
				canvas1.draw_plot(history1["train_loss"])
				canvas1.draw_plot(history1["test_acc"])
				canvas1.draw_image(history1["hidden_weight"])