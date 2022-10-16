import torch
import numpy as np
import torchvision
from visdom import Visdom
import torch.utils.data as Data
from sklearn.datasets import load_iris

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

iris_x, iris_y = load_iris(return_X_y=True)
print(iris_x.shape)
print(iris_y.shape)
# 2D散点图
vis = Visdom()
vis.scatter(iris_x[:, 0:2], Y=iris_y + 1, win="windows1", env="main")
# 3D散点图
vis.scatter(iris_x[:, 0:3], Y=iris_y + 1, win="3D 散点图", env="main",
			opts=dict(markersize=4, xlabel="特征1", ylabel="特征2")
			)
# 添加折线图
x = torch.linspace(-6, 6, 100).view((-1, 1))
sigmoid = torch.nn.Sigmoid()
sigmoidy = sigmoid(x)
tanh = torch.nn.Tanh()
tanhy = tanh(x)
relu = torch.nn.ReLU()
reluy = relu(x)
# 连接3个张量
ploty = torch.cat((sigmoidy, tanhy, reluy), dim=1)
plotx = torch.cat((x, x, x), dim=1)
vis.line(Y=ploty, X=plotx, win="line plot", env="main",
		 opts=dict(dash=np.array(["solid", "dash", "dashdot"]),
				   legend=["Sigmoid", "Tanh", "Relu"]))

# 添加茎叶图
x = torch.linspace(-6, 6, 100).view((-1, 1))
y1 = torch.sin(x)
y2 = torch.cos(x)
# 连接两个张量
plotx = torch.cat((y1, y2), dim=1)
ploty = torch.cat((x, x), dim=1)
vis.stem(X=plotx, Y=ploty, win="stem plot", env="main",
		 opts=dict(legend=["sin", "cos"], title="茎叶图"))

# 添加热力图，计算鸢尾花数据的相关系数
iris_corr = torch.from_numpy(np.corrcoef(iris_x, rowvar=False))
vis.heatmap(iris_corr, win="heatmap", env="main",
			opts=dict(rownames=["x1", "x2", "x3", "x4"],
					  columnames=["x1", "x2", "x3", "x4"],
					  title="热力图"))
# # 创建新的可视化图像环境，可视化图像，获得一个batch的数据
# for step, (b_x, b_y) in enumerate(train_loader):
# 	if step > 0:
# 		break
# 	# 输出训练图像和标签的尺寸
# 	print(b_x.shape)
# 	print(b_y.shape)
#
# # 可视化其中的一张图片
# vis.image(b_x[0, :, :, :], win="one image", env="MyimagePlot",
# 		  opts=dict(titlle="一张图像"))
# # 它形成一个大小为（B/nrow,nrow）的图像网络
# vis.image(b_x, win="my batch iamge", env="MyimagePlot",
# 		  nrow=16, opts=dict(title="一个批次的图像"))
# 可视化一段文本
texts = """A flexible tool for creating"""
vis.text(texts, win="text plot", env="MyimagePlot",
		 opts=dict(title="可视化文本"))
