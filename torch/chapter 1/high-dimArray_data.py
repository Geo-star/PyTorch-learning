import torch
import torch.utils.data as Data
import numpy as np
from sklearn.datasets import load_boston, load_iris

# regression
boston_X, boston_y = load_boston(return_X_y=True)
print("boston_X.dtype:", boston_X.dtype)  # boston_X.dtype: float64
print("boston_y.dtype:", boston_y.dtype)  # boston_y.dtype: float64

# 训练集X转化为张量，训练集y转化为张量
# 将数据集boston_X和boston_y从numpy的64位浮点型数据转化为torch的32位浮点型张量
train_xt = torch.from_numpy(boston_X.astype(np.float32))
train_yt = torch.from_numpy(boston_y.astype(np.float32))
print("train_xt.dtype", train_xt.dtype)  # torch.float32
print("train_yt.dtype", train_yt.dtype)  # torch.float32

# 将训练集转化为张量后，使用TensorDataset将X和y整理到一起
train_data = Data.TensorDataset(train_xt, train_yt)
# 定义一个数据加载器，将训练数据集进行批量化处理
train_loader = Data.DataLoader(
	dataset=train_data,  # 使用的数据集
	batch_size=64,  # 批处理样本大小
	shuffle=True,  # 每次迭代前打乱数据
	num_workers=0,  # 使用两个进程
)
# 检查训练数据集的一个batch的样本的维度是否正确
for step, (b_x, b_y) in enumerate(train_loader):
	if step > 0:
		break
# 输出训练图像的尺寸和标签的尺寸及数据类型
print("b_x.shape", b_x.shape)  # torch.Size([64, 13])
print("b_y.shape", b_y.shape)  # torch.Size([64])
print("b_x.dtype", b_x.dtype)  # torch.float32
print("b_y.dtype", b_y.dtype)  # torch.float32

# classification
iris_x, irisy = load_iris(return_X_y=True)
print("iris_x.dtype", iris_x.dtype)  # float64
print("irisy.dtype", irisy.dtype)  # int32

# 训练集X转化为张量，训练集y转化为张量
train_xt = torch.from_numpy(iris_x.astype(np.float32))
train_yt = torch.from_numpy(irisy.astype(np.int64))
print("train_xt.dtype", train_xt.dtype)  # torch.float32
print("train_yt.dtype", train_yt.dtype)  # torch.int64

# 使用Data.TensorDataset()和Data.DataLoader()定义数据加载器
# 将训练集转化为张量后，使用TensorDataset将X和y整理到一起
train_data = Data.TensorDataset(train_xt, train_yt)
# 定义一个数据加载器，将训练数据集进行批量处理
train_loader = Data.DataLoader(
	dataset=train_data,  # 使用的数据集
	batch_size=10,  # 批处理样本大小
	shuffle=True,  # 每次迭代前打乱数据
	num_workers=0,  # 使用两个进程
)
# 检查训练数据集的一个batch的样本的维度是否正确
for step, (b_x, b_y) in enumerate(train_loader):
	if step > 0:
		break
# 输出训练图像的尺寸和标签的尺寸及数据类型
print("b_x.shape", b_x.shape)  # torch.Size([10, 4])
print("b_y.shape", b_y.shape)  # torch.Size([10])
print("b_x.dtype", b_x.dtype)  # torch.float32
print("b_y.dtype", b_y.dtype)  # torch.int64
