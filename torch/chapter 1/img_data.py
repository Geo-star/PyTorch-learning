import torch
import torch.utils.data as Data
from torchvision.datasets import FashionMNIST
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

# 从torchvision的datasets模块中导入数据并预处理
# 使用FashionMNIST数据，准备训练数据集
train_data = FashionMNIST(
	root="./data/FashionMNIST",  # 数据的路径
	train=True,  # True只使用训练数据集，False导入测试集
	transform=transforms.ToTensor(),
	# 将数据中的像素值转换到0~1之间，并且将图像数据从形状为[H,W,C]转换成[C,H,W]
	download=False  # False指定路径下已经有数据集；True没有数据集，会自动下载
)
# 定义一个加载数据集，将整个数据集切分为多个batch，用于网络优化时利用梯度下降算法求解
train_loader = Data.DataLoader(
	dataset=train_data,  # 指定使用的数据集
	batch_size=64,  # 批处理样本大小
	shuffle=True,  # 每次从数据集获取批量图片进行迭代前打乱顺序
	num_workers=2,  # 使用两个进程
)
# 计算train_loader有多少个batch
print("train_loader的batch数量为：", len(train_loader))  # 938

# 对测试集进行处理
test_data = FashionMNIST(
	root="./data/FashionMNIST",  # 数据的路径
	train=False,  # 使用测试集数据
	download=False
)
# 为数据添加一个通道维度，并将取值范围缩放到0~1之间
test_data_x = test_data.data.type(torch.FloatTensor) / 255.0
test_data_x = torch.unsqueeze(test_data_x, dim=1)
test_data_y = test_data.targets  # 测试集的标签
print("test_data_x.shape:", test_data_x.shape)  # torch.Size([10000, 1, 28, 28])
print("test_data_y.shape:", test_data_y.shape)  # torch.Size([10000])

# 使用ImageFolder()函数从文件夹中导入数据并进行预处理
# 对训练集进行预处理
train_data_transforms = transforms.Compose([
	transforms.RandomResizedCrop(224),  # 随即长宽比裁剪为224*224
	transforms.RandomHorizontalFlip(),  # 依概率p=0.5水平翻转
	transforms.ToTensor(),  # 转化为张量并归一化至[0~1]
	# 图像标准化处理
	transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
# 读取图像
train_data_dir = "data/chap2/imagedata/"
train_data = ImageFolder(train_data_dir, transform=train_data_transforms)
train_data_loader = Data.DataLoader(
	train_data,
	batch_size=4,
	shuffle=True,
	num_workers=0
)
print("数据集的label", train_data.targets)
# 获得一个batch的数据
for step, (b_x, b_y) in enumerate(train_data_loader):
	if step > 0:
		break
# 输出训练图像的尺寸和标签的尺寸
print(b_x.shape)
print(b_y.shape)
print("图像的取值范围为：", b_x.min(), "~", b_x.max())
