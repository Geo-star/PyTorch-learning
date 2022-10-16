import torch
import numpy as np

# 张量
# 张量的生成
# 1> torch.tensor
A = torch.tensor([[1.0, 1.0], [2, 2]])
A.size()  # 形状大小
var = A.shape  # 维度
A.numel()  # 包含元素数量
B = torch.tensor((1, 2, 3), dtype=torch.float32, requires_grad=True)  # dtype指定数据类型
# 计算梯度大小
y = B.pow(2).sum()
y.backward()
# print(B.grad)

# 2> torch.Tensor
C = torch.Tensor([1, 2, 3, 4])
D = torch.Tensor(2, 3)
torch.ones_like(D)
torch.zeros_like(D)
torch.rand_like(D)
E = [[1, 2], [3, 4]]
# print(D.new_empty((3, 3)))

# 3> 张量和NumPy数据相互转换
F = np.ones((3, 3))
Ftensor = torch.from_numpy(F)
# print(Ftensor.numpy())

# 4> 随机数生成张量
torch.manual_seed(123)
A = torch.normal(mean=torch.arange(1, 5.0), std=torch.arange(1, 5.0))
B = torch.rand(3, 4)
C = torch.ones(2, 3)
D = torch.rand_like(C)
# print(torch.randperm(10))

# 5> 其他
torch.arange(start=0, end=10, step=2)
torch.linspace(start=1, end=10, steps=5)
torch.logspace(start=0.1, end=1.0, steps=5)

# 张量操作
# 1> 改变张量的形状
A = torch.arange(12.0).reshape(3, 4)
torch.reshape(input=A, shape=(2, -1))
A.resize_(2, 6)
B = torch.arange(10.0, 19.0).reshape(3, 3)
A.resize_as_(B)

A = torch.arange(12.0).reshape(2, 6)
B = torch.unsqueeze(A, dim=0)
C = B.unsqueeze(dim=3)
D = torch.squeeze(C)
E = torch.squeeze(C, dim=0)

A = torch.arange(3)
B = A.expand(3, -1)
C = torch.arange(6).reshape(2, 3)
B = A.expand_as(C)
D = B.repeat(1, 2, 2)

# 2> 获取张量中的元素
A = torch.arange(12).reshape(1, 3, 4)
print(torch.diag((torch.tensor([1, 2, 3]))))

# 3> 拼接和拆分
A = torch.arange(6.0).reshape(2, 3)
B = torch.linspace(0, 10, 6).reshape(2, 3)
C = torch.cat((A, B), dim=0)
print(A[:, 0:2])
D = torch.cat((A, B), dim=1)
E = torch.cat((A[:, 1:2], A, B), dim=1)

F = torch.stack((A, B), dim=0)
torch.chunk(E, 2, dim=0)

# 张量计算
# 1>比较大小
torch.allclose(torch.tensor([10000., 1e-07]), torch.tensor([10000.1, 1e-08]))  # False
torch.allclose(torch.tensor([10000., 1e-08]), torch.tensor([10000.1, 1e-09]))  # True
torch.allclose(torch.tensor([1.0, float('nan')]), torch.tensor([1.0, float('nan')]))  # False
torch.allclose(torch.tensor([1.0, float('nan')]), torch.tensor([1.0, float('nan')]), equal_nan=True)  # True

A = torch.tensor([1, 2, 3, 4, 5, 6])
B = torch.arange(1, 7)
C = torch.unsqueeze(B, dim=0)
print(torch.eq(A, B), torch.eq(A, C))
print(torch.equal(A, B), torch.equal(A, C))

# 2>基本运算
print(torch.max(torch.arange(16)))
print(torch.argmax(torch.arange(16)))
print(torch.min(torch.arange(16)))
print(torch.argmin(torch.arange(16)))

A = torch.tensor([12, 34, 25, 11, 67, 32, 29, 30, 99, 55, 23, 44])
print(torch.sort(A))
print(torch.sort(A, descending=True))
print(torch.topk(A, 8))
print(torch.kthvalue(A, 8))
B = A.reshape(3, 4)


