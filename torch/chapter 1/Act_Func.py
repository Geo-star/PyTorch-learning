import torch
import torch.nn as nn
import matplotlib.pyplot as plt

x = torch.linspace(-6, 6, 100)
sigmoid = nn.Sigmoid()  # Sigmoid激活函数
ysigmoid = sigmoid(x)
tanh = nn.Tanh()  # Tanh激活函数
ytanh = tanh(x)
relu = nn.ReLU()  # ReLU激活函数
yrelu = relu(x)
softplus = nn.Softplus()  # Softplus激活函数
ysoftplus = softplus(x)
plt.figure(figsize=(14, 3))  # 可视化激活函数
plt.subplot(1, 4, 1)
plt.plot(x.data.numpy(), ysigmoid.data.numpy(), "r-")
plt.title("Sigmoid")
plt.grid()
plt.subplot(1, 4, 2)
plt.plot(x.data.numpy(), ytanh.data.numpy(), "r-")
plt.title("Tanh")
plt.grid()
plt.subplot(1, 4, 3)
plt.plot(x.data.numpy(), yrelu.data.numpy(), "r-")
plt.title("ReLU")
plt.grid()
plt.subplot(1, 4, 4)
plt.plot(x.data.numpy(), ysoftplus.data.numpy(), "r-")
plt.title("Softplus")
plt.grid()
plt.show()
