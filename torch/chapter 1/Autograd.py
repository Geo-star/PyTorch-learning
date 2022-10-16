import torch

x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
y = torch.sum(x ** 2 + 2 * x + 1)
print("x.requires_grad:", x.requires_grad)
print("y.requires_grad:", y.requires_grad)
print("x:", x)
print("y:", y)
print(y.backward(), x.grad)
