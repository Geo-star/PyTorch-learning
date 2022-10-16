# 网络结构的可视化：层与层之间通常会有并联、串联等连接方式
# hiddenlayer库和torchviz库
import torch
import hiddenlayer as hl
from torchviz import make_dot
from MyConvNet import ConvNet

if __name__ == '__main__':
	# 初始化网络并输出网络的结构
	MyConvnet = ConvNet()
	print(MyConvnet)

	# 可视化卷积神经网络:使用hiddenlayer库中build_graph()函数
	hl_graph = hl.build_graph(MyConvnet, torch.zeros([1, 1, 28, 28]))
	hl_graph.theme = hl.graph.THEMES["blue"].copy()
	# 将可视化的网络保存为图片
	hl_graph.save("../data/chap4/MyConvNet_hl.png", format="png")

	# 可视化卷积神经网络:使用PyTorchViz(torchviz)库中make_dot()函数
	x = torch.randn(1, 1, 28, 28).requires_grad_(True)
	y = MyConvnet(x)

	MyConvnetVis = make_dot(y, params=dict(list(MyConvnet.named_parameters()) + [('x', x)]))
	# 将MyConvnetVis保存为图片
	MyConvnetVis.format = "png"
	# 指定文件保存位置
	MyConvnetVis.directory = "../data/chap4/MyConNet_vis"
	MyConvnetVis.view()  # 会自动在当前文件夹生成文件
