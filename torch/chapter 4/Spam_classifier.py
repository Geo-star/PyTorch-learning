import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as Data
import matplotlib.pyplot as plt
import seaborn as sns
import hiddenlayer as hl
from torchviz import make_dot
from torch.optim import SGD, Adam
# 用于数据标准化处理
from sklearn.preprocessing import StandardScaler, MinMaxScaler
# 用于数据集的切分
from sklearn.model_selection import train_test_split
# 用于评价模型的预测效果
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report
# 用于数据降维及可视化
from sklearn.manifold import TSNE

# 垃圾邮件分类
# 读取数据显示数据的前几行
spam = pd.read_csv("../data/chap5/spambase.csv")
spam.head(n=10)
# 计算垃圾邮件和非垃圾邮件的数量
pd.value_counts(spam.label)

