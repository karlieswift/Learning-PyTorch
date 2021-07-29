"""
 Env: /anaconda3/python3.7
 Time: 2021/7/13 19:39
 Author: karlieswfit
 File: mnist1.py
 Describe: 手写数字识别
 基本思路：1-加载数据 2-模型构建 3-模型的训练 4-模型保存 5-模型评估
"""

from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision.transforms import Normalize, ToTensor, Compose
from torch import nn
from torch.functional import F
from torch.optim import Adam

BATCH_SIZE = 128

tranform_fn = Compose([
    ToTensor(),
    Normalize(mean=(0.1307,), std=(0.3081,))  # mean和std的形状和通道数
])
data_set = MNIST(root='./data', train=True, transform=tranform_fn)
data_loader = DataLoader(dataset=data_set, batch_size=3, shuffle=True)


for i in enumerate(data_loader):
    print(i)

