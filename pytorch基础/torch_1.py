"""
 Env: /anaconda3/python3.7
 Time: 2021/7/10 9:52
 Author: karlieswfit
 File: torch_1.py
 Describe: pytorch的api使用
"""
#1-nn.Module  需要继承
    # __init__() 需要继承super 继承父类方法属性
    # forward() 前向计算
    # nn.Linear(input特征数量,output特性数量)
#2-optimizer
    # torch.optim

#3-loss
    #nn.MESloss
    #nn.xxx


import torch
import torch.nn as nn
from torch.optim import SGD

#准备数据
x=torch.rand([500,1])
y_ture=3*x+0.8

#定义模型
class MyLinear(nn.Module):
    def __init__(self):
        super(MyLinear,self).__init__()
        self.linear=nn.Linear(1,1)

    def forward(self,x):
        out=self.linear(x)
        return out

#实例化模型 优化器模型 loss模型
my_linear=MyLinear()
optimizer=SGD(my_linear.parameters(),0.001)
loss_fn=nn.MSELoss()

#循环 进行梯度下降 参数更新
for i in range(4000):
    #得到预测值
    y_predict=my_linear(x)
    #计算损失
    loss=loss_fn(y_ture,y_predict)
    #梯度重置为0
    optimizer.zero_grad()
    #反向传播
    loss.backward()
    #参数更新
    optimizer.step()

    print(loss.item(),list(my_linear.parameters()))


























