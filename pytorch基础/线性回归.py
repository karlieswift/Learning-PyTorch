"""
 Env: /anaconda3/python3.7
 Time: 2021/7/5 8:22
 Author: karlieswfit
 File: 线性回归.py
 Describe: 
"""
#手动线性回归简单实现
import torch

l=0.01
#1-准备数据 y=3x+0.8
x=torch.rand([500,1])
y_true=x*3+0.8
w=torch.rand([1,1],requires_grad=True)
b=torch.tensor(0,requires_grad=True,dtype=float)

#4-循环 反向传播 更新参数 循环2000次
for i in range(2000):
    y_predict=torch.matmul(x,w)+b
    loss=(y_true-y_predict).pow(2).mean()

    if w.grad is not None:
        w.grad.data.zero_()
    if b.grad is not None:
        b.grad.data.zero_()
    loss.backward()
    #  更新参数
    w.data=w.data-l*w.grad
    b.data=b.data-l*b.grad
    print('w=',w.data,' b=',b.data,' loss=',loss)




