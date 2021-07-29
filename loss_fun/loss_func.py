"""
 Env: /anaconda3/python3.7
 Time: 2021/7/14 20:43
 Author: karlieswfit
 File: loss_func.py
 Describe: pytorch的损失函数NLLLoss（nll_loss()）和CrossEntropyLoss
 CrossEntropyLoss是将LogSoftmax和NLLLoss集成到一个类别中去
 Softmax+log+NLLLoss=CrossEntropyLoss

"""

from torch import nn
import torch
from torch.functional import F

target=torch.tensor([0,2,1])
data=torch.tensor([
        [-2.4300,  2.0737,  1.0382],
        [-0.4114,  0.5158, -0.8912],
        [-0.1612,  1.3473,  1.3313]
])
print(data)
st=nn.Softmax(dim=-1)
p=st(data)
print(p)
'''
tensor([[0.0081, 0.7320, 0.2599],
        [0.2412, 0.6096, 0.1493],
        [0.1003, 0.4534, 0.4462]])
'''
l=torch.log(p)
print(l)
'''
tensor([[-4.8157, -0.3120, -1.3475],
        [-1.4222, -0.4950, -1.9020],
        [-2.2994, -0.7909, -0.8069]])
'''

s=torch.tensor(0)
for i in range(3):
    temp=l[i][target[i]]
    s=s+torch.abs(temp)
    print(temp)
    '''
tensor(-4.8157)
tensor(-1.9020)
tensor(-0.7909)
    '''

print(s/3) #tensor(2.5029)

print(nn.NLLLoss()(l, target)) #tensor(2.5029)
print(F.nll_loss(l, target)) #tensor(2.5029)



print(nn.CrossEntropyLoss()(l,target))#tensor(2.5029)
print(nn.CrossEntropyLoss()(data,target))#tensor(2.5029)




