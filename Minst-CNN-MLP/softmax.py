"""
 Env: /anaconda3/python3.7
 Time: 2021/7/15 19:53
 Author: karlieswfit
 File: softmax.py
 Describe: 
"""

import torch
from torch import nn

x=torch.tensor([
    [1,2,3],
    [2,3,5]
],dtype=torch.float)
print(x)
sm=nn.Softmax(dim=1)
print(sm(x))
'''
tensor([[0.0900, 0.2447, 0.6652],
        [0.0420, 0.1142, 0.8438]])
'''

sm=nn.Softmax(dim=-1)
print(sm(x))
'''
tensor([[0.0900, 0.2447, 0.6652],
        [0.0420, 0.1142, 0.8438]])
'''

sm=nn.Softmax(dim=0)
print(sm(x))
'''
tensor([[0.2689, 0.2689, 0.1192],
        [0.7311, 0.7311, 0.8808]])
'''