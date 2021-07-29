"""
 Env: /anaconda3/python3.7
 Time: 2021/7/15 21:13
 Author: karlieswfit
 File: one_hot_scatter.py
 Describe: 
"""

import torch

target=torch.tensor([0,1,0,2])
x=torch.zeros(4, 3)
print(x)
'''
tensor([[0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.]])
'''
x=x.scatter(1,target.reshape(len(target),-1),1)
print(x)
'''
tensor([[1., 0., 0.],
        [0., 1., 0.],
        [1., 0., 0.],
        [0., 0., 1.]])
'''

target=torch.tensor([0,1,0,2])
x=torch.zeros(4, 3)
x=x.scatter(0,target.reshape(len(target),-1),1)
print(x)
'''
tensor([[1., 0., 0.],
        [1., 0., 0.],
        [1., 0., 0.],
        [0., 0., 0.]])
'''