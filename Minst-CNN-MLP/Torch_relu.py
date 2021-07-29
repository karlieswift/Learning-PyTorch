"""
 Env: /anaconda3/python3.7
 Time: 2021/7/14 12:55
 Author: karlieswfit
 File: Torch_relu.py
 Describe: relu(x)=max(0,x)
"""

import torch.nn.functional as F
import torch
data=torch.tensor([-2,-1,0,1,2])
print(data)  #tensor([-2, -1,  0,  1,  2])

print(F.relu(data))#tensor([0, 0, 0, 1, 2])
