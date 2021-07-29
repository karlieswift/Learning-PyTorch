"""
 Env: /anaconda3/python3.7
 Time: 2021/7/23 10:25
 Author: karlieswfit
 File: embedding.py
 Describe: word embedding:将单词转化为vector
"""

import torch.nn as nn
import torch
'''
Embedding的理解
'''
em=nn.Embedding(100,3)  #100 代表我们的词库有多少的单词 3代表一个单词用一个大小为3的向量表示
print(torch.randint(1,100,[2, 4]))
x=torch.randint(1,100,[2, 4])  #随机生成[2,4] 2代表有2句话，每句话4个单词
print(em(x))
print(em(x).shape)  #torch.Size([2, 4, 3])

"""
所有的输出
tensor([[ 6, 42, 38, 76],
        [23, 90, 61, 35]])
tensor([[[-0.0918, -1.1488,  0.1969],
         [-1.2042,  0.3921, -0.6837],
         [-1.0383, -0.0724,  0.3366],
         [-0.7508, -0.3722, -2.3488]],

        [[ 0.6909,  2.0500,  0.3942],
         [ 0.3200,  0.8164, -0.2128],
         [-0.7853, -0.5349,  1.3111],
         [-1.1541,  0.6997,  0.3449]]], grad_fn=<EmbeddingBackward>)
torch.Size([2, 4, 3])
"""


