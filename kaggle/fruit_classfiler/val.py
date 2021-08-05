"""
 Env: /anaconda3/python3.7
 Time: 2021/8/5 14:13
 Author: karlieswfit
 File: val.py
 Describe: 通过模型加载进行数据预测
"""

import torchvision
import torch.nn as nn
import torch
import os
from PIL import Image
import matplotlib.pyplot as plt


from fruit_classfiler import Fruit_vgg16 as vgg

name=['freshapple','freshbanana','freshoranges','rottenapple','rottenbanana','rottenoranges',]


model=vgg.Net()
optimizer=torch.optim.Adam(model.parameters(),lr=0.001,weight_decay=0.0001)
if os.path.exists('./vgg_model/model.pkl'):
    model.load_state_dict(torch.load('./vgg_model/model.pkl'))
    optimizer.load_state_dict(torch.load('./vgg_model/optimizer.pkl'))


data=Image.open('./data/freshapple.png')
plt.imshow(data)
plt.show()

data=Image.open('./data/freshapple.png')
data=torchvision.transforms.ToTensor()(data)
input=torch.unsqueeze(data,dim=0)
output=model(input)
print(output)
index=torch.max(output,dim=1)[1]
print(name[index])


data=Image.open('./data/rottenapple.png')
data=torchvision.transforms.ToTensor()(data)
input=torch.unsqueeze(data,dim=0)
output=model(input)
print(output)
index=torch.max(output,dim=1)[1]
print(name[index])













