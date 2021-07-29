"""
 Env: /anaconda3/python3.7
 Time: 2021/7/13 19:45
 Author: karlieswfit
 File: transform实例.py
 Describe:
"""

from torchvision import transforms
import numpy as np

data=np.random.randint(0,255,size=24)
img=data.reshape(2,3,4)
print(img)
print(img.shape) #(2, 3, 4)

img_tensor=transforms.ToTensor()(img) #装置并且转化为tensor
print(img_tensor)
print(img_tensor.shape)#torch.Size([4, 2, 3])

from torchvision.datasets import MNIST

data=MNIST('./data',train=True,download=False)
print(transforms.ToTensor()(data[0][0]).shape)  #torch.Size([1, 28, 28])

