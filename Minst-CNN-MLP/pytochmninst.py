"""
 Env: /anaconda3/python3.7
 Time: 2021/7/13 19:35
 Author: karlieswfit
 File: pytochmninst.py
 Describe: pytroch自带的2个数据集api torchvision 和 torchtext
"""

'''
MNIST(
            self,
            root: str,  #数据存放路径
            train: bool = True, #是否用训练数据集进行测试
            transform: Optional[Callable] = None, #实现对图片的处理函数
            target_transform: Optional[Callable] = None,
            download: bool = False,#是否需要下载到root目录
    ) 
'''

from torchvision.datasets import MNIST

data_mnist=MNIST(root='./data',train=True,download=True)

# print(data_mnist)
print(data_mnist[0]) #(<PIL.Image.Image image mode=L size=28x28 at 0x17067733FD0>, 5)
data_mnist[0][0].show() #显示图片


