"""
 Env: /anaconda3/python3.7
 Time: 2021/7/14 11:31
 Author: karlieswfit
 File: 手写数字识别.py
 Describe: 基本思路：1-加载数据 2-模型构建 3-模型的训练 4-模型保存 5-模型评估
"""
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision.transforms import Normalize, ToTensor, Compose
from torch import nn
from torch.functional import F
from torch.optim import Adam
import os
import torch
import numpy as np

BATCH_SIZE = 128


# 1-准备数据集
def get_dataloader(train=True):
    tranform_fn = Compose([
        ToTensor(),
        Normalize(mean=(0.1307,), std=(0.3081,))  # mean和std的形状和通道数
    ])
    data_set = MNIST(root='./data', train=train, transform=tranform_fn)
    data_loader = DataLoader(dataset=data_set, batch_size=BATCH_SIZE, shuffle=True)
    return data_loader


# 2-构建模型
'''
激活函数relu(x)=max(0,x)
'''


class MnistModel(nn.Module):
    def __init__(self):
        super(MnistModel,self).__init__()
        self.fc1 = nn.Linear(1 * 28 * 28, 28)
        self.fc2 = nn.Linear(28, 10)

    def forward(self, input):
        # 1-改变数据的形状[batch_size,1,28,28]——>[batch_size,28*28]
        x = input.view(-1, 28 * 28)
        # 2-全连接层
        x = self.fc1(x)
        # 3-激活函数
        x = F.relu(x)
        # 4-全连接层
        out = self.fc2(x)
        return F.log_softmax(out,dim=-1)


modle=MnistModel()  #实例化模型
optimizer=Adam(modle.parameters(),lr=0.001)
if os.path.exists('./model/model.pkl'):
    modle.load_state_dict(torch.load('./model/model.pkl'))
    optimizer.load_state_dict(torch.load('./model/optimizer.pkl'))

#3-模型的训练
def train(epoch):
    data=get_dataloader() #获取数据
    for index,(input,target) in enumerate(data):
        optimizer.zero_grad() #梯度归零
        output=modle(input)
        loss=F.nll_loss(output,target)
        loss.backward() #反向传播
        optimizer.step() #参数更新
        if index%10==0:
            print(epoch,index,loss.item())

        #模型的保存 每个100次保存一次
        if index%100==0:
            torch.save(modle.state_dict(),'./model/model.pkl')
            torch.save(optimizer.state_dict(),'./model/optimizer.pkl')


#4-模型的评估
def test():
    acc_list=[]
    loss_list=[]
    '''
    在使用pytorch时，并不是所有的操作都需要进行计算图的生成（计算过程的构建，以便梯度反向传播等操作）。
    而对于tensor的计算操作，默认是要进行计算图的构建的，在这种情况下，可以使用 with torch.no_grad():，
    强制之后的内容不进行计算图构建。
    '''
    with torch.no_grad():
        test_dataloader=get_dataloader(train=False)
        for index,(input,target) in enumerate(test_dataloader):
            output=modle(input)
            # print(max(output[0])) #tensor(-0.0358)
            # print(target.shape) #torch.Size([128])
            # print(output.shape) #torch.Size([128, 10])
            loss=F.nll_loss(output,target)
            loss_list.append(loss)
            pred=output.max(dim=-1)[-1] #max返回一个元组(values,indices) values代表最大的值，indices代表最大值所在的index
            # print(pred.shape)  #torch.Size([128]) 128个0-9数字
            cur_acc=pred.eq(target).float().mean()
            acc_list.append(cur_acc)
    print("acc=",np.mean(acc_list),"loss=",np.mean(loss_list))  #acc= 0.9532239 loss= 0.15750977


if __name__ == '__main__':
    # for i in range(3):  #训练三次
    #     train(i)
    test()












