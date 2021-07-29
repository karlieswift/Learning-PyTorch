"""
 Env: /anaconda3/python3.7
 Time: 2021/7/17 19:04
 Author: karlieswfit
 File: Dropout和正则化.py
 Describe: 
"""


import  torch
from torch import  nn
from torch.utils.data import DataLoader
from torchvision import transforms,datasets

BATCH_SIZE=64
#加载数据集mnist
train_data=datasets.MNIST(root='./data',train=True,transform=transforms.ToTensor(),download=True)
test_data=datasets.MNIST(root='./data',train=False,transform=transforms.ToTensor(),download=True)

#分批次传入
train_loader=DataLoader(train_data,batch_size=BATCH_SIZE,shuffle=True)
test_loader=DataLoader(test_data,batch_size=BATCH_SIZE,shuffle=True)

#构建模型
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        #Dropout(p=0.5) p=0.5 代表舍弃的神经元 p=舍弃率 初始化0
        self.layer1=nn.Sequential(nn.Linear(28*28,500),nn.Dropout(p=0.5),nn.Tanh())
        self.layer2=nn.Sequential(nn.Linear(500,300),nn.Dropout(p=0.5),nn.Tanh())
        self.layer3=nn.Sequential(nn.Linear(300,10),nn.Softmax(dim=1))

    def forward(self,x):
        # x的形状[batchsize,1,28,28]
        x=x.view(x.size()[0],-1) #x.size()[0]=BATCH_SIZE
        x=self.layer1(x)  #torch.Size([64, 500])
        x=self.layer2(x)
        output=self.layer3(x)
        return output

#实例化model
model=Net()
mes_loss=nn.MSELoss()
crossEntropyLoss=nn.CrossEntropyLoss()
#在优化器里加入正则化weight_decay L2正则项系数
optimizer=torch.optim.SGD(model.parameters(),lr=0.05, weight_decay=0.0001)


def train():
    model.train()  #模型的训练状态  这样在训练的时候Dropout起作用(也就是说神经元部分工作)
    for i,(input,target) in enumerate(train_loader):
        output=model(input)  #torch.Size([64, 10])
        # target.shape=torch.Size([64])
        #1-MSELoss 这里使用MSELoss的时候需要将output和target的shape转化为同一尺寸,通过one-hot进行编码
        # target=target.reshape(-1,1)#torch.Size([64, 1])
        # one_hot=torch.zeros(target.size()[0],10).scatter(1,target,1)
        # loss=mes_loss(output,one_hot)

        #2-CrossEntropyLoss 使用CrossEntropyLoss的时候注意target的尺寸torch.Size([64]) 而不是torch.Size([64, 1])
        loss = crossEntropyLoss(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("loss:",loss.item())


def test():
    model.eval() #模型的测试状态  这样在训练的时候Dropout不起作用(也就是说神经元全部工作)

    #todo 测试集的准确率
    acc_num=0  #记录预测正确的个数
    for i, (input, target) in enumerate(test_loader):
        output=model(input)
        #output是一个[64,10]的矩阵 取出每一行的最大的概率值 也就是预测值对应的位置
        #解释:torch.max() 返回元组(values,indices) indices代表max的位置
        #这里返回的位置indices0-9位置 正好对应图片类别的0-9
        index=torch.max(output,dim=1)[1] #取出某行图片最大概率对应的位置 正好对应图片的类别
        acc_num+=(index == target).sum()

    print("测试数据正确录acc:", acc_num / len(test_data))

    # todo 训练集的准确率
    acc_num = 0  # 记录预测正确的个数
    for i, (input, target) in enumerate(train_loader):
        output = model(input)
        index = torch.max(output, dim=1)[1]
        acc_num += (index == target).sum()

    print("训练数据正确录acc:", acc_num / len(train_data))


if __name__ == '__main__':
    for i in range(6):
        train()

    print("---------------------------")
    test()



