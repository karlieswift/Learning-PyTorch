"""
 Env: /anaconda3/python3.7
 Time: 2021/7/16 17:15
 Author: karlieswfit
 File: 手写数字识别3.py
 Describe: CrossEntropyLoss和NLLLoss不需要one-hot
"""



import  torch
from torch import  nn
from torch.functional import F
from torch.utils.data import DataLoader
from torchvision import transforms,datasets

device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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
        self.fc1=nn.Linear(28*28,10)
        self.softmax=nn.Softmax(dim=1)

    def forward(self,x):
        # x的形状[batchsize,1,28,28]
        x=x.view(x.size()[0],-1) #x.size()[0]=BATCH_SIZE
        # print(x.shape) # torch.Size([64, 784])
        x=self.fc1(x)
        # print(x.shape)  # torch.Size([64, 10])
        output=self.softmax(x)
        # output是一个[64,10]代表64张图片,每一行代表一个图片 每一行的总和为 1即概率
        # print(output.shape) # torch.Size([64, 10])
        return output

#实例化model
model=Net().to(device)
mes_loss=nn.MSELoss()
crossEntropyLoss=nn.CrossEntropyLoss()
optimizer=torch.optim.SGD(model.parameters(),lr=0.05)


def train(epoch):
    for i,(input,target) in enumerate(train_loader):
        # GPU
        input = input.to(device)
        target = target.to(device)

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
        print("第",epoch+1,"次训练：loss:",loss.item())


def test():
    acc_num=0  #记录预测正确的个数
    for i, (input, target) in enumerate(test_loader):
        # GPU
        input = input.to(device)
        target = target.to(device)
        output=model(input)
        #output是一个[64,10]的矩阵 取出每一行的最大的概率值 也就是预测值对应的位置
        #解释:torch.max() 返回元组(values,indices) indices代表max的位置
        #这里返回的位置indices0-9位置 正好对应图片类别的0-9
        index=torch.max(output,dim=1)[1] #取出某行图片最大概率对应的位置 正好对应图片的类别
        acc_num+=(index == target).sum()

    print("正确录acc:", acc_num / len(test_data))


if __name__ == '__main__':
    for i in range(3):
        train(i)

    print("---------------------------")
    test()










