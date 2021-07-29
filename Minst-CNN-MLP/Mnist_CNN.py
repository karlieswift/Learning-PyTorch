"""
 Env: /anaconda3/python3.7
 Time: 2021/7/19 17:38
 Author: karlieswfit
 File: Mnist_CNN.py
 Describe:nn.Conv2d(in_channels=1,out_channels=32,kernel_size=(5,5),stride=1,padding=2)
 in_channels:上一层图片的通道数(原始图片是彩色为3，黑白色为1)
 out_channels：可以理解为过滤器(卷积核)的数目 ，通过卷积后的结果叫做特征图
 kernel_size：卷积核的大小用元组表示(3,4) ，如果是方阵直接写大小n
 stride:卷积核移动的步长
 padding：在图片外层填充几层0
"""


import  torch
from torch import  nn
from torch.utils.data import DataLoader
from torchvision import transforms,datasets

device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE=64
#加载数据集mnist 28*28的数据
train_data=datasets.MNIST(root='./data',train=True,transform=transforms.ToTensor(),download=True)
test_data=datasets.MNIST(root='./data',train=False,transform=transforms.ToTensor(),download=True)

#分批次传入
train_loader=DataLoader(train_data,batch_size=BATCH_SIZE,shuffle=True)
test_loader=DataLoader(test_data,batch_size=BATCH_SIZE,shuffle=True)

#构建模型
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        # nn.Conv2d(in_channels=1,out_channels=32,kernel_size=(5,5),stride=1,padding=2)
        self.conv1=nn.Sequential(nn.Conv2d(1,32,5,1,2),nn.ReLU(),nn.MaxPool2d(2,2))
        #第一次卷积得到是一个14*14的特征图 过程：(28+padding*2)=32 然后32-(5-1)=28 变为28*28,经过MaxPool2d变为 14*14
        self.conv2=nn.Sequential(nn.Conv2d(32,64,5,1,2),nn.ReLU(),nn.MaxPool2d(2,2))
        # 第二次卷积得到的7*7的特征图 过程：(14+padding*2)=18 然后18-(5-1)=14 变为14*14,经过MaxPool2d变为 7*7
        self.fc1=nn.Sequential(nn.Linear(64*7*7,1000),nn.Dropout(p=0.5),nn.ReLU())
        self.fc2=nn.Sequential(nn.Linear(1000,10),nn.Softmax(dim=1))

    def forward(self,x):
        # x的形状[64,1,28,28]
        x=self.conv1(x)  #[64, 32, 14, 14]
        x=self.conv2(x) #[64, 64, 7, 7]
        x = x.view(x.size()[0], -1)  # 得到的x的大小[64, 3136] 其中x.size()[0]=BATCH_SIZE
        x=self.fc1(x) #[64, 1000]
        output = self.fc2(x)   #[64, 10]
        return output

#实例化model
model=Net().to(device)
mes_loss=nn.MSELoss()
crossEntropyLoss=nn.CrossEntropyLoss()
#在优化器里加入正则化weight_decay L2正则项系数
optimizer=torch.optim.Adam(model.parameters(),lr=0.001, weight_decay=0.0001)


def train(epoch):
    model.train()  #模型的训练状态  这样在训练的时候Dropout起作用(也就是说神经元部分工作)
    for i,(input,target) in enumerate(train_loader):
        #input的大小[64, 1, 28, 28] target的大小[64]
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
    model.eval() #模型的测试状态  这样在训练的时候Dropout不起作用(也就是说神经元全部工作)

    #todo 测试集的准确率
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

    print("测试数据正确录acc:", acc_num / len(test_data))

    # todo 训练集的准确率
    acc_num = 0  # 记录预测正确的个数
    for i, (input, target) in enumerate(train_loader):
        # GPU
        input = input.to(device)
        target = target.to(device)
        output = model(input)
        index = torch.max(output, dim=1)[1]
        acc_num += (index == target).sum()

    print("训练数据正确录acc:", acc_num / len(train_data))


if __name__ == '__main__':
    for i in range(3):
        train(i)

    print("---------------------------")
    test()



