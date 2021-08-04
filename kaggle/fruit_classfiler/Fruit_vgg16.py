"""
 Env: /anaconda3/python3.7
 Time: 2021/8/4 18:11
 Author: karlieswfit
 File: Fruit_vgg16.py
 Describe: 数据来源：https://www.kaggle.com/sriramr/fruits-fresh-and-rotten-for-classification
"""


import torch
import torch.nn as nn
import torchvision
import os
from torchvision.models import vgg16

device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


batch_size=64
# 数据加载
transform=torchvision.transforms.Compose([
    torchvision.transforms.Resize((100,100)),
    torchvision.transforms.RandomHorizontalFlip(p=0.5),
    torchvision.transforms.RandomRotation(30), #随机旋转30度
    torchvision.transforms.ToTensor()
])
train_dataset=torchvision.datasets.ImageFolder(root=r'C:\Users\karlieswift\Python\kaggle\data\archive\dataset\dataset\train',transform=transform)
train_dataloader=torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True)

test_dataset=torchvision.datasets.ImageFolder(root=r'C:\Users\karlieswift\Python\kaggle\data\archive\dataset\dataset\test',transform=transform)
test_dataloader=torch.utils.data.DataLoader(test_dataset,batch_size=batch_size,shuffle=True)

# 模型构造
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg16=vgg16(pretrained=True)  #第一次运行需要联网下载
        self.my_classifier=nn.Sequential(
            nn.Linear(in_features=25088, out_features=100),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=100, out_features=6),
            nn.Sigmoid()
        )
        self.vgg16.classifier=self.my_classifier
        # 冻结vgg16 classifier前的参数
        for param in self.vgg16.parameters():
            param.requires_grad=False
        for param in self.vgg16.classifier.parameters():
            param.requires_grad=True


    def forward(self, input):
        output = self.vgg16(input)
        return output


# 模型训练
#实例化model
model=Net().to(device)
#在优化器里加入正则化weight_decay L2正则项系数
optimizer=torch.optim.Adam(model.parameters(),lr=0.001, weight_decay=0.0001)
if os.path.exists('vgg_model/optimizer.pkl'):
    model.load_state_dict(torch.load('./vgg_model/model.pkl'))
    optimizer.load_state_dict(torch.load('./vgg_model/optimizer.pkl'))
crossEntropyLoss=nn.CrossEntropyLoss()

def train(epoch):
    for i,(input,target) in enumerate(train_dataloader):
        input = input.to(device)
        target = target.to(device)
        output=model(input)
        loss = crossEntropyLoss(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if(i%10==0):
            print("第", epoch + 1, "次训练：loss:", loss.item())
            torch.save(model.state_dict(),'./vgg_model/model.pkl')
            torch.save(optimizer.state_dict(),'./vgg_model/optimizer.pkl')



def test():
    acc_sum=0
    for i,(input,target) in enumerate(test_dataloader):
        input = input.to(device)
        target = target.to(device)
        output=model(input)
        y_pre=torch.max(output, dim=1)[1]
        acc_sum=acc_sum+(y_pre == target).sum()

    acc =acc_sum/len(test_dataset)
    print("正确率:",acc.item())


# 模型评估
if __name__ == '__main__':
    # for i in range(1):
    #     train(i)
    #     break
    test()

