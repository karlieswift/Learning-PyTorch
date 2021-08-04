"""
 Env: /anaconda3/python3.7
 Time: 2021/8/4 14:04
 Author: karlieswfit
 File: fruit_classifier.py
 Describe: 数据来源：https://www.kaggle.com/sriramr/fruits-fresh-and-rotten-for-classification

 数据加载
 模型构造
 模型训练
 模型评估
"""

import torch
import torch.nn as nn
import torchvision
import os

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
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d((32)),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d((64)),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        # self.conv3 = nn.Sequential(
        #     nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0),
        #     nn.BatchNorm2d((128)),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2, 2)
        # )

        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=33856, out_features=6),
            nn.Sigmoid()
        )

    def forward(self, input):
        conv1_out = self.conv1(input)
        conv2_out = self.conv2(conv1_out)
        # conv3_out = self.conv3(conv2_out)
        output = self.fc1(conv2_out)
        return output


# 模型训练
#实例化model
model=Net().to(device)
#在优化器里加入正则化weight_decay L2正则项系数
optimizer=torch.optim.Adam(model.parameters(),lr=0.001, weight_decay=0.0001)
if os.path.exists('model/optimizer.pkl'):
    model.load_state_dict(torch.load('./model/model.pkl'))
    optimizer.load_state_dict(torch.load('./model/optimizer.pkl'))
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
            torch.save(model.state_dict(),'./model/model.pkl')
            torch.save(optimizer.state_dict(),'./model/optimizer.pkl')



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
    for i in range(10):
        train(i)
        break
    test()



# from torchsummary import summary
# summary(model=model,input_size=[(3,100,100)],batch_size=2,device='cuda')

'''
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [2, 32, 98, 98]             896
       BatchNorm2d-2            [2, 32, 98, 98]              64
              ReLU-3            [2, 32, 98, 98]               0
         MaxPool2d-4            [2, 32, 49, 49]               0
            Conv2d-5            [2, 64, 47, 47]          18,496
       BatchNorm2d-6            [2, 64, 47, 47]             128
              ReLU-7            [2, 64, 47, 47]               0
         MaxPool2d-8            [2, 64, 23, 23]               0
           Flatten-9                 [2, 33856]               0
           Linear-10                     [2, 6]         203,142
          Sigmoid-11                     [2, 6]               0
================================================================
Total params: 222,726
Trainable params: 222,726
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.23
Forward/backward pass size (MB): 22.75
Params size (MB): 0.85
Estimated Total Size (MB): 23.82
----------------------------------------------------------------
'''
