"""
 Env: /anaconda3/python3.7
 Time: 2021/8/5 13:26
 Author: karlieswfit
 File: cnn可视化图层.py
 Describe: 
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision
from PIL import Image
from torchvision.models import vgg16

device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
transform=torchvision.transforms.Compose([
    torchvision.transforms.Resize((100,100)),
    torchvision.transforms.RandomHorizontalFlip(p=0.5),
    torchvision.transforms.RandomRotation(30), #随机旋转30度
    torchvision.transforms.ToTensor()
])
train_dataset=torchvision.datasets.ImageFolder(root=r'C:\Users\karlieswift\Python\kaggle\data\archive\dataset\dataset\train',transform=transform)
train_dataloader=torch.utils.data.DataLoader(train_dataset,batch_size=32,shuffle=True)

# 可视化
# for i,(input,target) in enumerate(train_dataloader):
#     for j in range(len(target)):
#         image=input[j]  #[3, 100, 100]
#         label=target[j]
#         plt.subplot(4,8,j+1)
#         image=np.transpose(image,(1,2,0))
#         plt.imshow(image)
#     plt.show()
#     break

class VggNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg16 = vgg16(pretrained=True)
        self.my_classifier = nn.Sequential(
            nn.Linear(in_features=25088, out_features=100),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=100, out_features=6),
            nn.Sigmoid()
        )
        self.vgg16.classifier = self.my_classifier
        # 冻结vgg16 classifier前的参数
        for param in self.vgg16.parameters():
            param.requires_grad = False
        for param in self.vgg16.classifier.parameters():
            param.requires_grad = True

    def forward(self, input):
        output = self.vgg16(input)
        return output


#实例化model
model=VggNet().to(device)
crossEntropyLoss=nn.CrossEntropyLoss()
#在优化器里加入正则化weight_decay L2正则项系数
optimizer=torch.optim.Adam(model.parameters(),lr=0.001, weight_decay=0.0001)

loss_list=[]
acc_list=[]
def train(epoch):
    for i,(input,target) in enumerate(train_dataloader):
        input = input.to(device)
        target = target.to(device)
        output=model(input)
        loss = crossEntropyLoss(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())
        pre=torch.max(output,dim=-1)[1]
        acc=(pre==target).sum()/(len(target))
        acc_list.append(acc)
        print("第",epoch+1,"次训练：loss:",loss.item(),',acc:',acc.item())
        break

for i in range(6):
    train(i)


class LayerActivation():
    features = None

    def __init__(self, model, layer_num):
        self.hook = model[layer_num].register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features = output.cpu()

    def remove(self):
        self.hook.remove()

data=Image.open('./data/1.png')
#需要将Image格式转换为tensor
input_image=torchvision.transforms.ToTensor()(data)
# input_image=torchvision.transforms.Resize((50,50))(input_image)
#这里的input_image是三维数据，需要加一维(bacth_size)
input_image=torch.unsqueeze(input_image,0)

#实例化可视化
conv_out=LayerActivation(model.vgg16.features,0) #第0层
#在调用register_forward_hook函数之前，需要做一次前向传播forward
o=model.vgg16(torch.autograd.Variable(input_image.cuda()))
#移除hook
conv_out.remove()
# 把register_forward_hook捕捉的信息传给act
act=conv_out.features

#对act 进行可视化
act=act.detach()
fig=plt.figure(figsize=(20,50))
fig.subplots_adjust(left=0,right=1,bottom=0,top=0.8,hspace=0,wspace=0.2)
for i in range(30):
    ax=fig.add_subplot(12,5,i+1,xticks=[],yticks=[])
    ax.imshow(act[0][i])
plt.show()





