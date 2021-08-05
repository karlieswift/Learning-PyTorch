"""
 Env: /anaconda3/python3.7
 Time: 2021/8/5 14:08
 Author: karlieswfit
 File: vgg16迁移学习.py
 Describe: vgg16模型
"""

import torch.nn as nn
from torchvision.models import vgg16
from torchsummary import summary



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
        # todo 让前三层参与参数更新
        for i in range(3):
            for param in self.vgg16.features[i].parameters():
                param.requires_grad = True

        for param in self.vgg16.classifier.parameters():
            param.requires_grad = True

    def forward(self, input):
        output = self.vgg16(input)
        return output

model=VggNet()
print(summary(model=model, input_size=[(3, 100, 100)], batch_size=32, device='cpu'))