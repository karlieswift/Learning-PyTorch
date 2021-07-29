"""
 Env: /anaconda3/python3.7
 Time: 2021/7/24 12:22
 Author: karlieswfit
 File: model.py
 Describe: torch.nn.LSTM(input_size,hidden_size,num_layers,batch_first,dropout,bidirectional)
"""
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from lib import max_len,ws
from dataset import get_dataloader
import lib





class MyModel(nn.Module):
    def __init__(self):
        super(MyModel,self).__init__()
        self.embedding=nn.Embedding(len(ws),100)  #len(ws)=所有单词,100代表一个单词用一个多大的向量表示

        self.fc=nn.Linear(max_len*100,2)

    def forward(self,input):
        #input的大小[10, 20]，是get_dataloader()得到的数据，10=batch_size，20=max_len最大的单词数量
        x=self.embedding(input)#embedding后数据大小 [10, 20, 100] 10代表一个批次大小 20代表一个列表有几个单词(也就是一句话我们提取单词的数量)
        # 100代表一个单词用一个多大的向量表示，例如hello=[0.1,2.2,1,0,0]
        #embedding后数据大小 [10, 20, 100]可以理解 一个批次处理10句话 ，每句话我们提取20个出现次数最多的单词，每个单词用一个大小为100的向量表示

        x=x.view(-1,max_len*100)#大小[10, 2000]
        out=self.fc(x) #大小[10, 2] 10=batch_size，2=分类的个数
        return F.log_softmax(out,dim=-1)


model=MyModel()
optmizer=Adam(model.parameters(),0.001)


def train(enpoch):
    for index,(input,target) in enumerate(get_dataloader()):

        optmizer.zero_grad()
        output=model(input)
        loss=F.nll_loss(output,target)
        loss.backward()
        optmizer.step()
        print(index,loss.item())



if __name__ == '__main__':
    for i in range(3):
        train(i)
        break