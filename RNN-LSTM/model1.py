"""
 Env: /anaconda3/python3.7
 Time: 2021/7/25 16:27
 Author: karlieswfit
 File: model1.py
 Describe:
 torch.nn.LSTM(input_size,hidden_size,num_layers,batch_first,dropout,bidirectional)
 input_size `：输入数据的形状，即embedding_dim
 hidden_size=128 #隐藏层神经元的数量，即每一层有多少个LSTM单元
 num_layers=2 #即RNN的中LSTM单元的层数
 bidirectional=True #是否使用双向LSTM,默认是False
 dropout=0.4  是否在除最后一个RNN层外的RNN层后面加dropout层 dropout的比例，默认值为0。dropout是一种训练过程中让部分参数随机失活的一种方式，能够提高训练速度，同时能够解决过拟合的问题。
 batch_first=True  是否输入输出的第一维为batchsize 默认值为False，输入的数据需要`[seq_len,batch,feature]`,如果为True，则为`[batch,seq_len,feature]`
"""
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from lib import max_len, ws
from dataset import get_dataloader
import lib
import os
import torch
import numpy as np

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.embedding = nn.Embedding(len(ws), embedding_dim=100)  # len(ws)=所有单词,100代表一个单词用一个多大的向量表示
        # LSTM  input_size=embedding_dim=100
        self.lstm = nn.LSTM(input_size=100, hidden_size=lib.hidden_size, num_layers=lib.num_layers,
                            batch_first=lib.batch_first,
                            bidirectional=lib.bidirectional, dropout=lib.dropout)
        self.fc = nn.Linear(lib.hidden_size * 2, 2)

    def forward(self, input):
        # input的大小[10, 20]，是get_dataloader()得到的数据，10=batch_size，20=max_len最大的单词数量
        x = self.embedding(input)  # embedding后数据大小 [10, 20, 100] 10代表一个批次大小 20代表一个列表有几个单词(也就是一句话我们提取单词的数量)
        # 100代表一个单词用一个多大的向量表示，例如hello=[0.1,2.2,1,0,0]
        # embedding后数据大小 [10, 20, 100]可以理解 一个批次处理10句话 ，每句话我们提取20个出现次数最多的单词，每个单词用一个大小为100的向量表示

        x, (h_n, c_n) = self.lstm(x)  # x的大小[10, 20, 256] h_n和c_n的大小[4, 10, 128] 这里的4=num_layers*2 双向bidirectional=True
        # h_n=W.tanh(c_n)

        # 获取两个方向最后一次的output，进行contact
        #[1,2,3,4]==[-4,-3,-2,-1]=[第一层正向，第一层反向，第二层正向，第二层反向]
        output_fw = h_n[-2, :, :]  # 正向最后一次输出 大小[10, 128]
        output_bw = h_n[-1, :, :]  # 反向最后一次输出 大小[10, 128]
        output = torch.cat([output_fw, output_bw], dim=-1)  # output的大小[10, 256]
        out = self.fc(output)  # 大小[10, 2] 10=batch_size，2=分类的个数
        return F.log_softmax(out, dim=-1)


model = MyModel().to(lib.device)
optmizer = Adam(model.parameters(), 0.001)

if os.path.exists('./model/model.pkl'):
    model.load_state_dict(torch.load('./model/model.pkl'))
    optmizer.load_state_dict(torch.load('./model/optmizer.pkl'))


def train(enpoch):
    for index, (input, target) in enumerate(get_dataloader()):

        # GPU
        input = input.to(lib.device)
        target = target.to(lib.device)

        optmizer.zero_grad()
        output = model(input)
        loss = F.nll_loss(output, target)
        loss.backward()
        optmizer.step()
        print(enpoch, index, loss.item())

        if index % 100 == 0:
            torch.save(model.state_dict(), './model/model.pkl')
            torch.save(optmizer.state_dict(), './model/optmizer.pkl')


def eval():
    loss_list = []
    acc_list = []
    for index, (input, target) in enumerate(get_dataloader(train=False)):
        input = input.to(lib.device)
        target = target.to(lib.device)
        with torch.no_grad():
            output = model(input)
            cur_loss = F.nll_loss(output, target)
            loss_list.append(cur_loss.cpu().item())
            # 准确率
            pred = output.max(dim=-1)[-1]
            acc = pred.eq(target).float().mean()
            acc_list.append(acc.cpu().item())

    print("total_loss,acc", np.mean(loss_list), np.mean(acc_list))
    # total_loss,acc 0.6349090121507645 0.6300800080955029


if __name__ == '__main__':
    for i in range(3):
        train(i)
        break
    eval()
