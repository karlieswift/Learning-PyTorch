"""
 Env: /anaconda3/python3.7
 Time: 2021/7/25 12:48
 Author: karlieswfit
 File: lstm1.py
 Describe:
 torch.nn.LSTM(input_size,hidden_size,num_layers,batch_first,dropout,bidirectional)
1. input_size `：输入数据的形状，即embedding_dim
2. `hidden_size`：隐藏层神经元的数量，即每一层有多少个LSTM单元
3. `num_layer` ：即RNN的中LSTM单元的层数
4. `batch_first`：默认值为False，输入的数据需要`[seq_len,batch,feature]`,如果为True，则为`[batch,seq_len,feature]`
5. `dropout`:dropout的比例，默认值为0。dropout是一种训练过程中让部分参数随机失活的一种方式，能够提高训练速度，同时能够解决过拟合的问题。
这里是在LSTM的最后一层，对每个输出进行dropout
6. `bidirectional`：是否使用双向LSTM,默认是False
"""

import torch.nn as nn
import torch

batch_size = 10
seq_len = 20  # 句子的长度
vocab_size = 100  # 词典的数量
embedding_dim = 30  # 一个单词的向量大小
hidden_size = 18
num_layers = 2

# 创建一个batch数据input
input = torch.randint(low=0, high=100, size=[batch_size, seq_len])
print("1", input.size())  # torch.Size([10, 20])  10句话 每句话20个单词
# 将input数据进行embedding处理
embedding = nn.Embedding(vocab_size, embedding_dim)
input_embedding = embedding(input)
print("2", input_embedding.size())  # torch.Size([10, 20, 30])

# 将数据传入LSTM
lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=num_layers, batch_first=True,bidirectional=False)
output, (h_n, c_n) = lstm(input_embedding)
print("3", output.size())  # torch.Size([10, 20, 18])
print("4", h_n.size())  # torch.Size([2, 10, 18])  num_layers=2
print("5", c_n.size())  # torch.Size([2, 10, 18])
# 最后一次的h_1应该和output的最后一个time step的输出是一样的
#获取最后一个时间步的输出
last_output=output[:,-1,:]  #大小[10, 18]
#获取最后一次hidden_state
last_state=h_n[-1,:,:]    #大小[10, 18]
print(last_output==last_state)  #全部为True


# 将数据传入LSTM
lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=num_layers, batch_first=True,bidirectional=True)
output, (h_n, c_n) = lstm(input_embedding)
print("6", output.size())  # torch.Size([10, 20, 36])
print("7", h_n.size())  # torch.Size([4, 10, 18])  4=2*num_layers=2*2
print("8", c_n.size())  # torch.Size([4, 10, 18])





