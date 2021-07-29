"""
 Env: /anaconda3/python3.7
 Time: 2021/7/28 10:41
 Author: karlieswfit
 File: lib.py
 Describe:
"""


import pickle

max_len=20
ws=pickle.load(open('./model/ws.pkl','rb'))



 # torch.nn.LSTM(input_size,hidden_size,num_layers,batch_first,dropout,bidirectional)

hidden_size=128 #隐藏层神经元的数量，即每一层有多少个LSTM单元
num_layers=2 #即RNN的中LSTM单元的层数
bidirectional=True #是否使用双向LSTM,默认是False
dropout=0.4
batch_first=True
import torch
device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')