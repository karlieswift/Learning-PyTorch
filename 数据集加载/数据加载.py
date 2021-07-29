"""
 Env: /anaconda3/python3.7
 Time: 2021/7/11 13:32
 Author: karlieswfit
 File: 数据加载.py
 Describe: 对数据的加载 处理
"""

import torch
from torch.utils.data import Dataset,DataLoader

data_path=r'D:\Project\Python\torchData\SMSSpamCollection'

#数据集的加载
class MyDataSet(Dataset):
    def __init__(self):
        self.lines=open(data_path,'r',encoding='UTF-8').readlines()

    def __getitem__(self, index):
        #获取对应的数据的一条的索引index
        cur_lines=self.lines[index].strip()  #获取当前行
        #对label和content进行分离
        label=cur_lines[:4].strip()
        content=cur_lines[4:].strip()
        return label,content

    def __len__(self):
        #返回数据的长度
        return len(self.lines)


mydata=MyDataSet()

# if __name__ == '__main__':
#     print(mydata[1000])
#     print(len(mydata))

data_loader=DataLoader(dataset=mydata,batch_size=2,shuffle=True)
if __name__ == '__main__':
    for i in data_loader:
        print(i)
        break
    print(len(data_loader))